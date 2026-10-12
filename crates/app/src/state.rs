//! Application state: portable folders, persisted settings, open document, loaded model.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64};
use std::sync::{Arc, Mutex};

use openscissors_core::document::Document;
use openscissors_core::model::{Embeddings, ModelRegistry, ModelSpec, ModelState, Predictor, Sam2};
pub use openscissors_core::settings::Settings;
use serde::Serialize;

/// Environment variable pointing to a local ONNX file (development only).
pub const LOCAL_MODEL_ENV: &str = "OPENSCISSORS_MODEL";
pub const LOCAL_MODEL_ID: &str = "local-file";

/// Everything the app stores lives under `root`, so the folder can be moved or copied.
#[derive(Clone, Debug)]
pub struct Paths {
    pub root: PathBuf,
    pub data: PathBuf,
    pub models: PathBuf,
    pub plugins: PathBuf,
    /// Scratch files of the application and of the plug-ins it starts; emptied at each start.
    pub temp: PathBuf,
}

impl Paths {
    pub fn resolve() -> Self {
        let root = if let Ok(home) = std::env::var("OPENSCISSORS_HOME") {
            PathBuf::from(home)
        } else if cfg!(debug_assertions) {
            Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..").join("portable-dev")
        } else {
            let exe_dir =
                std::env::current_exe().ok().and_then(|p| p.parent().map(Path::to_path_buf)).unwrap_or_else(|| PathBuf::from("."));
            if is_writable(&exe_dir) {
                exe_dir
            } else {
                fallback_dir()
            }
        };
        let data = root.join("data");
        let models = root.join("models");
        let plugins = root.join("plugins");
        let temp = root.join("temp");
        let _ = std::fs::create_dir_all(&data);
        let _ = std::fs::create_dir_all(&models);
        Self { root, data, models, plugins, temp }
    }
}

/// Keep every scratch file inside the application folder: `temp/` becomes the temporary folder of the
/// process (and of the plug-ins it starts). Call once, before any other thread exists.
pub fn use_portable_temp() {
    let temp = Paths::resolve().temp;
    // Leftovers of earlier runs go; recent files stay, another copy of the application may be using them.
    let old = std::time::SystemTime::now() - std::time::Duration::from_secs(24 * 3600);
    for entry in std::fs::read_dir(&temp).into_iter().flatten().flatten() {
        if entry.metadata().and_then(|m| m.modified()).is_ok_and(|t| t < old) {
            let p = entry.path();
            let _ = if p.is_dir() { std::fs::remove_dir_all(p) } else { std::fs::remove_file(p) };
        }
    }
    if std::fs::create_dir_all(&temp).is_ok() {
        for var in ["TMP", "TEMP", "TMPDIR"] {
            std::env::set_var(var, &temp);
        }
    }
}

fn is_writable(dir: &Path) -> bool {
    let probe = dir.join(".osc-write-test");
    let ok = std::fs::write(&probe, b"").is_ok();
    let _ = std::fs::remove_file(probe);
    ok
}

/// Only used when the folder of the executable cannot be written (a read-only drive, Program Files):
/// the application then keeps its data in the user's profile instead of failing.
fn fallback_dir() -> PathBuf {
    let base = std::env::var_os("LOCALAPPDATA")
        .or_else(|| std::env::var_os("XDG_DATA_HOME"))
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local").join("share")))
        .unwrap_or_else(|| PathBuf::from("."));
    base.join("OpenScissors")
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct ModelStatus {
    pub active: Option<String>,
    pub loading: bool,
    pub ready: bool,
    pub error: Option<String>,
    /// "CPU" or "GPU": where the loaded model runs.
    pub backend: Option<String>,
}

pub struct AppState {
    pub paths: Paths,
    pub registry: ModelRegistry,
    pub settings: Mutex<Settings>,
    pub doc: Mutex<Option<Document>>,
    pub next_doc_id: AtomicU64,
    pub model: Mutex<Option<Arc<Predictor>>>,
    pub model_status: Mutex<ModelStatus>,
    /// SAM 2 (smart select), loaded on first use.
    pub sam: Mutex<Option<Arc<Sam2>>>,
    /// Encoder output for the document it was computed for.
    pub sam_cache: Mutex<Option<(u64, Arc<Embeddings>)>>,
    /// Previous reading of the processor meter: when, and the process CPU time then (ms).
    pub cpu_sample: Mutex<Option<(std::time::Instant, u64)>>,
    pub downloads: Mutex<HashMap<String, Arc<AtomicBool>>>,
    /// Cancel flag of the running batch, if any.
    pub batch: Mutex<Option<Arc<AtomicBool>>>,
    /// Document id a prediction is running (or was already run) for.
    pub predicted_doc: Mutex<Option<u64>>,
}

impl Default for AppState {
    fn default() -> Self {
        Self::new()
    }
}

impl AppState {
    pub fn new() -> Self {
        let paths = Paths::resolve();
        let settings = Settings::load(&paths.data.join("settings.json"));
        let registry = ModelRegistry::new(&paths.models);
        openscissors_core::sysmem::set_budget_percent(settings.ram_percent as u64);
        Self {
            paths,
            registry,
            settings: Mutex::new(settings),
            doc: Mutex::new(None),
            next_doc_id: AtomicU64::new(1),
            model: Mutex::new(None),
            model_status: Mutex::new(ModelStatus::default()),
            sam: Mutex::new(None),
            sam_cache: Mutex::new(None),
            cpu_sample: Mutex::new(None),
            downloads: Mutex::new(HashMap::new()),
            batch: Mutex::new(None),
            predicted_doc: Mutex::new(None),
        }
    }

    pub fn settings_path(&self) -> PathBuf {
        self.paths.data.join("settings.json")
    }

    pub fn save_settings(&self) {
        self.settings.lock().expect("settings lock").save(&self.settings_path());
    }

    /// Catalogue plus the development-only local file entry.
    pub fn models(&self) -> Vec<ModelSpec> {
        let mut specs = self.registry.available();
        if let Some(spec) = self.local_model_spec() {
            specs.push(spec);
        }
        specs
    }

    pub fn local_model_spec(&self) -> Option<ModelSpec> {
        std::env::var_os(LOCAL_MODEL_ENV)?;
        Some(ModelSpec {
            id: LOCAL_MODEL_ID.into(),
            name: "Local file (OPENSCISSORS_MODEL)".into(),
            description: "Model file given by the environment, for development.".into(),
            url: String::new(),
            sha256: String::new(),
            size_bytes: 0,
            license: "see file".into(),
            file_name: "model.onnx".into(),
            input_size: [1024, 1024],
            mean: [0.5, 0.5, 0.5],
            std: [1.0, 1.0, 1.0],
            input_name: None,
            output_name: None,
            output: openscissors_core::model::OutputNorm::Minmax,
            recommended: false,
            kind: openscissors_core::model::ModelKind::Segmenter,
            files: vec![],
            ram_mb: 0,
            folder: None,
        })
    }

    pub fn model_path(&self, spec: &ModelSpec) -> PathBuf {
        if spec.id == LOCAL_MODEL_ID {
            return PathBuf::from(std::env::var_os(LOCAL_MODEL_ENV).unwrap_or_default());
        }
        self.registry.model_path(spec)
    }

    pub fn is_ready(&self, spec: &ModelSpec) -> bool {
        if spec.id == LOCAL_MODEL_ID {
            return self.model_path(spec).is_file();
        }
        self.registry.state(spec) == ModelState::Ready
    }
}
