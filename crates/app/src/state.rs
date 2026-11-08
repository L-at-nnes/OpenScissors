//! Application state: portable folders, persisted settings, open document, loaded model.

use openscissors_core::sync::LockExt;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
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
        Self::under(openscissors_core::paths::root())
    }

    /// The folders under `root`, created when missing.
    pub fn under(root: PathBuf) -> Self {
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

/// The image encoding of smart select and the (document id, geometry) it was computed for.
pub type SamCache = ((u64, u64), Arc<Embeddings>);

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
    /// Keyed by the document id and its geometry (a turned picture needs a new encoding).
    pub sam_cache: Mutex<Option<SamCache>>,
    /// Previous reading of the processor meter: when, and the process CPU time then (ms).
    pub cpu_sample: Mutex<Option<(std::time::Instant, u64)>>,
    pub downloads: Mutex<HashMap<String, Arc<AtomicBool>>>,
    /// Cancel flag of the running batch, if any.
    pub batch: Mutex<Option<Arc<AtomicBool>>>,
    /// Document id a prediction is running (or was already run) for.
    pub predicted_doc: Mutex<Option<u64>>,
    /// Bumped by every smart select run: only the latest one is shown.
    pub smart_gen: AtomicU64,
    /// Bumped by every prediction that starts: the result of an older one (another model) is dropped.
    pub predict_gen: AtomicU64,
    /// The catalogue and the models found on disk, read once and again when the models panel opens or a model is
    /// downloaded or deleted (the interface asks for the list at every refresh).
    pub models_cache: Mutex<Option<Vec<ModelSpec>>>,
    /// The settings changed and are not written yet (see `Ctl::save_settings_soon`).
    pub settings_dirty: AtomicBool,
    /// Bumped by every delayed settings write that is asked for: only the last one writes.
    pub settings_gen: AtomicU64,
    /// The content last written to the recovery file: document id and `Document::edit_mark`.
    pub autosaved: Mutex<Option<(u64, u64)>>,
    /// Exports and project saves in progress (closing the window asks first while one runs).
    pub saving: AtomicUsize,
    /// Bumped by every model load that is asked for: a load that is no longer the latest is skipped or dropped.
    pub model_gen: AtomicU64,
    /// Held while a model loads, so that two models are never loaded at the same time.
    pub model_load: Mutex<()>,
}

/// See [`AppState::saving`].
pub struct SavingGuard<'a>(&'a AtomicUsize);

impl Drop for SavingGuard<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

impl Default for AppState {
    fn default() -> Self {
        Self::new()
    }
}

impl AppState {
    pub fn new() -> Self {
        Self::with_paths(Paths::resolve())
    }

    /// The state of an application whose folder is `root` (tests use a folder of their own).
    pub fn at(root: &std::path::Path) -> Self {
        Self::with_paths(Paths::under(root.to_path_buf()))
    }

    fn with_paths(paths: Paths) -> Self {
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
            smart_gen: AtomicU64::new(0),
            predict_gen: AtomicU64::new(0),
            models_cache: Mutex::new(None),
            settings_dirty: AtomicBool::new(false),
            settings_gen: AtomicU64::new(0),
            autosaved: Mutex::new(None),
            saving: AtomicUsize::new(0),
            model_gen: AtomicU64::new(0),
            model_load: Mutex::new(()),
        }
    }

    /// Whether a batch, a download or a save is still running (they stop unfinished if the application quits).
    pub fn tasks_running(&self) -> bool {
        self.batch.guard().is_some() || !self.downloads.guard().is_empty() || self.saving.load(Ordering::SeqCst) > 0
    }

    /// Count a save as running until the returned guard is dropped.
    pub fn saving(&self) -> SavingGuard<'_> {
        self.saving.fetch_add(1, Ordering::SeqCst);
        SavingGuard(&self.saving)
    }

    /// Where the work in progress is saved every minute, to be offered back after a crash.
    pub fn recovery_path(&self) -> PathBuf {
        self.paths.data.join("recovery.osproj")
    }

    pub fn settings_path(&self) -> PathBuf {
        self.paths.data.join("settings.json")
    }

    pub fn save_settings(&self) -> std::io::Result<()> {
        self.settings_dirty.store(false, Ordering::SeqCst);
        self.settings.guard().save(&self.settings_path())
    }

    /// Write the settings if a delayed write is still pending (the application is closing).
    pub fn flush_settings(&self) -> std::io::Result<()> {
        if self.settings_dirty.load(Ordering::SeqCst) {
            self.save_settings()?;
        }
        Ok(())
    }

    /// Catalogue plus the development-only local file entry.
    pub fn models(&self) -> Vec<ModelSpec> {
        self.models_cache
            .guard()
            .get_or_insert_with(|| {
                let mut specs = self.registry.available();
                if let Some(spec) = self.local_model_spec() {
                    specs.push(spec);
                }
                specs
            })
            .clone()
    }

    /// Read the models folder again at the next [`AppState::models`] (a model was added, downloaded or deleted).
    pub fn rescan_models(&self) {
        *self.models_cache.guard() = None;
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

#[cfg(test)]
mod tests {
    use super::*;
    use openscissors_core::sync::LockExt;

    #[test]
    fn the_model_list_is_read_once_until_it_is_rescanned() {
        let home = std::env::temp_dir().join(format!("osc-models-cache-{}", std::process::id()));
        std::fs::create_dir_all(home.join("models").join("mine")).unwrap();
        let st = AppState::at(&home);
        let before = st.models().len();
        std::fs::write(home.join("models/mine/model.json"), r#"{"id":"mine","name":"Mine","license":"MIT"}"#).unwrap();
        assert_eq!(st.models().len(), before, "not read again at every call");
        st.rescan_models();
        assert_eq!(st.models().len(), before + 1, "a rescan finds the new model");
        assert!(st.models_cache.guard().is_some());
        std::fs::remove_dir_all(home).ok();
    }
}
