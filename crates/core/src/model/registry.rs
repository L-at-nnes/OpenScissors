//! Model catalogue, local model folder and checked downloads.
//!
//! A model is described by a [`ModelSpec`]. The built-in catalogue below can be extended
//! without code: drop `<models>/<id>/model.json` (a `ModelSpec`) next to the `.onnx` file.

use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::error::{Error, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OutputNorm {
    /// Rescale the raw output to 0..255 by its min and max.
    Minmax,
    /// Raw output is a logit.
    Sigmoid,
    /// Raw output is already 0..1.
    None,
}

fn default_size() -> [usize; 2] {
    [1024, 1024]
}
fn default_mean() -> [f32; 3] {
    [0.5, 0.5, 0.5]
}
fn default_std() -> [f32; 3] {
    [1.0, 1.0, 1.0]
}
fn default_file() -> String {
    "model.onnx".into()
}
fn default_norm() -> OutputNorm {
    OutputNorm::Minmax
}

/// What a model does.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelKind {
    /// Predicts the foreground mask of a whole image (the active model).
    #[default]
    Segmenter,
    /// SAM 2 prompt segmentation ("Smart select"): an encoder and a decoder, loaded on first use.
    Sam2,
}

/// One file of a model (several for models that ship weights in a side file).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelFile {
    pub name: String,
    pub url: String,
    pub sha256: String,
    #[serde(default)]
    pub size_bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelSpec {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub description: String,
    /// Empty for models the user placed by hand.
    #[serde(default)]
    pub url: String,
    /// Lower-case hex SHA-256 of the file; required to download.
    #[serde(default)]
    pub sha256: String,
    #[serde(default)]
    pub size_bytes: u64,
    pub license: String,
    #[serde(default = "default_file")]
    pub file_name: String,
    /// Network input `[width, height]`.
    #[serde(default = "default_size")]
    pub input_size: [usize; 2],
    /// Per-channel mean/std applied to `pixel / 255` (RGB order).
    #[serde(default = "default_mean")]
    pub mean: [f32; 3],
    #[serde(default = "default_std")]
    pub std: [f32; 3],
    #[serde(default)]
    pub input_name: Option<String>,
    #[serde(default)]
    pub output_name: Option<String>,
    #[serde(default = "default_norm")]
    pub output: OutputNorm,
    #[serde(default)]
    pub recommended: bool,
    /// Peak memory in MB when running in the default memory-lean mode (0 = not measured).
    #[serde(default)]
    pub ram_mb: u32,
    #[serde(default)]
    pub kind: ModelKind,
    /// Files of a multi-file model. Empty for the usual single-file model, which is described
    /// by `file_name`, `url`, `sha256` and `size_bytes`.
    #[serde(default)]
    pub files: Vec<ModelFile>,
}

impl ModelSpec {
    /// Every file that must be present for the model to be usable.
    pub fn all_files(&self) -> Vec<ModelFile> {
        if self.files.is_empty() {
            vec![ModelFile {
                name: self.file_name.clone(),
                url: self.url.clone(),
                sha256: self.sha256.clone(),
                size_bytes: self.size_bytes,
            }]
        } else {
            self.files.clone()
        }
    }

    /// Total download size in bytes.
    pub fn total_size(&self) -> u64 {
        self.all_files().iter().map(|f| f.size_bytes).sum()
    }
}

const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Models shipped in the catalogue (downloaded on demand, never bundled).
pub fn builtin_catalog() -> Vec<ModelSpec> {
    vec![
        ModelSpec {
            id: "birefnet-lite".into(),
            name: "BiRefNet lite".into(),
            description: "Best quality-to-speed balance. Strong on hair and thin structures.".into(),
            url: "https://huggingface.co/onnx-community/BiRefNet_lite-ONNX/resolve/main/onnx/model.onnx".into(),
            sha256: "5600024376f572a557870a5eb0afb1e5961636bef4e1e22132025467d0f03333".into(),
            size_bytes: 224_005_088,
            license: "MIT".into(),
            file_name: "model.onnx".into(),
            input_size: [1024, 1024],
            mean: IMAGENET_MEAN,
            std: IMAGENET_STD,
            input_name: None,
            output_name: None,
            output: OutputNorm::Sigmoid,
            recommended: false,
            ram_mb: 5200,
            kind: ModelKind::Segmenter,
            files: vec![],
        },
        ModelSpec {
            id: "birefnet-lite-fp16".into(),
            name: "BiRefNet lite (half precision)".into(),
            description: "Half-size file with the same quality. Made for the GPU; slower than the regular one on a CPU.".into(),
            url: "https://huggingface.co/onnx-community/BiRefNet_lite-ONNX/resolve/main/onnx/model_fp16.onnx".into(),
            sha256: "d39b897ceb16ae654c1731f3dba0cf9b368d9cae74b5a57459b455cc8bfec402".into(),
            size_bytes: 114_538_221,
            license: "MIT".into(),
            file_name: "model.onnx".into(),
            input_size: [1024, 1024],
            mean: IMAGENET_MEAN,
            std: IMAGENET_STD,
            input_name: None,
            output_name: None,
            output: OutputNorm::Sigmoid,
            recommended: false,
            ram_mb: 5300,
            kind: ModelKind::Segmenter,
            files: vec![],
        },
        ModelSpec {
            id: "birefnet".into(),
            name: "BiRefNet full".into(),
            description: "Highest quality, several times slower on CPU. Best with a GPU.".into(),
            url: "https://huggingface.co/onnx-community/BiRefNet-ONNX/resolve/main/onnx/model.onnx".into(),
            sha256: "58f621f00f5d756097615970a88a791584600dcf7c45b18a0a6267535a1ebd3c".into(),
            size_bytes: 972_666_916,
            license: "MIT".into(),
            file_name: "model.onnx".into(),
            input_size: [1024, 1024],
            mean: IMAGENET_MEAN,
            std: IMAGENET_STD,
            input_name: None,
            output_name: None,
            output: OutputNorm::Sigmoid,
            recommended: false,
            ram_mb: 0,
            kind: ModelKind::Segmenter,
            files: vec![],
        },
        ModelSpec {
            id: "isnet-general-use".into(),
            name: "IS-Net general use".into(),
            description: "Fastest option (under a second on CPU). Good on clear subjects, softer on hair.".into(),
            url: "https://github.com/danielgatis/rembg/releases/download/v0.0.0/isnet-general-use.onnx".into(),
            sha256: "60920e99c45464f2ba57bee2ad08c919a52bbf852739e96947fbb4358c0d964a".into(),
            size_bytes: 178_648_008,
            license: "Apache-2.0".into(),
            file_name: "model.onnx".into(),
            input_size: [1024, 1024],
            mean: [0.5, 0.5, 0.5],
            std: [1.0, 1.0, 1.0],
            input_name: None,
            output_name: None,
            output: OutputNorm::Minmax,
            recommended: true,
            ram_mb: 600,
            kind: ModelKind::Segmenter,
            files: vec![],
        },
        ModelSpec {
            id: "sam2-tiny".into(),
            name: "SAM 2.1 tiny (Smart select)".into(),
            description: "Point at a subject to select it: click what to keep, right-click what to leave out, or drag a box. Used by the Smart select tool.".into(),
            url: String::new(),
            sha256: String::new(),
            size_bytes: 0,
            license: "Apache-2.0".into(),
            file_name: "vision_encoder.onnx".into(),
            input_size: [1024, 1024],
            mean: IMAGENET_MEAN,
            std: IMAGENET_STD,
            input_name: None,
            output_name: None,
            output: OutputNorm::None,
            recommended: false,
            ram_mb: 0,
            kind: ModelKind::Sam2,
            files: vec![
                ModelFile {
                    name: "vision_encoder.onnx".into(),
                    url: "https://huggingface.co/onnx-community/sam2.1-hiera-tiny-ONNX/resolve/main/onnx/vision_encoder.onnx".into(),
                    sha256: "4f30aacd3aaefbca81a0b7fe4c1fc96345570ea0a6f80ced599493d1b3be2e8c".into(),
                    size_bytes: 354_238,
                },
                ModelFile {
                    name: "vision_encoder.onnx_data".into(),
                    url: "https://huggingface.co/onnx-community/sam2.1-hiera-tiny-ONNX/resolve/main/onnx/vision_encoder.onnx_data".into(),
                    sha256: "e83df9866a5afe68ea7f0f721f18f65137fc3acbf0da1c74e946d363e09c69cc".into(),
                    size_bytes: 134_084_864,
                },
                ModelFile {
                    name: "prompt_encoder_mask_decoder.onnx".into(),
                    url: "https://huggingface.co/onnx-community/sam2.1-hiera-tiny-ONNX/resolve/main/onnx/prompt_encoder_mask_decoder.onnx".into(),
                    sha256: "874414704c5d686db7d206a35f6e15d26563d50c8c4468fccc6739bd7e491dcf".into(),
                    size_bytes: 213_114,
                },
                ModelFile {
                    name: "prompt_encoder_mask_decoder.onnx_data".into(),
                    url: "https://huggingface.co/onnx-community/sam2.1-hiera-tiny-ONNX/resolve/main/onnx/prompt_encoder_mask_decoder.onnx_data".into(),
                    sha256: "e9874d900dd4134ed60eab1e97910327c2419e0b2954485d8fd6e7f1a1470f47".into(),
                    size_bytes: 20_958_208,
                },
            ],
        },
    ]
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelState {
    NotDownloaded,
    Ready,
}

pub struct ModelRegistry {
    dir: PathBuf,
}

pub type Progress<'a> = &'a mut dyn FnMut(u64, u64);

impl ModelRegistry {
    pub fn new(models_dir: impl Into<PathBuf>) -> Self {
        Self { dir: models_dir.into() }
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Built-in catalogue plus user-provided `model.json` specs found on disk.
    pub fn available(&self) -> Vec<ModelSpec> {
        let mut specs = builtin_catalog();
        if let Ok(entries) = fs::read_dir(&self.dir) {
            for e in entries.flatten() {
                let json = e.path().join("model.json");
                if let Ok(text) = fs::read_to_string(&json) {
                    if let Ok(spec) = serde_json::from_str::<ModelSpec>(&text) {
                        if !specs.iter().any(|s| s.id == spec.id) {
                            specs.push(spec);
                        }
                    }
                }
            }
        }
        specs
    }

    pub fn get(&self, id: &str) -> Option<ModelSpec> {
        self.available().into_iter().find(|s| s.id == id)
    }

    pub fn model_path(&self, spec: &ModelSpec) -> PathBuf {
        self.dir.join(&spec.id).join(&spec.file_name)
    }

    /// Folder holding the files of `spec`.
    pub fn model_dir(&self, spec: &ModelSpec) -> PathBuf {
        self.dir.join(&spec.id)
    }

    pub fn state(&self, spec: &ModelSpec) -> ModelState {
        if spec.all_files().iter().all(|f| self.model_dir(spec).join(&f.name).is_file()) {
            ModelState::Ready
        } else {
            ModelState::NotDownloaded
        }
    }

    pub fn delete(&self, spec: &ModelSpec) -> Result<()> {
        let folder = self.dir.join(&spec.id);
        if folder.is_dir() {
            fs::remove_dir_all(folder)?;
        }
        Ok(())
    }

    /// Download every file of `spec` (HTTPS only) into its folder, verifying each SHA-256.
    /// `progress(done, total)` counts bytes over all files; set `cancel` to abort.
    pub fn download(&self, spec: &ModelSpec, progress: Progress, cancel: &AtomicBool) -> Result<PathBuf> {
        let files = spec.all_files();
        if let Some(f) = files.iter().find(|f| f.sha256.len() != 64) {
            return Err(Error::Download(format!("file '{}' of model '{}' has no pinned sha256", f.name, spec.id)));
        }
        if files.iter().any(|f| !f.url.starts_with("https://")) {
            return Err(Error::Download("model URL must use HTTPS".into()));
        }
        let dir = self.model_dir(spec);
        fs::create_dir_all(&dir)?;
        let total: u64 = files.iter().map(|f| f.size_bytes).sum();
        let mut before = 0u64;
        for f in &files {
            let dest = dir.join(&f.name);
            if dest.is_file() {
                before += f.size_bytes;
                progress(before, total);
                continue;
            }
            let base = before;
            let mut report = |done: u64, _file_total: u64| progress(base + done, total.max(base + done));
            self.download_file(&spec.id, f, &dest, &mut report, cancel)?;
            before += f.size_bytes;
        }
        Ok(self.model_path(spec))
    }

    fn download_file(&self, id: &str, file: &ModelFile, dest: &Path, progress: Progress, cancel: &AtomicBool) -> Result<()> {
        let part = dest.with_extension("part");
        let resp = ureq::get(&file.url).call().map_err(|e| Error::Download(e.to_string()))?;
        let total =
            resp.headers().get("content-length").and_then(|v| v.to_str().ok()).and_then(|v| v.parse().ok()).unwrap_or(file.size_bytes);
        let mut reader = resp.into_body().into_reader();
        let mut out = fs::File::create(&part)?;
        let mut hasher = Sha256::new();
        let mut buf = vec![0u8; 1 << 20];
        let mut done = 0u64;
        loop {
            if cancel.load(Ordering::Relaxed) {
                drop(out);
                let _ = fs::remove_file(&part);
                return Err(Error::Cancelled);
            }
            let n = reader.read(&mut buf).map_err(|e| Error::Download(e.to_string()))?;
            if n == 0 {
                break;
            }
            out.write_all(&buf[..n])?;
            hasher.update(&buf[..n]);
            done += n as u64;
            progress(done, total);
        }
        out.flush()?;
        drop(out);
        if hex::encode(hasher.finalize()) != file.sha256 {
            let _ = fs::remove_file(&part);
            return Err(Error::Checksum(id.to_string()));
        }
        fs::rename(&part, dest)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("osc-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn catalog_has_unique_ids_pinned_hashes_and_https() {
        let cat = builtin_catalog();
        for s in &cat {
            for f in s.all_files() {
                assert_eq!(f.sha256.len(), 64, "{} / {}", s.id, f.name);
                assert!(f.url.starts_with("https://"), "{} / {}", s.id, f.name);
                assert!(f.sha256.chars().all(|c| c.is_ascii_hexdigit()));
                assert!(f.size_bytes > 0);
            }
        }
        let sam = cat.iter().find(|s| s.kind == ModelKind::Sam2).expect("SAM 2 entry");
        assert_eq!(sam.all_files().len(), 4);
        assert!(sam.total_size() > 150_000_000);
        let mut ids: Vec<_> = cat.iter().map(|s| s.id.clone()).collect();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), cat.len());
        assert_eq!(cat.iter().filter(|s| s.recommended).count(), 1);
    }

    #[test]
    fn state_follows_the_file_on_disk_and_delete_removes_it() {
        let dir = tmp("state");
        let reg = ModelRegistry::new(&dir);
        let spec = reg.get("birefnet-lite").unwrap();
        assert_eq!(reg.state(&spec), ModelState::NotDownloaded);
        fs::create_dir_all(dir.join("birefnet-lite")).unwrap();
        fs::write(reg.model_path(&spec), b"x").unwrap();
        assert_eq!(reg.state(&spec), ModelState::Ready);
        reg.delete(&spec).unwrap();
        assert_eq!(reg.state(&spec), ModelState::NotDownloaded);
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn user_models_are_discovered_from_model_json() {
        let dir = tmp("custom");
        fs::create_dir_all(dir.join("mine")).unwrap();
        fs::write(dir.join("mine/model.json"), r#"{"id":"mine","name":"Mine","license":"MIT","output":"sigmoid"}"#).unwrap();
        let reg = ModelRegistry::new(&dir);
        let mine = reg.get("mine").expect("custom model listed");
        assert_eq!(mine.input_size, [1024, 1024]);
        assert_eq!(mine.output, OutputNorm::Sigmoid);
        assert!(reg.available().len() > builtin_catalog().len());
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn download_refuses_unpinned_or_non_https() {
        let dir = tmp("dl");
        let reg = ModelRegistry::new(&dir);
        let cancel = AtomicBool::new(false);
        let mut spec = reg.get("birefnet-lite").unwrap();
        spec.sha256.clear();
        assert!(matches!(reg.download(&spec, &mut |_, _| {}, &cancel), Err(Error::Download(_))));
        let mut spec = reg.get("birefnet-lite").unwrap();
        spec.url = "http://example.invalid/m.onnx".into();
        assert!(matches!(reg.download(&spec, &mut |_, _| {}, &cancel), Err(Error::Download(_))));
        fs::remove_dir_all(dir).ok();
    }
}
