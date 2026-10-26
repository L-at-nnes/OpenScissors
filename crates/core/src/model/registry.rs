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
    /// Folder under the models folder when it differs from `id` (a model dropped in by hand). Set from
    /// the disk by [`ModelRegistry::available`], never read from `model.json`.
    #[serde(skip)]
    pub folder: Option<String>,
}

/// An id or folder name is one plain path component: no separator, no `..`, not hidden.
fn is_plain_name(name: &str) -> bool {
    !name.is_empty() && !name.starts_with('.') && name.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
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
            folder: None,
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
            folder: None,
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
            folder: None,
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
            folder: None,
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
            folder: None,
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
                let folder = e.file_name().to_string_lossy().into_owned();
                let json = e.path().join("model.json");
                if let Ok(text) = fs::read_to_string(&json) {
                    if let Ok(mut spec) = serde_json::from_str::<ModelSpec>(&text) {
                        if is_plain_name(&spec.id) && is_plain_name(&folder) && !specs.iter().any(|s| s.id == spec.id) {
                            spec.folder = (folder != spec.id).then_some(folder);
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
        self.model_dir(spec).join(&spec.file_name)
    }

    /// Folder holding the files of `spec`: its `id`, or the folder it was found in.
    pub fn model_dir(&self, spec: &ModelSpec) -> PathBuf {
        self.dir.join(spec.folder.as_deref().unwrap_or(&spec.id))
    }

    pub fn state(&self, spec: &ModelSpec) -> ModelState {
        if spec.all_files().iter().all(|f| self.model_dir(spec).join(&f.name).is_file()) {
            ModelState::Ready
        } else {
            ModelState::NotDownloaded
        }
    }

    pub fn delete(&self, spec: &ModelSpec) -> Result<()> {
        // Never remove anything outside the models folder, whatever the spec says.
        if !is_plain_name(spec.folder.as_deref().unwrap_or(&spec.id)) {
            return Err(Error::msg(format!("refusing to delete the folder of model '{}'", spec.id)));
        }
        let folder = self.model_dir(spec);
        if folder.is_dir() {
            fs::remove_dir_all(folder)?;
        }
        Ok(())
    }

    /// Download every file of `spec` (HTTPS only) into its folder, verifying each SHA-256. A file already there is
    /// checked and kept if intact; a download that was cut resumes where it stopped.
    /// `progress(done, total)` counts bytes over all files; set `cancel` to abort.
    pub fn download(&self, spec: &ModelSpec, progress: Progress, cancel: &AtomicBool) -> Result<PathBuf> {
        self.download_files(spec, progress, cancel, true)
    }

    fn download_files(&self, spec: &ModelSpec, progress: Progress, cancel: &AtomicBool, https_only: bool) -> Result<PathBuf> {
        let files = spec.all_files();
        if let Some(f) = files.iter().find(|f| f.sha256.len() != 64) {
            return Err(Error::Download(format!("file '{}' of model '{}' has no pinned sha256", f.name, spec.id)));
        }
        if https_only && files.iter().any(|f| !f.url.starts_with("https://")) {
            return Err(Error::Download("model URL must use HTTPS".into()));
        }
        if !is_plain_name(spec.folder.as_deref().unwrap_or(&spec.id)) {
            return Err(Error::Download(format!("model id '{}' is not a valid folder name", spec.id)));
        }
        let dir = self.model_dir(spec);
        fs::create_dir_all(&dir)?;
        let total: u64 = files.iter().map(|f| f.size_bytes).sum();
        let mut before = 0u64;
        for f in &files {
            let dest = dir.join(&f.name);
            if dest.is_file() {
                // Present is not the same as intact: a damaged file is downloaded again.
                if file_matches(&dest, &f.sha256, cancel)? {
                    before += f.size_bytes;
                    progress(before, total);
                    continue;
                }
                fs::remove_file(&dest)?;
            }
            let base = before;
            let mut report = |done: u64, _file_total: u64| progress(base + done, total.max(base + done));
            self.download_file(&spec.id, f, &dest, &mut report, cancel)?;
            before += f.size_bytes;
        }
        Ok(self.model_path(spec))
    }

    fn download_file(&self, id: &str, file: &ModelFile, dest: &Path, progress: Progress, cancel: &AtomicBool) -> Result<()> {
        let part = part_path(dest);
        // A download that was cut (cancelled, connection lost) left its start behind: ask for the rest only.
        let kept = fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
        let mut resume_from = if kept > 0 && (file.size_bytes == 0 || kept < file.size_bytes) { kept } else { 0 };
        let agent = download_agent();
        let resp = loop {
            let mut request = agent.get(&file.url);
            if resume_from > 0 {
                request = request.header("Range", format!("bytes={resume_from}-"));
            }
            match request.call() {
                Ok(r) => break r,
                // The server does not accept that range (the file changed, or the partial file is wrong): start over.
                Err(ureq::Error::StatusCode(416)) if resume_from > 0 => {
                    let _ = fs::remove_file(&part);
                    resume_from = 0;
                }
                Err(e) => return Err(Error::Download(e.to_string())),
            }
        };
        let announced: Option<u64> = resp.headers().get("content-length").and_then(|v| v.to_str().ok()).and_then(|v| v.parse().ok());
        // 206 with a Content-Range that starts where the partial file ends: the body is the rest of the file.
        let resuming = resume_from > 0 && resp.status().as_u16() == 206 && content_range_start(&resp) == Some(resume_from);
        let start = if resuming { resume_from } else { 0 };
        let total = announced.map_or(file.size_bytes, |n| start + n);
        if !within_expected(file.size_bytes, total) {
            return Err(Error::Download(format!(
                "the server announces {total} bytes for '{}', the catalogue says {}",
                file.name, file.size_bytes
            )));
        }
        let mut hasher = Sha256::new();
        let mut out = if resuming {
            hash_into(&mut hasher, &part, cancel)?;
            fs::OpenOptions::new().append(true).open(&part)?
        } else {
            fs::File::create(&part)?
        };
        let mut reader = resp.into_body().into_reader();
        let mut buf = vec![0u8; 1 << 20];
        let mut done = start;
        loop {
            if cancel.load(Ordering::Relaxed) {
                // The part stays: the next attempt continues from it.
                return Err(Error::Cancelled);
            }
            let n = reader.read(&mut buf).map_err(|e| Error::Download(e.to_string()))?;
            if n == 0 {
                break;
            }
            out.write_all(&buf[..n])?;
            hasher.update(&buf[..n]);
            done += n as u64;
            if !within_expected(file.size_bytes, done) {
                drop(out);
                let _ = fs::remove_file(&part);
                return Err(Error::Download(format!("'{}' is larger than the {} bytes the catalogue says", file.name, file.size_bytes)));
            }
            progress(done, total);
        }
        out.flush()?;
        drop(out);
        if announced.is_some() && done != total {
            return Err(Error::Download(format!("the connection closed after {done} of {total} bytes; the download can be resumed")));
        }
        if hex::encode(hasher.finalize()) != file.sha256 {
            let _ = fs::remove_file(&part);
            return Err(Error::Checksum(id.to_string()));
        }
        fs::rename(&part, dest)?;
        Ok(())
    }

    /// Check the files of `spec` that are on disk against the pinned SHA-256: the names of the ones that are missing
    /// or damaged (empty when the model is intact).
    pub fn verify(&self, spec: &ModelSpec, cancel: &AtomicBool) -> Result<Vec<String>> {
        let dir = self.model_dir(spec);
        let mut bad = Vec::new();
        for f in spec.all_files() {
            let path = dir.join(&f.name);
            if !path.is_file() || !file_matches(&path, &f.sha256, cancel)? {
                bad.push(f.name);
            }
        }
        Ok(bad)
    }
}

/// Where a file is downloaded before it is checked and moved into place (never shared between two files of a model).
fn part_path(dest: &Path) -> PathBuf {
    let mut name = dest.file_name().map(|n| n.to_os_string()).unwrap_or_default();
    name.push(".part");
    dest.with_file_name(name)
}

/// First byte of the answer to a range request, from `Content-Range: bytes <first>-<last>/<size>`.
fn content_range_start(resp: &ureq::http::Response<ureq::Body>) -> Option<u64> {
    let v = resp.headers().get("content-range")?.to_str().ok()?;
    v.strip_prefix("bytes ")?.split('-').next()?.trim().parse().ok()
}

/// Feed the whole file to `hasher`.
fn hash_into(hasher: &mut Sha256, path: &Path, cancel: &AtomicBool) -> Result<()> {
    let mut f = fs::File::open(path)?;
    let mut buf = vec![0u8; 1 << 20];
    loop {
        if cancel.load(Ordering::Relaxed) {
            return Err(Error::Cancelled);
        }
        let n = f.read(&mut buf)?;
        if n == 0 {
            return Ok(());
        }
        hasher.update(&buf[..n]);
    }
}

/// Whether the SHA-256 of the file is `sha256`.
fn file_matches(path: &Path, sha256: &str, cancel: &AtomicBool) -> Result<bool> {
    let mut hasher = Sha256::new();
    hash_into(&mut hasher, path, cancel)?;
    Ok(hex::encode(hasher.finalize()).eq_ignore_ascii_case(sha256))
}

/// A client that gives up on a server that does not answer, instead of waiting for ever. The body is not given a
/// time limit: a model is up to 1 GB and the connection may be slow, the user can cancel.
fn download_agent() -> ureq::Agent {
    use std::time::Duration;
    ureq::Agent::new_with_config(
        ureq::Agent::config_builder()
            .timeout_connect(Some(Duration::from_secs(30)))
            .timeout_recv_response(Some(Duration::from_secs(60)))
            .build(),
    )
}

/// Whether `received` bytes are no more than the catalogue says (`expected` 0 = size not recorded). A server that
/// sends more is refused before it fills the disk.
fn within_expected(expected: u64, received: u64) -> bool {
    expected == 0 || received <= expected
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_download_may_not_exceed_the_catalogue_size() {
        assert!(within_expected(1000, 1000));
        assert!(within_expected(1000, 1));
        assert!(!within_expected(1000, 1001));
        assert!(within_expected(0, u64::MAX), "unknown size: no limit");
    }

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
    fn a_user_model_in_a_folder_named_differently_from_its_id_is_found() {
        let dir = tmp("folder");
        fs::create_dir_all(dir.join("my_folder")).unwrap();
        fs::write(dir.join("my_folder/model.json"), r#"{"id":"mine","name":"Mine","license":"MIT"}"#).unwrap();
        fs::write(dir.join("my_folder/model.onnx"), b"x").unwrap();
        let reg = ModelRegistry::new(&dir);
        let mine = reg.get("mine").expect("listed");
        assert_eq!(reg.model_path(&mine), dir.join("my_folder").join("model.onnx"));
        assert_eq!(reg.state(&mine), ModelState::Ready);
        reg.delete(&mine).unwrap();
        assert!(!dir.join("my_folder").exists());
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn ids_that_could_leave_the_models_folder_are_refused() {
        let dir = tmp("traversal");
        fs::create_dir_all(dir.join("evil")).unwrap();
        fs::write(dir.join("evil/model.json"), r#"{"id":"../victim","name":"Evil","license":"MIT"}"#).unwrap();
        let reg = ModelRegistry::new(&dir);
        assert!(reg.get("../victim").is_none(), "an id with a separator is not listed");
        let mut spec = builtin_catalog().remove(0);
        spec.id = "../victim".into();
        let victim = dir.parent().unwrap().join("victim");
        fs::create_dir_all(&victim).unwrap();
        assert!(reg.delete(&spec).is_err());
        assert!(victim.exists(), "nothing outside the models folder was removed");
        assert!(matches!(reg.download(&spec, &mut |_, _| {}, &AtomicBool::new(false)), Err(Error::Download(_))));
        fs::remove_dir_all(victim).ok();
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

    /// A one-file HTTP server on localhost for the tests. The first `cut_first` requests are answered with the start of
    /// the body only (the connection then drops); the others with the whole body, or from the requested offset when
    /// `ranges` is on. Returns the address and the `Range` headers it received.
    fn serve(body: Vec<u8>, cut_first: usize, ranges: bool) -> (String, std::sync::Arc<std::sync::Mutex<Vec<String>>>) {
        use std::io::{BufRead, BufReader};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/model.onnx", listener.local_addr().unwrap());
        let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let log = seen.clone();
        std::thread::spawn(move || {
            for (n, stream) in listener.incoming().enumerate() {
                let Ok(mut stream) = stream else { return };
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut range = None;
                loop {
                    let mut line = String::new();
                    if reader.read_line(&mut line).unwrap_or(0) == 0 || line.trim().is_empty() {
                        break;
                    }
                    if let Some(v) = line.to_ascii_lowercase().strip_prefix("range: bytes=") {
                        range = v.trim().trim_end_matches('-').parse::<usize>().ok();
                    }
                }
                log.lock().unwrap().push(range.map_or("none".to_string(), |r| format!("from {r}")));
                let from = if ranges { range.unwrap_or(0) } else { 0 };
                let rest = &body[from..];
                let head = if from > 0 {
                    format!(
                        "HTTP/1.1 206 Partial Content\r\nContent-Range: bytes {from}-{}/{}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        body.len() - 1,
                        body.len(),
                        rest.len()
                    )
                } else {
                    format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", rest.len())
                };
                let _ = stream.write_all(head.as_bytes());
                let _ = stream.write_all(if n < cut_first { &rest[..rest.len() / 3] } else { rest });
            }
        });
        (url, seen)
    }

    fn file_for(url: String, body: &[u8]) -> ModelFile {
        ModelFile { name: "model.onnx".into(), url, sha256: hex::encode(Sha256::digest(body)), size_bytes: body.len() as u64 }
    }

    fn body() -> Vec<u8> {
        (0..3_000_000u32).map(|i| (i.wrapping_mul(2_654_435_761) >> 24) as u8).collect()
    }

    #[test]
    fn a_cut_download_resumes_from_the_bytes_it_has() {
        let dir = tmp("resume");
        let data = body();
        let (url, seen) = serve(data.clone(), 1, true);
        let file = file_for(url, &data);
        let dest = dir.join("model.onnx");
        let reg = ModelRegistry::new(&dir);
        let no = AtomicBool::new(false);
        assert!(reg.download_file("m", &file, &dest, &mut |_, _| {}, &no).is_err(), "the connection dropped");
        let kept = fs::metadata(part_path(&dest)).unwrap().len();
        assert_eq!(kept, data.len() as u64 / 3, "the start of the file is kept");
        assert!(!dest.exists());

        let mut last = 0;
        reg.download_file("m", &file, &dest, &mut |done, _| last = done, &no).unwrap();
        assert_eq!(fs::read(&dest).unwrap(), data, "the file is whole and its hash matches");
        assert_eq!(last, data.len() as u64, "progress counts the bytes of the first attempt too");
        assert!(!part_path(&dest).exists());
        assert_eq!(*seen.lock().unwrap(), vec!["none".to_string(), format!("from {kept}")]);
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn a_server_that_ignores_the_range_makes_the_download_start_over() {
        let dir = tmp("noranges");
        let data = body();
        let (url, seen) = serve(data.clone(), 1, false);
        let file = file_for(url, &data);
        let dest = dir.join("model.onnx");
        let reg = ModelRegistry::new(&dir);
        let no = AtomicBool::new(false);
        assert!(reg.download_file("m", &file, &dest, &mut |_, _| {}, &no).is_err());
        reg.download_file("m", &file, &dest, &mut |_, _| {}, &no).unwrap();
        assert_eq!(fs::read(&dest).unwrap(), data);
        assert_eq!(seen.lock().unwrap().len(), 2);
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn a_wrong_hash_is_refused_and_leaves_nothing_to_resume() {
        let dir = tmp("badhash");
        let data = body();
        let (url, _) = serve(data.clone(), 0, true);
        let mut file = file_for(url, &data);
        file.sha256 = "0".repeat(64);
        let dest = dir.join("model.onnx");
        let reg = ModelRegistry::new(&dir);
        assert!(matches!(reg.download_file("m", &file, &dest, &mut |_, _| {}, &AtomicBool::new(false)), Err(Error::Checksum(_))));
        assert!(!dest.exists() && !part_path(&dest).exists());
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn a_file_that_is_already_there_is_checked_before_it_is_trusted() {
        let dir = tmp("verify");
        let data = body();
        let (url, seen) = serve(data.clone(), 0, true);
        let file = file_for(url, &data);
        let spec = ModelSpec { id: "m".into(), files: vec![], ..builtin_catalog().remove(0) };
        let spec = ModelSpec {
            file_name: file.name.clone(),
            url: file.url.clone(),
            sha256: file.sha256.clone(),
            size_bytes: file.size_bytes,
            ..spec
        };
        let reg = ModelRegistry::new(&dir);
        let folder = reg.model_dir(&spec);
        fs::create_dir_all(&folder).unwrap();
        let no = AtomicBool::new(false);
        // Right size, wrong content: it is found damaged and downloaded again.
        fs::write(folder.join(&file.name), vec![7u8; data.len()]).unwrap();
        assert_eq!(reg.verify(&spec, &no).unwrap(), vec![file.name.clone()]);
        reg.download_files(&spec, &mut |_, _| {}, &no, false).unwrap();
        assert_eq!(fs::read(folder.join(&file.name)).unwrap(), data);
        assert!(reg.verify(&spec, &no).unwrap().is_empty());
        // An intact file is kept without contacting the server again.
        let requests = seen.lock().unwrap().len();
        reg.download_files(&spec, &mut |_, _| {}, &no, false).unwrap();
        assert_eq!(seen.lock().unwrap().len(), requests);
        fs::remove_dir_all(dir).ok();
    }
}
