#![allow(clippy::chunks_exact_to_as_chunks)]
//! Command line front end: `openscissors-cli predict|cut|models`.
//!
//! The models folder is `$OPENSCISSORS_HOME/models`, or `./models` next to the executable.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::atomic::AtomicBool;

use openscissors_core::batch::{self, BatchOptions};
use openscissors_core::compare;
use openscissors_core::io::{load_image, save_mask_png, save_rgba, write_atomic, Format};
use openscissors_core::model::{ModelRegistry, ModelState, Predictor};
use openscissors_core::params::Params;
use openscissors_core::settings::Settings;
use openscissors_core::types::Rgba;

const USAGE: &str = "usage:
  openscissors-cli models                      list models and their state
  openscissors-cli models download <id>        download a model (HTTPS, sha256 checked; run it again to resume)
  openscissors-cli models verify <id>          check the downloaded files against their sha256
  openscissors-cli compare <in> <out-folder> [--models <id,id,...>]
      runs several models one after the other on one picture (every downloaded model by default), writes each mask
      and a contact sheet (the picture, then each cut-out) into <out-folder>, and prints how much each model keeps
      and how much each pair agrees
  openscissors-cli predict <in> <mask.png> [--model <id|file.onnx>]
  openscissors-cli cut <in> <out.png|webp|avif|tiff> [--model <id|file.onnx>]
  openscissors-cli batch <src-folder> <dst-folder> [--format png|jpg|webp|avif] [--quality 95]
                         [--model <id|file.onnx>] [--defaults] [--resume] [--report <report.json>]
      cuts out every image of the folder; --defaults applies the layer settings saved in the app;
      --resume leaves alone the images whose result is already in <dst-folder>, so that running the same
      command again after an error or an interruption does only what is left; --report writes a JSON
      report (one entry per image: done, skipped or failed, the file written, the error, the time)";

/// The portable folder: `$OPENSCISSORS_HOME`, or the folder of the executable.
fn home_dir() -> PathBuf {
    if let Ok(home) = std::env::var("OPENSCISSORS_HOME") {
        return PathBuf::from(home);
    }
    std::env::current_exe().ok().and_then(|p| p.parent().map(Path::to_path_buf)).unwrap_or_else(|| PathBuf::from("."))
}

fn models_dir() -> PathBuf {
    home_dir().join("models")
}

fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
}

fn load_predictor(registry: &ModelRegistry, choice: Option<String>) -> Result<Predictor, String> {
    let choice = choice.or_else(|| std::env::var("OPENSCISSORS_MODEL").ok());
    if let Some(c) = &choice {
        if Path::new(c).is_file() {
            // A bare ONNX file: assume an IS-Net style model (x/255 - 0.5, min-max output).
            let mut spec = registry.available().remove(0);
            spec.mean = [0.5; 3];
            spec.std = [1.0; 3];
            spec.output = openscissors_core::model::OutputNorm::Minmax;
            return Predictor::load(&spec, Path::new(c)).map_err(|e| e.to_string());
        }
    }
    let spec = match &choice {
        Some(id) => registry.get(id).ok_or_else(|| format!("unknown model '{id}'"))?,
        None => registry
            .available()
            .into_iter()
            .find(|s| registry.state(s) == ModelState::Ready)
            .ok_or("no model downloaded; run `openscissors-cli models download isnet-general-use`")?,
    };
    if registry.state(&spec) != ModelState::Ready {
        return Err(format!("model '{}' is not downloaded", spec.id));
    }
    Predictor::load(&spec, &registry.model_path(&spec)).map_err(|e| e.to_string())
}

fn run(args: Vec<String>) -> Result<(), String> {
    let registry = ModelRegistry::new(models_dir());
    match args.first().map(String::as_str) {
        Some("models") if args.get(1).map(String::as_str) == Some("download") => {
            let id = args.get(2).ok_or("model id expected")?;
            let spec = registry.get(id).ok_or_else(|| format!("unknown model '{id}'"))?;
            let mut last = 0;
            let mut progress = |done: u64, total: u64| {
                let pct = (done * 100).checked_div(total).unwrap_or(0);
                if pct != last {
                    last = pct;
                    eprint!("\r{} {}%", spec.id, pct);
                }
            };
            registry.download(&spec, &mut progress, &AtomicBool::new(false)).map_err(|e| e.to_string())?;
            eprintln!("\ndownloaded to {}", registry.model_path(&spec).display());
            Ok(())
        }
        Some("models") if args.get(1).map(String::as_str) == Some("verify") => {
            let id = args.get(2).ok_or("model id expected")?;
            let spec = registry.get(id).ok_or_else(|| format!("unknown model '{id}'"))?;
            let bad = registry.verify(&spec, &AtomicBool::new(false)).map_err(|e| e.to_string())?;
            if bad.is_empty() {
                eprintln!("{} is intact", spec.id);
                Ok(())
            } else {
                Err(format!("{}: missing or damaged: {}", spec.id, bad.join(", ")))
            }
        }
        Some("compare") => {
            let (input, out_dir) = (args.get(1).ok_or("input expected")?, args.get(2).ok_or("output folder expected")?);
            let ids: Vec<String> = match flag(&args, "--models") {
                Some(list) => list.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect(),
                None => registry
                    .available()
                    .into_iter()
                    .filter(|s| s.kind == openscissors_core::model::ModelKind::Segmenter && registry.state(s) == ModelState::Ready)
                    .map(|s| s.id)
                    .collect(),
            };
            if ids.len() < 2 {
                return Err("compare needs at least two models: download them, or name them with --models a,b".into());
            }
            let image = load_image(Path::new(input)).map_err(|e| e.to_string())?;
            std::fs::create_dir_all(out_dir).map_err(|e| e.to_string())?;
            let stem = Path::new(input).file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "image".into());
            let mut results: Vec<(String, openscissors_core::types::Mask)> = Vec::new();
            for id in &ids {
                // One model in memory at a time: the next is loaded once this one is dropped.
                let started = std::time::Instant::now();
                let mask = {
                    let predictor = load_predictor(&registry, Some(id.clone()))?;
                    predictor.predict(&image, true).map_err(|e| e.to_string())?
                };
                let millis = started.elapsed().as_millis();
                save_mask_png(&Path::new(out_dir).join(format!("{stem}.{id}.png")), &mask).map_err(|e| e.to_string())?;
                println!("{id:<24} {:>5.1}% kept  {millis:>6} ms", compare::coverage(&mask) * 100.0);
                results.push((id.clone(), mask));
            }
            println!("\nagreement (intersection over union):");
            for (i, (a, ma)) in results.iter().enumerate() {
                for (b, mb) in &results[i + 1..] {
                    println!("  {a} / {b}: {:.1}%", compare::iou(ma, mb) * 100.0);
                }
            }
            let sheet = compare::contact_sheet(&image, &results);
            let sheet_path = Path::new(out_dir).join(format!("{stem}.compare.png"));
            save_rgba(&sheet_path, &sheet, Format::Png, 95, [255; 3]).map_err(|e| e.to_string())?;
            println!("\ncontact sheet: {} (picture, then {})", sheet_path.display(), ids.join(", "));
            Ok(())
        }
        Some("models") => {
            for s in registry.available() {
                let state = if registry.state(&s) == ModelState::Ready { "ready" } else { "not downloaded" };
                println!("{:<18} {:<16} {:>6} MB  {}  {}", s.id, state, s.size_bytes / 1_000_000, s.license, s.name);
            }
            Ok(())
        }
        Some("batch") => {
            let (src, dst) = (args.get(1).ok_or("source folder expected")?, args.get(2).ok_or("destination folder expected")?);
            let format = match flag(&args, "--format").as_deref().unwrap_or("png") {
                "png" => Format::Png,
                "jpg" | "jpeg" => Format::Jpeg,
                "webp" => Format::Webp,
                "avif" => Format::Avif,
                other => return Err(format!("unsupported format '{other}'")),
            };
            let quality = flag(&args, "--quality")
                .map(|q| q.parse::<u8>().map_err(|_| "quality must be a number from 10 to 100"))
                .transpose()?
                .unwrap_or(95);
            let params = if args.iter().any(|a| a == "--defaults") {
                Settings::load(&home_dir().join("data").join("settings.json")).defaults
            } else {
                Params::default()
            };
            let files = batch::validate(Path::new(src), Path::new(dst)).map_err(|e| e.to_string())?;
            let predictor = load_predictor(&registry, flag(&args, "--model"))?;
            let opts =
                BatchOptions { dst: PathBuf::from(dst), format, quality, params, skip_existing: args.iter().any(|a| a == "--resume") };
            let predict = |img: &openscissors_core::types::Image| predictor.predict(img, true);
            let report = batch::run_batch(
                &files,
                &opts,
                &predict,
                &mut |p| {
                    if let Some(f) = p.current {
                        eprintln!("[{}/{}] {}", p.done + 1, p.total, f.file_name().map(|n| n.to_string_lossy()).unwrap_or_default());
                    }
                },
                &AtomicBool::new(false),
            );
            for f in &report.failed {
                eprintln!("failed: {}: {}", f.file, f.error);
            }
            if let Some(path) = flag(&args, "--report") {
                write_atomic(Path::new(&path), report.to_json().as_bytes()).map_err(|e| format!("cannot write the report {path}: {e}"))?;
            }
            eprintln!("{} processed, {} skipped, {} failed", report.processed, report.skipped, report.failed.len());
            if report.failed.is_empty() {
                Ok(())
            } else {
                Err(format!("{} file(s) failed", report.failed.len()))
            }
        }
        Some(cmd @ ("predict" | "cut")) => {
            let (input, output) = (args.get(1).ok_or("input expected")?, args.get(2).ok_or("output expected")?);
            let predictor = load_predictor(&registry, flag(&args, "--model"))?;
            let image = load_image(Path::new(input)).map_err(|e| e.to_string())?;
            let mask = predictor.predict(&image, true).map_err(|e| e.to_string())?;
            if cmd == "predict" {
                return save_mask_png(Path::new(output), &mask).map_err(|e| e.to_string());
            }
            let mut rgba = Rgba::new(image.w, image.h);
            for (i, p) in rgba.data.chunks_exact_mut(4).enumerate() {
                p[..3].copy_from_slice(&image.data[i * 3..i * 3 + 3]);
                p[3] = mask.data[i];
            }
            let format =
                Format::from_path(Path::new(output)).filter(|f| f.supports_alpha()).ok_or("output must be png, webp, avif or tiff")?;
            save_rgba(Path::new(output), &rgba, format, 95, [255; 3]).map_err(|e| e.to_string())
        }
        _ => Err(USAGE.to_string()),
    }
}

fn main() -> ExitCode {
    match run(std::env::args().skip(1).collect()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}
