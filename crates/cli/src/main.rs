#![allow(clippy::chunks_exact_to_as_chunks)]
//! Command line front end: `openscissors-cli predict|cut|models`.
//!
//! The models and settings are those of the application (`openscissors_core::paths::root`): `$OPENSCISSORS_HOME`, the
//! folder of the executable, or the user's profile when that folder is read-only.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::atomic::AtomicBool;

use openscissors_core::batch::{self, BatchOptions};
use openscissors_core::compare;
use openscissors_core::io::{load_image, save_mask_png, save_rgba, write_atomic, Format};
use openscissors_core::model::{Accelerator, ModelKind, ModelRegistry, ModelState, Predictor};
use openscissors_core::params::{BgMode, Params, PRESETS};
use openscissors_core::settings::Settings;

const USAGE: &str = "usage:
  openscissors-cli models [--json]             list models and their state (--json: every field, for a script)
  openscissors-cli models download <id>        download a model (HTTPS, sha256 checked; run it again to resume)
  openscissors-cli models verify <id>          check the downloaded files against their sha256
  openscissors-cli compare <in> <out-folder> [--models <id,id,...>] [--gpu]
      runs several models one after the other on one picture (every downloaded model by default), writes each mask
      and a contact sheet (the picture, then each cut-out) into <out-folder>, and prints how much each model keeps
      and how much each pair agrees
  openscissors-cli predict <in> <mask.png> [--model <id|file.onnx>] [--gpu]
  openscissors-cli cut <in> <out.png|jpg|webp|avif|tiff|bmp> [--model <id|file.onnx>] [--gpu] [--quality 95]
                       [--defaults] [--preset <id>] [--bg <colour>] [--width <px>] [--height <px>] [--8bit] [--dpi 300]
      cuts out one picture, as every image of a batch; - as <in> reads it from the standard input, - as <out> writes
      the result to the standard output (in PNG, or --format jpg|webp|avif|tiff|bmp)
  openscissors-cli batch <src-folder> <dst-folder> [--format png|jpg|webp|avif|tiff|bmp] [--quality 95]
                         [--model <id|file.onnx>] [--defaults] [--resume] [--report <report.json>] [--8bit]
                         [--dpi 300] [--gpu] [--preset <id>] [--bg <colour>] [--width <px>] [--height <px>]
                         [--recursive]
      cuts out every image of the folder (a picture that is already cut out keeps its transparency, a 16-bit one
      stays 16-bit in PNG unless --8bit); --defaults applies the layer settings saved in the app;
      --dpi writes the print resolution in PNG and JPEG files; --preset applies a preset of the application
      (marketplace, marketplace-large, thumbnail, banner), --bg sets the background (transparent, white, black or
      #rrggbb), --width and --height the size of the result (both: exactly that size), after --defaults;
      --recursive takes the pictures of the subfolders too and writes each result in the same subfolder of
      <dst-folder>; --resume leaves alone the images whose result is already in <dst-folder>, so that running the same
      command again after an error or an interruption does only what is left; --report writes a JSON
      report (one entry per image: done, skipped or failed, the file written, the error, the time)
  --gpu (compare, predict, cut, batch) runs the model on the graphics card when one can (DirectML on Windows), and
      on the processor otherwise; the one used is printed";

/// The folder of the application, the same one the window uses.
fn home_dir() -> PathBuf {
    openscissors_core::paths::root()
}

fn models_dir() -> PathBuf {
    home_dir().join("models")
}

fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
}

/// Refuse an option the command does not know (a typo such as `--resum` would otherwise be ignored without a word)
/// and an option that needs a value but has none. `with_value` options take the next argument.
fn check_options(args: &[String], with_value: &[&str], switches: &[&str]) -> Result<(), String> {
    let mut i = 0;
    while i < args.len() {
        let a = args[i].as_str();
        if a.starts_with("--") {
            if with_value.contains(&a) {
                if args.get(i + 1).is_none_or(|v| v.starts_with("--")) {
                    return Err(format!("{a} needs a value"));
                }
                i += 1;
            } else if !switches.contains(&a) {
                return Err(format!("unknown option '{a}'\n\n{USAGE}"));
            }
        }
        i += 1;
    }
    Ok(())
}

/// The quality option: 10 to 100.
fn parse_quality(value: Option<String>) -> Result<u8, String> {
    match value {
        None => Ok(95),
        Some(v) => {
            v.parse::<u8>().ok().filter(|q| (10..=100).contains(q)).ok_or_else(|| "quality must be a number from 10 to 100".to_string())
        }
    }
}

/// The print resolution option: 1 to 65535 dots per inch, none when absent.
fn parse_dpi(value: Option<String>) -> Result<u16, String> {
    match value {
        None => Ok(0),
        Some(v) => v.parse::<u16>().ok().filter(|&d| d > 0).ok_or_else(|| "dpi must be a number from 1 to 65535".to_string()),
    }
}

/// The catalogue for a script: every field of each model, whether it is downloaded and, when it is, where.
fn models_json(registry: &ModelRegistry) -> String {
    let list: Vec<serde_json::Value> = registry
        .available()
        .iter()
        .map(|s| {
            let mut v = serde_json::to_value(s).unwrap_or_default();
            let ready = registry.state(s) == ModelState::Ready;
            v["ready"] = ready.into();
            if ready {
                v["path"] = registry.model_path(s).display().to_string().into();
            }
            v
        })
        .collect();
    serde_json::to_string_pretty(&list).unwrap_or_default()
}

/// The layer settings of `cut` and `batch`: those saved in the application with `--defaults` (the default ones
/// otherwise), then `--preset`, `--bg`, `--width` and `--height`, in that order.
fn layer_params(args: &[String]) -> Result<Params, String> {
    let mut p = if args.iter().any(|a| a == "--defaults") {
        Settings::load(&home_dir().join("data").join("settings.json")).defaults
    } else {
        Params::default()
    };
    if let Some(id) = flag(args, "--preset") {
        let ids = || PRESETS.iter().map(|p| p.id).collect::<Vec<_>>().join(", ");
        PRESETS.iter().find(|p| p.id == id).ok_or_else(|| format!("unknown preset '{id}' (one of {})", ids()))?.apply(&mut p);
    }
    if let Some(bg) = flag(args, "--bg") {
        match parse_colour(&bg)? {
            None => p.background.mode = BgMode::Transparent,
            Some(c) => {
                p.background.mode = BgMode::Color;
                p.background.color = c;
            }
        }
    }
    if let Some(w) = flag(args, "--width") {
        p.background.output_width = parse_size("--width", &w)?;
    }
    if let Some(h) = flag(args, "--height") {
        p.background.output_height = parse_size("--height", &h)?;
    }
    p.sanitize();
    Ok(p)
}

/// The background option: `transparent`, `white`, `black` or a colour `#rrggbb`. `None` is transparent.
fn parse_colour(value: &str) -> Result<Option<[u8; 3]>, String> {
    match value.to_ascii_lowercase().as_str() {
        "transparent" => Ok(None),
        "white" => Ok(Some([255; 3])),
        "black" => Ok(Some([0; 3])),
        s => {
            let hex = s.strip_prefix('#').unwrap_or(s);
            if hex.len() != 6 || !hex.chars().all(|c| c.is_ascii_hexdigit()) {
                return Err(format!("--bg must be transparent, white, black or a colour such as #ff8800, not '{value}'"));
            }
            let byte = |i: usize| u8::from_str_radix(&hex[i..i + 2], 16).unwrap_or(0);
            Ok(Some([byte(0), byte(2), byte(4)]))
        }
    }
}

/// A size option in pixels, at least 1.
fn parse_size(name: &str, value: &str) -> Result<u32, String> {
    value.parse::<u32>().ok().filter(|&n| n > 0).ok_or_else(|| format!("{name} must be a number of pixels, not '{value}'"))
}

/// The format of `cut`: from the name of the output file, or from `--format` (PNG without it) for the standard output.
fn cut_format(output: &Path, format: Option<&str>) -> Result<Format, String> {
    match (output == Path::new("-"), format) {
        (true, f) => parse_format(f),
        (false, None) => Format::from_path(output).ok_or_else(|| "output must be png, jpg, webp, avif, tiff or bmp".to_string()),
        (false, Some(_)) => Err("--format is for the standard output (-): the name of the file gives the format".to_string()),
    }
}

/// The format option of a batch.
fn parse_format(value: Option<&str>) -> Result<Format, String> {
    match value.unwrap_or("png") {
        "png" => Ok(Format::Png),
        "jpg" | "jpeg" => Ok(Format::Jpeg),
        "webp" => Ok(Format::Webp),
        "avif" => Ok(Format::Avif),
        "tif" | "tiff" => Ok(Format::Tiff),
        "bmp" => Ok(Format::Bmp),
        other => Err(format!("unsupported format '{other}'")),
    }
}

/// The model chosen with `--model` (or `$OPENSCISSORS_MODEL`, or as the application picks), on the graphics card
/// when `gpu` is set and one can run it.
fn load_predictor(registry: &ModelRegistry, choice: Option<String>, gpu: bool) -> Result<Predictor, String> {
    let accelerator = if gpu { Accelerator::Auto } else { Accelerator::Cpu };
    let predictor = choose_predictor(registry, choice, accelerator)?;
    if gpu {
        eprintln!("{} runs on the {}", predictor.spec().id, predictor.backend());
    }
    Ok(predictor)
}

fn choose_predictor(registry: &ModelRegistry, choice: Option<String>, accelerator: Accelerator) -> Result<Predictor, String> {
    let choice = choice.or_else(|| std::env::var("OPENSCISSORS_MODEL").ok());
    if let Some(c) = &choice {
        if Path::new(c).is_file() {
            // A bare ONNX file: assume an IS-Net style model (x/255 - 0.5, min-max output).
            let mut spec = registry.available().remove(0);
            spec.mean = [0.5; 3];
            spec.std = [1.0; 3];
            spec.output = openscissors_core::model::OutputNorm::Minmax;
            return Predictor::load_with(&spec, Path::new(c), accelerator).map_err(|e| e.to_string());
        }
    }
    let spec = match &choice {
        Some(id) => registry.get(id).ok_or_else(|| format!("unknown model '{id}'"))?,
        None => {
            // As the application picks: the model chosen there, otherwise the lightest one that fits the memory.
            let budget = openscissors_core::sysmem::budget_mb();
            let ready: Vec<_> = registry
                .available()
                .into_iter()
                .filter(|s| s.kind == ModelKind::Segmenter && registry.state(s) == ModelState::Ready)
                .filter(|s| budget.is_none_or(|b| s.ram_mb as u64 <= b))
                .collect();
            let chosen = Settings::load(&home_dir().join("data").join("settings.json")).chosen_model;
            openscissors_core::model::default_model(&ready, chosen.as_deref())
                .cloned()
                .ok_or("no model downloaded; run `openscissors-cli models download isnet-general-use`")?
        }
    };
    if registry.state(&spec) != ModelState::Ready {
        return Err(format!("model '{}' is not downloaded", spec.id));
    }
    Predictor::load_with(&spec, &registry.model_path(&spec), accelerator).map_err(|e| e.to_string())
}

fn run(args: Vec<String>) -> Result<(), String> {
    let registry = ModelRegistry::new(models_dir());
    let rest = args.get(1..).unwrap_or_default();
    let gpu = args.iter().any(|a| a == "--gpu");
    match args.first().map(String::as_str) {
        Some("--help" | "-h" | "help") => {
            println!("{USAGE}");
            Ok(())
        }
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
            check_options(rest, &["--models"], &["--gpu"])?;
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
                    let predictor = load_predictor(&registry, Some(id.clone()), gpu)?;
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
            check_options(rest, &[], &["--json"])?;
            if args.iter().any(|a| a == "--json") {
                println!("{}", models_json(&registry));
                return Ok(());
            }
            for s in registry.available() {
                let state = if registry.state(&s) == ModelState::Ready { "ready" } else { "not downloaded" };
                println!("{:<18} {:<16} {:>6} MB  {}  {}", s.id, state, s.size_bytes / 1_000_000, s.license, s.name);
            }
            Ok(())
        }
        Some("batch") => {
            check_options(
                rest,
                &["--format", "--quality", "--model", "--report", "--dpi", "--preset", "--bg", "--width", "--height"],
                &["--defaults", "--resume", "--8bit", "--gpu", "--recursive"],
            )?;
            let (src, dst) = (args.get(1).ok_or("source folder expected")?, args.get(2).ok_or("destination folder expected")?);
            let format = parse_format(flag(&args, "--format").as_deref())?;
            let quality = parse_quality(flag(&args, "--quality"))?;
            let dpi = parse_dpi(flag(&args, "--dpi"))?;
            let params = layer_params(&args)?;
            let recursive = args.iter().any(|a| a == "--recursive");
            let files = batch::validate(Path::new(src), Path::new(dst), recursive).map_err(|e| e.to_string())?;
            let predictor = load_predictor(&registry, flag(&args, "--model"), gpu)?;
            let opts = BatchOptions {
                dst: PathBuf::from(dst),
                format,
                quality,
                params,
                skip_existing: args.iter().any(|a| a == "--resume"),
                export_16bit: !args.iter().any(|a| a == "--8bit"),
                dpi,
                source: recursive.then(|| PathBuf::from(src)),
            };
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
        Some("predict") => {
            check_options(rest, &["--model"], &["--gpu"])?;
            let (input, output) = (args.get(1).ok_or("input expected")?, args.get(2).ok_or("output expected")?);
            let predictor = load_predictor(&registry, flag(&args, "--model"), gpu)?;
            let image = load_image(Path::new(input)).map_err(|e| e.to_string())?;
            let mask = predictor.predict(&image, true).map_err(|e| e.to_string())?;
            save_mask_png(Path::new(output), &mask).map_err(|e| e.to_string())
        }
        Some("cut") => {
            check_options(
                rest,
                &["--model", "--quality", "--dpi", "--preset", "--bg", "--width", "--height", "--format"],
                &["--defaults", "--8bit", "--gpu"],
            )?;
            let (input, output) = (args.get(1).ok_or("input expected")?, Path::new(args.get(2).ok_or("output expected")?));
            let format = cut_format(output, flag(&args, "--format").as_deref())?;
            let opts = BatchOptions {
                dst: output.parent().map(Path::to_path_buf).unwrap_or_default(),
                format,
                quality: parse_quality(flag(&args, "--quality"))?,
                params: layer_params(&args)?,
                skip_existing: false,
                export_16bit: !args.iter().any(|a| a == "--8bit"),
                dpi: parse_dpi(flag(&args, "--dpi"))?,
                source: None,
            };
            // `-` reads the picture from the standard input and writes the result to the standard output.
            let picture = if input == "-" {
                let mut bytes = Vec::new();
                std::io::Read::read_to_end(&mut std::io::stdin().lock(), &mut bytes)
                    .map_err(|e| format!("cannot read the standard input: {e}"))?;
                openscissors_core::io::load_image_bytes(&bytes)
            } else {
                openscissors_core::io::load_image_deep(Path::new(input))
            }
            .map_err(|e| e.to_string())?;
            let predictor = load_predictor(&registry, flag(&args, "--model"), gpu)?;
            let result = batch::cut_loaded(picture, &opts, &|img| predictor.predict(img, true)).map_err(|e| e.to_string())?;
            if output == Path::new("-") {
                let mut out = std::io::stdout().lock();
                std::io::Write::write_all(&mut out, &result).and_then(|_| std::io::Write::flush(&mut out))
            } else {
                write_atomic(output, &result)
            }
            .map_err(|e| format!("cannot write {}: {e}", output.display()))
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

#[cfg(test)]
mod tests {
    use super::*;

    fn args(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn unknown_or_incomplete_options_are_refused() {
        let known = (&["--format", "--model"][..], &["--resume"][..]);
        assert!(check_options(&args(&["a", "b", "--format", "png", "--resume"]), known.0, known.1).is_ok());
        assert!(check_options(&args(&["a", "--resum"]), known.0, known.1).unwrap_err().contains("--resum"));
        assert!(check_options(&args(&["a", "--format"]), known.0, known.1).unwrap_err().contains("needs a value"));
        assert!(check_options(&args(&["a", "--model", "--resume"]), known.0, known.1).is_err());
    }

    #[test]
    fn the_format_of_cut_comes_from_the_name_or_from_the_option_for_a_pipe() {
        assert_eq!(cut_format(Path::new("a.webp"), None), Ok(Format::Webp));
        assert_eq!(cut_format(Path::new("-"), None), Ok(Format::Png));
        assert_eq!(cut_format(Path::new("-"), Some("jpg")), Ok(Format::Jpeg));
        assert!(cut_format(Path::new("a.png"), Some("jpg")).is_err(), "the name already says it");
        assert!(cut_format(Path::new("a.txt"), None).is_err());
    }

    #[test]
    fn the_catalogue_reads_as_json() {
        let dir = std::env::temp_dir().join(format!("osc-cli-models-{}", std::process::id()));
        let list: Vec<serde_json::Value> = serde_json::from_str(&models_json(&ModelRegistry::new(dir))).unwrap();
        assert!(list.len() >= 2);
        assert!(list.iter().all(|m| m["id"].is_string() && m["sha256"].is_string() && m["ready"] == false && m.get("path").is_none()));
    }

    #[test]
    fn layer_options_apply_in_order_and_are_checked() {
        let p = layer_params(&args(&["batch", "a", "b", "--preset", "banner", "--bg", "#FF8000", "--height", "500"])).unwrap();
        assert_eq!((p.background.mode, p.background.color), (BgMode::Color, [255, 128, 0]));
        assert_eq!(p.background.aspect_ratio, "16:9", "the preset");
        assert_eq!((p.background.output_width, p.background.output_height), (1920, 500), "--height after the preset");
        let p = layer_params(&args(&["cut", "a", "b", "--preset", "marketplace", "--bg", "transparent"])).unwrap();
        assert_eq!(p.background.mode, BgMode::Transparent, "--bg after the preset");
        assert_eq!(layer_params(&args(&["cut", "a", "b"])).unwrap().background.mode, Params::default().background.mode);
        assert!(layer_params(&args(&["cut", "a", "b", "--preset", "poster"])).unwrap_err().contains("marketplace"), "the ids are listed");
        for bad in [["--bg", "orange"], ["--bg", "#ff80"], ["--bg", "#gg0000"], ["--width", "0"], ["--height", "tall"]] {
            assert!(layer_params(&args(&["cut", "a", "b", bad[0], bad[1]])).is_err(), "{bad:?}");
        }
        assert_eq!((parse_colour("white"), parse_colour("000000")), (Ok(Some([255; 3])), Ok(Some([0; 3]))));
    }

    #[test]
    fn quality_and_format_are_checked() {
        assert_eq!((parse_dpi(None), parse_dpi(Some("300".into()))), (Ok(0), Ok(300)));
        for bad in ["0", "-1", "70000", "high"] {
            assert!(parse_dpi(Some(bad.into())).is_err(), "{bad}");
        }
        assert_eq!(parse_quality(None), Ok(95));
        assert_eq!(parse_quality(Some("80".into())), Ok(80));
        for bad in ["5", "101", "200", "x"] {
            assert!(parse_quality(Some(bad.into())).is_err(), "{bad}");
        }
        assert_eq!(parse_format(Some("tiff")), Ok(Format::Tiff));
        assert_eq!(parse_format(Some("bmp")), Ok(Format::Bmp));
        assert_eq!(parse_format(None), Ok(Format::Png));
        assert!(parse_format(Some("gif")).is_err());
    }

    #[test]
    fn help_is_not_an_error() {
        assert!(run(args(&["--help"])).is_ok());
        assert!(run(args(&["nonsense"])).is_err());
    }
}
