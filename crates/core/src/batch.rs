//! Batch processing: cut out every image of a folder with the same settings and no markers.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use serde::Serialize;

use crate::document::Document;
use crate::error::{Error, Result};
use crate::io::{load_image, save_rgba, Format, IMAGE_EXTENSIONS};
use crate::params::{BgMode, Params};
use crate::types::{Image, Mask};

pub struct BatchOptions {
    pub dst: PathBuf,
    pub format: Format,
    /// JPEG quality, 10-100.
    pub quality: u8,
    /// Layer settings applied to every image (normally the user's saved defaults).
    pub params: Params,
}

#[derive(Clone, Debug, Serialize)]
pub struct BatchFailure {
    pub file: String,
    pub error: String,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct BatchReport {
    pub total: usize,
    pub processed: usize,
    pub failed: Vec<BatchFailure>,
    pub cancelled: bool,
}

#[derive(Clone, Debug)]
pub struct BatchProgress<'a> {
    /// Files finished so far (successfully or not).
    pub done: usize,
    pub total: usize,
    /// File about to be processed, `None` at the end.
    pub current: Option<&'a Path>,
}

/// Image files directly inside `dir`, sorted by name.
pub fn list_images(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.is_file()
                && p.extension().and_then(|e| e.to_str()).is_some_and(|e| IMAGE_EXTENSIONS.contains(&e.to_ascii_lowercase().as_str()))
        })
        .collect();
    files.sort_by_key(|p| p.file_name().map(|n| n.to_string_lossy().to_lowercase()));
    Ok(files)
}

/// Check the folders and return the images to process. The destination is created when missing.
pub fn validate(src: &Path, dst: &Path) -> Result<Vec<PathBuf>> {
    if !src.is_dir() {
        return Err(Error::msg("The source folder does not exist."));
    }
    std::fs::create_dir_all(dst)?;
    if std::fs::canonicalize(src)? == std::fs::canonicalize(dst)? {
        return Err(Error::msg("The source and destination folders must be different."));
    }
    let files = list_images(src)?;
    if files.is_empty() {
        return Err(Error::msg("The source folder contains no supported image (jpg, png, bmp, tif, webp)."));
    }
    Ok(files)
}

/// `dir/stem.ext`, or `stem_1.ext`, `stem_2.ext`... when that name is taken. Nothing is overwritten.
pub fn unique_path(dir: &Path, stem: &str, ext: &str) -> PathBuf {
    let first = dir.join(format!("{stem}.{ext}"));
    if !first.exists() {
        return first;
    }
    (1..).map(|n| dir.join(format!("{stem}_{n}.{ext}"))).find(|p| !p.exists()).expect("an unused name exists")
}

fn process_one(file: &Path, opts: &BatchOptions, predict: &dyn Fn(&Image) -> Result<Mask>) -> Result<()> {
    let image = load_image(file)?;
    let mask = predict(&image)?;
    let mut params = opts.params.clone();
    params.crop = None; // a crop belongs to one image, not to a style
    let flatten = if params.background.mode == BgMode::Color { params.background.color } else { [255; 3] };
    let mut doc = Document::new(0, image, Some(file.to_path_buf())).with_params(params);
    doc.set_prediction(Some(mask));
    let out = doc.export();
    let stem = file.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "image".into());
    save_rgba(&unique_path(&opts.dst, &stem, opts.format.extension()), &out, opts.format, opts.quality, flatten)
}

/// Process `files` one after the other. `predict` produces the mask of an image; `cancel` is
/// polled between files. A failing file is reported and does not stop the batch.
pub fn run_batch(
    files: &[PathBuf],
    opts: &BatchOptions,
    predict: &dyn Fn(&Image) -> Result<Mask>,
    progress: &mut dyn FnMut(BatchProgress),
    cancel: &AtomicBool,
) -> BatchReport {
    let mut report = BatchReport { total: files.len(), ..Default::default() };
    for (i, file) in files.iter().enumerate() {
        if cancel.load(Ordering::Relaxed) {
            report.cancelled = true;
            break;
        }
        progress(BatchProgress { done: i, total: files.len(), current: Some(file) });
        match process_one(file, opts, predict) {
            Ok(()) => report.processed += 1,
            Err(e) => report.failed.push(BatchFailure {
                file: file.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(),
                error: e.to_string(),
            }),
        }
    }
    progress(BatchProgress { done: report.processed + report.failed.len(), total: files.len(), current: None });
    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Rgba;

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("osc-batch-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn write_png(path: &Path, w: usize, h: usize) {
        let mut rgba = Rgba::new(w, h);
        for p in rgba.data.chunks_exact_mut(4) {
            p.copy_from_slice(&[200, 100, 50, 255]);
        }
        save_rgba(path, &rgba, Format::Png, 95, [255; 3]).unwrap();
    }

    /// Everything left of the middle is foreground.
    fn half_mask(img: &Image) -> Result<Mask> {
        let mut m = Mask::new(img.w, img.h);
        for y in 0..img.h {
            for x in 0..img.w / 2 {
                m.data[y * img.w + x] = 255;
            }
        }
        Ok(m)
    }

    fn opts(dst: &Path, format: Format) -> BatchOptions {
        BatchOptions { dst: dst.into(), format, quality: 90, params: Params::default() }
    }

    #[test]
    fn validation_rules() {
        let root = tmp("val");
        let (src, dst) = (root.join("src"), root.join("dst"));
        std::fs::create_dir_all(&src).unwrap();
        assert!(validate(&root.join("missing"), &dst).is_err(), "source must exist");
        assert!(validate(&src, &src).is_err(), "folders must differ");
        assert!(validate(&src, &dst).is_err(), "no image yet");
        std::fs::write(src.join("notes.txt"), "x").unwrap();
        assert!(validate(&src, &dst).is_err(), "text files are not images");
        write_png(&src.join("B.PNG"), 8, 8);
        write_png(&src.join("a.png"), 8, 8);
        let files = validate(&src, &dst).unwrap();
        assert_eq!(files.iter().map(|f| f.file_name().unwrap().to_string_lossy().into_owned()).collect::<Vec<_>>(), ["a.png", "B.PNG"]);
        assert!(dst.is_dir(), "destination is created");
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn unique_names_never_overwrite() {
        let dir = tmp("uniq");
        assert_eq!(unique_path(&dir, "cat", "png"), dir.join("cat.png"));
        std::fs::write(dir.join("cat.png"), "x").unwrap();
        assert_eq!(unique_path(&dir, "cat", "png"), dir.join("cat_1.png"));
        std::fs::write(dir.join("cat_1.png"), "x").unwrap();
        assert_eq!(unique_path(&dir, "cat", "png"), dir.join("cat_2.png"));
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn processes_every_image_reports_failures_and_keeps_going() {
        let root = tmp("run");
        let (src, dst) = (root.join("src"), root.join("dst"));
        std::fs::create_dir_all(&src).unwrap();
        write_png(&src.join("one.png"), 20, 10);
        write_png(&src.join("two.png"), 30, 12);
        std::fs::write(src.join("broken.png"), "not an image").unwrap();
        let files = validate(&src, &dst).unwrap();
        let mut seen = Vec::new();
        let report = run_batch(
            &files,
            &opts(&dst, Format::Png),
            &half_mask,
            &mut |p| seen.push((p.done, p.total, p.current.is_some())),
            &AtomicBool::new(false),
        );
        assert_eq!((report.total, report.processed, report.failed.len(), report.cancelled), (3, 2, 1, false));
        assert_eq!(report.failed[0].file, "broken.png");
        assert_eq!(seen.last(), Some(&(3, 3, false)), "final progress event");
        let out = image::open(dst.join("one.png")).unwrap().to_rgba8();
        assert_eq!(out.dimensions(), (20, 10));
        assert_eq!(out.get_pixel(2, 5).0[3], 255);
        assert_eq!(out.get_pixel(18, 5).0[3], 0, "right half is transparent");

        // Running again never overwrites.
        let again = run_batch(&files, &opts(&dst, Format::Png), &half_mask, &mut |_| {}, &AtomicBool::new(false));
        assert_eq!(again.processed, 2);
        assert!(dst.join("one_1.png").exists() && dst.join("two_1.png").exists());
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn jpeg_output_is_flattened_and_cancel_stops_between_files() {
        let root = tmp("jpg");
        let (src, dst) = (root.join("src"), root.join("dst"));
        std::fs::create_dir_all(&src).unwrap();
        for n in ["a", "b", "c"] {
            write_png(&src.join(format!("{n}.png")), 16, 16);
        }
        let files = validate(&src, &dst).unwrap();
        let cancel = AtomicBool::new(false);
        let report = run_batch(
            &files,
            &opts(&dst, Format::Jpeg),
            &half_mask,
            &mut |p| {
                if p.done == 1 {
                    cancel.store(true, Ordering::Relaxed); // user presses Cancel during the second file
                }
            },
            &cancel,
        );
        assert!(report.cancelled && report.processed == 2, "{report:?}");
        let jpg = image::open(dst.join("a.jpg")).unwrap().to_rgb8();
        assert!(jpg.get_pixel(14, 8).0.iter().all(|&v| v > 235), "transparent area flattened on white");
        assert!(!dst.join("c.jpg").exists());
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn styles_apply_and_a_failing_predictor_is_reported() {
        let root = tmp("style");
        let (src, dst) = (root.join("src"), root.join("dst"));
        std::fs::create_dir_all(&src).unwrap();
        write_png(&src.join("x.png"), 10, 10);
        let files = validate(&src, &dst).unwrap();
        let mut o = opts(&dst, Format::Png);
        o.params.background.mode = BgMode::Color;
        o.params.background.color = [1, 2, 3];
        let r = run_batch(&files, &o, &half_mask, &mut |_| {}, &AtomicBool::new(false));
        assert_eq!(r.processed, 1);
        assert_eq!(image::open(dst.join("x.png")).unwrap().to_rgba8().get_pixel(9, 9).0, [1, 2, 3, 255]);
        let r = run_batch(&files, &o, &|_| Err(Error::msg("model exploded")), &mut |_| {}, &AtomicBool::new(false));
        assert_eq!((r.processed, r.failed.len()), (0, 1));
        assert!(r.failed[0].error.contains("exploded"));
        std::fs::remove_dir_all(root).ok();
    }
}
