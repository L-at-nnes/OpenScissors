//! Batch processing: cut out every image of a folder with the same settings and no markers.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;

use rayon::prelude::*;

use serde::Serialize;

use crate::document::Document;
use crate::error::{Error, Result};
use crate::io::{encode_rgba, encode_rgba16, encode_rgba_within, load_image_deep, write_atomic, Format, Loaded, IMAGE_EXTENSIONS};
use crate::params::{BgMode, Params};
use crate::types::{Image, Mask};

pub struct BatchOptions {
    pub dst: PathBuf,
    pub format: Format,
    /// JPEG quality, 10-100.
    pub quality: u8,
    /// Layer settings applied to every image (normally the user's saved defaults).
    pub params: Params,
    /// Leave alone the images whose result (`name.ext`) is already in the destination: running the same batch again
    /// after an interruption or a failure then only does what is left.
    pub skip_existing: bool,
    /// Write 16-bit PNG or TIFF for a 16-bit source when the settings allow it, as the interactive export does.
    pub export_16bit: bool,
    /// Print resolution written in PNG, JPEG and TIFF files, 0 for none.
    pub dpi: u16,
    /// The source folder when its subfolders are processed too ([`validate`] with `subfolders`): the result of
    /// `source/a/b.jpg` is written to `dst/a/b.png`. `None` for the pictures of one folder.
    pub source: Option<PathBuf>,
    /// How many pictures are processed at once (1: one after the other). The model runs one picture at a time; the
    /// reading, the rendering and the writing of the others go on meanwhile. Each picture needs its own memory.
    pub jobs: usize,
    /// Added to the name of every result: `photo.jpg` gives `photo-cutout.png` with `-cutout`.
    pub suffix: String,
    /// The largest size of a JPEG, WebP or AVIF result in bytes, reached by lowering the quality; 0 for none.
    pub max_bytes: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct BatchFailure {
    pub file: String,
    pub error: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum BatchStatus {
    Done,
    Skipped,
    Failed,
}

/// What happened to one image.
#[derive(Clone, Debug, Serialize)]
pub struct BatchItem {
    pub file: String,
    pub status: BatchStatus,
    /// The file written (done) or the one that was already there (skipped).
    pub output: Option<String>,
    pub error: Option<String>,
    pub millis: u64,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct BatchReport {
    pub total: usize,
    pub processed: usize,
    pub skipped: usize,
    pub failed: Vec<BatchFailure>,
    pub cancelled: bool,
    /// One entry per image that was looked at, in order.
    pub items: Vec<BatchItem>,
}

impl BatchReport {
    /// The report as JSON text, for a machine to read.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }
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

/// Image files inside `dir` and all its subfolders but `skip` (the destination, when it is inside), sorted by path.
pub fn list_images_tree(dir: &Path, skip: &Path) -> Result<Vec<PathBuf>> {
    let skip = std::fs::canonicalize(skip).ok();
    let mut files = Vec::new();
    let mut folders = vec![dir.to_path_buf()];
    while let Some(folder) = folders.pop() {
        if skip.is_some() && std::fs::canonicalize(&folder).ok() == skip {
            continue;
        }
        files.extend(list_images(&folder)?);
        folders.extend(std::fs::read_dir(&folder)?.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.is_dir()));
    }
    files.sort_by_key(|p| p.strip_prefix(dir).unwrap_or(p).to_string_lossy().to_lowercase());
    Ok(files)
}

/// Check the folders and return the images to process, those of the subfolders too with `subfolders`. The destination
/// is created when missing, once everything else is known to be right (a refused batch leaves no empty folder behind).
pub fn validate(src: &Path, dst: &Path, subfolders: bool) -> Result<Vec<PathBuf>> {
    if !src.is_dir() {
        return Err(Error::msg("The source folder does not exist."));
    }
    // A destination that does not exist yet cannot be the source, which does.
    if dst.exists() && std::fs::canonicalize(src)? == std::fs::canonicalize(dst)? {
        return Err(Error::msg("The source and destination folders must be different."));
    }
    let files = if subfolders { list_images_tree(src, dst)? } else { list_images(src)? };
    if files.is_empty() {
        return Err(Error::msg("The source folder contains no supported image (jpg, png, bmp, tif, webp, gif)."));
    }
    std::fs::create_dir_all(dst)?;
    Ok(files)
}

/// The suffix ([`BatchOptions::suffix`]) must leave every result a plain file name in its folder.
pub fn check_suffix(suffix: &str) -> Result<()> {
    if suffix.chars().any(|c| c.is_control() || r#"\/:*?"<>|"#.contains(c)) {
        return Err(Error::msg(r#"The suffix cannot hold \ / : * ? " < > | (it is added to the file names)."#));
    }
    Ok(())
}

/// `dir/stem.ext`, or `stem_1.ext`, `stem_2.ext`... when that name is taken. Nothing is overwritten.
pub fn unique_path(dir: &Path, stem: &str, ext: &str) -> PathBuf {
    let first = dir.join(format!("{stem}.{ext}"));
    if !first.exists() {
        return first;
    }
    (1..).map(|n| dir.join(format!("{stem}_{n}.{ext}"))).find(|p| !p.exists()).expect("an unused name exists")
}

/// The file name of the result of each of `files`: `stem<suffix>.ext`, or `stem_<source extension><suffix>.ext` when
/// several sources share a stem (`a.jpg` and `a.png`; letter case is ignored, as on Windows). Every source then has a
/// result of its own, always the same one, so that [`BatchOptions::skip_existing`] finds it on the next run.
pub fn output_names(files: &[PathBuf], suffix: &str, ext: &str) -> Vec<String> {
    let stem = |p: &Path| p.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "image".into());
    let mut count: HashMap<String, usize> = HashMap::new();
    for f in files {
        *count.entry(stem(f).to_lowercase()).or_default() += 1;
    }
    let mut taken: HashSet<String> = HashSet::new();
    files
        .iter()
        .map(|f| {
            let s = stem(f);
            let base = if count[&s.to_lowercase()] > 1 {
                format!("{s}_{}{suffix}", f.extension().map(|e| e.to_string_lossy().to_lowercase()).unwrap_or_default())
            } else {
                format!("{s}{suffix}")
            };
            let mut name = format!("{base}.{ext}");
            let mut n = 1;
            while !taken.insert(name.to_lowercase()) {
                name = format!("{base}_{n}.{ext}");
                n += 1;
            }
            name
        })
        .collect()
}

/// What `process_one` did.
enum Outcome {
    Written(PathBuf),
    AlreadyThere(PathBuf),
}

/// The folder of `file` under the source, when the subfolders are processed ([`BatchOptions::source`]).
fn subfolder<'a>(file: &'a Path, opts: &BatchOptions) -> &'a Path {
    opts.source.as_deref().and_then(|s| file.parent()?.strip_prefix(s).ok()).unwrap_or(Path::new(""))
}

fn process_one(file: &Path, name: &str, opts: &BatchOptions, predict: &dyn Fn(&Image) -> Result<Mask>) -> Result<Outcome> {
    let dir = opts.dst.join(subfolder(file, opts));
    let existing = dir.join(name);
    if opts.skip_existing && existing.is_file() {
        return Ok(Outcome::AlreadyThere(existing));
    }
    std::fs::create_dir_all(&dir)?;
    let stem = name.strip_suffix(&format!(".{}", opts.format.extension())).unwrap_or(name);
    let target = unique_path(&dir, stem, opts.format.extension());
    cut_file(file, &target, opts, predict)?;
    Ok(Outcome::Written(target))
}

/// Cut out `file` and write the result to `target`, as every image of a batch (`opts.dst` and `opts.skip_existing`
/// play no part).
pub fn cut_file(file: &Path, target: &Path, opts: &BatchOptions, predict: &dyn Fn(&Image) -> Result<Mask>) -> Result<()> {
    // Opened as the window opens it: a picture that is already cut out keeps its own transparency (no model runs on
    // it), a colour profile is converted to sRGB and said in the result, 16-bit samples are kept.
    write_atomic(target, &cut_loaded(load_image_deep(file)?, opts, predict)?)?;
    Ok(())
}

/// Cut out a picture already read (from a pipe, say) as [`cut_file`] does, and return the file of the result in
/// `opts.format`.
pub fn cut_loaded(loaded: Loaded, opts: &BatchOptions, predict: &dyn Fn(&Image) -> Result<Mask>) -> Result<Vec<u8>> {
    let mask = match loaded.alpha {
        Some(alpha) => alpha,
        None => predict(&loaded.image)?,
    };
    let mut params = opts.params.clone();
    params.crop = None; // a crop belongs to one image, not to a style
    let flatten = if params.background.mode == BgMode::Color { params.background.color } else { [255; 3] };
    let mut doc = Document::new(0, loaded.image, None).with_deep(loaded.deep).with_params(params);
    doc.set_prediction(Some(mask));
    let job = doc.export_job();
    job.check_memory()?;
    let deep = (opts.export_16bit && matches!(opts.format, Format::Png | Format::Tiff)).then(|| job.run_deep()).flatten();
    match deep {
        Some(deep) => encode_rgba16(&deep, opts.format, loaded.converted, opts.dpi),
        None if opts.max_bytes > 0 => encode_rgba_within(&job.run(), opts.format, opts.max_bytes, flatten, loaded.converted, opts.dpi),
        None => encode_rgba(&job.run(), opts.format, opts.quality, flatten, loaded.converted, opts.dpi),
    }
}

/// Process `files`, [`BatchOptions::jobs`] at a time. `predict` produces the mask of an image; `cancel` is polled
/// before each file. A failing file is reported and does not stop the batch. `progress` is called on the calling
/// thread when a file starts and at the end; the report lists the files in their order.
pub fn run_batch(
    files: &[PathBuf],
    opts: &BatchOptions,
    predict: &(dyn Fn(&Image) -> Result<Mask> + Sync),
    progress: &mut dyn FnMut(BatchProgress),
    cancel: &AtomicBool,
) -> BatchReport {
    let mut report = BatchReport { total: files.len(), ..Default::default() };
    // Names are chosen folder by folder: `a/x.jpg` and `b/x.jpg` both give `x.png`, each in its own folder.
    let mut names = vec![String::new(); files.len()];
    let mut by_folder: HashMap<&Path, Vec<usize>> = HashMap::new();
    for (i, f) in files.iter().enumerate() {
        by_folder.entry(subfolder(f, opts)).or_default().push(i);
    }
    for indices in by_folder.values() {
        let group: Vec<PathBuf> = indices.iter().map(|&i| files[i].clone()).collect();
        for (&i, name) in indices.iter().zip(output_names(&group, &opts.suffix, opts.format.extension())) {
            names[i] = name;
        }
    }
    // One file: `(i, None)` is sent when it starts, `(i, Some(item))` when it is done.
    let work = |i: usize, events: &mpsc::Sender<(usize, Option<BatchItem>)>| {
        if cancel.load(Ordering::Relaxed) {
            return;
        }
        let _ = events.send((i, None));
        let file = &files[i];
        let started = std::time::Instant::now();
        // In the report: the name of the file, with its subfolder when there is one (`a/b.jpg`).
        let name =
            subfolder(file, opts).join(file.file_name().unwrap_or_default()).to_string_lossy().replace(std::path::MAIN_SEPARATOR, "/");
        let (status, output, error) = match process_one(file, &names[i], opts, predict) {
            Ok(Outcome::Written(p)) => (BatchStatus::Done, Some(p.display().to_string()), None),
            Ok(Outcome::AlreadyThere(p)) => (BatchStatus::Skipped, Some(p.display().to_string()), None),
            Err(e) => (BatchStatus::Failed, None, Some(e.to_string())),
        };
        let _ = events.send((i, Some(BatchItem { file: name, status, output, error, millis: started.elapsed().as_millis() as u64 })));
    };
    let jobs = opts.jobs.clamp(1, files.len().max(1));
    let mut items: Vec<Option<BatchItem>> = vec![None; files.len()];
    let (events, received) = mpsc::channel();
    std::thread::scope(|s| {
        s.spawn(move || {
            let pool = (jobs > 1).then(|| rayon::ThreadPoolBuilder::new().num_threads(jobs).build().ok()).flatten();
            match pool {
                Some(pool) => pool.install(|| (0..files.len()).into_par_iter().for_each_with(events, |events, i| work(i, events))),
                None => (0..files.len()).for_each(|i| work(i, &events)),
            }
        });
        let mut done = 0;
        for (i, item) in received {
            match item {
                None => progress(BatchProgress { done, total: files.len(), current: Some(&files[i]) }),
                Some(item) => {
                    items[i] = Some(item);
                    done += 1;
                }
            }
        }
    });
    for item in items.into_iter().flatten() {
        match item.status {
            BatchStatus::Done => report.processed += 1,
            BatchStatus::Skipped => report.skipped += 1,
            BatchStatus::Failed => {
                report.failed.push(BatchFailure { file: item.file.clone(), error: item.error.clone().unwrap_or_default() })
            }
        }
        report.items.push(item);
    }
    report.cancelled = report.items.len() < files.len();
    progress(BatchProgress { done: report.items.len(), total: files.len(), current: None });
    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::save_rgba;
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
        BatchOptions {
            dst: dst.into(),
            format,
            quality: 90,
            params: Params::default(),
            skip_existing: false,
            export_16bit: true,
            dpi: 0,
            source: None,
            jobs: 1,
            suffix: String::new(),
            max_bytes: 0,
        }
    }

    #[test]
    fn several_pictures_at_once_give_what_one_at_a_time_gives() {
        let src = tmp("jobs-src");
        for i in 0..6 {
            write_png(&src.join(format!("p{i}.png")), 8 + i, 5);
        }
        std::fs::write(src.join("broken.png"), b"not a picture").unwrap();
        let files = list_images(&src).unwrap();
        let run = |jobs: usize| {
            let dst = tmp(&format!("jobs-dst-{jobs}"));
            let mut starts = 0;
            let report = run_batch(
                &files,
                &BatchOptions { jobs, ..opts(&dst, Format::Png) },
                &half_mask,
                &mut |p| starts += p.current.is_some() as usize,
                &AtomicBool::new(false),
            );
            let written: Vec<Vec<u8>> = (0..6).map(|i| std::fs::read(dst.join(format!("p{i}.png"))).unwrap()).collect();
            std::fs::remove_dir_all(dst).ok();
            (report, written, starts)
        };
        let (one, one_files, _) = run(1);
        let (four, four_files, starts) = run(4);
        assert_eq!((four.processed, four.failed.len(), four.cancelled), (6, 1, false));
        assert_eq!(starts, 7, "every file says when it starts");
        let order = |r: &BatchReport| r.items.iter().map(|i| (i.file.clone(), i.status)).collect::<Vec<_>>();
        assert_eq!(order(&four), order(&one), "the report keeps the order of the files");
        assert!(four_files == one_files, "the same results");
        // Cancelled before it starts: nothing is done, and the report says so.
        let dst = tmp("jobs-cancel");
        let r = run_batch(&files, &BatchOptions { jobs: 3, ..opts(&dst, Format::Png) }, &half_mask, &mut |_| {}, &AtomicBool::new(true));
        assert!(r.cancelled && r.items.is_empty());
        std::fs::remove_dir_all(dst).ok();
        std::fs::remove_dir_all(src).ok();
    }

    #[test]
    fn subfolders_keep_their_tree_and_the_destination_inside_is_left_out() {
        let src = tmp("tree");
        for f in ["a.png", "sub/b.png", "sub/deeper/c.png", "other/b.png"] {
            std::fs::create_dir_all(src.join(f).parent().unwrap()).unwrap();
            write_png(&src.join(f), 6, 4);
        }
        let dst = src.join("out");
        assert_eq!(validate(&src, &dst, false).unwrap().len(), 1, "without subfolders: only a.png");
        let files = validate(&src, &dst, true).unwrap();
        assert_eq!(files.len(), 4);
        let o = BatchOptions { source: Some(src.clone()), ..opts(&dst, Format::Png) };
        let report = run_batch(&files, &o, &half_mask, &mut |_| {}, &AtomicBool::new(false));
        assert_eq!(report.processed, 4, "{:?}", report.failed);
        for f in ["a.png", "sub/b.png", "sub/deeper/c.png", "other/b.png"] {
            assert!(dst.join(f).is_file(), "{f}: the same subfolder, the same name (b.png twice, each in its folder)");
        }
        let names: Vec<&str> = report.items.iter().map(|i| i.file.as_str()).collect();
        assert_eq!(names, ["a.png", "other/b.png", "sub/b.png", "sub/deeper/c.png"], "sorted by path, with the subfolder");
        // Run again: the results in out/ are not taken as pictures to cut out, and --resume finds them.
        let again = validate(&src, &dst, true).unwrap();
        assert_eq!(again.len(), 4, "the destination inside the source is left out");
        let resumed = run_batch(&again, &BatchOptions { skip_existing: true, ..o }, &half_mask, &mut |_| {}, &AtomicBool::new(false));
        assert_eq!((resumed.processed, resumed.skipped), (0, 4));
        std::fs::remove_dir_all(src).ok();
    }

    #[test]
    fn validation_rules() {
        let root = tmp("val");
        let (src, dst) = (root.join("src"), root.join("dst"));
        std::fs::create_dir_all(&src).unwrap();
        assert!(validate(&root.join("missing"), &dst, false).is_err(), "source must exist");
        assert!(validate(&src, &src, false).is_err(), "folders must differ");
        assert!(validate(&src, &dst, false).is_err(), "no image yet");
        assert!(!dst.exists(), "a refused batch creates no folder");
        std::fs::write(src.join("notes.txt"), "x").unwrap();
        assert!(validate(&src, &dst, false).is_err(), "text files are not images");
        write_png(&src.join("B.PNG"), 8, 8);
        write_png(&src.join("a.png"), 8, 8);
        let files = validate(&src, &dst, false).unwrap();
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
        let files = validate(&src, &dst, false).unwrap();
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
    fn running_again_with_skip_existing_only_does_what_is_left_and_the_report_says_so() {
        let root = tmp("resume");
        let (src, dst) = (root.join("src"), root.join("dst"));
        std::fs::create_dir_all(&src).unwrap();
        write_png(&src.join("one.png"), 12, 12);
        write_png(&src.join("two.png"), 12, 12);
        std::fs::write(src.join("three.png"), "not an image yet").unwrap();
        let files = validate(&src, &dst, false).unwrap();
        let mut o = opts(&dst, Format::Png);
        o.skip_existing = true;
        let first = run_batch(&files, &o, &half_mask, &mut |_| {}, &AtomicBool::new(false));
        assert_eq!((first.processed, first.skipped, first.failed.len()), (2, 0, 1));

        // The broken file is repaired; the next run leaves the two finished results alone.
        write_png(&src.join("three.png"), 12, 12);
        let second = run_batch(&files, &o, &half_mask, &mut |_| {}, &AtomicBool::new(false));
        assert_eq!((second.processed, second.skipped, second.failed.len()), (1, 2, 0));
        assert!(!dst.join("one_1.png").exists(), "nothing was written twice");
        let statuses: Vec<_> = second.items.iter().map(|i| (i.file.as_str(), i.status)).collect();
        assert_eq!(statuses, [("one.png", BatchStatus::Skipped), ("three.png", BatchStatus::Done), ("two.png", BatchStatus::Skipped)]);

        let json: serde_json::Value = serde_json::from_str(&second.to_json()).unwrap();
        assert_eq!((json["processed"].as_u64(), json["skipped"].as_u64()), (Some(1), Some(2)));
        assert_eq!(json["items"][1]["status"], "done");
        assert!(json["items"][1]["output"].as_str().unwrap().ends_with("three.png"));
        assert!(json["items"][0]["error"].is_null());
        let failed = serde_json::from_str::<serde_json::Value>(&first.to_json()).unwrap();
        assert_eq!(failed["items"][1]["status"], "failed");
        assert!(failed["items"][1]["error"].as_str().is_some());
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn sources_sharing_a_name_each_get_their_own_result_and_resume_finds_it() {
        let root = tmp("samestem");
        let (src, dst) = (root.join("src"), root.join("dst"));
        std::fs::create_dir_all(&src).unwrap();
        write_png(&src.join("a.png"), 12, 12);
        write_png(&src.join("A.jpg"), 14, 14); // the content decides the decoder, the name only the output
        write_png(&src.join("b.png"), 10, 10);
        let files = validate(&src, &dst, false).unwrap();
        let mut o = opts(&dst, Format::Png);
        o.skip_existing = true;
        let first = run_batch(&files, &o, &half_mask, &mut |_| {}, &AtomicBool::new(false));
        assert_eq!((first.processed, first.skipped, first.failed.len()), (3, 0, 0), "{first:?}");
        let mut names: Vec<String> =
            std::fs::read_dir(&dst).unwrap().map(|e| e.unwrap().file_name().to_string_lossy().into_owned()).collect();
        names.sort();
        assert_eq!(names, ["A_jpg.png", "a_png.png", "b.png"]);
        assert_eq!(image::open(dst.join("A_jpg.png")).unwrap().width(), 14);
        let again = run_batch(&files, &o, &half_mask, &mut |_| {}, &AtomicBool::new(false));
        assert_eq!((again.processed, again.skipped), (0, 3), "everything is found on the second run");
        assert_eq!(output_names(&[PathBuf::from("x.png"), PathBuf::from("y.png")], "", "webp"), ["x.webp", "y.webp"]);
        let names = output_names(&[PathBuf::from("a.jpg"), PathBuf::from("a.png"), PathBuf::from("b.png")], "-cutout", "png");
        assert_eq!(names, ["a_jpg-cutout.png", "a_png-cutout.png", "b-cutout.png"], "the suffix comes last");
        assert!(check_suffix(" (cut out)").is_ok() && check_suffix("").is_ok());
        for bad in ["/../x", "\\x", "a:b", "x?", "\n"] {
            assert!(check_suffix(bad).is_err(), "{bad:?} could leave the folder or is no file name");
        }
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn a_cut_out_or_16_bit_source_comes_out_as_the_window_would_export_it() {
        let root = tmp("asopen");
        let (src, dst) = (root.join("src"), root.join("dst"));
        std::fs::create_dir_all(&src).unwrap();
        // Already cut out: the model is not asked, the transparency is kept.
        let mut cut = Rgba::new(8, 4);
        for (i, p) in cut.data.chunks_exact_mut(4).enumerate() {
            p.copy_from_slice(&[10, 20, 30, if i % 8 < 3 { 255 } else { 0 }]);
        }
        save_rgba(&src.join("cut.png"), &cut, Format::Png, 95, [255; 3]).unwrap();
        // A 16-bit picture: its samples survive in a PNG result.
        let deep = crate::types::Rgba16 { w: 6, h: 3, data: [40_000, 40_000, 40_000, 65_535].repeat(6 * 3) };
        crate::io::save_rgba16(&src.join("deep.png"), &deep, Format::Png).unwrap();
        let files = validate(&src, &dst, false).unwrap();
        let asked = std::sync::atomic::AtomicUsize::new(0);
        let predict = |img: &Image| {
            asked.fetch_add(1, Ordering::Relaxed);
            Ok(Mask::filled(img.w, img.h, 255))
        };
        let r = run_batch(&files, &opts(&dst, Format::Png), &predict, &mut |_| {}, &AtomicBool::new(false));
        assert_eq!((r.processed, asked.load(Ordering::Relaxed)), (2, 1), "only the picture without transparency needs the model");
        let out = image::open(dst.join("cut.png")).unwrap().to_rgba8();
        assert_eq!((out.get_pixel(1, 0).0[3], out.get_pixel(5, 0).0[3]), (255, 0));
        let d = image::open(dst.join("deep.png")).unwrap();
        assert!(matches!(d, image::DynamicImage::ImageRgba16(_)), "16-bit result");
        assert_eq!(d.to_rgba16().get_pixel(1, 1).0[0], 40_000);
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
        let files = validate(&src, &dst, false).unwrap();
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
        let files = validate(&src, &dst, false).unwrap();
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
