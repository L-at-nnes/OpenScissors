//! Performance audit against the targets of the specification (section 9).
//!
//! `cargo run --release -p openscissors-core --example perf_audit [-- <models_dir> <model_id>]`
//!
//! Memory is the peak of heap bytes held at once (a counting allocator), image buffers included.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use openscissors_core::document::Document;
use openscissors_core::mask::graphcut::{graphcut_alpha, GraphcutOptions};
use openscissors_core::mask::matting::hair_matting;
use openscissors_core::mask::{Stroke, StrokeKind};
use openscissors_core::model::predict::upsample_guided;
use openscissors_core::types::{Image, Mask, Rect};

struct Counting;
static CURRENT: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let p = System.alloc(l);
        if !p.is_null() {
            let now = CURRENT.fetch_add(l.size(), Ordering::Relaxed) + l.size();
            PEAK.fetch_max(now, Ordering::Relaxed);
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        System.dealloc(p, l);
        CURRENT.fetch_sub(l.size(), Ordering::Relaxed);
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
        let q = System.realloc(p, l, new);
        if !q.is_null() {
            if new >= l.size() {
                let now = CURRENT.fetch_add(new - l.size(), Ordering::Relaxed) + (new - l.size());
                PEAK.fetch_max(now, Ordering::Relaxed);
            } else {
                CURRENT.fetch_sub(l.size() - new, Ordering::Relaxed);
            }
        }
        q
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

fn photo(w: usize, h: usize) -> Image {
    let mut img = Image::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let inside = (x as f64 - w as f64 / 2.0).powi(2) + (y as f64 - h as f64 / 2.0).powi(2) < (h as f64 / 3.0).powi(2);
            let n = ((x * 31 + y * 17) % 13) as u8;
            let c: [u8; 3] = if inside { [200 + n, 70, 60] } else { [50, 90 + n, 170] };
            img.data[(y * w + x) * 3..(y * w + x) * 3 + 3].copy_from_slice(&c);
        }
    }
    img
}

fn disc_mask(w: usize, h: usize) -> Mask {
    let mut m = Mask::new(w, h);
    for y in 0..h {
        for x in 0..w {
            if (x as f64 - w as f64 / 2.0).powi(2) + (y as f64 - h as f64 / 2.0).powi(2) < (h as f64 / 3.0).powi(2) {
                m.data[y * w + x] = 255;
            }
        }
    }
    m
}

fn best_of<F: FnMut()>(n: usize, mut f: F) -> Duration {
    (0..n)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed()
        })
        .min()
        .expect("n > 0")
}

struct Row {
    what: &'static str,
    target: &'static str,
    got: String,
    ok: bool,
}

fn row(what: &'static str, target_ms: f64, got: Duration) -> Row {
    let ms = got.as_secs_f64() * 1000.0;
    Row { what, target: Box::leak(format!("<= {target_ms} ms").into_boxed_str()), got: format!("{ms:.0} ms"), ok: ms <= target_ms }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut rows = Vec::new();
    let (w, h) = (4000usize, 3000usize);
    let img = photo(w, h);
    let mask = disc_mask(w, h);

    // Guided upsampling of a low-resolution mask to full format.
    let low = openscissors_core::imaging::resize::resize_mask(&mask, 1024, 768, openscissors_core::imaging::resize::Interp::Area);
    rows.push(row("guided upsampling, 12 Mpx", 500.0, best_of(3, || drop(upsample_guided(&img, &low)))));

    // Graph-cut (reduced to 600 px internally).
    let (mut fg, mut bg) = (Mask::new(w, h), Mask::new(w, h));
    for y in 1400..1600 {
        for x in 1900..2100 {
            fg.data[y * w + x] = 255;
        }
    }
    for y in 50..250 {
        for x in 50..250 {
            bg.data[y * w + x] = 255;
        }
    }
    rows.push(row("graph-cut, 12 Mpx", 800.0, best_of(3, || drop(graphcut_alpha(&img, None, &fg, &bg, &GraphcutOptions::default())))));

    // Hair matting on a band along the subject's edge.
    let mut hair = Mask::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let d = ((x as f64 - 2000.0).powi(2) + (y as f64 - 1500.0).powi(2)).sqrt();
            if (d - 1000.0).abs() < 60.0 {
                hair.data[y * w + x] = 255;
            }
        }
    }
    rows.push(row("hair matting, 12 Mpx", 1500.0, best_of(3, || drop(hair_matting(&img, &mask, &hair, 0)))));

    // Interactive preview: a stroke, then the visible region of both views.
    let mut doc = Document::new(1, img.clone(), None);
    doc.set_prediction(Some(mask.clone()));
    doc.set_param("shadow.enabled", serde_json::json!(true)).unwrap();
    doc.set_param("foreground.feather", serde_json::json!(3)).unwrap();
    let view = Rect::new(500.0, 400.0, 2500.0, 1900.0); // what a 1000 x 750 view shows at 50 %
    let _ = doc.render_result(view, 1000, 750); // warm caches
    let _ = doc.render_source(view, 1000, 750, false);
    let mut n = 0.0f32;
    rows.push(row(
        "preview after a stroke (both views), 12 Mpx",
        50.0,
        best_of(5, || {
            n += 40.0;
            doc.add_stroke(Stroke { kind: StrokeKind::Keep, width: 40, points: vec![[1500.0 + n, 1500.0], [1600.0 + n, 1600.0]] });
            let _ = doc.render_result(view, 1000, 750);
            let _ = doc.render_source(view, 1000, 750, false);
        }),
    ));
    rows.push(row("pan / zoom redraw of the result view, 12 Mpx", 50.0, best_of(5, || drop(doc.render_result(view, 1000, 750)))));
    drop(doc);

    // Full-resolution export of a 12 Mpx image with effects.
    let mut doc = Document::new(2, img.clone(), None);
    doc.set_prediction(Some(mask.clone()));
    doc.set_param("shadow.enabled", serde_json::json!(true)).unwrap();
    doc.set_param("background.mode", serde_json::json!("color")).unwrap();
    rows.push(row("full export with shadow, 12 Mpx", 3000.0, best_of(2, || drop(doc.export()))));
    drop(doc);
    drop(img);

    // Peak memory for 24 Mpx: open, predict, edit, preview, export.
    CURRENT.store(0, Ordering::Relaxed);
    PEAK.store(0, Ordering::Relaxed);
    {
        let (w2, h2) = (6000usize, 4000usize);
        let big = photo(w2, h2);
        let big_mask = disc_mask(w2, h2);
        let mut doc = Document::new(3, big, None);
        doc.set_prediction(Some(big_mask));
        doc.set_param("shadow.enabled", serde_json::json!(true)).unwrap();
        doc.set_param("background.mode", serde_json::json!("color")).unwrap();
        doc.add_stroke(Stroke { kind: StrokeKind::Erase, width: 60, points: vec![[3000.0, 2000.0]] });
        let _ = doc.render_result(Rect::new(0.0, 0.0, w2 as f64, h2 as f64), 1500, 1000);
        let _ = doc.render_source(Rect::new(0.0, 0.0, w2 as f64, h2 as f64), 1500, 1000, false);
        let _ = doc.export();
    }
    let peak = PEAK.load(Ordering::Relaxed) as f64 / (1024.0 * 1024.0 * 1024.0);
    rows.push(Row { what: "peak memory, 24 Mpx edit and export", target: "<= 2.0 GB", got: format!("{peak:.2} GB"), ok: peak <= 2.0 });

    // Network inference, when a model is given.
    if let (Some(dir), Some(id)) = (args.get(1), args.get(2)) {
        use openscissors_core::model::{ModelRegistry, Predictor};
        let reg = ModelRegistry::new(dir);
        if let Some(spec) = reg.get(id) {
            if let Ok(p) = Predictor::load(&spec, &reg.model_path(&spec)) {
                p.warmup().expect("warmup");
                // The specification's target is the network itself: a 1024 x 768 image needs no resizing.
                let small = photo(1024, 768);
                let net = best_of(3, || drop(p.predict(&small, true).expect("predict")));
                rows.push(Row {
                    what: "model inference, 1024 px image",
                    target: if id.starts_with("isnet") { "<= 1000 ms" } else { "(informative)" },
                    got: format!("{:.0} ms", net.as_secs_f64() * 1000.0),
                    ok: !id.starts_with("isnet") || net.as_secs_f64() <= 1.0,
                });
                let big = photo(w, h);
                rows.push(Row {
                    what: "prediction end to end, 12 Mpx photo",
                    target: "(informative)",
                    got: format!("{:.0} ms", best_of(2, || drop(p.predict(&big, true).expect("predict"))).as_secs_f64() * 1000.0),
                    ok: true,
                });
            }
        }
    }

    println!("{:<48} {:<14} {:<12} result", "step", "target", "measured");
    for r in &rows {
        println!("{:<48} {:<14} {:<12} {}", r.what, r.target, r.got, if r.ok { "ok" } else { "ABOVE TARGET" });
    }
}
