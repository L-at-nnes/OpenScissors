#![allow(clippy::chunks_exact_to_as_chunks)]
//! End to end on the synthetic image with a real model:
//! open -> predict -> strokes -> properties -> export.
//! Needs `OPENSCISSORS_MODEL` (IS-Net style ONNX), otherwise it is skipped.

use std::path::{Path, PathBuf};

use openscissors_core::document::Document;
use openscissors_core::io::{load_image, save_rgba, Format};
use openscissors_core::mask::{Stroke, StrokeKind};
use openscissors_core::model::{ModelSpec, OutputNorm, Predictor};
use serde_json::json;

fn data(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/data").join(name)
}

#[test]
fn open_predict_edit_style_export() {
    let Ok(model) = std::env::var("OPENSCISSORS_MODEL") else {
        eprintln!("OPENSCISSORS_MODEL not set: skipped");
        return;
    };
    let spec = ModelSpec {
        id: "isnet-local".into(),
        name: "local".into(),
        description: String::new(),
        url: String::new(),
        sha256: String::new(),
        size_bytes: 0,
        license: String::new(),
        file_name: "model.onnx".into(),
        input_size: [1024, 1024],
        mean: [0.5; 3],
        std: [1.0; 3],
        input_name: None,
        output_name: None,
        output: OutputNorm::Minmax,
        recommended: false,
        kind: openscissors_core::model::ModelKind::Segmenter,
        files: vec![],
        ram_mb: 0,
    };
    let predictor = Predictor::load(&spec, Path::new(&model)).unwrap();
    let image = load_image(&data("synthetic_figure.png")).unwrap();
    let (w, h) = (image.w, image.h);
    let mut doc = Document::new(1, image, None);

    let t = std::time::Instant::now();
    let mask = predictor.predict(&doc.image, true).unwrap();
    eprintln!("predict: {:?}", t.elapsed());
    doc.set_prediction(Some(mask));
    let figure_alpha = doc.alpha().get(w / 2, h / 2 - 40);
    assert!(figure_alpha > 200, "figure should be foreground, got {figure_alpha}");

    // Erase the centre of the figure, then restyle.
    doc.add_stroke(Stroke { kind: StrokeKind::Erase, width: 60, points: vec![[(w / 2) as f32, (h / 2 - 40) as f32]] });
    assert_eq!(doc.alpha().get(w / 2, h / 2 - 40), 0);
    doc.set_param("background.mode", json!("color")).unwrap();
    doc.set_param("background.color", json!([240, 240, 240])).unwrap();
    doc.set_param("shadow.enabled", json!(true)).unwrap();
    doc.set_param("foreground.feather", json!(2)).unwrap();
    doc.set_param("background.fit_to_result", json!(true)).unwrap();
    doc.set_param("background.aspect_ratio", json!("1:1")).unwrap();

    let t = std::time::Instant::now();
    let out = doc.export();
    eprintln!("export {}x{}: {:?}", out.w, out.h, t.elapsed());
    assert_eq!(out.w, out.h, "1:1 crop");
    assert!(out.w < w.max(h));
    assert!(out.data.chunks_exact(4).all(|p| p[3] == 255), "solid background is opaque");

    let dir = std::env::temp_dir().join("osc-pipeline");
    std::fs::create_dir_all(&dir).unwrap();
    for (name, fmt) in [("o.png", Format::Png), ("o.jpg", Format::Jpeg), ("o.webp", Format::Webp)] {
        save_rgba(&dir.join(name), &out, fmt, 90, [255; 3]).unwrap();
        assert!(dir.join(name).metadata().unwrap().len() > 100);
    }
}

#[test]
fn graphcut_with_real_prediction_and_a_large_image_timing() {
    use openscissors_core::mask::graphcut::{graphcut_alpha, GraphcutOptions};
    use openscissors_core::types::{Image, Mask};

    // Large synthetic photo: GrabCut works on a reduced copy, so a 12 Mpx image must stay fast.
    let (w, h) = (4000usize, 3000usize);
    let mut img = Image::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let inside = (x as f64 - 2000.0).powi(2) + (y as f64 - 1500.0).powi(2) < 900.0_f64.powi(2);
            let c: [u8; 3] = if inside { [200, 70, 60] } else { [50, 90, 170] };
            img.data[(y * w + x) * 3..(y * w + x) * 3 + 3].copy_from_slice(&c);
        }
    }
    let mut fg = Mask::new(w, h);
    let mut bg = Mask::new(w, h);
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
    let t = std::time::Instant::now();
    let m = graphcut_alpha(&img, None, &fg, &bg, &GraphcutOptions::default()).unwrap();
    eprintln!("graphcut 12 Mpx: {:?}", t.elapsed());
    assert_eq!((m.w, m.h), (w, h));
    assert_eq!(m.get(2000, 1500), 255);
    assert_eq!(m.get(100, 100), 0);
    let inside = m.data.iter().filter(|&&v| v > 127).count() as f64;
    let disc = std::f64::consts::PI * 900.0 * 900.0;
    assert!((inside - disc).abs() < disc * 0.05, "{inside} vs {disc}");

    // With the real model, a stroke on a distractor bubble flips it to background.
    let Ok(model) = std::env::var("OPENSCISSORS_MODEL") else { return };
    let spec = openscissors_core::model::ModelSpec {
        id: "local".into(),
        name: "local".into(),
        description: String::new(),
        url: String::new(),
        sha256: String::new(),
        size_bytes: 0,
        license: String::new(),
        file_name: "model.onnx".into(),
        input_size: [1024, 1024],
        mean: [0.5; 3],
        std: [1.0; 3],
        input_name: None,
        output_name: None,
        output: OutputNorm::Minmax,
        recommended: false,
        kind: openscissors_core::model::ModelKind::Segmenter,
        files: vec![],
        ram_mb: 0,
    };
    let predictor = Predictor::load(&spec, Path::new(&model)).unwrap();
    let image = load_image(&data("synthetic_figure.png")).unwrap();
    let (iw, ih) = (image.w, image.h);
    let mut doc = Document::new(9, image, None);
    let pred = predictor.predict(&doc.image, true).unwrap();
    doc.set_prediction(Some(pred));
    doc.set_param("mask.graphcut", json!(true)).unwrap();
    doc.add_stroke(Stroke { kind: StrokeKind::Keep, width: 20, points: vec![[(iw / 2) as f32, (ih / 2 - 80) as f32]] });
    doc.add_stroke(Stroke { kind: StrokeKind::Remove, width: 20, points: vec![[20.0, 20.0]] });
    let t = std::time::Instant::now();
    let alpha = doc.alpha().clone();
    eprintln!("document graphcut at {iw}x{ih}: {:?}", t.elapsed());
    assert!(alpha.get(iw / 2, ih / 2 - 80) > 200);
    assert_eq!(alpha.get(20, 20), 0);
}

#[test]
fn hair_matting_on_a_12_mpx_photo_is_fast_and_soft() {
    use openscissors_core::mask::matting::hair_matting;
    use openscissors_core::types::{Image, Mask};

    // A vertical colour transition 120 px wide in the middle of a 4000 x 3000 image.
    let (w, h) = (4000usize, 3000usize);
    let mut img = Image::new(w, h);
    let mut alpha = Mask::new(w, h);
    let mut hair = Mask::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let a = ((x as f64 - 1940.0) / 120.0).clamp(0.0, 1.0);
            for (c, (f, b)) in [(230.0, 20.0), (200.0, 40.0), (60.0, 150.0)].into_iter().enumerate() {
                img.data[(y * w + x) * 3 + c] = (a * f + (1.0 - a) * b) as u8;
            }
            alpha.data[y * w + x] = if a > 0.5 { 255 } else { 0 };
            if (1900..2100).contains(&x) {
                hair.data[y * w + x] = 255;
            }
        }
    }
    let t = std::time::Instant::now();
    let out = hair_matting(&img, &alpha, &hair, 0);
    eprintln!("hair matting 12 Mpx: {:?}", t.elapsed());
    let mut err = 0.0;
    let mut n = 0.0;
    for y in (100..h - 100).step_by(7) {
        for x in 1940..2060 {
            let truth = (x as f64 - 1940.0) / 120.0;
            err += (out.data[y * w + x] as f64 / 255.0 - truth).abs();
            n += 1.0;
        }
    }
    assert!(err / n < 0.08, "mean error {}", err / n);
}
