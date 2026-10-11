//! 16-bit sources: kept at full precision through load, edit and export.

use std::path::PathBuf;

use openscissors_core::document::Document;
use openscissors_core::io::{load_image, load_image_deep, save_rgba16, Format};
use openscissors_core::types::{Image, Image16, Mask, Rect, Rgba16};
use serde_json::json;

fn tmp(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("osc-deep-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// 16-bit gradient whose low byte is never zero, so 8-bit conversion would lose information.
fn gradient16(w: usize, h: usize) -> Image16 {
    let mut data = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            data.extend_from_slice(&[(x * 997 + 13) as u16, (y * 1511 + 77) as u16, ((x + y) * 733 + 201) as u16]);
        }
    }
    Image16 { w, h, data }
}

fn write_png16(path: &std::path::Path, img: &Image16) {
    let buf: image::ImageBuffer<image::Rgb<u16>, Vec<u16>> =
        image::ImageBuffer::from_raw(img.w as u32, img.h as u32, img.data.clone()).unwrap();
    image::DynamicImage::ImageRgb16(buf).save_with_format(path, image::ImageFormat::Png).unwrap();
}

fn left_half(w: usize, h: usize) -> Mask {
    let mut m = Mask::new(w, h);
    for y in 0..h {
        for x in 0..w / 2 {
            m.data[y * w + x] = 255;
        }
    }
    m
}

fn doc_from(dir: &std::path::Path) -> (Document, Image16) {
    let deep = gradient16(40, 20);
    write_png16(&dir.join("src.png"), &deep);
    let loaded = load_image_deep(&dir.join("src.png")).unwrap();
    let mut doc = Document::new(1, loaded.image, None).with_deep(loaded.deep);
    doc.set_prediction(Some(left_half(40, 20)));
    (doc, deep)
}

#[test]
fn sixteen_bit_files_keep_their_samples_and_eight_bit_files_do_not() {
    let dir = tmp("load");
    let deep = gradient16(8, 4);
    write_png16(&dir.join("a.png"), &deep);
    let loaded = load_image_deep(&dir.join("a.png")).unwrap();
    assert_eq!(loaded.deep.as_ref(), Some(&deep));
    assert_eq!(loaded.image.data[0], (deep.data[0] >> 8) as u8, "working copy is the 8-bit version");

    let eight = Image::filled(4, 4, [10, 20, 30]);
    let buf = image::RgbImage::from_raw(4, 4, eight.data.clone()).unwrap();
    buf.save(dir.join("b.png")).unwrap();
    assert!(load_image_deep(&dir.join("b.png")).unwrap().deep.is_none());
    assert_eq!(load_image(&dir.join("b.png")).unwrap(), eight);
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn transparent_background_exports_the_original_16_bit_samples() {
    let dir = tmp("transparent");
    let (mut doc, deep) = doc_from(&dir);
    assert!(doc.has_deep_source() && doc.deep_blocker().is_none());
    let out = doc.export_deep().expect("16-bit export is possible");
    assert_eq!((out.w, out.h), (40, 20));
    let px = |x: usize, y: usize| out.data[(y * 40 + x) * 4..(y * 40 + x) * 4 + 4].to_vec();
    let i = (5 * 40 + 10) * 3;
    assert_eq!(&px(10, 5)[..3], &deep.data[i..i + 3], "colours are bit-exact");
    assert_eq!(px(10, 5)[3], 65535);
    assert_eq!(px(30, 5)[3], 0, "outside the subject is transparent");
    assert!(px(10, 5)[0] & 0xFF != 0, "the low byte survived");
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn solid_background_blends_in_16_bits_and_crop_applies() {
    let dir = tmp("color");
    let (mut doc, deep) = doc_from(&dir);
    doc.set_param("background.mode", json!("color")).unwrap();
    doc.set_param("background.color", json!([255, 0, 128])).unwrap();
    doc.set_param("crop", json!({ "x0": 5.0, "y0": 2.0, "x1": 35.0, "y1": 12.0 })).unwrap();
    let out = doc.export_deep().unwrap();
    assert_eq!((out.w, out.h), (30, 10));
    let px = |x: usize, y: usize| out.data[(y * 30 + x) * 4..(y * 30 + x) * 4 + 4].to_vec();
    // Crop starts at (5, 2): output (0, 0) is source (5, 2), inside the subject.
    let i = (2 * 40 + 5) * 3;
    assert_eq!(&px(0, 0)[..3], &deep.data[i..i + 3]);
    assert_eq!(px(0, 0)[3], 65535);
    assert_eq!(px(29, 9), vec![65535, 0, 128 * 257, 65535], "background colour at 16 bits");
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn effects_force_eight_bits_and_say_why() {
    let dir = tmp("blocked");
    let (mut doc, _) = doc_from(&dir);
    doc.set_param("shadow.enabled", json!(true)).unwrap();
    assert!(doc.export_deep().is_none());
    assert_eq!(doc.deep_blocker(), Some("the shadow is on"));
    doc.set_param("shadow.enabled", json!(false)).unwrap();
    doc.set_param("foreground.filter.kind", json!("blur")).unwrap();
    assert_eq!(doc.deep_blocker(), Some("a foreground effect is active"));
    doc.set_param("foreground.filter.kind", json!("none")).unwrap();
    doc.set_param("background.mode", json!("original")).unwrap();
    assert!(doc.deep_blocker().is_some() && doc.export_deep().is_none());

    // An 8-bit source never offers 16-bit output.
    let mut plain = Document::new(2, Image::filled(4, 4, [1, 2, 3]), None);
    assert!(!plain.has_deep_source() && plain.deep_blocker().is_none() && plain.export_deep().is_none());
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn sixteen_bit_png_and_tiff_files_roundtrip() {
    let dir = tmp("save");
    let data: Vec<u16> = (0..6 * 4 * 4).map(|i| (i as u32 * 2711 + 5) as u16).collect();
    let img = Rgba16 { w: 6, h: 4, data: data.clone() };
    for (name, fmt) in [("o.png", Format::Png), ("o.tiff", Format::Tiff)] {
        save_rgba16(&dir.join(name), &img, fmt).unwrap();
        let back = image::open(dir.join(name)).unwrap().to_rgba16();
        assert_eq!(back.into_raw(), data, "{name}");
    }
    assert!(save_rgba16(&dir.join("o.jpg"), &img, Format::Jpeg).is_err());
    let _ = Rect::new(0.0, 0.0, 1.0, 1.0);
    std::fs::remove_dir_all(dir).ok();
}
