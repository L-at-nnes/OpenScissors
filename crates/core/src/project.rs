//! File formats: the project (`.osproj`) and the marks (`.psm`).
//!
//! Both are JSON. A project holds everything needed to continue editing: the source image,
//! the AI prediction, every stroke and all properties. Marks files hold the strokes only and
//! follow the layout used by other tools of this kind, so they can be exchanged.

use std::io::Cursor;
use std::path::Path;

use base64::engine::general_purpose::STANDARD as B64;
use base64::Engine;
use image::{ImageFormat, ImageReader};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::document::Document;
use crate::error::{Error, Result};
use crate::mask::{Stroke, StrokeKind};
use crate::params::{BackgroundParams, ForegroundParams, MaskParams, Params, ShadowParams};
use crate::types::{Image, Mask, Rect};

pub const PROJECT_FORMAT: &str = "openscissors-project";
pub const PROJECT_VERSION: u32 = 1;
/// Project files larger than this are refused (they are read fully into memory).
const MAX_PROJECT_BYTES: u64 = 1 << 30;

#[derive(Serialize, Deserialize)]
struct EncodedImage {
    encoding: String,
    data: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    orig_path: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ModelRef {
    pub id: String,
    #[serde(default)]
    pub sha256: String,
}

#[derive(Default, Serialize, Deserialize)]
#[serde(default)]
struct Layers {
    mask: MaskParams,
    foreground: ForegroundParams,
    background: BackgroundParams,
    shadow: ShadowParams,
}

#[derive(Serialize, Deserialize)]
#[serde(default)]
struct Ui {
    graphcut: bool,
    marker_size: u32,
}

impl Default for Ui {
    fn default() -> Self {
        Self { graphcut: false, marker_size: 30 }
    }
}

#[derive(Serialize, Deserialize)]
struct ProjectFile {
    format: String,
    version: u32,
    image: EncodedImage,
    #[serde(default)]
    predicted: Option<EncodedImage>,
    #[serde(default)]
    model: Option<ModelRef>,
    #[serde(default)]
    strokes: Vec<Stroke>,
    #[serde(default)]
    layers: Layers,
    /// Manual crop rectangle (an extension of the base layout).
    #[serde(default)]
    crop: Option<Rect>,
    #[serde(default)]
    ui: Ui,
}

/// Everything restored from a project file.
pub struct LoadedProject {
    pub image: Image,
    pub orig_path: Option<String>,
    pub predicted: Option<Mask>,
    pub strokes: Vec<Stroke>,
    pub params: Params,
    pub model: Option<ModelRef>,
    pub marker_size: u32,
    /// Set when the file comes from a newer version: it opened, but parts may be ignored.
    pub warning: Option<String>,
}

fn png_base64(data: &[u8], w: usize, h: usize, color: image::ExtendedColorType) -> Result<String> {
    use image::ImageEncoder;
    let mut out = Vec::new();
    image::codecs::png::PngEncoder::new(&mut out).write_image(data, w as u32, h as u32, color)?;
    Ok(B64.encode(out))
}

fn decode(e: &EncodedImage) -> Result<image::DynamicImage> {
    if e.encoding != "png-base64" {
        return Err(Error::msg(format!("unsupported image encoding '{}'", e.encoding)));
    }
    let bytes = B64.decode(&e.data).map_err(|err| Error::msg(format!("corrupt image data: {err}")))?;
    Ok(ImageReader::with_format(Cursor::new(bytes), ImageFormat::Png).decode()?)
}

/// Write `doc` as a project. `model` records which model produced the prediction.
pub fn save_project(doc: &Document, model: Option<ModelRef>, marker_size: u32, path: &Path) -> Result<()> {
    let p = doc.params();
    let file = ProjectFile {
        format: PROJECT_FORMAT.into(),
        version: PROJECT_VERSION,
        image: EncodedImage {
            encoding: "png-base64".into(),
            data: png_base64(&doc.image.data, doc.image.w, doc.image.h, image::ExtendedColorType::Rgb8)?,
            orig_path: doc.path().map(|p| p.display().to_string()),
        },
        predicted: match doc.predicted() {
            Some(m) => Some(EncodedImage {
                encoding: "png-base64".into(),
                data: png_base64(&m.data, m.w, m.h, image::ExtendedColorType::L8)?,
                orig_path: None,
            }),
            None => None,
        },
        model,
        strokes: doc.strokes().to_vec(),
        layers: Layers {
            mask: p.mask.clone(),
            foreground: p.foreground.clone(),
            background: p.background.clone(),
            shadow: p.shadow.clone(),
        },
        crop: p.crop,
        ui: Ui { graphcut: p.mask.graphcut, marker_size },
    };
    crate::io::write_atomic(path, &serde_json::to_vec(&file)?)?;
    Ok(())
}

/// A file can be edited by hand or damaged: keep only strokes the brush could have made (width 1 to 300, finite points).
fn clean_strokes(strokes: Vec<Stroke>) -> Vec<Stroke> {
    strokes
        .into_iter()
        .filter_map(|mut s| {
            s.points.retain(|p| p[0].is_finite() && p[1].is_finite());
            s.width = s.width.clamp(1, 300);
            (!s.points.is_empty()).then_some(s)
        })
        .collect()
}

/// A crop must be a real rectangle inside the image; anything else means "no manual crop".
fn clean_crop(crop: Option<Rect>, w: usize, h: usize) -> Option<Rect> {
    let r = crop?;
    if ![r.x0, r.y0, r.x1, r.y1].iter().all(|v| v.is_finite()) {
        return None;
    }
    let (x0, y0) = (r.x0.clamp(0.0, w as f64), r.y0.clamp(0.0, h as f64));
    let (x1, y1) = (r.x1.clamp(0.0, w as f64), r.y1.clamp(0.0, h as f64));
    (x1 - x0 >= 1.0 && y1 - y0 >= 1.0).then(|| Rect::new(x0, y0, x1, y1))
}

pub fn load_project(path: &Path) -> Result<LoadedProject> {
    if std::fs::metadata(path)?.len() > MAX_PROJECT_BYTES {
        return Err(Error::msg("project file is too large"));
    }
    let value: Value = serde_json::from_slice(&std::fs::read(path)?)?;
    if value.get("format").and_then(Value::as_str) != Some(PROJECT_FORMAT) {
        return Err(Error::msg("this is not an OpenScissors project"));
    }
    let file: ProjectFile = serde_json::from_value(value)?;
    let img = decode(&file.image)?.to_rgb8();
    let image = Image { w: img.width() as usize, h: img.height() as usize, data: img.into_raw() };
    let predicted = match &file.predicted {
        Some(e) => {
            let m = decode(e)?.to_luma8();
            if (m.width() as usize, m.height() as usize) != (image.w, image.h) {
                return Err(Error::msg("the stored prediction does not match the image size"));
            }
            Some(Mask { w: image.w, h: image.h, data: m.into_raw() })
        }
        None => None,
    };
    let mut params = Params {
        mask: file.layers.mask,
        foreground: file.layers.foreground,
        background: file.layers.background,
        shadow: file.layers.shadow,
        crop: clean_crop(file.crop, image.w, image.h),
    };
    params.sanitize();
    let warning = (file.version > PROJECT_VERSION)
        .then(|| format!("This project was made by a newer version (format {}); some settings may be ignored.", file.version));
    Ok(LoadedProject {
        image,
        orig_path: file.image.orig_path,
        predicted,
        strokes: clean_strokes(file.strokes),
        params,
        model: file.model,
        marker_size: file.ui.marker_size.clamp(1, 300),
        warning,
    })
}

// -- marks (.psm)

const KEEP_COLOR: &str = "#3dbe4a";
const REMOVE_COLOR: &str = "#e04840";
const HAIR_COLOR: &str = "#f0c828";
const ERASER_COLOR: &str = "#ffffff";

fn action(kind: &str, stroke: &Stroke, color: &str) -> Value {
    serde_json::json!({
        "type": kind,
        "lineWidth": stroke.width,
        "color": color,
        "points": stroke.points.iter().map(|p| [p[0], p[1]]).collect::<Vec<_>>(),
    })
}

/// Serialise strokes to the marks layout (`version` 2). Foreground and background strokes go to
/// `scribble`, hair strokes to `matting`; erasers are stored as `eraser` actions.
pub fn marks_to_json(strokes: &[Stroke], w: usize, h: usize) -> String {
    let mut scribble = Vec::new();
    let mut matting = Vec::new();
    for s in strokes {
        match s.kind {
            StrokeKind::Keep => scribble.push(action("marker", s, KEEP_COLOR)),
            StrokeKind::Remove => scribble.push(action("marker", s, REMOVE_COLOR)),
            StrokeKind::Erase => scribble.push(action("eraser", s, ERASER_COLOR)),
            StrokeKind::Hair => matting.push(action("marker", s, HAIR_COLOR)),
            StrokeKind::HairErase => matting.push(action("eraser", s, ERASER_COLOR)),
        }
    }
    serde_json::json!({
        "version": 2, "width": w, "height": h,
        "scribble": { "actions": scribble },
        "matting": { "actions": matting },
    })
    .to_string()
}

fn parse_color(c: &str) -> Option<(u8, u8, u8)> {
    let h = c.strip_prefix('#')?;
    (h.len() == 6).then_some(())?;
    let n = u32::from_str_radix(h, 16).ok()?;
    Some(((n >> 16) as u8, (n >> 8) as u8, n as u8))
}

fn parse_actions(list: &Value, hair: bool, out: &mut Vec<Stroke>) {
    for a in list.get("actions").and_then(Value::as_array).into_iter().flatten() {
        let width = a.get("lineWidth").and_then(Value::as_u64).unwrap_or(30).clamp(1, 300) as u32;
        let points: Vec<[f32; 2]> = a
            .get("points")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|p| Some([p.get(0)?.as_f64()? as f32, p.get(1)?.as_f64()? as f32]))
            .collect();
        let kind = match a.get("type").and_then(Value::as_str) {
            Some("eraser") => {
                if hair {
                    StrokeKind::HairErase
                } else {
                    StrokeKind::Erase
                }
            }
            Some("marker") if hair => StrokeKind::Hair,
            Some("marker") => {
                // Greenish markers keep, reddish ones remove.
                let (r, g, _) = a.get("color").and_then(Value::as_str).and_then(parse_color).unwrap_or((0, 255, 0));
                if g >= r {
                    StrokeKind::Keep
                } else {
                    StrokeKind::Remove
                }
            }
            _ => continue,
        };
        if !points.is_empty() {
            out.push(Stroke { kind, width, points });
        }
    }
}

/// Parse a marks file for an image of `w` x `h`. Refuses other versions and other sizes.
pub fn marks_from_json(text: &str, w: usize, h: usize) -> Result<Vec<Stroke>> {
    let v: Value = serde_json::from_str(text)?;
    if v.get("version").and_then(Value::as_u64) != Some(2) {
        return Err(Error::msg("unsupported marks file version"));
    }
    let dims = (v.get("width").and_then(Value::as_u64), v.get("height").and_then(Value::as_u64));
    if dims != (Some(w as u64), Some(h as u64)) {
        return Err(Error::msg("these marks were made for an image of a different size"));
    }
    let mut strokes = Vec::new();
    parse_actions(&v["scribble"], false, &mut strokes);
    parse_actions(&v["matting"], true, &mut strokes);
    Ok(strokes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn sample_doc() -> Document {
        let mut img = Image::new(24, 16);
        for (i, v) in img.data.iter_mut().enumerate() {
            *v = (i * 7 % 251) as u8;
        }
        let mut d = Document::new(1, img, Some("C:/photos/me.jpg".into()));
        let mut pred = Mask::new(24, 16);
        for (i, v) in pred.data.iter_mut().enumerate() {
            *v = (i * 3 % 256) as u8;
        }
        d.set_prediction(Some(pred));
        d.add_stroke(Stroke { kind: StrokeKind::Keep, width: 7, points: vec![[3.5, 4.0], [10.25, 8.75]] });
        d.add_stroke(Stroke { kind: StrokeKind::Hair, width: 11, points: vec![[1.0, 2.0]] });
        d.set_param("background.mode", json!("color")).unwrap();
        d.set_param("background.color", json!([12, 34, 56])).unwrap();
        d.set_param("shadow.enabled", json!(true)).unwrap();
        d.set_param("mask.graphcut", json!(true)).unwrap();
        d.set_param("crop", json!({ "x0": 2.0, "y0": 1.0, "x1": 20.0, "y1": 12.0 })).unwrap();
        d
    }

    fn tmp(name: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("osc-proj-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn project_roundtrip_restores_everything_and_resaves_identically() {
        let dir = tmp("rt");
        let d = sample_doc();
        let model = Some(ModelRef { id: "birefnet-lite".into(), sha256: "abc".into() });
        save_project(&d, model.clone(), 42, &dir.join("a.osproj")).unwrap();

        let loaded = load_project(&dir.join("a.osproj")).unwrap();
        assert_eq!(loaded.image, *d.image);
        assert_eq!(loaded.predicted.as_ref(), d.predicted());
        assert_eq!(loaded.strokes, d.strokes());
        assert_eq!(&loaded.params, d.params());
        assert_eq!(loaded.model, model);
        assert_eq!(loaded.marker_size, 42);
        assert_eq!(loaded.orig_path.as_deref(), Some("C:/photos/me.jpg"));
        assert!(loaded.warning.is_none());

        let restored =
            Document::restore(2, loaded.image, loaded.orig_path.map(Into::into), loaded.predicted, loaded.strokes, loaded.params);
        save_project(&restored, model, 42, &dir.join("b.osproj")).unwrap();
        assert_eq!(
            std::fs::read(dir.join("a.osproj")).unwrap(),
            std::fs::read(dir.join("b.osproj")).unwrap(),
            "same bytes after a round trip"
        );
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn a_damaged_project_cannot_ask_for_absurd_strokes_or_crops() {
        let dir = tmp("bad");
        let d = Document::new(1, Image::filled(40, 30, [1, 2, 3]), None);
        save_project(&d, None, 30, &dir.join("p.osproj")).unwrap();
        let mut v: Value = serde_json::from_slice(&std::fs::read(dir.join("p.osproj")).unwrap()).unwrap();
        v["strokes"] = json!([
            { "kind": "keep", "width": 4_000_000_000u64, "points": [[5.0, 5.0], [1e60, 2.0]] },
            { "kind": "keep", "width": 5, "points": [[1e60, 2.0]] }
        ]);
        v["crop"] = json!({ "x0": -50.0, "y0": -5.0, "x1": 9999.0, "y1": 12.0 });
        std::fs::write(dir.join("q.osproj"), v.to_string()).unwrap();
        let l = load_project(&dir.join("q.osproj")).unwrap();
        assert_eq!(l.strokes.len(), 1, "the stroke with no usable point is dropped");
        assert_eq!(l.strokes[0].width, 300);
        assert_eq!(l.strokes[0].points, vec![[5.0, 5.0]]);
        assert_eq!(l.params.crop, Some(Rect::new(0.0, 0.0, 40.0, 12.0)), "the crop is brought inside the image");
        v["crop"] = json!({ "x0": 10.0, "y0": 10.0, "x1": 10.0, "y1": 20.0 });
        std::fs::write(dir.join("r.osproj"), v.to_string()).unwrap();
        assert_eq!(load_project(&dir.join("r.osproj")).unwrap().params.crop, None, "an empty crop means no crop");
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn project_without_prediction_and_unknown_keys_are_accepted() {
        let dir = tmp("unk");
        let d = Document::new(1, Image::filled(6, 5, [1, 2, 3]), None);
        save_project(&d, None, 30, &dir.join("p.osproj")).unwrap();
        let mut v: Value = serde_json::from_slice(&std::fs::read(dir.join("p.osproj")).unwrap()).unwrap();
        v["future_feature"] = json!({ "x": 1 });
        v["layers"]["shadow"]["glow"] = json!(5);
        std::fs::write(dir.join("q.osproj"), v.to_string()).unwrap();
        let l = load_project(&dir.join("q.osproj")).unwrap();
        assert!(l.predicted.is_none() && l.strokes.is_empty());
        assert_eq!(l.image.w, 6);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn newer_versions_open_with_a_warning_and_bad_files_are_refused() {
        let dir = tmp("ver");
        let d = Document::new(1, Image::filled(4, 4, [9, 9, 9]), None);
        save_project(&d, None, 30, &dir.join("p.osproj")).unwrap();
        let mut v: Value = serde_json::from_slice(&std::fs::read(dir.join("p.osproj")).unwrap()).unwrap();
        v["version"] = json!(99);
        std::fs::write(dir.join("new.osproj"), v.to_string()).unwrap();
        assert!(load_project(&dir.join("new.osproj")).unwrap().warning.is_some());

        std::fs::write(dir.join("other.json"), r#"{"format":"something-else"}"#).unwrap();
        assert!(load_project(&dir.join("other.json")).is_err());
        std::fs::write(dir.join("broken.osproj"), "{ nope").unwrap();
        assert!(load_project(&dir.join("broken.osproj")).is_err());
        v = json!({ "format": PROJECT_FORMAT, "version": 1, "image": { "encoding": "png-base64", "data": "!!!" } });
        std::fs::write(dir.join("badimg.osproj"), v.to_string()).unwrap();
        assert!(load_project(&dir.join("badimg.osproj")).is_err());
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn marks_roundtrip_all_stroke_kinds() {
        let strokes = vec![
            Stroke { kind: StrokeKind::Keep, width: 30, points: vec![[1.0, 2.0], [3.5, 4.5]] },
            Stroke { kind: StrokeKind::Remove, width: 12, points: vec![[9.0, 9.0]] },
            Stroke { kind: StrokeKind::Erase, width: 5, points: vec![[2.0, 2.0]] },
            Stroke { kind: StrokeKind::Hair, width: 8, points: vec![[5.0, 5.0], [6.0, 6.0]] },
            Stroke { kind: StrokeKind::HairErase, width: 4, points: vec![[7.0, 7.0]] },
        ];
        let text = marks_to_json(&strokes, 100, 80);
        let back = marks_from_json(&text, 100, 80).unwrap();
        // scribble actions come first, then matting ones: same kinds, same data.
        let kinds: Vec<_> = back.iter().map(|s| s.kind).collect();
        assert_eq!(kinds, vec![StrokeKind::Keep, StrokeKind::Remove, StrokeKind::Erase, StrokeKind::Hair, StrokeKind::HairErase]);
        assert_eq!(back, strokes);
    }

    #[test]
    fn marks_layout_matches_the_documented_format_and_is_validated() {
        let strokes = vec![Stroke { kind: StrokeKind::Keep, width: 30, points: vec![[1.0, 2.0]] }];
        let v: Value = serde_json::from_str(&marks_to_json(&strokes, 100, 80)).unwrap();
        assert_eq!((v["version"].as_u64(), v["width"].as_u64(), v["height"].as_u64()), (Some(2), Some(100), Some(80)));
        let a = &v["scribble"]["actions"][0];
        assert_eq!((a["type"].as_str(), a["lineWidth"].as_u64()), (Some("marker"), Some(30)));
        assert!(a["color"].as_str().unwrap().starts_with('#') && a["points"][0][0].as_f64() == Some(1.0));

        assert!(marks_from_json(&marks_to_json(&strokes, 100, 80), 99, 80).is_err(), "size mismatch");
        assert!(marks_from_json(r#"{"version":1,"width":1,"height":1}"#, 1, 1).is_err(), "other version");
        // Colours decide: a red marker removes, a green one keeps; unknown types are skipped.
        let foreign = r##"{"version":2,"width":10,"height":10,"scribble":{"actions":[
            {"type":"marker","lineWidth":9,"color":"#ff0000","points":[[1,1]]},
            {"type":"marker","lineWidth":9,"color":"#00ff00","points":[[2,2]]},
            {"type":"lasso","points":[[3,3]]}]},"matting":{"actions":[]}}"##;
        let got = marks_from_json(foreign, 10, 10).unwrap();
        assert_eq!(got.iter().map(|s| s.kind).collect::<Vec<_>>(), vec![StrokeKind::Remove, StrokeKind::Keep]);
    }
}
