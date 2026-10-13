//! Document properties (layers): everything the right-hand panel edits.
//! The JSON shape of these types is the contract with the frontend.

use serde::{Deserialize, Serialize};

use crate::types::Rect;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FilterKind {
    #[default]
    None,
    Blur,
    Pixelate,
    Colorize,
    Grayscale,
    Negative,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct FilterSpec {
    pub kind: FilterKind,
    /// Blur and pixelate size, 0-100.
    pub radius: u32,
    /// Colorize strength, 0-100.
    pub strength: u32,
    pub color: [u8; 3],
}

impl Default for FilterSpec {
    fn default() -> Self {
        Self { kind: FilterKind::None, radius: 10, strength: 100, color: [0x00, 0x00, 0xC0] }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BgMode {
    #[default]
    Transparent,
    Color,
    Image,
    Original,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct MaskParams {
    /// Boundary offset in pixels; negative erodes.
    pub offset: i32,
    /// "Group by Color Similarity" (graph-cut), not implemented yet.
    pub graphcut: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ForegroundParams {
    /// Feathering radius in pixels.
    pub feather: u32,
    /// Edge clean-up reach in pixels, 0 = off: soft edges take the colour of the subject next to them.
    pub defringe: u32,
    pub filter: FilterSpec,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct BackgroundParams {
    pub mode: BgMode,
    pub color: [u8; 3],
    pub image_path: Option<String>,
    pub fill_hole: bool,
    /// "free" or "W:H".
    pub aspect_ratio: String,
    pub fit_to_result: bool,
    pub padding: u32,
    /// Width of the exported picture in pixels, 0 = keep the size of the cropped result. The height follows.
    pub output_width: u32,
    pub filter: FilterSpec,
}

impl Default for BackgroundParams {
    fn default() -> Self {
        Self {
            mode: BgMode::Transparent,
            color: [255, 255, 255],
            image_path: None,
            fill_hole: false,
            aspect_ratio: "free".into(),
            fit_to_result: false,
            padding: 5,
            output_width: 0,
            filter: FilterSpec::default(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ShadowParams {
    pub enabled: bool,
    pub opacity: u32,
    pub blur_radius: u32,
    pub color: [u8; 3],
}

impl Default for ShadowParams {
    fn default() -> Self {
        Self { enabled: false, opacity: 75, blur_radius: 10, color: [0, 0, 0] }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Params {
    pub mask: MaskParams,
    pub foreground: ForegroundParams,
    pub background: BackgroundParams,
    pub shadow: ShadowParams,
    /// Manual crop in image coordinates; `None` = whole image.
    pub crop: Option<Rect>,
}

impl Params {
    /// Clamp every value to the range the UI allows.
    pub fn sanitize(&mut self) {
        self.mask.offset = self.mask.offset.clamp(-100, 100);
        self.foreground.feather = self.foreground.feather.min(50);
        self.foreground.defringe = self.foreground.defringe.min(20);
        for f in [&mut self.foreground.filter, &mut self.background.filter] {
            f.radius = f.radius.min(100);
            f.strength = f.strength.min(100);
        }
        self.background.padding = self.background.padding.min(100);
        self.background.output_width = self.background.output_width.min(16_000);
        self.shadow.opacity = self.shadow.opacity.min(100);
        self.shadow.blur_radius = self.shadow.blur_radius.min(100);
    }
}

/// Ratios offered by the "Aspect Ratio" list, in display order (`None` = unconstrained).
pub const ASPECT_RATIOS: &[(&str, Option<(u32, u32)>)] = &[
    ("free", None),
    ("1:1", Some((1, 1))),
    ("3:1", Some((3, 1))),
    ("3:2", Some((3, 2))),
    ("4:3", Some((4, 3))),
    ("5:3", Some((5, 3))),
    ("5:4", Some((5, 4))),
    ("6:5", Some((6, 5))),
    ("7:5", Some((7, 5))),
    ("7:4", Some((7, 4))),
    ("8:5", Some((8, 5))),
    ("10:7", Some((10, 7))),
    ("13:10", Some((13, 10))),
    ("14:9", Some((14, 9))),
    ("14:11", Some((14, 11))),
    ("16:9", Some((16, 9))),
    ("16:11", Some((16, 11))),
    ("17:11", Some((17, 11))),
    ("19:13", Some((19, 13))),
    ("22:17", Some((22, 17))),
    ("24:19", Some((24, 19))),
];

/// Widths offered by the "Output width" list (0 = keep the size).
pub const OUTPUT_WIDTHS: &[u32] = &[0, 500, 800, 1000, 1080, 1200, 1500, 1920, 2000, 3000, 4000];

/// A ready-made result for a common destination: crop to the subject with a margin, a ratio, a background
/// and an output width. It only touches the background tab; the foreground and the shadow are left as they are.
pub struct Preset {
    pub id: &'static str,
    pub ratio: &'static str,
    pub padding: u32,
    pub width: u32,
    /// Background colour, `None` for a transparent background.
    pub color: Option<[u8; 3]>,
}

pub const PRESETS: &[Preset] = &[
    Preset { id: "marketplace", ratio: "1:1", padding: 10, width: 1000, color: Some([255, 255, 255]) },
    Preset { id: "marketplace-large", ratio: "1:1", padding: 10, width: 2000, color: Some([255, 255, 255]) },
    Preset { id: "thumbnail", ratio: "1:1", padding: 5, width: 500, color: None },
    Preset { id: "banner", ratio: "16:9", padding: 10, width: 1920, color: Some([255, 255, 255]) },
];

impl Preset {
    pub fn apply(&self, params: &mut Params) {
        let b = &mut params.background;
        b.mode = if self.color.is_some() { BgMode::Color } else { BgMode::Transparent };
        if let Some(c) = self.color {
            b.color = c;
        }
        b.aspect_ratio = self.ratio.into();
        b.fit_to_result = true;
        b.padding = self.padding;
        b.output_width = self.width;
        params.crop = None;
    }

    /// Whether `params` is exactly what [`Preset::apply`] produces (on the settings it controls).
    pub fn matches(&self, params: &Params) -> bool {
        let b = &params.background;
        let background = match self.color {
            Some(c) => b.mode == BgMode::Color && b.color == c,
            None => b.mode == BgMode::Transparent,
        };
        background
            && b.aspect_ratio == self.ratio
            && b.fit_to_result
            && b.padding == self.padding
            && b.output_width == self.width
            && params.crop.is_none()
    }
}

/// Parse an aspect-ratio key ("free", "16:9") to `(w, h)`.
pub fn parse_ratio(key: &str) -> Option<(u32, u32)> {
    let (a, b) = key.split_once(':')?;
    let (a, b): (u32, u32) = (a.trim().parse().ok()?, b.trim().parse().ok()?);
    (a > 0 && b > 0).then_some((a, b))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_the_spec() {
        let p = Params::default();
        assert_eq!((p.shadow.opacity, p.shadow.blur_radius, p.shadow.color), (75, 10, [0, 0, 0]));
        assert_eq!(p.background.padding, 5);
        assert_eq!(p.foreground.filter.color, [0, 0, 0xC0]);
        assert_eq!((p.foreground.filter.radius, p.foreground.filter.strength), (10, 100));
        assert_eq!(p.mask.offset, 0);
    }

    #[test]
    fn partial_json_fills_defaults() {
        let p: Params = serde_json::from_str(r#"{"shadow":{"enabled":true}}"#).unwrap();
        assert!(p.shadow.enabled);
        assert_eq!(p.shadow.opacity, 75);
    }

    #[test]
    fn presets_are_valid_and_recognised_once_applied() {
        for (i, preset) in PRESETS.iter().enumerate() {
            assert!(ASPECT_RATIOS.iter().any(|(k, _)| *k == preset.ratio), "{}", preset.id);
            assert!(OUTPUT_WIDTHS.contains(&preset.width), "{}", preset.id);
            let mut p = Params { crop: Some(Rect::new(1.0, 1.0, 2.0, 2.0)), ..Params::default() };
            assert!(!preset.matches(&p));
            preset.apply(&mut p);
            let mut sane = p.clone();
            sane.sanitize();
            assert_eq!(p, sane, "{} stays inside the allowed ranges", preset.id);
            let found: Vec<usize> = PRESETS.iter().enumerate().filter(|(_, q)| q.matches(&p)).map(|(j, _)| j).collect();
            assert_eq!(found, [i]);
        }
    }

    #[test]
    fn ratio_parsing_and_list() {
        assert_eq!(parse_ratio("16:9"), Some((16, 9)));
        assert_eq!(parse_ratio("free"), None);
        assert_eq!(parse_ratio("0:3"), None);
        assert!(ASPECT_RATIOS.iter().all(|(k, v)| parse_ratio(k) == *v));
        assert_eq!(ASPECT_RATIOS.len(), 21);
    }
}
