//! Brush strokes and their rasterisation (hard edges, image coordinates).

use serde::{Deserialize, Serialize};

use crate::types::Mask;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StrokeKind {
    Keep,
    Remove,
    Erase,
    Hair,
    HairErase,
}

impl StrokeKind {
    /// Value a stroke of this kind paints on its own canvas (erasers paint 0).
    pub fn value(self) -> u8 {
        match self {
            Self::Keep | Self::Remove | Self::Hair => 255,
            Self::Erase | Self::HairErase => 0,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Stroke {
    pub kind: StrokeKind,
    /// Brush diameter in image pixels.
    pub width: u32,
    pub points: Vec<[f32; 2]>,
}

/// Paint `stroke` onto `mask` in place as a round-capped polyline with hard edges.
/// A pixel is covered when its centre lies within `width / 2` of the polyline.
pub fn paint_stroke(mask: &mut Mask, stroke: &Stroke, value: u8) {
    let r = stroke.width.max(1) as f32 / 2.0;
    let r2 = r * r;
    let mut paint_capsule = |a: [f32; 2], b: [f32; 2]| {
        let x0 = ((a[0].min(b[0]) - r).floor().max(0.0)) as usize;
        let y0 = ((a[1].min(b[1]) - r).floor().max(0.0)) as usize;
        let x1 = ((a[0].max(b[0]) + r).ceil().min(mask.w as f32)).max(0.0) as usize;
        let y1 = ((a[1].max(b[1]) + r).ceil().min(mask.h as f32)).max(0.0) as usize;
        let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
        let len2 = dx * dx + dy * dy;
        for y in y0..y1 {
            let py = y as f32 + 0.5;
            for x in x0..x1 {
                let px = x as f32 + 0.5;
                let t = if len2 > 0.0 { (((px - a[0]) * dx + (py - a[1]) * dy) / len2).clamp(0.0, 1.0) } else { 0.0 };
                let (cx, cy) = (a[0] + t * dx - px, a[1] + t * dy - py);
                if cx * cx + cy * cy <= r2 {
                    mask.data[y * mask.w + x] = value;
                }
            }
        }
    };
    match stroke.points.as_slice() {
        [] => {}
        [p] => paint_capsule(*p, *p),
        pts => {
            for pair in pts.windows(2) {
                paint_capsule(pair[0], pair[1]);
            }
        }
    }
}

/// Strokes of `kind` and `width` (picture pixels) that fill the closed `outline` (a lasso): horizontal runs inside it,
/// close enough for their round ends to leave no gap. They are ordinary strokes, so files and the undo history keep them
/// as any other mark.
pub fn fill_polygon(outline: &[[f32; 2]], kind: StrokeKind, width: u32) -> Vec<Stroke> {
    if outline.len() < 3 {
        return Vec::new();
    }
    let width = width.max(2);
    let step = width as f32 * 0.6;
    let (y0, y1) = outline.iter().fold((f32::MAX, f32::MIN), |(a, b), p| (a.min(p[1]), b.max(p[1])));
    let mut out = Vec::new();
    let mut y = y0 + step / 2.0;
    while y < y1 {
        // Where the scanline crosses the edges (even-odd rule), then the runs between pairs of crossings.
        let mut xs: Vec<f32> = (0..outline.len())
            .filter_map(|i| {
                let (a, b) = (outline[i], outline[(i + 1) % outline.len()]);
                ((a[1] <= y) != (b[1] <= y)).then(|| a[0] + (y - a[1]) / (b[1] - a[1]) * (b[0] - a[0]))
            })
            .collect();
        xs.sort_by(f32::total_cmp);
        for pair in xs.chunks_exact(2) {
            out.push(Stroke { kind, width, points: vec![[pair[0], y], [pair[1], y]] });
        }
        y += step;
    }
    out
}

/// Paint the strokes whose kind is in `kinds`, in order: keep-like = 255, eraser-like = 0.
pub fn rasterize(strokes: &[Stroke], w: usize, h: usize, kinds: &[StrokeKind]) -> Mask {
    let mut m = Mask::new(w, h);
    for s in strokes.iter().filter(|s| kinds.contains(&s.kind)) {
        paint_stroke(&mut m, s, s.kind.value());
    }
    m
}

/// Pixels (255) whose last green, red or eraser stroke is an eraser: what the user wiped out and did not mark again.
pub fn erased_mask(strokes: &[Stroke], w: usize, h: usize) -> Mask {
    let mut m = Mask::new(w, h);
    for s in strokes.iter().filter(|s| matches!(s.kind, StrokeKind::Keep | StrokeKind::Remove | StrokeKind::Erase)) {
        paint_stroke(&mut m, s, if s.kind == StrokeKind::Erase { 255 } else { 0 });
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    fn keep(points: &[[f32; 2]], width: u32) -> Stroke {
        Stroke { kind: StrokeKind::Keep, width, points: points.to_vec() }
    }
    fn erase(points: &[[f32; 2]], width: u32) -> Stroke {
        Stroke { kind: StrokeKind::Erase, width, points: points.to_vec() }
    }

    #[test]
    fn single_point_is_a_disc() {
        let mut m = Mask::new(21, 21);
        paint_stroke(&mut m, &keep(&[[10.5, 10.5]], 9), 255);
        assert_eq!(m.get(10, 10), 255);
        assert_eq!(m.get(14, 10), 255);
        assert_eq!(m.get(15, 10), 0);
        assert_eq!(m.get(6, 6), 0); // outside the disc
    }

    #[test]
    fn polyline_covers_segment_and_clips_at_borders() {
        let mut m = Mask::new(20, 20);
        paint_stroke(&mut m, &keep(&[[-5.0, 10.0], [30.0, 10.0]], 3), 255);
        assert!((0..20).all(|x| m.get(x, 10) == 255));
        assert_eq!(m.get(5, 5), 0);
    }

    #[test]
    fn rasterize_respects_chronological_order() {
        let strokes = vec![keep(&[[10.5, 10.5]], 9), erase(&[[10.5, 10.5]], 3), keep(&[[10.5, 10.5]], 1)];
        let m = rasterize(&strokes, 21, 21, &[StrokeKind::Keep, StrokeKind::Erase]);
        assert_eq!(m.get(10, 10), 255); // keep after erase wins
        assert_eq!(m.get(11, 10), 0); // erased ring stays erased
        assert_eq!(m.get(13, 10), 255); // untouched part of the first keep
    }

    #[test]
    fn erased_pixels_are_those_whose_last_stroke_is_an_eraser() {
        let remove = Stroke { kind: StrokeKind::Remove, width: 3, points: vec![[3.5, 3.5]] };
        let strokes = vec![keep(&[[10.5, 10.5]], 9), erase(&[[10.5, 10.5]], 3), keep(&[[10.5, 10.5]], 1), erase(&[[3.5, 3.5]], 5), remove];
        let m = erased_mask(&strokes, 21, 21);
        assert_eq!(m.get(10, 10), 0, "marked again after the eraser");
        assert_eq!(m.get(11, 10), 255, "erased ring");
        assert_eq!(m.get(13, 10), 0, "never erased");
        assert_eq!(m.get(3, 3), 0, "a red mark after the eraser");
        assert_eq!(m.get(1, 3), 255, "erased by the second eraser, outside the red mark");
    }

    #[test]
    fn empty_and_offscreen_strokes_are_harmless() {
        let mut m = Mask::new(8, 8);
        paint_stroke(&mut m, &keep(&[], 5), 255);
        paint_stroke(&mut m, &keep(&[[100.0, 100.0]], 5), 255);
        assert!(m.is_empty_mask());
    }

    #[test]
    fn a_lasso_fills_its_outline_and_nothing_else() {
        // An L shape: concave, so a run must not bridge the notch.
        let outline = [[10.0, 10.0], [50.0, 10.0], [50.0, 30.0], [30.0, 30.0], [30.0, 60.0], [10.0, 60.0]];
        let strokes = fill_polygon(&outline, StrokeKind::Keep, 6);
        assert!(!strokes.is_empty() && strokes.iter().all(|s| s.kind == StrokeKind::Keep && s.points.len() == 2));
        let m = rasterize(&strokes, 70, 70, &[StrokeKind::Keep]);
        for (x, y) in [(20, 20), (45, 15), (15, 55), (25, 45), (12, 12)] {
            assert_eq!(m.get(x, y), 255, "inside at {x},{y}");
        }
        for (x, y) in [(45, 50), (40, 40), (60, 20), (5, 5)] {
            assert_eq!(m.get(x, y), 0, "outside at {x},{y}");
        }
        assert!(fill_polygon(&outline[..2], StrokeKind::Keep, 6).is_empty(), "two points enclose nothing");
    }
}
