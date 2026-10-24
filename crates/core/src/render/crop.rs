//! Crop rectangle logic: aspect ratios, "fit to result" and padding.

use crate::params::{parse_ratio, Params};
use crate::types::{Mask, Rect};

/// Bounding box of the pixels with alpha > 0.
pub fn alpha_bbox(alpha: &Mask) -> Option<Rect> {
    let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0usize, 0usize);
    for y in 0..alpha.h {
        let row = &alpha.data[y * alpha.w..(y + 1) * alpha.w];
        if let (Some(a), Some(b)) = (row.iter().position(|&v| v > 0), row.iter().rposition(|&v| v > 0)) {
            x0 = x0.min(a);
            x1 = x1.max(b);
            y0 = y0.min(y);
            y1 = y;
        }
    }
    (x0 != usize::MAX).then(|| Rect::new(x0 as f64, y0 as f64, (x1 + 1) as f64, (y1 + 1) as f64))
}

/// How far the canvas may grow past the picture: at most this many times its size along each axis.
pub const MAX_EXTENT: f64 = 2.0;

/// Place `r` on the picture. Along an axis where it is smaller than the picture it is shifted to lie inside it; where it
/// is larger, the canvas is extended past the picture (the background fills the extra area) and `r` keeps the whole
/// picture in view, as close to its own centre as that allows. Nothing is cut away from `r`, except above
/// [`MAX_EXTENT`], where it is scaled down (keeping its proportions).
pub fn place_on_canvas(r: Rect, canvas: (f64, f64)) -> Rect {
    let (w, h) = (r.width(), r.height());
    let k = (MAX_EXTENT * canvas.0 / w).min(MAX_EXTENT * canvas.1 / h).min(1.0);
    let (nw, nh) = (w * k, h * k);
    let cx = (r.x0 + r.x1) / 2.0;
    let cy = (r.y0 + r.y1) / 2.0;
    let place = |c: f64, size: f64, room: f64| {
        let (lo, hi) = ((room - size).min(0.0), (room - size).max(0.0));
        let x0 = (c - size / 2.0).clamp(lo, hi);
        (x0, x0 + size)
    };
    let (x0, x1) = place(cx, nw, canvas.0);
    let (y0, y1) = place(cy, nh, canvas.1);
    Rect::new(x0, y0, x1, y1)
}

/// `r` grown around its centre until it has the ratio `ratio.0 : ratio.1`.
fn grow_to_ratio(r: Rect, ratio: (u32, u32)) -> Rect {
    let target = ratio.0 as f64 / ratio.1 as f64;
    let (mut w, mut h) = (r.width(), r.height());
    if w / h < target {
        w = h * target;
    } else {
        h = w / target;
    }
    let cx = (r.x0 + r.x1) / 2.0;
    let cy = (r.y0 + r.y1) / 2.0;
    Rect::new(cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)
}

/// Grow `r` around its centre until it has the ratio `ratio.0 : ratio.1`, then place it on the canvas (see
/// [`place_on_canvas`]: the canvas is extended rather than cutting `r`).
pub fn constrain_ratio(r: Rect, ratio: (u32, u32), canvas: (f64, f64)) -> Rect {
    place_on_canvas(grow_to_ratio(r, ratio), canvas)
}

/// Where a hand-made crop may lie: the picture and, around it, a margin as wide as the picture. Together with
/// [`MAX_EXTENT`] this bounds the canvas the export has to render.
pub fn allowed_area(canvas: (f64, f64)) -> Rect {
    Rect::new(-canvas.0, -canvas.1, 2.0 * canvas.0, 2.0 * canvas.1)
}

/// A crop chosen by hand, kept where it is: only brought within [`allowed_area`] and [`MAX_EXTENT`].
fn keep_within_limits(r: Rect, canvas: (f64, f64)) -> Rect {
    let area = allowed_area(canvas);
    let k = (MAX_EXTENT * canvas.0 / r.width()).min(MAX_EXTENT * canvas.1 / r.height()).min(1.0);
    let (cx, cy) = ((r.x0 + r.x1) / 2.0, (r.y0 + r.y1) / 2.0);
    let (w, h) = (r.width() * k, r.height() * k);
    let x0 = (cx - w / 2.0).clamp(area.x0, area.x1 - w);
    let y0 = (cy - h / 2.0).clamp(area.y0, area.y1 - h);
    Rect::new(x0, y0, x0 + w, y0 + h)
}

/// Largest rectangle with the given ratio that fits the canvas, centred.
pub fn largest_ratio_rect(canvas: (f64, f64), ratio: (u32, u32)) -> Rect {
    let target = ratio.0 as f64 / ratio.1 as f64;
    let (w, h) = if canvas.0 / canvas.1 > target { (canvas.1 * target, canvas.1) } else { (canvas.0, canvas.0 / target) };
    let x0 = (canvas.0 - w) / 2.0;
    let y0 = (canvas.1 - h) / 2.0;
    Rect::new(x0, y0, x0 + w, y0 + h)
}

/// The crop actually applied by the preview overlay and by export.
/// `finished_alpha` is only read when `fit_to_result` is on.
pub fn effective_crop(params: &Params, finished_alpha: Option<&Mask>, canvas: (f64, f64)) -> Rect {
    let full = Rect::new(0.0, 0.0, canvas.0, canvas.1);
    let ratio = parse_ratio(&params.background.aspect_ratio);
    if params.background.fit_to_result {
        if let Some(bb) = finished_alpha.and_then(alpha_bbox) {
            // A visible shadow is part of the result: keep it inside the crop.
            let bb = if params.shadow.enabled {
                let (dx, dy, r) = (params.shadow.offset_x as f64, params.shadow.offset_y as f64, params.shadow.blur_radius as f64);
                let s = Rect::new(bb.x0 + dx - r, bb.y0 + dy - r, bb.x1 + dx + r, bb.y1 + dy + r);
                Rect::new(bb.x0.min(s.x0), bb.y0.min(s.y0), bb.x1.max(s.x1), bb.y1.max(s.y1))
            } else {
                bb
            };
            let p = params.background.padding as f64 / 100.0;
            let padded = Rect::new(bb.x0 - bb.width() * p, bb.y0 - bb.height() * p, bb.x1 + bb.width() * p, bb.y1 + bb.height() * p);
            return match ratio {
                Some(r) => constrain_ratio(padded, r, canvas),
                None => place_on_canvas(padded, canvas),
            };
        }
        // Nothing to fit to (no subject yet): keep the requested ratio on the whole picture.
        return ratio.map_or(full, |r| largest_ratio_rect(canvas, r));
    }
    match (params.crop, ratio) {
        // A crop that already has the ratio (as the crop tool leaves it) stays where the user put it.
        (Some(c), Some(r)) if (c.width() / c.height() * r.1 as f64 / r.0 as f64 - 1.0).abs() < 1e-3 => keep_within_limits(c, canvas),
        (Some(c), Some(r)) => constrain_ratio(c, r, canvas),
        (Some(c), None) => keep_within_limits(c, canvas),
        (None, Some(r)) => largest_ratio_rect(canvas, r),
        (None, None) => full,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ratio_of(r: Rect) -> f64 {
        r.width() / r.height()
    }

    #[test]
    fn fit_to_result_without_a_subject_still_honours_the_ratio() {
        let mut p = Params::default();
        p.background.fit_to_result = true;
        let empty = Mask::new(200, 100);
        assert_eq!(effective_crop(&p, Some(&empty), (200.0, 100.0)), Rect::new(0.0, 0.0, 200.0, 100.0));
        p.background.aspect_ratio = "1:1".into();
        assert_eq!(effective_crop(&p, Some(&empty), (200.0, 100.0)), Rect::new(50.0, 0.0, 150.0, 100.0));
    }

    #[test]
    fn fit_to_result_keeps_the_shadow_inside() {
        let mut p = Params::default();
        p.background.fit_to_result = true;
        p.background.padding = 0;
        let mut alpha = Mask::new(200, 100);
        for y in 40..60 {
            for x in 80..100 {
                alpha.data[y * 200 + x] = 255;
            }
        }
        let plain = effective_crop(&p, Some(&alpha), (200.0, 100.0));
        assert_eq!(plain, Rect::new(80.0, 40.0, 100.0, 60.0));
        p.shadow.enabled = true;
        p.shadow.blur_radius = 5;
        p.shadow.offset_x = 20;
        p.shadow.offset_y = 0;
        let with = effective_crop(&p, Some(&alpha), (200.0, 100.0));
        assert_eq!(with, Rect::new(80.0, 35.0, 125.0, 65.0), "the box grows right by the offset plus the blur, up and down by the blur");
    }

    #[test]
    fn bbox_of_alpha() {
        let mut m = Mask::new(10, 10);
        assert!(alpha_bbox(&m).is_none());
        m.data[2 * 10 + 3] = 1;
        m.data[6 * 10 + 8] = 255;
        assert_eq!(alpha_bbox(&m), Some(Rect::new(3.0, 2.0, 9.0, 7.0)));
    }

    #[test]
    fn constrain_ratio_hits_the_ratio_and_stays_inside() {
        let canvas = (200.0, 100.0);
        for &(a, b) in &[(1, 1), (16, 9), (4, 3), (3, 1), (5, 4)] {
            let r = constrain_ratio(Rect::new(150.0, 60.0, 190.0, 95.0), (a, b), canvas);
            assert!((ratio_of(r) - a as f64 / b as f64).abs() < 1e-9, "{a}:{b} -> {r:?}");
            assert!(r.x0 >= -1e-9 && r.y0 >= -1e-9 && r.x1 <= 200.0 + 1e-9 && r.y1 <= 100.0 + 1e-9);
        }
    }

    #[test]
    fn largest_rect_fills_one_dimension_and_is_centred() {
        let r = largest_ratio_rect((200.0, 100.0), (1, 1));
        assert_eq!(r, Rect::new(50.0, 0.0, 150.0, 100.0));
        let r = largest_ratio_rect((100.0, 200.0), (2, 1));
        assert_eq!((r.width(), r.height()), (100.0, 50.0));
    }

    #[test]
    fn fit_to_result_pads_the_bbox() {
        let mut alpha = Mask::new(100, 100);
        for y in 40..60 {
            for x in 40..60 {
                alpha.data[y * 100 + x] = 255;
            }
        }
        let mut p = Params::default();
        p.background.fit_to_result = true;
        p.background.padding = 10;
        let r = effective_crop(&p, Some(&alpha), (100.0, 100.0));
        assert_eq!(r, Rect::new(38.0, 38.0, 62.0, 62.0));
        p.background.aspect_ratio = "2:1".into();
        let r = effective_crop(&p, Some(&alpha), (100.0, 100.0));
        assert!((ratio_of(r) - 2.0).abs() < 1e-9);
    }

    #[test]
    fn manual_and_default_crops() {
        let mut p = Params::default();
        assert_eq!(effective_crop(&p, None, (80.0, 60.0)), Rect::new(0.0, 0.0, 80.0, 60.0));
        p.crop = Some(Rect::new(10.0, 10.0, 30.0, 40.0));
        assert_eq!(effective_crop(&p, None, (80.0, 60.0)), Rect::new(10.0, 10.0, 30.0, 40.0));
        p.background.aspect_ratio = "1:1".into();
        assert!((ratio_of(effective_crop(&p, None, (80.0, 60.0))) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn a_hand_made_crop_may_reach_past_the_picture_and_stays_where_it_was_put() {
        let canvas = (100.0, 100.0);
        let mut p = Params { crop: Some(Rect::new(-30.0, 10.0, 70.0, 60.0)), ..Params::default() };
        assert_eq!(effective_crop(&p, None, canvas), Rect::new(-30.0, 10.0, 70.0, 60.0), "no ratio: untouched");
        p.background.aspect_ratio = "2:1".into();
        assert_eq!(effective_crop(&p, None, canvas), Rect::new(-30.0, 10.0, 70.0, 60.0), "already 2:1: untouched");
        p.background.aspect_ratio = "1:1".into();
        let r = effective_crop(&p, None, canvas);
        assert!((ratio_of(r) - 1.0).abs() < 1e-9 && r.x0 >= 0.0 && r.x1 <= 100.0, "another ratio: grown and shifted inside: {r:?}");
        // Far-fetched rectangles are brought back to something the export can render.
        p.background.aspect_ratio = "free".into();
        p.crop = Some(Rect::new(-900.0, -900.0, 900.0, 900.0));
        let r = effective_crop(&p, None, canvas);
        assert!(r.width() <= MAX_EXTENT * 100.0 + 1e-9 && r.x0 >= -100.0 && r.x1 <= 200.0, "{r:?}");
    }

    #[test]
    fn placing_never_panics_stays_in_view_and_extends_only_when_the_rectangle_is_larger() {
        // Pseudo-random rectangles, sometimes larger than the canvas: `canvas - w * (canvas / w)` is sometimes a tiny negative.
        let mut seed = 0x2545_F491_4F6C_DD1Du64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f64 / (1u64 << 53) as f64
        };
        for _ in 0..20_000 {
            let canvas = (100.0 + next() * 3000.0, 100.0 + next() * 3000.0);
            let (w, h) = (50.0 + next() * 5000.0, 50.0 + next() * 5000.0);
            let r = Rect::new(10.0, 10.0, 10.0 + w, 10.0 + h);
            let placed = place_on_canvas(r, canvas);
            let k = placed.width() / w;
            assert!(k <= 1.0 + 1e-9 && ((placed.height() / h) - k).abs() < 1e-9, "proportions are kept: {r:?} {canvas:?}");
            for (lo, hi, size) in [(placed.x0, placed.x1, canvas.0), (placed.y0, placed.y1, canvas.1)] {
                let inside = lo >= -1e-6 && hi <= size + 1e-6;
                let covers = lo <= 1e-6 && hi >= size - 1e-6;
                assert!(inside || covers, "{r:?} {canvas:?} -> {placed:?}");
                assert!(hi - lo <= MAX_EXTENT * size + 1e-6);
            }
            let ratio = constrain_ratio(r, (16, 9), canvas);
            assert!(ratio.width() > 0.0 && ratio.height() > 0.0);
        }
    }

    #[test]
    fn a_ratio_taller_than_the_picture_adds_margin_instead_of_cutting_the_subject() {
        // A square picture, a subject as tall as the picture, the 16:9 banner preset.
        let mut alpha = Mask::new(100, 100);
        for y in 0..100 {
            for x in 40..60 {
                alpha.data[y * 100 + x] = 255;
            }
        }
        let mut p = Params::default();
        p.background.fit_to_result = true;
        p.background.padding = 0;
        p.background.aspect_ratio = "16:9".into();
        let r = effective_crop(&p, Some(&alpha), (100.0, 100.0));
        assert!((ratio_of(r) - 16.0 / 9.0).abs() < 1e-9);
        assert!((r.height() - 100.0).abs() < 1e-9, "the subject keeps its full height: {r:?}");
        assert!(r.x0 < 0.0 && r.x1 > 100.0, "the canvas grows on both sides: {r:?}");
        // Without a ratio, the padding of a subject that touches the picture is no longer lost either.
        p.background.aspect_ratio = "free".into();
        p.background.padding = 10;
        let r = effective_crop(&p, Some(&alpha), (100.0, 100.0));
        assert!((r.height() - 120.0).abs() < 1e-9 && r.y0 < 0.0 && r.y1 > 100.0, "{r:?}");
        // The growth is bounded.
        let sliver = Mask::filled(100, 10, 255);
        p.background.padding = 0;
        p.background.aspect_ratio = "1:1".into();
        let r = effective_crop(&p, Some(&sliver), (100.0, 10.0));
        assert!(r.width() <= MAX_EXTENT * 100.0 + 1e-9 && r.height() <= MAX_EXTENT * 10.0 + 1e-9, "{r:?}");
        assert!((ratio_of(r) - 1.0).abs() < 1e-9, "the ratio is kept");
    }
}
