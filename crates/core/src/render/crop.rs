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

/// Shrink (keeping the aspect) if larger than the canvas, then shift inside it.
pub fn fit_inside(r: Rect, canvas: (f64, f64)) -> Rect {
    let (w, h) = (r.width(), r.height());
    let k = (canvas.0 / w).min(canvas.1 / h).min(1.0);
    let (nw, nh) = (w * k, h * k);
    let cx = (r.x0 + r.x1) / 2.0;
    let cy = (r.y0 + r.y1) / 2.0;
    // `w * (canvas / w)` can exceed `canvas` by an ulp: the room left may be a tiny negative number.
    let x0 = (cx - nw / 2.0).clamp(0.0, (canvas.0 - nw).max(0.0));
    let y0 = (cy - nh / 2.0).clamp(0.0, (canvas.1 - nh).max(0.0));
    Rect::new(x0, y0, x0 + nw, y0 + nh)
}

/// Grow `r` around its centre until it has the ratio `ratio.0 : ratio.1`, then keep it in the canvas.
pub fn constrain_ratio(r: Rect, ratio: (u32, u32), canvas: (f64, f64)) -> Rect {
    let target = ratio.0 as f64 / ratio.1 as f64;
    let (mut w, mut h) = (r.width(), r.height());
    if w / h < target {
        w = h * target;
    } else {
        h = w / target;
    }
    let cx = (r.x0 + r.x1) / 2.0;
    let cy = (r.y0 + r.y1) / 2.0;
    fit_inside(Rect::new(cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0), canvas)
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
            let p = params.background.padding as f64 / 100.0;
            let padded = Rect::new(bb.x0 - bb.width() * p, bb.y0 - bb.height() * p, bb.x1 + bb.width() * p, bb.y1 + bb.height() * p);
            return match ratio {
                Some(r) => constrain_ratio(padded, r, canvas),
                None => fit_inside(padded, canvas),
            };
        }
        // Nothing to fit to (no subject yet): keep the requested ratio on the whole picture.
        return ratio.map_or(full, |r| largest_ratio_rect(canvas, r));
    }
    match (params.crop, ratio) {
        (Some(c), Some(r)) => constrain_ratio(c, r, canvas),
        (Some(c), None) => fit_inside(c, canvas),
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
    fn fit_inside_never_panics_when_rounding_leaves_no_room() {
        // Pseudo-random rectangles larger than the canvas: `canvas - w * (canvas / w)` is sometimes a tiny negative.
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
            let fitted = fit_inside(r, canvas);
            assert!(
                fitted.x0 >= 0.0 && fitted.y0 >= 0.0 && fitted.x1 <= canvas.0 + 1e-6 && fitted.y1 <= canvas.1 + 1e-6,
                "{r:?} {canvas:?}"
            );
            let ratio = constrain_ratio(r, (16, 9), canvas);
            assert!(ratio.x0 >= 0.0 && ratio.y0 >= 0.0);
        }
    }
}
