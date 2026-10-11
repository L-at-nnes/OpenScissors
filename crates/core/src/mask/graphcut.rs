//! "Group by color similarity": GrabCut seeded by the AI prediction and the user's strokes
//! (spec 4.4, graph-cut mode).

use super::grabcut::{grabcut, BGD, FGD, PR_BGD, PR_FGD};
use crate::imaging::filter::median_blur_u8;
use crate::imaging::morph::{morph_ellipse, MorphOp};
use crate::imaging::resize::{resize_image, resize_mask, Interp};
use crate::model::predict::upsample_guided;
use crate::types::{Image, Mask};

#[derive(Clone, Debug)]
pub struct GraphcutOptions {
    /// Smoothness weight of the energy.
    pub gamma: f64,
    /// Prediction values above this seed "probably foreground", others "probably background".
    pub prior_threshold: u8,
    pub iterations: usize,
    /// Images larger than this many pixels are processed with their short side set to `small_side`.
    pub max_pixels: usize,
    pub small_side: usize,
}

impl Default for GraphcutOptions {
    fn default() -> Self {
        Self { gamma: 100.0, prior_threshold: 127, iterations: 1, max_pixels: 360_000, small_side: 600 }
    }
}

/// Compute the mask with GrabCut. `fg` / `bg` are the keep / remove marks (255 = marked).
/// Returns `None` when a class has no mark: there is nothing to separate yet.
pub fn graphcut_alpha(image: &Image, predicted: Option<&Mask>, fg: &Mask, bg: &Mask, opts: &GraphcutOptions) -> Option<Mask> {
    if fg.is_empty_mask() || bg.is_empty_mask() {
        return None;
    }
    let (w, h) = (image.w, image.h);
    let resized = w * h > opts.max_pixels;
    let (sw, sh) = if resized {
        let s = opts.small_side as f64 / w.min(h) as f64;
        (((w as f64 * s).round() as usize).max(1), ((h as f64 * s).round() as usize).max(1))
    } else {
        (w, h)
    };
    let (small, fg_s, bg_s, pred_s) = if resized {
        (
            resize_image(image, sw, sh, Interp::Area),
            resize_mask(fg, sw, sh, Interp::Nearest),
            resize_mask(bg, sw, sh, Interp::Nearest),
            predicted.map(|p| resize_mask(p, sw, sh, Interp::Area)),
        )
    } else {
        (image.clone(), fg.clone(), bg.clone(), predicted.cloned())
    };

    let mut labels: Vec<u8> = (0..sw * sh)
        .map(|i| {
            if fg_s.data[i] > 127 {
                FGD
            } else if bg_s.data[i] > 127 {
                BGD
            } else {
                match &pred_s {
                    Some(p) if p.data[i] > opts.prior_threshold => PR_FGD,
                    _ => PR_BGD,
                }
            }
        })
        .collect();
    grabcut(&small, &mut labels, opts.iterations, opts.gamma);

    let binary = Mask { w: sw, h: sh, data: labels.iter().map(|&l| if l == FGD || l == PR_FGD { 255 } else { 0 }).collect() };
    let cleaned = morph_ellipse(&morph_ellipse(&binary, 3, MorphOp::Close), 3, MorphOp::Open);
    let smooth = Mask { w: sw, h: sh, data: median_blur_u8(&cleaned.data, sw, sh, 5) };

    let mut out = if resized { upsample_guided(image, &smooth) } else { smooth };
    // The user's own marks are certain: restore them exactly.
    for i in 0..w * h {
        if fg.data[i] > 127 {
            out.data[i] = 255;
        } else if bg.data[i] > 127 {
            out.data[i] = 0;
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scene(w: usize, h: usize) -> Image {
        // Red disc on a blue background, mild noise.
        let mut img = Image::new(w, h);
        let mut s = 7u32;
        for y in 0..h {
            for x in 0..w {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                let n = ((s >> 24) % 20) as i32 - 10;
                let inside = (x as f64 - w as f64 / 2.0).powi(2) + (y as f64 - h as f64 / 2.0).powi(2) < (h as f64 / 3.0).powi(2);
                let base: [i32; 3] = if inside { [210, 60, 50] } else { [40, 80, 180] };
                for c in 0..3 {
                    img.data[(y * w + x) * 3 + c] = (base[c] + n).clamp(0, 255) as u8;
                }
            }
        }
        img
    }

    fn truth(w: usize, h: usize) -> Vec<bool> {
        (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as f64, (i / w) as f64);
                (x - w as f64 / 2.0).powi(2) + (y - h as f64 / 2.0).powi(2) < (h as f64 / 3.0).powi(2)
            })
            .collect()
    }

    fn marks(w: usize, h: usize, pts: &[(usize, usize)]) -> Mask {
        let mut m = Mask::new(w, h);
        for &(px, py) in pts {
            for y in py.saturating_sub(2)..(py + 3).min(h) {
                for x in px.saturating_sub(2)..(px + 3).min(w) {
                    m.data[y * w + x] = 255;
                }
            }
        }
        m
    }

    fn accuracy(m: &Mask, t: &[bool]) -> f64 {
        m.data.iter().zip(t).filter(|(&a, &b)| (a > 127) == b).count() as f64 / t.len() as f64
    }

    #[test]
    fn needs_both_classes() {
        let img = scene(40, 30);
        let fg = marks(40, 30, &[(20, 15)]);
        assert!(graphcut_alpha(&img, None, &fg, &Mask::new(40, 30), &GraphcutOptions::default()).is_none());
        assert!(graphcut_alpha(&img, None, &Mask::new(40, 30), &fg, &GraphcutOptions::default()).is_none());
    }

    #[test]
    fn one_green_and_one_red_stroke_isolate_the_subject() {
        let (w, h) = (80, 60);
        let img = scene(w, h);
        let fg = marks(w, h, &[(40, 30)]);
        let bg = marks(w, h, &[(5, 5)]);
        let m = graphcut_alpha(&img, None, &fg, &bg, &GraphcutOptions::default()).unwrap();
        let acc = accuracy(&m, &truth(w, h));
        assert!(acc > 0.97, "accuracy {acc}");
        assert_eq!(m.get(40, 30), 255);
        assert_eq!(m.get(5, 5), 0);
    }

    #[test]
    fn a_wrong_prediction_is_corrected_by_colour_similarity() {
        let (w, h) = (80, 60);
        let img = scene(w, h);
        // Prediction is shifted by 12 pixels: right shape, wrong place.
        let mut pred = Mask::new(w, h);
        for (i, t) in truth(w, h).iter().enumerate() {
            let (x, y) = (i % w, i / w);
            if *t && x + 12 < w {
                pred.data[y * w + x + 12] = 255;
            }
        }
        let before = accuracy(&pred, &truth(w, h));
        let m = graphcut_alpha(&img, Some(&pred), &marks(w, h, &[(40, 30)]), &marks(w, h, &[(5, 5)]), &GraphcutOptions::default()).unwrap();
        let after = accuracy(&m, &truth(w, h));
        assert!(after > before + 0.05 && after > 0.9, "before {before}, after {after}");
    }

    #[test]
    fn large_images_are_processed_reduced_and_returned_at_full_size() {
        let (w, h) = (800, 640); // 512 000 px > 360 000
        let img = scene(w, h);
        let m = graphcut_alpha(&img, None, &marks(w, h, &[(400, 320)]), &marks(w, h, &[(10, 10)]), &GraphcutOptions::default()).unwrap();
        assert_eq!((m.w, m.h), (w, h));
        let acc = accuracy(&m, &truth(w, h));
        assert!(acc > 0.97, "accuracy {acc}");
    }
}
