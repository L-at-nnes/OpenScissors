//! "Fill Hole": synthesise the background behind the removed subject.
//!
//! Current implementation is a pull-push diffusion fill at reduced resolution. It gives a
//! smooth, plausible fill but no texture; the spec's PatchMatch is the planned upgrade.

use crate::imaging::resize::{resize_image, resize_mask, Interp};
use crate::types::{Image, Mask};

const WORK_SIZE: usize = 768;

struct Level {
    w: usize,
    h: usize,
    /// Mean colour of the known pixels (0..255 per channel).
    c: Vec<[f32; 3]>,
    /// Fraction of known pixels, 0..1.
    k: Vec<f32>,
}

fn pull(l: &Level) -> Level {
    let (w, h) = (l.w.div_ceil(2), l.h.div_ceil(2));
    let mut c = vec![[0f32; 3]; w * h];
    let mut k = vec![0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let (mut acc, mut wsum, mut n) = ([0f32; 3], 0f32, 0f32);
            for dy in 0..2 {
                for dx in 0..2 {
                    let (sx, sy) = (2 * x + dx, 2 * y + dy);
                    if sx < l.w && sy < l.h {
                        let i = sy * l.w + sx;
                        for ch in 0..3 {
                            acc[ch] += l.c[i][ch] * l.k[i];
                        }
                        wsum += l.k[i];
                        n += 1.0;
                    }
                }
            }
            if wsum > 0.0 {
                c[y * w + x] = [acc[0] / wsum, acc[1] / wsum, acc[2] / wsum];
            }
            k[y * w + x] = wsum / n;
        }
    }
    Level { w, h, c, k }
}

fn sample(l: &Level, fx: f32, fy: f32) -> [f32; 3] {
    let x = fx.clamp(0.0, (l.w - 1) as f32);
    let y = fy.clamp(0.0, (l.h - 1) as f32);
    let (x0, y0) = (x.floor() as usize, y.floor() as usize);
    let (x1, y1) = ((x0 + 1).min(l.w - 1), (y0 + 1).min(l.h - 1));
    let (tx, ty) = (x - x0 as f32, y - y0 as f32);
    let mut out = [0f32; 3];
    for ch in 0..3 {
        let top = l.c[y0 * l.w + x0][ch] * (1.0 - tx) + l.c[y0 * l.w + x1][ch] * tx;
        let bot = l.c[y1 * l.w + x0][ch] * (1.0 - tx) + l.c[y1 * l.w + x1][ch] * tx;
        out[ch] = top * (1.0 - ty) + bot * ty;
    }
    out
}

/// Fill the pixels of `hole` (non-zero) from their surroundings.
pub fn fill_hole(image: &Image, hole: &Mask) -> Image {
    if hole.is_empty_mask() {
        return image.clone();
    }
    let s = (WORK_SIZE as f64 / image.w.max(image.h) as f64).min(1.0);
    let (ww, wh) = (((image.w as f64 * s).round() as usize).max(1), ((image.h as f64 * s).round() as usize).max(1));
    let small = resize_image(image, ww, wh, Interp::Area);
    let small_hole = resize_mask(hole, ww, wh, Interp::Area);

    let mut levels = vec![Level {
        w: ww,
        h: wh,
        c: small.data.chunks_exact(3).map(|p| [p[0] as f32, p[1] as f32, p[2] as f32]).collect(),
        k: small_hole.data.iter().map(|&v| if v > 0 { 0.0 } else { 1.0 }).collect(),
    }];
    while levels.last().is_some_and(|l| l.w > 1 || l.h > 1) {
        let next = pull(levels.last().unwrap());
        levels.push(next);
    }
    // Push: fill unknown pixels of each level from the (already filled) coarser one.
    for i in (0..levels.len() - 1).rev() {
        let (fine, coarse) = levels.split_at_mut(i + 1);
        let (f, c) = (&mut fine[i], &coarse[0]);
        for y in 0..f.h {
            for x in 0..f.w {
                let idx = y * f.w + x;
                if f.k[idx] < 1.0 {
                    let up = sample(c, (x as f32 + 0.5) / 2.0 - 0.5, (y as f32 + 0.5) / 2.0 - 0.5);
                    let k = f.k[idx];
                    for ch in 0..3 {
                        f.c[idx][ch] = f.c[idx][ch] * k + up[ch] * (1.0 - k);
                    }
                }
            }
        }
    }
    let filled_small = Image { w: ww, h: wh, data: levels[0].c.iter().flat_map(|c| c.map(|v| v.clamp(0.0, 255.0) as u8)).collect() };
    let filled =
        if (ww, wh) == (image.w, image.h) { filled_small } else { resize_image(&filled_small, image.w, image.h, Interp::Bilinear) };
    let mut out = image.clone();
    for (i, &hv) in hole.data.iter().enumerate() {
        if hv > 0 {
            out.data[i * 3..i * 3 + 3].copy_from_slice(&filled.data[i * 3..i * 3 + 3]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hole_in_flat_image_is_filled_with_the_same_colour() {
        let mut img = Image::filled(40, 30, [50, 100, 150]);
        let mut hole = Mask::new(40, 30);
        for y in 10..20 {
            for x in 15..25 {
                hole.data[y * 40 + x] = 255;
                img.data[(y * 40 + x) * 3..(y * 40 + x) * 3 + 3].copy_from_slice(&[255, 0, 0]);
            }
        }
        let out = fill_hole(&img, &hole);
        let p = &out.data[(15 * 40 + 20) * 3..(15 * 40 + 20) * 3 + 3];
        for (a, b) in p.iter().zip([50u8, 100, 150]) {
            assert!((*a as i32 - b as i32).abs() <= 3, "{p:?}");
        }
    }

    #[test]
    fn pixels_outside_the_hole_are_untouched_and_gradient_is_continued() {
        let (w, h) = (60, 20);
        let mut img = Image::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let v = (x * 4) as u8;
                img.data[(y * w + x) * 3..(y * w + x) * 3 + 3].copy_from_slice(&[v, v, v]);
            }
        }
        let mut hole = Mask::new(w, h);
        for y in 0..h {
            for x in 25..35 {
                hole.data[y * w + x] = 255;
            }
        }
        let out = fill_hole(&img, &hole);
        assert_eq!(out.data[..3 * 25], img.data[..3 * 25]);
        let mid = out.data[(10 * w + 30) * 3] as i32;
        assert!((mid - 120).abs() < 25, "{mid}");
    }

    #[test]
    fn empty_hole_returns_the_image() {
        let img = Image::filled(5, 5, [1, 2, 3]);
        assert_eq!(fill_hole(&img, &Mask::new(5, 5)), img);
    }
}
