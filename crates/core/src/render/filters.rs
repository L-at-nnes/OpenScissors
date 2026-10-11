//! Foreground / background effects. All sizes are given in image pixels and multiplied
//! by `scale` (output pixels per image pixel), so previews and exports look the same.

use rayon::prelude::*;

use crate::imaging::filter::gaussian_blur_u8;
use crate::params::{FilterKind, FilterSpec};
use crate::types::Image;

#[inline]
fn luma(p: &[u8]) -> u32 {
    (p[0] as u32 * 299 + p[1] as u32 * 587 + p[2] as u32 * 114 + 500) / 1000
}

/// Apply `f` to `img`. `origin` is the position of the canvas origin in `img` pixels
/// (used to keep the pixelate grid stable while panning).
pub fn apply_filter(img: &mut Image, f: &FilterSpec, scale: f64, origin: (f64, f64)) {
    match f.kind {
        FilterKind::None => {}
        FilterKind::Blur => {
            let sigma = f.radius as f64 * scale / 2.0;
            img.data = gaussian_blur_u8(&img.data, img.w, img.h, 3, sigma);
        }
        FilterKind::Pixelate => pixelate(img, (f.radius as f64 * scale).max(1.0), origin),
        FilterKind::Grayscale => img.data.par_chunks_exact_mut(3).for_each(|p| {
            let l = luma(p) as u8;
            p.copy_from_slice(&[l, l, l]);
        }),
        FilterKind::Negative => img.data.par_iter_mut().for_each(|v| *v = 255 - *v),
        FilterKind::Colorize => colorize(img, f.color, f.strength.min(100) as f32 / 100.0),
    }
}

/// Tint the luminance towards `color` (overlay blend), mixed with the original by `strength`.
fn colorize(img: &mut Image, color: [u8; 3], strength: f32) {
    img.data.par_chunks_exact_mut(3).for_each(|p| {
        let l = luma(p) as f32;
        for c in 0..3 {
            let k = color[c] as f32;
            let tinted = if l < 128.0 { 2.0 * l * k / 255.0 } else { 255.0 - 2.0 * (255.0 - l) * (255.0 - k) / 255.0 };
            let orig = p[c] as f32;
            p[c] = (orig + (tinted - orig) * strength).clamp(0.0, 255.0) as u8;
        }
    });
}

/// Replace each `block` x `block` cell (grid anchored at `origin`) by its mean colour.
fn pixelate(img: &mut Image, block: f64, origin: (f64, f64)) {
    let (w, h) = (img.w, img.h);
    let cell = |v: usize, o: f64| ((v as f64 + 0.5 - o) / block).floor() as i64;
    let xs: Vec<i64> = (0..w).map(|x| cell(x, origin.0)).collect();
    let ys: Vec<i64> = (0..h).map(|y| cell(y, origin.1)).collect();
    let (x_min, y_min) = (xs[0], ys[0]);
    let nx = (xs[w - 1] - x_min + 1) as usize;
    let ny = (ys[h - 1] - y_min + 1) as usize;
    let mut sum = vec![[0u64; 4]; nx * ny];
    for y in 0..h {
        let row = (ys[y] - y_min) as usize * nx;
        for x in 0..w {
            let s = &mut sum[row + (xs[x] - x_min) as usize];
            let p = &img.data[(y * w + x) * 3..(y * w + x) * 3 + 3];
            s[0] += p[0] as u64;
            s[1] += p[1] as u64;
            s[2] += p[2] as u64;
            s[3] += 1;
        }
    }
    for y in 0..h {
        let row = (ys[y] - y_min) as usize * nx;
        for x in 0..w {
            let s = &sum[row + (xs[x] - x_min) as usize];
            let o = &mut img.data[(y * w + x) * 3..(y * w + x) * 3 + 3];
            for c in 0..3 {
                o[c] = ((s[c] + s[3] / 2) / s[3]) as u8;
            }
        }
    }
}

/// Extra border (output pixels) a filter needs around the visible area to avoid edge artefacts.
pub fn filter_margin(f: &FilterSpec, scale: f64) -> usize {
    match f.kind {
        FilterKind::Blur => (3.0 * f.radius as f64 * scale / 2.0).ceil() as usize,
        FilterKind::Pixelate => (f.radius as f64 * scale).ceil() as usize,
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat(rgb: [u8; 3]) -> Image {
        Image::filled(8, 8, rgb)
    }

    fn spec(kind: FilterKind) -> FilterSpec {
        FilterSpec { kind, ..FilterSpec::default() }
    }

    #[test]
    fn negative_twice_is_identity_and_grayscale_is_neutral() {
        let mut a = flat([10, 120, 250]);
        apply_filter(&mut a, &spec(FilterKind::Negative), 1.0, (0.0, 0.0));
        assert_eq!(&a.data[..3], &[245, 135, 5]);
        apply_filter(&mut a, &spec(FilterKind::Negative), 1.0, (0.0, 0.0));
        assert_eq!(&a.data[..3], &[10, 120, 250]);
        apply_filter(&mut a, &spec(FilterKind::Grayscale), 1.0, (0.0, 0.0));
        assert!(a.data[0] == a.data[1] && a.data[1] == a.data[2]);
    }

    #[test]
    fn colorize_strength_zero_keeps_and_full_tints() {
        let mut a = flat([200, 200, 200]);
        let mut f = spec(FilterKind::Colorize);
        f.strength = 0;
        apply_filter(&mut a, &f, 1.0, (0.0, 0.0));
        assert_eq!(&a.data[..3], &[200, 200, 200]);
        f.strength = 100;
        apply_filter(&mut a, &f, 1.0, (0.0, 0.0));
        assert!(a.data[2] > a.data[0], "blue tint expected: {:?}", &a.data[..3]);
    }

    #[test]
    fn pixelate_makes_uniform_cells_and_keeps_mean() {
        let mut img = Image::new(8, 8);
        for (i, p) in img.data.chunks_exact_mut(3).enumerate() {
            let v = (i % 8 * 30) as u8;
            p.copy_from_slice(&[v, v, v]);
        }
        let before: u64 = img.data.iter().map(|&v| v as u64).sum();
        let mut f = spec(FilterKind::Pixelate);
        f.radius = 4;
        apply_filter(&mut img, &f, 1.0, (0.0, 0.0));
        assert_eq!(img.data[0..3], img.data[3 * 3..3 * 3 + 3]); // x=0 and x=3 share a cell
        assert_ne!(img.data[3 * 3], img.data[4 * 3]); // x=3 and x=4 do not
        let after: u64 = img.data.iter().map(|&v| v as u64).sum();
        assert!((before as i64 - after as i64).unsigned_abs() < before / 50);
    }

    #[test]
    fn blur_keeps_flat_images_and_none_is_noop() {
        let mut a = flat([40, 80, 120]);
        let orig = a.clone();
        apply_filter(&mut a, &spec(FilterKind::None), 1.0, (0.0, 0.0));
        assert_eq!(a, orig);
        apply_filter(&mut a, &spec(FilterKind::Blur), 1.0, (0.0, 0.0));
        assert_eq!(a, orig);
    }
}
