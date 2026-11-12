//! Hair matting: soft alpha inside the hair strokes by closed-form matting
//! (Levin, Lischinski, Weiss 2008), solved matrix-free with conjugate gradients on the
//! unknown pixels only.

use rayon::prelude::*;

use crate::imaging::filter::box_blur_u8;
use crate::imaging::guided::guided_filter;
use crate::imaging::linalg::{det3, inv3};
use crate::imaging::resize::{crop_resize_mask, resize_image, resize_mask, working_size, Interp};
use crate::types::{Image, Mask};

/// Images with more than this many pixels are solved on a reduced copy (see [`working_size`]).
const MAX_PIXELS: usize = 360_000;
const SMALL_SIDE: usize = 600;
/// Pixels of known foreground and background kept around the hair strokes, for the solve to lean on.
const STROKE_MARGIN: usize = 48;
/// The region around the strokes is solved at full resolution up to this many pixels, reduced less past it.
const REGION_MAX_PIXELS: usize = 1_000_000;
const REGION_SMALL_SIDE: usize = 800;
/// Regularisation of the local colour model (colours in 0..1).
const EPSILON: f64 = 1e-7;
const CG_ITERATIONS: usize = 150;
/// Below this many unknowns (or this image side) the coarse-to-fine recursion stops.
const COARSEST_UNKNOWNS: usize = 1500;
const COARSEST_SIDE: usize = 48;
const CG_TOLERANCE: f64 = 1e-6;
/// Guided refinement back at full resolution.
const REFINE_RADIUS: usize = 10;
const REFINE_EPS: f32 = 1e-5;

/// One 3x3 window of the matting Laplacian.
struct Window {
    pixels: [usize; 9],
    mean: [f64; 3],
    /// Inverse of (covariance + epsilon / 9), symmetric.
    inv: [[f64; 3]; 3],
}

struct Problem<'a> {
    img: &'a [[f64; 3]],
    windows: Vec<Window>,
    /// Pixel -> index among the unknowns, or `usize::MAX`.
    unknown_index: Vec<usize>,
    unknowns: usize,
}

const KNOWN: usize = usize::MAX;

impl<'a> Problem<'a> {
    fn new(img: &'a [[f64; 3]], w: usize, h: usize, unknown: &[bool]) -> Self {
        let mut unknown_index = vec![KNOWN; w * h];
        let mut n = 0;
        for (i, &u) in unknown.iter().enumerate() {
            if u {
                unknown_index[i] = n;
                n += 1;
            }
        }
        let mut windows = Vec::new();
        for cy in 1..h.saturating_sub(1) {
            for cx in 1..w.saturating_sub(1) {
                let mut pixels = [0usize; 9];
                let mut any = false;
                for (k, p) in pixels.iter_mut().enumerate() {
                    *p = (cy + k / 3 - 1) * w + cx + k % 3 - 1;
                    any |= unknown[*p];
                }
                if !any {
                    continue;
                }
                let mut mean = [0f64; 3];
                for &p in &pixels {
                    for c in 0..3 {
                        mean[c] += img[p][c] / 9.0;
                    }
                }
                let mut cov = [[0f64; 3]; 3];
                for &p in &pixels {
                    for a in 0..3 {
                        for b in 0..3 {
                            cov[a][b] += (img[p][a] - mean[a]) * (img[p][b] - mean[b]) / 9.0;
                        }
                    }
                }
                for (a, row) in cov.iter_mut().enumerate() {
                    row[a] += EPSILON / 9.0;
                }
                windows.push(Window { pixels, mean, inv: inv3(&cov, det3(&cov)) });
            }
        }
        Problem { img, windows, unknown_index, unknowns: n }
    }

    /// `out[u] = (L x)[pixel of unknown u]`.
    fn apply(&self, x: &[f64], out: &mut [f64]) {
        out.iter_mut().for_each(|o| *o = 0.0);
        for win in &self.windows {
            let mut m = 0.0;
            let mut c = [0f64; 3];
            for &p in &win.pixels {
                m += x[p] / 9.0;
                for k in 0..3 {
                    c[k] += (self.img[p][k] - win.mean[k]) * x[p] / 9.0;
                }
            }
            let a = [
                win.inv[0][0] * c[0] + win.inv[0][1] * c[1] + win.inv[0][2] * c[2],
                win.inv[1][0] * c[0] + win.inv[1][1] * c[1] + win.inv[1][2] * c[2],
                win.inv[2][0] * c[0] + win.inv[2][1] * c[1] + win.inv[2][2] * c[2],
            ];
            for &p in &win.pixels {
                let u = self.unknown_index[p];
                if u != KNOWN {
                    let d = [self.img[p][0] - win.mean[0], self.img[p][1] - win.mean[1], self.img[p][2] - win.mean[2]];
                    out[u] += (x[p] - m) - (d[0] * a[0] + d[1] * a[1] + d[2] * a[2]);
                }
            }
        }
    }

    /// Diagonal of the unknown-unknown block, for Jacobi preconditioning.
    fn diagonal(&self) -> Vec<f64> {
        let mut diag = vec![0f64; self.unknowns];
        for win in &self.windows {
            for &p in &win.pixels {
                let u = self.unknown_index[p];
                if u != KNOWN {
                    let d = [self.img[p][0] - win.mean[0], self.img[p][1] - win.mean[1], self.img[p][2] - win.mean[2]];
                    let mut q = 0.0;
                    for a in 0..3 {
                        for b in 0..3 {
                            q += d[a] * win.inv[a][b] * d[b];
                        }
                    }
                    diag[u] += 1.0 - 1.0 / 9.0 - q / 9.0;
                }
            }
        }
        diag
    }
}

/// Solve for alpha on the `unknown` pixels given `known` values (0..1) elsewhere.
/// `guess` seeds the solver. Returns a full-size alpha (known values kept).
fn solve_level(img: &[[f64; 3]], w: usize, h: usize, unknown: &[bool], known: &[f64], guess: &[f64], iterations: usize) -> Vec<f64> {
    let prob = Problem::new(img, w, h, unknown);
    let n = prob.unknowns;
    let mut alpha: Vec<f64> = (0..w * h).map(|i| if unknown[i] { guess[i] } else { known[i] }).collect();
    if n == 0 {
        return alpha;
    }
    // Right-hand side: -(L x0)_U with x0 = known values and 0 on the unknowns.
    let x0: Vec<f64> = (0..w * h).map(|i| if unknown[i] { 0.0 } else { known[i] }).collect();
    let mut b = vec![0f64; n];
    prob.apply(&x0, &mut b);
    b.iter_mut().for_each(|v| *v = -*v);

    let diag = prob.diagonal();
    let precond = |r: &[f64], z: &mut [f64]| {
        z.par_iter_mut().zip(r).zip(&diag).for_each(|((z, r), d)| *z = r / d.max(1e-9));
    };

    // Unknown values as a vector, and a scratch full-size vector for matvecs.
    let mut x: Vec<f64> = (0..w * h).filter(|&i| unknown[i]).map(|i| alpha[i]).collect();
    let mut full = vec![0f64; w * h];
    let pixels: Vec<usize> = (0..w * h).filter(|&i| unknown[i]).collect();
    let matvec = |v: &[f64], full: &mut Vec<f64>, out: &mut [f64]| {
        for (u, &p) in pixels.iter().enumerate() {
            full[p] = v[u];
        }
        prob.apply(full, out);
    };

    let mut ax = vec![0f64; n];
    matvec(&x, &mut full, &mut ax);
    let mut r: Vec<f64> = b.iter().zip(&ax).map(|(b, a)| b - a).collect();
    let mut z = vec![0f64; n];
    precond(&r, &mut z);
    let mut p = z.clone();
    let mut rz: f64 = r.iter().zip(&z).map(|(a, b)| a * b).sum();
    let b_norm = b.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-12);
    let mut ap = vec![0f64; n];
    for _ in 0..iterations {
        if r.iter().map(|v| v * v).sum::<f64>().sqrt() / b_norm < CG_TOLERANCE {
            break;
        }
        matvec(&p, &mut full, &mut ap);
        // `full` keeps the previous `p` on unknowns only; known entries stay 0 as required.
        let pap: f64 = p.iter().zip(&ap).map(|(a, b)| a * b).sum();
        if pap <= 0.0 {
            break;
        }
        let step = rz / pap;
        for i in 0..n {
            x[i] += step * p[i];
            r[i] -= step * ap[i];
        }
        precond(&r, &mut z);
        let rz_new: f64 = r.iter().zip(&z).map(|(a, b)| a * b).sum();
        let beta = rz_new / rz;
        rz = rz_new;
        for i in 0..n {
            p[i] = z[i] + beta * p[i];
        }
    }
    for (u, &pix) in pixels.iter().enumerate() {
        alpha[pix] = x[u].clamp(0.0, 1.0);
    }
    alpha
}

/// Coarse-to-fine solve: the coarse solution seeds the fine one, which is what makes long
/// thin bands converge in a few conjugate-gradient steps per level.
fn solve(img: &[[f64; 3]], w: usize, h: usize, unknown: &[bool], known: &[f64], guess: &[f64]) -> Vec<f64> {
    let n_unknown = unknown.iter().filter(|&&u| u).count();
    if n_unknown <= COARSEST_UNKNOWNS || w.min(h) < COARSEST_SIDE * 2 {
        return solve_level(img, w, h, unknown, known, guess, CG_ITERATIONS * 4);
    }
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    let mut c_img = vec![[0f64; 3]; cw * ch];
    let mut c_unknown = vec![false; cw * ch];
    let mut c_known = vec![0f64; cw * ch];
    let mut c_guess = vec![0f64; cw * ch];
    for cy in 0..ch {
        for cx in 0..cw {
            let (mut n, mut k_sum, mut k_n, mut g_sum) = (0.0, 0.0, 0.0, 0.0);
            let mut col = [0f64; 3];
            let mut any_unknown = false;
            for dy in 0..2 {
                for dx in 0..2 {
                    let (x, y) = (2 * cx + dx, 2 * cy + dy);
                    if x < w && y < h {
                        let i = y * w + x;
                        n += 1.0;
                        g_sum += guess[i];
                        for c in 0..3 {
                            col[c] += img[i][c];
                        }
                        if unknown[i] {
                            any_unknown = true;
                        } else {
                            k_sum += known[i];
                            k_n += 1.0;
                        }
                    }
                }
            }
            let o = cy * cw + cx;
            c_img[o] = [col[0] / n, col[1] / n, col[2] / n];
            c_unknown[o] = any_unknown;
            c_known[o] = if k_n > 0.0 { (k_sum / k_n).round() } else { 0.0 };
            c_guess[o] = g_sum / n;
        }
    }
    let coarse = solve(&c_img, cw, ch, &c_unknown, &c_known, &c_guess);
    let seeded: Vec<f64> = (0..w * h).map(|i| coarse[(i / w / 2).min(ch - 1) * cw + (i % w / 2).min(cw - 1)]).collect();
    solve_level(img, w, h, unknown, known, &seeded, CG_ITERATIONS)
}

/// Refine `alpha` inside the `hair` strokes. Elsewhere `alpha` is returned unchanged.
///
/// Pixels with `alpha > 128` (after an optional feather blur) are certain foreground, the
/// others certain background; the hair strokes are the unknown band where the soft alpha is
/// estimated from the image colours.
///
/// Only the region around the strokes is solved, at full resolution when it is small enough (a hair band on a large
/// photo keeps its single-pixel strands), on a copy reduced less than the whole picture would be otherwise.
pub fn hair_matting(image: &Image, alpha: &Mask, hair: &Mask, feather: u32) -> Mask {
    let (w, h) = (image.w, image.h);
    if hair.is_empty_mask() {
        return alpha.clone();
    }
    let strokes = Mask { w, h, data: hair.data.iter().map(|&v| if v > 0 { 255 } else { 0 }).collect() };
    if let Some(bb) = crate::render::crop::alpha_bbox(&strokes) {
        let (x0, y0) = ((bb.x0 as usize).saturating_sub(STROKE_MARGIN), (bb.y0 as usize).saturating_sub(STROKE_MARGIN));
        let (x1, y1) = ((bb.x1 as usize + STROKE_MARGIN).min(w), (bb.y1 as usize + STROKE_MARGIN).min(h));
        if (x1 - x0) * (y1 - y0) < w * h {
            let (cw, ch) = (x1 - x0, y1 - y0);
            let rows = |data: &[u8], ch_: usize| -> Vec<u8> {
                (y0..y1).flat_map(|y| data[(y * w + x0) * ch_..(y * w + x1) * ch_].iter().copied()).collect()
            };
            let sub_img = Image { w: cw, h: ch, data: rows(&image.data, 3) };
            let sub_alpha = Mask { w: cw, h: ch, data: rows(&alpha.data, 1) };
            let sub_hair = Mask { w: cw, h: ch, data: rows(&hair.data, 1) };
            if let Some(sub) = matte(&sub_img, &sub_alpha, &sub_hair, feather, REGION_MAX_PIXELS, REGION_SMALL_SIDE) {
                let mut out = alpha.clone();
                for y in y0..y1 {
                    for x in x0..x1 {
                        if hair.data[y * w + x] > 0 {
                            out.data[y * w + x] = sub.data[(y - y0) * cw + x - x0];
                        }
                    }
                }
                return out;
            }
        }
    }
    // The strokes cover the picture, or their region holds only one side: the whole picture, reduced.
    matte(image, alpha, hair, feather, MAX_PIXELS, SMALL_SIDE).unwrap_or_else(|| alpha.clone())
}

/// The matting of `image` inside `hair`, solved on a copy reduced past `max_pixels` (see [`working_size`]). `None`
/// when there is nothing to solve: no stroke, or no known foreground or background around them.
fn matte(image: &Image, alpha: &Mask, hair: &Mask, feather: u32, max_pixels: usize, small_side: usize) -> Option<Mask> {
    let (w, h) = (image.w, image.h);
    let decided = if feather > 0 { box_blur_u8(&alpha.data, w, h, 1, feather as usize) } else { alpha.data.clone() };

    let working = working_size(w, h, max_pixels, small_side);
    let reduced = working.is_some();
    let (sw, sh) = working.unwrap_or((w, h));
    let (small, dec_s, hair_s) = if reduced {
        // Area average keeps thin strokes alive when shrinking.
        (
            resize_image(image, sw, sh, Interp::Area),
            resize_mask(&Mask { w, h, data: decided }, sw, sh, Interp::Area),
            resize_mask(hair, sw, sh, Interp::Area),
        )
    } else {
        (image.clone(), Mask { w, h, data: decided }, hair.clone())
    };

    let unknown: Vec<bool> = hair_s.data.iter().map(|&v| v > 0).collect();
    let known: Vec<f64> = dec_s.data.iter().map(|&v| if v > 128 { 1.0 } else { 0.0 }).collect();
    let has_fg = (0..sw * sh).any(|i| !unknown[i] && known[i] > 0.5);
    let has_bg = (0..sw * sh).any(|i| !unknown[i] && known[i] < 0.5);
    if !unknown.iter().any(|&u| u) || !has_fg || !has_bg {
        return None;
    }

    let img: Vec<[f64; 3]> = small.data.chunks_exact(3).map(|p| [p[0] as f64 / 255.0, p[1] as f64 / 255.0, p[2] as f64 / 255.0]).collect();
    let guess: Vec<f64> = dec_s.data.iter().map(|&v| v as f64 / 255.0).collect();
    let soft = solve(&img, sw, sh, &unknown, &known, &guess);

    let mut out = alpha.clone();
    if !reduced {
        for (i, &v) in soft.iter().enumerate() {
            if hair.data[i] > 0 {
                out.data[i] = (v.clamp(0.0, 1.0) * 255.0).round() as u8;
            }
        }
        return Some(out);
    }
    // Back to full resolution: guided refinement on the image edges, band by band, only around the hair strokes (the
    // only pixels kept), with the reach of the filter around them so that the result is the one of the whole picture.
    let low = Mask { w: sw, h: sh, data: soft.iter().map(|&v| (v * 255.0).round() as u8).collect() };
    let Some(bb) = crate::render::crop::alpha_bbox(&Mask { w, h, data: hair.data.iter().map(|&v| if v > 0 { 255 } else { 0 }).collect() })
    else {
        return Some(out);
    };
    let reach = 2 * REFINE_RADIUS + 1;
    let (x0, y0) = ((bb.x0 as usize).saturating_sub(reach), (bb.y0 as usize).saturating_sub(reach));
    let (x1, y1) = ((bb.x1 as usize + reach).min(w), (bb.y1 as usize + reach).min(h));
    let (bw, bh) = (x1 - x0, y1 - y0);
    let (kx, ky) = (sw as f64 / w as f64, sh as f64 / h as f64);
    let region = crate::types::Rect::new(x0 as f64 * kx, y0 as f64 * ky, x1 as f64 * kx, y1 as f64 * ky);
    let up = crop_resize_mask(&low, region, bw, bh, Interp::Bilinear);
    let mut gray = Vec::with_capacity(bw * bh);
    for y in y0..y1 {
        let row = &image.data[(y * w + x0) * 3..(y * w + x1) * 3];
        gray.extend(row.chunks_exact(3).map(|p| ((p[0] as u32 * 299 + p[1] as u32 * 587 + p[2] as u32 * 114 + 500) / 1000) as u8));
    }
    let refined = refine_in_bands(&gray, &up.data, bw, bh, REFINE_BAND_ROWS);
    for y in y0..y1 {
        for x in x0..x1 {
            if hair.data[y * w + x] > 0 {
                out.data[y * w + x] = refined[(y - y0) * bw + x - x0];
            }
        }
    }
    Some(out)
}

/// Rows refined together by [`refine_in_bands`]. The filter's working buffers cost about 56 bytes per pixel of the
/// band instead of per pixel of the picture, which was the one big memory item left on 33 megapixels and more.
const REFINE_BAND_ROWS: usize = 128;

/// The guided filter of `soft` (a 0-255 mask) on the edges of `gray`, as 0-255 again, computed `band_rows` rows at a
/// time. The two box filters inside reach `2 * REFINE_RADIUS` rows, which each band reads beyond its own: the result
/// is the one of the whole picture.
fn refine_in_bands(gray: &[u8], soft: &[u8], w: usize, h: usize, band_rows: usize) -> Vec<u8> {
    let reach = 2 * REFINE_RADIUS + 1;
    let mut out = vec![0u8; w * h];
    for y0 in (0..h).step_by(band_rows.max(1)) {
        let y1 = (y0 + band_rows).min(h);
        let (a0, a1) = (y0.saturating_sub(reach), (y1 + reach).min(h));
        let unit = |s: &[u8]| s[a0 * w..a1 * w].iter().map(|&v| v as f32 / 255.0).collect::<Vec<f32>>();
        let filtered = guided_filter(&unit(gray), &unit(soft), w, a1 - a0, REFINE_RADIUS, REFINE_EPS, 1);
        for (o, &v) in out[y0 * w..y1 * w].iter_mut().zip(&filtered[(y0 - a0) * w..(y1 - a0) * w]) {
            *o = (v.clamp(0.0, 1.0) * 255.0).round() as u8;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Black/white scene where the true alpha ramps from 0 to 1 over columns `x0..x1`.
    fn ramp_scene(w: usize, h: usize, x0: usize, x1: usize) -> (Image, Mask, Mask, Vec<f64>) {
        let mut img = Image::new(w, h);
        let mut truth = vec![0f64; w * h];
        let mut s = 3u32;
        for y in 0..h {
            for x in 0..w {
                let a = if x < x0 {
                    0.0
                } else if x >= x1 {
                    1.0
                } else {
                    (x - x0) as f64 / (x1 - x0) as f64
                };
                truth[y * w + x] = a;
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                let n = ((s >> 24) % 6) as f64;
                let f = [230.0, 200.0, 60.0]; // foreground colour
                let b = [20.0, 40.0, 150.0]; // background colour
                for c in 0..3 {
                    img.data[(y * w + x) * 3 + c] = (a * f[c] + (1.0 - a) * b[c] + n).clamp(0.0, 255.0) as u8;
                }
            }
        }
        // Coarse binary mask (alpha > 0.5) and hair strokes over the ramp.
        let mut alpha = Mask::new(w, h);
        let mut hair = Mask::new(w, h);
        for y in 0..h {
            for x in 0..w {
                alpha.data[y * w + x] = if truth[y * w + x] > 0.5 { 255 } else { 0 };
                if x >= x0 && x < x1 {
                    hair.data[y * w + x] = 255;
                }
            }
        }
        (img, alpha, hair, truth)
    }

    #[test]
    fn recovers_a_soft_ramp_from_colours() {
        let (w, h) = (60, 30);
        let (img, alpha, hair, truth) = ramp_scene(w, h, 20, 40);
        let out = hair_matting(&img, &alpha, &hair, 0);
        let mut err = 0.0;
        let mut n = 0.0;
        for y in 2..h - 2 {
            for x in 20..40 {
                err += (out.data[y * w + x] as f64 / 255.0 - truth[y * w + x]).abs();
                n += 1.0;
            }
        }
        assert!(err / n < 0.05, "mean error {}", err / n);
        // Pixels outside the strokes are untouched.
        assert_eq!(out.data[5 * w + 10], alpha.data[5 * w + 10]);
        assert_eq!(out.data[5 * w + 50], alpha.data[5 * w + 50]);
    }

    #[test]
    fn refining_in_bands_gives_the_result_of_the_whole_picture() {
        // 90 x 77: not a multiple of the band height, so the last band is short; a soft ramp over a textured guide.
        let (w, h) = (90usize, 77usize);
        let mut seed = 5u32;
        let gray: Vec<u8> = (0..w * h)
            .map(|_| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                (seed >> 24) as u8
            })
            .collect();
        let soft: Vec<u8> = (0..w * h).map(|i| (i % w * 255 / w) as u8).collect();
        let whole = refine_in_bands(&gray, &soft, w, h, h);
        for rows in [1, 7, 20, 64] {
            assert_eq!(refine_in_bands(&gray, &soft, w, h, rows), whole, "{rows} rows per band");
        }
        let f = |s: &[u8]| s.iter().map(|&v| v as f32 / 255.0).collect::<Vec<f32>>();
        let direct = guided_filter(&f(&gray), &f(&soft), w, h, REFINE_RADIUS, REFINE_EPS, 1);
        let worst = whole.iter().zip(&direct).map(|(&b, &d)| (b as f32 - (d.clamp(0.0, 1.0) * 255.0)).abs()).fold(0f32, f32::max);
        assert!(worst <= 0.51, "{worst}");
    }

    #[test]
    fn no_strokes_or_one_sided_trimap_is_a_no_op() {
        let (w, h) = (30, 20);
        let (img, alpha, _hair, _) = ramp_scene(w, h, 10, 20);
        assert_eq!(hair_matting(&img, &alpha, &Mask::new(w, h), 0), alpha);
        let everywhere = Mask::filled(w, h, 255); // no known pixels left
        assert_eq!(hair_matting(&img, &alpha, &everywhere, 0), alpha);
    }

    #[test]
    fn result_is_softer_than_the_binary_input_and_bounded() {
        let (w, h) = (60, 30);
        let (img, alpha, hair, _) = ramp_scene(w, h, 20, 40);
        let out = hair_matting(&img, &alpha, &hair, 0);
        let mid = (0..w * h).filter(|&i| hair.data[i] > 0 && out.data[i] > 20 && out.data[i] < 235).count();
        assert!(mid > 100, "only {mid} soft pixels");
    }

    #[test]
    fn thin_strands_keep_their_width_when_the_strokes_cover_a_small_part_of_a_large_picture() {
        // One-pixel light strands every 4 px on a dark background, between a light subject and the background, on a
        // picture far over the reduction threshold: around the strokes it is solved at full size, so a strand stays a
        // strand instead of a grey blur.
        let (w, h) = (1600usize, 1200usize);
        let mut img = Image::new(w, h);
        let mut alpha = Mask::new(w, h);
        let mut hair = Mask::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let fg = x >= 820 || (x >= 760 && x % 4 == 0); // the subject, then its strands
                let v = if fg { 235 } else { 25 };
                img.data[(y * w + x) * 3..(y * w + x) * 3 + 3].copy_from_slice(&[v, v, v]);
                alpha.data[y * w + x] = if x >= 790 { 255 } else { 0 };
                if (500..700).contains(&y) && (760..820).contains(&x) {
                    hair.data[y * w + x] = 255;
                }
            }
        }
        let t = std::time::Instant::now();
        let out = hair_matting(&img, &alpha, &hair, 0);
        eprintln!("hair matting in a region of {w}x{h}: {:?}", t.elapsed());
        let (mut on, mut off, mut n) = (0f64, 0f64, 0f64);
        for y in 520..680 {
            for x in 768..800 {
                let a = out.data[y * w + x] as f64 / 255.0;
                if x % 4 == 0 {
                    on += a;
                } else {
                    off += a;
                }
                n += 1.0;
            }
        }
        let (on, off) = (on / (n / 4.0), off / (n * 3.0 / 4.0));
        assert!(on > 0.7 && off < 0.3, "strands {on:.2}, between them {off:.2}");
        assert_eq!(out.data[100 * w + 100], 0, "outside the strokes: untouched");
    }

    #[test]
    fn large_images_are_solved_reduced_and_returned_at_full_size() {
        let (w, h) = (800, 640);
        let (img, alpha, hair, truth) = ramp_scene(w, h, 360, 440);
        let t = std::time::Instant::now();
        let out = hair_matting(&img, &alpha, &hair, 0);
        eprintln!("hair matting {w}x{h}: {:?}", t.elapsed());
        assert_eq!((out.w, out.h), (w, h));
        let mut err = 0.0;
        let mut n = 0.0;
        for y in 20..h - 20 {
            for x in 360..440 {
                err += (out.data[y * w + x] as f64 / 255.0 - truth[y * w + x]).abs();
                n += 1.0;
            }
        }
        assert!(err / n < 0.08, "mean error {}", err / n);
    }
}
