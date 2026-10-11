//! Box and (approximate) Gaussian blurs with REFLECT_101 borders.

use rayon::prelude::*;

/// Index reflection without repeating the edge sample (`dcb|abcd|cba`).
#[inline]
pub(crate) fn reflect101(mut i: isize, n: isize) -> usize {
    if n == 1 {
        return 0;
    }
    loop {
        if i < 0 {
            i = -i;
        } else if i >= n {
            i = 2 * (n - 1) - i;
        } else {
            return i as usize;
        }
    }
}

/// Mean filter with a (2r+1)^2 window over `ch` interleaved channels.
/// Integer accumulation, one rounding at the end (matches a standard 8-bit box filter).
pub fn box_blur_u8(src: &[u8], w: usize, h: usize, ch: usize, r: usize) -> Vec<u8> {
    if r == 0 || w == 0 || h == 0 {
        return src.to_vec();
    }
    let row_len = w * ch;
    let (wi, hi) = (w as isize, h as isize);
    let ri = r as isize;

    // Horizontal pass: window sums per sample.
    let mut hsum = vec![0u32; w * h * ch];
    hsum.par_chunks_mut(row_len).zip(src.par_chunks(row_len)).for_each(|(out, row)| {
        for c in 0..ch {
            let mut sum: u32 = 0;
            for k in -ri..=ri {
                sum += row[reflect101(k, wi) * ch + c] as u32;
            }
            out[c] = sum;
            for x in 1..w {
                let add = reflect101(x as isize + ri, wi);
                let sub = reflect101(x as isize - ri - 1, wi);
                sum = sum + row[add * ch + c] as u32 - row[sub * ch + c] as u32;
                out[x * ch + c] = sum;
            }
        }
    });

    // Vertical pass: sliding column sums.
    let n = ((2 * r + 1) * (2 * r + 1)) as u64;
    let mut out = vec![0u8; w * h * ch];
    let mut col = vec![0u64; row_len];
    for k in -ri..=ri {
        let y = reflect101(k, hi);
        for (c, s) in col.iter_mut().enumerate() {
            *s += hsum[y * row_len + c] as u64;
        }
    }
    for y in 0..h {
        let orow = &mut out[y * row_len..(y + 1) * row_len];
        for (o, s) in orow.iter_mut().zip(&col) {
            *o = ((*s + n / 2) / n) as u8;
        }
        if y + 1 < h {
            let add = reflect101(y as isize + ri + 1, hi);
            let sub = reflect101(y as isize - ri, hi);
            for (c, s) in col.iter_mut().enumerate() {
                *s = *s + hsum[add * row_len + c] as u64 - hsum[sub * row_len + c] as u64;
            }
        }
    }
    out
}

/// Radii of three successive box blurs approximating a Gaussian of deviation `sigma`.
fn gauss_box_radii(sigma: f64) -> [usize; 3] {
    let n = 3.0;
    let ideal = (12.0 * sigma * sigma / n + 1.0).sqrt();
    let mut wl = ideal.floor();
    if wl % 2.0 == 0.0 {
        wl -= 1.0;
    }
    let wu = wl + 2.0;
    let m = ((12.0 * sigma * sigma - n * wl * wl - 4.0 * n * wl - 3.0 * n) / (-4.0 * wl - 4.0)).round();
    let mut out = [0usize; 3];
    for (i, o) in out.iter_mut().enumerate() {
        let width = if (i as f64) < m { wl } else { wu };
        *o = ((width - 1.0) / 2.0).max(0.0) as usize;
    }
    out
}

/// Gaussian blur approximated by three box blurs (O(1) per pixel in the radius).
pub fn gaussian_blur_u8(src: &[u8], w: usize, h: usize, ch: usize, sigma: f64) -> Vec<u8> {
    if sigma < 0.3 {
        return src.to_vec();
    }
    let mut cur = src.to_vec();
    for r in gauss_box_radii(sigma) {
        cur = box_blur_u8(&cur, w, h, ch, r);
    }
    cur
}

/// Float mean filter (window 2r+1), f64 accumulators for stability.
pub fn box_mean_f32(src: &[f32], w: usize, h: usize, r: usize) -> Vec<f32> {
    if r == 0 {
        return src.to_vec();
    }
    let (wi, hi, ri) = (w as isize, h as isize, r as isize);
    let mut hsum = vec![0f64; w * h];
    hsum.par_chunks_mut(w).zip(src.par_chunks(w)).for_each(|(out, row)| {
        let mut sum: f64 = (-ri..=ri).map(|k| row[reflect101(k, wi)] as f64).sum();
        out[0] = sum;
        for x in 1..w {
            sum += row[reflect101(x as isize + ri, wi)] as f64 - row[reflect101(x as isize - ri - 1, wi)] as f64;
            out[x] = sum;
        }
    });
    let n = ((2 * r + 1) * (2 * r + 1)) as f64;
    let mut out = vec![0f32; w * h];
    let mut col = vec![0f64; w];
    for k in -ri..=ri {
        let y = reflect101(k, hi);
        for (x, s) in col.iter_mut().enumerate() {
            *s += hsum[y * w + x];
        }
    }
    for y in 0..h {
        for (o, s) in out[y * w..(y + 1) * w].iter_mut().zip(&col) {
            *o = (*s / n) as f32;
        }
        if y + 1 < h {
            let add = reflect101(y as isize + ri + 1, hi);
            let sub = reflect101(y as isize - ri, hi);
            for (x, s) in col.iter_mut().enumerate() {
                *s += hsum[add * w + x] - hsum[sub * w + x];
            }
        }
    }
    out
}

/// Median filter over a `k` x `k` window (`k` odd) of a single-channel image, REFLECT_101 borders.
pub fn median_blur_u8(src: &[u8], w: usize, h: usize, k: usize) -> Vec<u8> {
    if k <= 1 || w == 0 || h == 0 {
        return src.to_vec();
    }
    let r = (k / 2) as isize;
    let (wi, hi) = (w as isize, h as isize);
    let mut out = vec![0u8; w * h];
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let mut win = vec![0u8; k * k];
        for (x, o) in row.iter_mut().enumerate() {
            let mut n = 0;
            for dy in -r..=r {
                let yy = reflect101(y as isize + dy, hi);
                for dx in -r..=r {
                    win[n] = src[yy * w + reflect101(x as isize + dx, wi)];
                    n += 1;
                }
            }
            let mid = n / 2;
            *o = *win[..n].select_nth_unstable(mid).1;
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_is_unchanged_and_impulse_spreads() {
        let c = vec![200u8; 9 * 9];
        assert_eq!(box_blur_u8(&c, 9, 9, 1, 2), c);
        let mut imp = vec![0u8; 11 * 11];
        imp[5 * 11 + 5] = 255;
        let o = box_blur_u8(&imp, 11, 11, 1, 1);
        assert_eq!(o[5 * 11 + 5], 28); // round(255 / 9)
        assert_eq!(o[4 * 11 + 4], 28);
        assert_eq!(o[3 * 11 + 3], 0);
    }

    #[test]
    fn median_removes_salt_noise_and_keeps_edges() {
        let mut m = vec![0u8; 20 * 20];
        for y in 0..20 {
            for x in 10..20 {
                m[y * 20 + x] = 255;
            }
        }
        let clean = m.clone();
        m[5 * 20 + 3] = 255; // speck
        m[8 * 20 + 15] = 0; // hole
        assert_eq!(median_blur_u8(&m, 20, 20, 5), clean);
    }

    #[test]
    fn radius_zero_is_identity() {
        let m: Vec<u8> = (0..64).collect();
        assert_eq!(box_blur_u8(&m, 8, 8, 1, 0), m);
    }

    #[test]
    fn channels_are_independent() {
        let mut m = vec![0u8; 5 * 5 * 3];
        for p in m.chunks_exact_mut(3) {
            p[1] = 90;
        }
        let o = box_blur_u8(&m, 5, 5, 3, 1);
        assert!(o.chunks_exact(3).all(|p| p == [0, 90, 0]));
    }

    #[test]
    fn gaussian_preserves_mean_roughly_and_smooths() {
        let mut m = vec![0u8; 41 * 41];
        for y in 15..26 {
            for x in 15..26 {
                m[y * 41 + x] = 255;
            }
        }
        let o = gaussian_blur_u8(&m, 41, 41, 1, 3.0);
        let a: u64 = m.iter().map(|&v| v as u64).sum();
        let b: u64 = o.iter().map(|&v| v as u64).sum();
        assert!((a as f64 - b as f64).abs() / a as f64 > -1.0 && (a as f64 - b as f64).abs() / (a as f64) < 0.03);
        assert!(o[20 * 41 + 20] < 255 && o[20 * 41 + 13] > 0);
    }

    #[test]
    fn float_mean_matches_integer_box() {
        let v: Vec<u8> = (0..100).map(|i| (i * 7 % 256) as u8).collect();
        let f: Vec<f32> = v.iter().map(|&x| x as f32).collect();
        let a = box_blur_u8(&v, 10, 10, 1, 2);
        let b = box_mean_f32(&f, 10, 10, 2);
        for (x, y) in a.iter().zip(&b) {
            assert!((*x as f32 - y).abs() <= 0.51);
        }
    }
}
