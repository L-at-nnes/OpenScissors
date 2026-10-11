//! Guided filter (He et al. 2010) and its fast variant (He & Sun 2015).

use super::filter::box_mean_f32;
use rayon::prelude::*;

/// Filter `src` using the edges of the single-channel `guide` (both `w` x `h`).
/// `eps` is in the squared value range of the inputs (0-255 data: `(0.05 * 255)^2`).
/// `subsample > 1` computes the linear coefficients on a reduced grid.
pub fn guided_filter(guide: &[f32], src: &[f32], w: usize, h: usize, r: usize, eps: f32, subsample: usize) -> Vec<f32> {
    assert_eq!(guide.len(), w * h);
    assert_eq!(src.len(), w * h);
    let s = subsample.max(1);
    let (gs, ps, sw, sh, rs) = if s > 1 {
        let (sw, sh) = ((w / s).max(1), (h / s).max(1));
        (downsample(guide, w, h, sw, sh), downsample(src, w, h, sw, sh), sw, sh, (r / s).max(1))
    } else {
        (guide.to_vec(), src.to_vec(), w, h, r)
    };

    let mean_g = box_mean_f32(&gs, sw, sh, rs);
    let mean_p = box_mean_f32(&ps, sw, sh, rs);
    let gp: Vec<f32> = gs.par_iter().zip(&ps).map(|(g, p)| g * p).collect();
    let gg: Vec<f32> = gs.par_iter().map(|g| g * g).collect();
    let mean_gp = box_mean_f32(&gp, sw, sh, rs);
    let mean_gg = box_mean_f32(&gg, sw, sh, rs);

    let mut a = vec![0f32; sw * sh];
    let mut b = vec![0f32; sw * sh];
    a.par_iter_mut().zip(b.par_iter_mut()).enumerate().for_each(|(i, (ai, bi))| {
        let cov = mean_gp[i] - mean_g[i] * mean_p[i];
        let var = mean_gg[i] - mean_g[i] * mean_g[i];
        *ai = cov / (var + eps);
        *bi = mean_p[i] - *ai * mean_g[i];
    });
    let mut mean_a = box_mean_f32(&a, sw, sh, rs);
    let mut mean_b = box_mean_f32(&b, sw, sh, rs);
    if s > 1 {
        mean_a = upsample(&mean_a, sw, sh, w, h);
        mean_b = upsample(&mean_b, sw, sh, w, h);
    }
    guide.par_iter().zip(mean_a.par_iter().zip(&mean_b)).map(|(g, (a, b))| a * g + b).collect()
}

fn downsample(src: &[f32], w: usize, h: usize, dw: usize, dh: usize) -> Vec<f32> {
    let mut out = vec![0f32; dw * dh];
    out.par_chunks_mut(dw).enumerate().for_each(|(y, row)| {
        let y0 = y * h / dh;
        let y1 = ((y + 1) * h / dh).max(y0 + 1).min(h);
        for (x, o) in row.iter_mut().enumerate() {
            let x0 = x * w / dw;
            let x1 = ((x + 1) * w / dw).max(x0 + 1).min(w);
            let mut acc = 0f32;
            for yy in y0..y1 {
                for xx in x0..x1 {
                    acc += src[yy * w + xx];
                }
            }
            *o = acc / ((y1 - y0) * (x1 - x0)) as f32;
        }
    });
    out
}

fn upsample(src: &[f32], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<f32> {
    let mut out = vec![0f32; dw * dh];
    out.par_chunks_mut(dw).enumerate().for_each(|(y, row)| {
        let fy = ((y as f32 + 0.5) * sh as f32 / dh as f32 - 0.5).clamp(0.0, (sh - 1) as f32);
        let y0 = fy.floor() as usize;
        let y1 = (y0 + 1).min(sh - 1);
        let ty = fy - y0 as f32;
        for (x, o) in row.iter_mut().enumerate() {
            let fx = ((x as f32 + 0.5) * sw as f32 / dw as f32 - 0.5).clamp(0.0, (sw - 1) as f32);
            let x0 = fx.floor() as usize;
            let x1 = (x0 + 1).min(sw - 1);
            let tx = fx - x0 as f32;
            let top = src[y0 * sw + x0] * (1.0 - tx) + src[y0 * sw + x1] * tx;
            let bot = src[y1 * sw + x0] * (1.0 - tx) + src[y1 * sw + x1] * tx;
            *o = top * (1.0 - ty) + bot * ty;
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn noise(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                (s >> 8) as f32 / (1u32 << 24) as f32 * 255.0
            })
            .collect()
    }

    #[test]
    fn constant_source_is_unchanged() {
        let g = noise(32 * 32, 1);
        let p = vec![100.0; 32 * 32];
        let o = guided_filter(&g, &p, 32, 32, 2, 10.0, 1);
        assert!(o.iter().all(|v| (v - 100.0).abs() < 1e-2));
    }

    #[test]
    fn source_linear_in_guide_is_recovered() {
        let g = noise(40 * 40, 2);
        let p: Vec<f32> = g.iter().map(|v| 0.5 * v + 10.0).collect();
        let o = guided_filter(&g, &p, 40, 40, 3, 1e-6, 1);
        for (a, b) in o.iter().zip(&p) {
            assert!((a - b).abs() < 0.05, "{a} vs {b}");
        }
    }

    #[test]
    fn huge_eps_smooths_towards_local_mean() {
        let g = noise(30 * 30, 3);
        let p = noise(30 * 30, 4);
        let o = guided_filter(&g, &p, 30, 30, 2, 1e12, 1);
        let sd = |v: &[f32]| {
            let m = v.iter().sum::<f32>() / v.len() as f32;
            (v.iter().map(|x| (x - m).powi(2)).sum::<f32>() / v.len() as f32).sqrt()
        };
        assert!(sd(&o) < sd(&p) * 0.5);
    }

    #[test]
    fn subsampled_is_close_to_full() {
        let (w, h) = (128, 128);
        let base = noise(16 * 16, 5);
        let mut g = vec![0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                g[y * w + x] = base[(y / 8) * 16 + x / 8];
            }
        }
        let p: Vec<f32> = g.iter().map(|v| v * 0.8 + 5.0).collect();
        let full = guided_filter(&g, &p, w, h, 4, 100.0, 1);
        let fast = guided_filter(&g, &p, w, h, 4, 100.0, 2);
        let mae = full.iter().zip(&fast).map(|(a, b)| (a - b).abs()).sum::<f32>() / full.len() as f32;
        assert!(mae < 2.0, "{mae}");
    }
}
