//! "Fill Hole": synthesise the background behind the removed subject.
//!
//! [`fill_hole`] is a pull-push diffusion at reduced resolution: smooth, without texture, done in a few milliseconds,
//! so that the preview shows a fill at once. [`fill_hole_texture`] continues the texture around the hole (grass, a
//! wall, a floor) with patches copied from the rest of the picture: the iterative completion of Wexler, Shechtman and
//! Irani (2007), from coarse to fine, with the PatchMatch search of Barnes et al. (2009) for the most similar patches.
//! The patches are searched at a working size; the colours are then copied from the full-resolution picture.

use rayon::prelude::*;

use crate::imaging::resize::{resize_image, resize_mask, Interp};
use crate::types::{Image, Mask};

const WORK_SIZE: usize = 768;

/// Patches are `2 R + 1` pixels square.
const R: i32 = 3;

/// The pyramid of the texture fill stops when the longer side gets under twice this.
const COARSEST: usize = 16;

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
    bilinear(&l.c, l.w, l.h, fx, fy)
}

/// The colour at `(fx, fy)` (pixel centres at whole numbers) of a `w` x `h` grid, interpolated.
fn bilinear(c: &[[f32; 3]], w: usize, h: usize, fx: f32, fy: f32) -> [f32; 3] {
    let x = fx.clamp(0.0, (w - 1) as f32);
    let y = fy.clamp(0.0, (h - 1) as f32);
    let (x0, y0) = (x.floor() as usize, y.floor() as usize);
    let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
    let (tx, ty) = (x - x0 as f32, y - y0 as f32);
    let mut out = [0f32; 3];
    for ch in 0..3 {
        let top = c[y0 * w + x0][ch] * (1.0 - tx) + c[y0 * w + x1][ch] * tx;
        let bot = c[y1 * w + x0][ch] * (1.0 - tx) + c[y1 * w + x1][ch] * tx;
        out[ch] = top * (1.0 - ty) + bot * ty;
    }
    out
}

/// Pull-push: the colours `c` whose known fraction `k` is below 1 are filled from the known ones around them.
fn pull_push(w: usize, h: usize, c: Vec<[f32; 3]>, k: Vec<f32>) -> Vec<[f32; 3]> {
    let mut levels = vec![Level { w, h, c, k }];
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
    levels.swap_remove(0).c
}

/// The working size of a `w` x `h` picture: its longer side at most [`WORK_SIZE`].
fn work_size(w: usize, h: usize) -> (usize, usize) {
    let s = (WORK_SIZE as f64 / w.max(h) as f64).min(1.0);
    (((w as f64 * s).round() as usize).max(1), ((h as f64 * s).round() as usize).max(1))
}

/// Fill the pixels of `hole` (non-zero) from their surroundings, smoothly (see the module).
pub fn fill_hole(image: &Image, hole: &Mask) -> Image {
    if hole.is_empty_mask() {
        return image.clone();
    }
    let (ww, wh) = work_size(image.w, image.h);
    let small = resize_image(image, ww, wh, Interp::Area);
    let small_hole = resize_mask(hole, ww, wh, Interp::Area);
    let c = small.data.chunks_exact(3).map(|p| [p[0] as f32, p[1] as f32, p[2] as f32]).collect();
    let k = small_hole.data.iter().map(|&v| if v > 0 { 0.0 } else { 1.0 }).collect();
    let filled_small = Image { w: ww, h: wh, data: pull_push(ww, wh, c, k).iter().flat_map(|c| c.map(to_u8)).collect() };
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

fn to_u8(v: f32) -> u8 {
    v.round().clamp(0.0, 255.0) as u8
}

/// A small, fast, reproducible random generator (xorshift64*): the same picture always gets the same fill.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// A whole number from `-r` to `r`.
    fn within(&mut self, r: i64) -> i64 {
        (self.next() % (2 * r as u64 + 1)) as i64 - r
    }
}

/// The picture at one level of the texture fill: colours as floats, and the pixels to fill.
struct Plane {
    w: usize,
    h: usize,
    px: Vec<[f32; 3]>,
    hole: Vec<bool>,
}

impl Plane {
    /// Half the size. A pixel is to fill when any of the four it covers is.
    fn half(&self) -> Plane {
        let (w, h) = (self.w.div_ceil(2), self.h.div_ceil(2));
        let mut px = vec![[0f32; 3]; w * h];
        let mut hole = vec![false; w * h];
        for y in 0..h {
            for x in 0..w {
                let (mut acc, mut n) = ([0f32; 3], 0f32);
                for (sx, sy) in [(2 * x, 2 * y), (2 * x + 1, 2 * y), (2 * x, 2 * y + 1), (2 * x + 1, 2 * y + 1)] {
                    if sx < self.w && sy < self.h {
                        let i = sy * self.w + sx;
                        hole[y * w + x] |= self.hole[i];
                        for ch in 0..3 {
                            acc[ch] += self.px[i][ch];
                        }
                        n += 1.0;
                    }
                }
                px[y * w + x] = acc.map(|v| v / n);
            }
        }
        Plane { w, h, px, hole }
    }

    /// For each pixel: whether a patch centred there lies inside the picture and outside the hole (a place to copy
    /// from), and whether the patch centred there touches the hole (a place to fill).
    fn roles(&self) -> (Vec<bool>, Vec<bool>) {
        let (w, h) = (self.w, self.h);
        // Summed-area table of the hole: the number of hole pixels in any rectangle in four reads.
        let mut sat = vec![0u32; (w + 1) * (h + 1)];
        for y in 0..h {
            let mut row = 0;
            for x in 0..w {
                row += self.hole[y * w + x] as u32;
                sat[(y + 1) * (w + 1) + x + 1] = sat[y * (w + 1) + x + 1] + row;
            }
        }
        let count = |x0: usize, y0: usize, x1: usize, y1: usize| {
            sat[(y1 + 1) * (w + 1) + x1 + 1] + sat[y0 * (w + 1) + x0] - sat[y0 * (w + 1) + x1 + 1] - sat[(y1 + 1) * (w + 1) + x0]
        };
        let r = R as usize;
        // Patches are taken near the hole, within its own size around it: what is next to the subject continues
        // behind it better than what is at the other end of the picture.
        let (mut bx0, mut by0, mut bx1, mut by1) = (w, h, 0, 0);
        for (i, _) in self.hole.iter().enumerate().filter(|(_, &v)| v) {
            let (x, y) = (i % w, i / w);
            (bx0, by0, bx1, by1) = (bx0.min(x), by0.min(y), bx1.max(x), by1.max(y));
        }
        let margin = (bx1.saturating_sub(bx0)).max(by1.saturating_sub(by0)).max(16);
        let near = |x: usize, y: usize| x + margin >= bx0 && x <= bx1 + margin && y + margin >= by0 && y <= by1 + margin;
        let mut source = vec![false; w * h];
        let mut target = vec![false; w * h];
        for y in 0..h {
            for x in 0..w {
                let (x0, y0, x1, y1) = (x.saturating_sub(r), y.saturating_sub(r), (x + r).min(w - 1), (y + r).min(h - 1));
                let inside = x >= r && y >= r && x + r < w && y + r < h && near(x, y);
                let holes = count(x0, y0, x1, y1);
                source[y * w + x] = inside && holes == 0;
                target[y * w + x] = holes > 0;
            }
        }
        (source, target)
    }

    /// The summed squared colour difference between the patch centred on `t` (cut by the edges of the picture) and
    /// the one centred on `s` (always inside). Stops counting once it exceeds `best`.
    fn distance(&self, t: usize, s: usize, best: f32) -> f32 {
        let (w, h) = (self.w as i32, self.h as i32);
        let (tx, ty) = ((t % self.w) as i32, (t / self.w) as i32);
        let (sx, sy) = ((s % self.w) as i32, (s / self.w) as i32);
        let mut d = 0f32;
        for dy in -R..=R {
            let y = ty + dy;
            if y < 0 || y >= h {
                continue;
            }
            let (trow, srow) = ((y * w) as usize, ((sy + dy) * w) as usize);
            for dx in -R..=R {
                let x = tx + dx;
                if x < 0 || x >= w {
                    continue;
                }
                let (a, b) = (self.px[trow + x as usize], self.px[srow + (sx + dx) as usize]);
                d += (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2);
            }
            if d >= best {
                return d;
            }
        }
        d
    }

    /// How much each pixel can be trusted: 1 when known, less and less deeper in the hole (1.3 to the power of minus
    /// the distance to the nearest known pixel, a chamfer distance, counted up to 40 pixels).
    fn confidence(&self) -> Vec<f32> {
        let (w, h) = (self.w, self.h);
        let far = u32::MAX / 2;
        let mut d: Vec<u32> = self.hole.iter().map(|&v| if v { far } else { 0 }).collect();
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                if d[i] == 0 {
                    continue;
                }
                let mut m = d[i];
                if x > 0 {
                    m = m.min(d[i - 1] + 3);
                }
                if y > 0 {
                    m = m.min(d[i - w] + 3);
                    if x > 0 {
                        m = m.min(d[i - w - 1] + 4);
                    }
                    if x + 1 < w {
                        m = m.min(d[i - w + 1] + 4);
                    }
                }
                d[i] = m;
            }
        }
        for y in (0..h).rev() {
            for x in (0..w).rev() {
                let i = y * w + x;
                if d[i] == 0 {
                    continue;
                }
                let mut m = d[i];
                if x + 1 < w {
                    m = m.min(d[i + 1] + 3);
                }
                if y + 1 < h {
                    m = m.min(d[i + w] + 3);
                    if x + 1 < w {
                        m = m.min(d[i + w + 1] + 4);
                    }
                    if x > 0 {
                        m = m.min(d[i + w - 1] + 4);
                    }
                }
                d[i] = m;
            }
        }
        d.iter().map(|&v| 1.3f32.powf(-(v.min(120) as f32) / 3.0)).collect()
    }

    /// The mean of `v` over the patch centred on each pixel (cut by the edges).
    fn patch_means(&self, v: &[f32]) -> Vec<f32> {
        let (w, h) = (self.w, self.h);
        let mut sat = vec![0f64; (w + 1) * (h + 1)];
        for y in 0..h {
            let mut row = 0f64;
            for x in 0..w {
                row += f64::from(v[y * w + x]);
                sat[(y + 1) * (w + 1) + x + 1] = sat[y * (w + 1) + x + 1] + row;
            }
        }
        let r = R as usize;
        (0..w * h)
            .map(|i| {
                let (x, y) = (i % w, i / w);
                let (x0, y0, x1, y1) = (x.saturating_sub(r), y.saturating_sub(r), (x + r).min(w - 1), (y + r).min(h - 1));
                let sum =
                    sat[(y1 + 1) * (w + 1) + x1 + 1] + sat[y0 * (w + 1) + x0] - sat[y0 * (w + 1) + x1 + 1] - sat[(y1 + 1) * (w + 1) + x0];
                (sum / ((x1 - x0 + 1) * (y1 - y0 + 1)) as f64) as f32
            })
            .collect()
    }

    /// The number of colour samples a patch centred on `t` holds once cut by the edges.
    fn samples(&self, t: usize) -> f32 {
        let (x, y) = ((t % self.w) as i32, (t / self.w) as i32);
        let span = |c: i32, n: i32| ((c + R).min(n - 1) - (c - R).max(0) + 1) as f32;
        span(x, self.w as i32) * span(y, self.h as i32) * 3.0
    }
}

/// The nearest-neighbour field of one level: for each pixel whose patch touches the hole (a target), the centre of
/// the most similar patch outside it, and their distance.
struct Field {
    w: usize,
    is_source: Vec<bool>,
    is_target: Vec<bool>,
    targets: Vec<u32>,
    nnf: Vec<u32>,
    dist: Vec<f32>,
}

impl Field {
    /// Every target starts from a random source. `None` when there is nothing to copy from.
    fn random(plane: &Plane, rng: &mut Rng) -> Option<Field> {
        let (is_source, is_target) = plane.roles();
        let sources: Vec<u32> = (0..is_source.len()).filter(|&i| is_source[i]).map(|i| i as u32).collect();
        if sources.is_empty() {
            return None;
        }
        let targets: Vec<u32> = (0..is_target.len()).filter(|&i| is_target[i]).map(|i| i as u32).collect();
        let mut nnf = vec![0u32; plane.w * plane.h];
        for &t in &targets {
            nnf[t as usize] = sources[(rng.next() % sources.len() as u64) as usize];
        }
        Some(Field { w: plane.w, is_source, is_target, targets, nnf, dist: vec![f32::INFINITY; plane.w * plane.h] })
    }

    /// The field of the level twice finer (`plane`), each target starting from where its coarser pixel pointed.
    fn upsample(&self, plane: &Plane, rng: &mut Rng) -> Option<Field> {
        let mut f = Field::random(plane, rng)?;
        let ch = self.nnf.len() / self.w;
        for &t in &f.targets {
            let (x, y) = (t as usize % plane.w, t as usize / plane.w);
            let c = (y / 2).min(ch - 1) * self.w + (x / 2).min(self.w - 1);
            if !self.is_target[c] {
                continue;
            }
            let q = self.nnf[c] as usize;
            let (qx, qy) = (2 * (q % self.w) + x % 2, 2 * (q / self.w) + y % 2);
            if qx < plane.w && qy < plane.h && f.is_source[qy * plane.w + qx] {
                f.nnf[t as usize] = (qy * plane.w + qx) as u32;
            }
        }
        Some(f)
    }

    fn refresh_distances(&mut self, plane: &Plane) {
        let d: Vec<f32> =
            self.targets.par_iter().map(|&t| plane.distance(t as usize, self.nnf[t as usize] as usize, f32::INFINITY)).collect();
        for (&t, d) in self.targets.iter().zip(d) {
            self.dist[t as usize] = d;
        }
    }

    /// One PatchMatch pass: each target tries the matches of its neighbours, shifted (propagation), then random
    /// places around its own match at shrinking distances (random search). Odd passes go backwards.
    fn improve(&mut self, plane: &Plane, pass: usize, rng: &mut Rng) {
        let (w, h) = (plane.w as i64, plane.h as i64);
        let step: i64 = if pass.is_multiple_of(2) { 1 } else { -1 };
        let n = self.targets.len();
        for k in 0..n {
            let t = self.targets[if step > 0 { k } else { n - 1 - k }] as usize;
            let (x, y) = ((t % plane.w) as i64, (t / plane.w) as i64);
            let (mut best, mut bd) = (self.nnf[t] as usize, self.dist[t]);
            for (dx, dy) in [(step, 0), (0, step)] {
                let (nx, ny) = (x - dx, y - dy);
                if nx < 0 || ny < 0 || nx >= w || ny >= h {
                    continue;
                }
                let nb = (ny * w + nx) as usize;
                if !self.is_target[nb] {
                    continue;
                }
                let q = self.nnf[nb] as i64;
                let (qx, qy) = (q % w + dx, q / w + dy);
                if qx < 0 || qy < 0 || qx >= w || qy >= h {
                    continue;
                }
                let cand = (qy * w + qx) as usize;
                if cand != best && self.is_source[cand] {
                    let d = plane.distance(t, cand, bd);
                    if d < bd {
                        (best, bd) = (cand, d);
                    }
                }
            }
            let (bx, by) = ((best % plane.w) as i64, (best / plane.w) as i64);
            let mut r = w.max(h);
            while r >= 1 {
                let (qx, qy) = (bx + rng.within(r), by + rng.within(r));
                if qx >= 0 && qy >= 0 && qx < w && qy < h {
                    let cand = (qy * w + qx) as usize;
                    if self.is_source[cand] {
                        let d = plane.distance(t, cand, bd);
                        if d < bd {
                            (best, bd) = (cand, d);
                        }
                    }
                }
                r /= 2;
            }
            self.nnf[t] = best as u32;
            self.dist[t] = bd;
        }
    }

    /// The weight of each target's match in the vote: close matches count much more than poor ones.
    /// Patches whose pixels are known, or close to known ones, count more than those deep in the hole (the
    /// confidence of Wexler et al.): what is around the hole leads, and the inside follows it.
    fn weights(&self, plane: &Plane) -> Vec<f32> {
        let mut errors: Vec<f32> = self.targets.iter().map(|&t| self.dist[t as usize] / plane.samples(t as usize)).collect();
        let mid = errors.len() / 2;
        let typical = if errors.is_empty() { 1.0 } else { *errors.select_nth_unstable_by(mid, f32::total_cmp).1 };
        let scale = typical.max(1.0);
        let confidence = plane.patch_means(&plane.confidence());
        let mut wt = vec![0f32; self.nnf.len()];
        for &t in &self.targets {
            let t = t as usize;
            wt[t] = ((-(self.dist[t] / plane.samples(t)) / scale).exp() * confidence[t]).max(1e-12);
        }
        wt
    }

    /// Each pixel to fill becomes the weighted mean of what the patches covering it copy there.
    fn vote(&self, plane: &mut Plane) {
        let wt = self.weights(plane);
        let holes: Vec<usize> = (0..plane.hole.len()).filter(|&i| plane.hole[i]).collect();
        let colours = self.copied(plane, &holes, &wt);
        for (p, c) in holes.into_iter().zip(colours) {
            plane.px[p] = c;
        }
    }

    /// Blending in the gradient domain (Pérez et al., 2003, with the guide voted from the patches as Darabi et al.,
    /// 2012, do): the fill keeps the differences between neighbours that the copied patches give, and is solved
    /// again (red-black over-relaxation) so that it meets the known pixels around it and joins patches copied from
    /// different places without a step. A pull towards the voted colours (screened Poisson) keeps the solution from
    /// drifting where the patches disagree, so a step is smoothed over a few pixels only. Returns the change to add
    /// to each pixel.
    fn blend(&self, plane: &Plane) -> Vec<[f32; 3]> {
        const ITERATIONS: usize = 100;
        const OMEGA: f32 = 1.8;
        const PULL: f32 = 0.05;
        let wt = self.weights(plane);
        let (w, h) = (plane.w as i32, plane.h as i32);
        let steps = [(1, 0), (-1, 0), (0, 1), (0, -1)];
        let holes: Vec<usize> = (0..plane.hole.len()).filter(|&i| plane.hole[i]).collect();
        // For each pixel to fill and each neighbour (inside the picture): what the patches say `f(p) - f(neighbour)`
        // should be, or `None` for a neighbour outside.
        let guide: Vec<[Option<[f32; 3]>; 4]> = holes
            .par_iter()
            .map(|&p| {
                let (x, y) = ((p % plane.w) as i32, (p / plane.w) as i32);
                steps.map(|(ex, ey)| {
                    let (nx, ny) = (x + ex, y + ey);
                    if nx < 0 || ny < 0 || nx >= w || ny >= h {
                        return None;
                    }
                    let (mut acc, mut sum) = ([0f32; 3], 0f32);
                    for dy in -R..=R {
                        for dx in -R..=R {
                            let (cx, cy) = (x - dx, y - dy);
                            if cx < 0 || cy < 0 || cx >= w || cy >= h || !self.is_target[(cy * w + cx) as usize] {
                                continue;
                            }
                            let c = (cy * w + cx) as usize;
                            let q = self.nnf[c] as usize;
                            let (sx, sy) = ((q % plane.w) as i32 + dx, (q / plane.w) as i32 + dy);
                            let (tx, ty) = (sx + ex, sy + ey);
                            if tx < 0 || ty < 0 || tx >= w || ty >= h {
                                continue;
                            }
                            let (a, b) = (plane.px[(sy * w + sx) as usize], plane.px[(ty * w + tx) as usize]);
                            for ch in 0..3 {
                                acc[ch] += wt[c] * (a[ch] - b[ch]);
                            }
                            sum += wt[c];
                        }
                    }
                    let (a, b) = (plane.px[p], plane.px[(ny * w + nx) as usize]);
                    Some(if sum > 0.0 { acc.map(|v| v / sum) } else { [a[0] - b[0], a[1] - b[1], a[2] - b[2]] })
                })
            })
            .collect();
        let mut f = plane.px.clone();
        for _ in 0..ITERATIONS {
            for colour in 0..2 {
                let updates: Vec<(usize, [f32; 3])> = holes
                    .par_iter()
                    .zip(&guide)
                    .filter(|(&p, _)| (p % plane.w + p / plane.w) % 2 == colour)
                    .map(|(&p, g)| {
                        let (x, y) = ((p % plane.w) as i32, (p / plane.w) as i32);
                        let (mut acc, mut n) = ([0f32; 3], 0f32);
                        for (k, (ex, ey)) in steps.iter().enumerate() {
                            if let Some(d) = g[k] {
                                let v = f[((y + ey) * w + x + ex) as usize];
                                for ch in 0..3 {
                                    acc[ch] += v[ch] + d[ch];
                                }
                                n += 1.0;
                            }
                        }
                        let (old, voted) = (f[p], plane.px[p]);
                        (p, std::array::from_fn(|ch| old[ch] + OMEGA * ((acc[ch] + PULL * voted[ch]) / (n + PULL) - old[ch])))
                    })
                    .collect();
                for (p, v) in updates {
                    f[p] = v;
                }
            }
        }
        // What is left of a mismatch along the border is spread smoothly over the whole hole (a membrane, as
        // seamless cloning does), so that the fill meets the known pixels without a band.
        let mut c = vec![[0f32; 3]; f.len()];
        let mut k = vec![0f32; f.len()];
        for (&p, g) in holes.iter().zip(&guide) {
            let (x, y) = ((p % plane.w) as i32, (p / plane.w) as i32);
            let (mut acc, mut n) = ([0f32; 3], 0f32);
            for (i, (ex, ey)) in steps.iter().enumerate() {
                let Some(d) = g[i] else { continue };
                let nb = ((y + ey) * w + x + ex) as usize;
                if !plane.hole[nb] {
                    for ch in 0..3 {
                        acc[ch] += f[nb][ch] + d[ch];
                    }
                    n += 1.0;
                }
            }
            if n > 0.0 {
                c[p] = std::array::from_fn(|ch| acc[ch] / n - f[p][ch]);
                k[p] = 1.0;
            }
        }
        let membrane = pull_push(plane.w, plane.h, c, k);
        // Where the patches disagree (across a sharp edge) the solution can overshoot: each pixel stays within the
        // colours voted around it, give or take a little.
        const SLACK: f32 = 8.0;
        let bounds: Vec<([f32; 3], [f32; 3])> = holes
            .par_iter()
            .map(|&p| {
                let (x, y) = ((p % plane.w) as i32, (p / plane.w) as i32);
                let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
                for ny in (y - R).max(0)..=(y + R).min(h - 1) {
                    for nx in (x - R).max(0)..=(x + R).min(w - 1) {
                        let v = plane.px[(ny * w + nx) as usize];
                        for ch in 0..3 {
                            lo[ch] = lo[ch].min(v[ch]);
                            hi[ch] = hi[ch].max(v[ch]);
                        }
                    }
                }
                (lo, hi)
            })
            .collect();
        for (&p, (lo, hi)) in holes.iter().zip(bounds) {
            for ch in 0..3 {
                f[p][ch] = (f[p][ch] + membrane[p][ch]).clamp(lo[ch] - SLACK, hi[ch] + SLACK);
            }
        }
        f.iter().zip(&plane.px).map(|(a, b)| [a[0] - b[0], a[1] - b[1], a[2] - b[2]]).collect()
    }

    /// The weighted mean of what the patches covering each of `pixels` copy there.
    fn copied(&self, plane: &Plane, pixels: &[usize], wt: &[f32]) -> Vec<[f32; 3]> {
        let (w, h) = (plane.w as i32, plane.h as i32);
        pixels
            .par_iter()
            .map(|&p| {
                let (x, y) = ((p % plane.w) as i32, (p / plane.w) as i32);
                let (mut acc, mut sum) = ([0f32; 3], 0f32);
                for dy in -R..=R {
                    for dx in -R..=R {
                        let (cx, cy) = (x - dx, y - dy);
                        if cx < 0 || cy < 0 || cx >= w || cy >= h {
                            continue;
                        }
                        let c = (cy * w + cx) as usize;
                        if !self.is_target[c] {
                            continue;
                        }
                        let q = self.nnf[c] as usize;
                        let s = (((q / plane.w) as i32 + dy) * w + (q % plane.w) as i32 + dx) as usize;
                        let v = plane.px[s];
                        for ch in 0..3 {
                            acc[ch] += wt[c] * v[ch];
                        }
                        sum += wt[c];
                    }
                }
                if sum > 0.0 {
                    acc.map(|a| a / sum)
                } else {
                    plane.px[p]
                }
            })
            .collect()
    }
}

/// Fill the pixels of `hole` (non-zero) by continuing the texture of the rest of the picture (see the module). Falls
/// back on [`fill_hole`] when nothing is left to copy from.
pub fn fill_hole_texture(image: &Image, hole: &Mask) -> Image {
    if hole.is_empty_mask() {
        return image.clone();
    }
    let (ww, wh) = work_size(image.w, image.h);
    let small = resize_image(image, ww, wh, Interp::Area);
    let small_hole = resize_mask(hole, ww, wh, Interp::Area);
    let base = Plane {
        w: ww,
        h: wh,
        px: small.data.chunks_exact(3).map(|p| [p[0] as f32, p[1] as f32, p[2] as f32]).collect(),
        hole: small_hole.data.iter().map(|&v| v > 0).collect(),
    };
    let mut planes = vec![base];
    loop {
        let last = planes.last().expect("the working size");
        if last.w.max(last.h) < 2 * COARSEST {
            break;
        }
        let half = last.half();
        if !half.roles().0.contains(&true) {
            break; // the hole covers that level: start from the finer one
        }
        planes.push(half);
    }
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    // The coarsest level starts from the smooth fill.
    let top = planes.last_mut().expect("the working size");
    let known = top.hole.iter().map(|&h| if h { 0.0 } else { 1.0 }).collect();
    top.px = pull_push(top.w, top.h, std::mem::take(&mut top.px), known);
    let mut field: Option<Field> = None;
    for level in (0..planes.len()).rev() {
        let (finer, coarser) = planes.split_at_mut(level + 1);
        let plane = &mut finer[level];
        let next = match (&field, coarser.first()) {
            (Some(f), Some(coarse)) => {
                // The pixels to fill start from the coarser result.
                for y in 0..plane.h {
                    for x in 0..plane.w {
                        if plane.hole[y * plane.w + x] {
                            plane.px[y * plane.w + x] = coarse.px[(y / 2).min(coarse.h - 1) * coarse.w + (x / 2).min(coarse.w - 1)];
                        }
                    }
                }
                f.upsample(plane, &mut rng)
            }
            _ => Field::random(plane, &mut rng),
        };
        let Some(mut f) = next else { return fill_hole(image, hole) };
        let rounds = if level == 0 { 3 } else { 5 };
        for _ in 0..rounds {
            f.refresh_distances(plane);
            for pass in 0..2 {
                f.improve(plane, pass, &mut rng);
            }
            f.vote(plane);
        }
        field = Some(f);
    }
    let (plane, field) = (&planes[0], field.expect("one level at least"));
    let change = field.blend(plane);
    let blended: Vec<[f32; 3]> = plane.px.iter().zip(&change).map(|(v, c)| [v[0] + c[0], v[1] + c[1], v[2] + c[2]]).collect();
    let mut out = image.clone();
    if (ww, wh) == (image.w, image.h) {
        for (i, &hv) in hole.data.iter().enumerate() {
            if hv > 0 {
                out.data[i * 3..i * 3 + 3].copy_from_slice(&blended[i].map(to_u8));
            }
        }
        return out;
    }
    let small = Image { w: ww, h: wh, data: blended.iter().flat_map(|c| c.map(to_u8)).collect() };
    let enlarged = resize_image(&small, image.w, image.h, Interp::Bilinear);
    // The colours copied at full size bring the detail finer than the working size; the blended result at the
    // working size brings the rest, so the seams between copies stay hidden.
    let mut copied = image.clone();
    full_resolution(image, hole, plane, &field, &enlarged, &mut copied);
    let coarse = resize_image(&resize_image(&copied, ww, wh, Interp::Area), image.w, image.h, Interp::Bilinear);
    for (i, &hv) in hole.data.iter().enumerate() {
        if hv > 0 {
            for ch in 0..3 {
                let k = i * 3 + ch;
                out.data[k] = to_u8(enlarged.data[k] as f32 + copied.data[k] as f32 - coarse.data[k] as f32);
            }
        }
    }
    out
}

/// Fill `hole` in `out` (a copy of `image`) at full resolution: each pixel takes, from the full-resolution picture,
/// the colours the matches found at the working size point to, weighted as in [`Field::vote`]. A pixel with no
/// usable match takes `enlarged`.
fn full_resolution(image: &Image, hole: &Mask, plane: &Plane, field: &Field, enlarged: &Image, out: &mut Image) {
    let wt = field.weights(plane);
    let (sx, sy) = (plane.w as f64 / image.w as f64, plane.h as f64 / image.h as f64);
    let (pw, ph) = (plane.w as i64, plane.h as i64);
    let (iw, ih) = (image.w as i64, image.h as i64);
    out.data.par_chunks_mut(image.w * 3).enumerate().for_each(|(y, row)| {
        for x in 0..image.w {
            let i = y * image.w + x;
            if hole.data[i] == 0 {
                continue;
            }
            let (px, py) = (((x as f64 * sx) as i64).min(pw - 1), ((y as f64 * sy) as i64).min(ph - 1));
            let (mut acc, mut sum) = ([0f32; 3], 0f32);
            for dy in -R as i64..=R as i64 {
                for dx in -R as i64..=R as i64 {
                    let (cx, cy) = (px - dx, py - dy);
                    if cx < 0 || cy < 0 || cx >= pw || cy >= ph {
                        continue;
                    }
                    let c = (cy * pw + cx) as usize;
                    if !field.is_target[c] {
                        continue;
                    }
                    // The match is shifted from its target by (q - c) working pixels: the same shift at full size.
                    let q = field.nnf[c] as i64;
                    let qx = x as i64 + (((q % pw) - cx) as f64 / sx).round() as i64;
                    let qy = y as i64 + (((q / pw) - cy) as f64 / sy).round() as i64;
                    if qx < 0 || qy < 0 || qx >= iw || qy >= ih {
                        continue;
                    }
                    let s = (qy * iw + qx) as usize;
                    if hole.data[s] > 0 {
                        continue;
                    }
                    for ch in 0..3 {
                        acc[ch] += wt[c] * image.data[s * 3 + ch] as f32;
                    }
                    sum += wt[c];
                }
            }
            let colour = if sum > 0.0 {
                acc.map(|a| to_u8(a / sum))
            } else {
                [enlarged.data[i * 3], enlarged.data[i * 3 + 1], enlarged.data[i * 3 + 2]]
            };
            row[x * 3..x * 3 + 3].copy_from_slice(&colour);
        }
    });
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
        for fill in [fill_hole, fill_hole_texture] {
            let out = fill(&img, &hole);
            let p = &out.data[(15 * 40 + 20) * 3..(15 * 40 + 20) * 3 + 3];
            for (a, b) in p.iter().zip([50u8, 100, 150]) {
                assert!((*a as i32 - b as i32).abs() <= 3, "{p:?}");
            }
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
        assert_eq!(fill_hole_texture(&img, &Mask::new(5, 5)), img);
    }

    /// Vertical stripes `period` pixels wide, black and white, with a square hole in the middle.
    fn stripes(w: usize, h: usize, period: usize, side: usize) -> (Image, Image, Mask) {
        let mut truth = Image::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let v = if (x / period).is_multiple_of(2) { 20 } else { 230 };
                truth.data[(y * w + x) * 3..(y * w + x) * 3 + 3].copy_from_slice(&[v, v, v]);
            }
        }
        let (mut img, mut hole) = (truth.clone(), Mask::new(w, h));
        let (x0, y0) = ((w - side) / 2, (h - side) / 2);
        for y in y0..y0 + side {
            for x in x0..x0 + side {
                hole.data[y * w + x] = 255;
                img.data[(y * w + x) * 3..(y * w + x) * 3 + 3].copy_from_slice(&[255, 0, 0]);
            }
        }
        (img, truth, hole)
    }

    /// Mean difference with the truth inside the hole, and the share of pixels there that are clearly dark or light
    /// (a smooth fill gives a grey in between).
    fn score(out: &Image, truth: &Image, hole: &Mask) -> (f64, f64) {
        let (mut diff, mut contrast, mut n) = (0f64, 0f64, 0f64);
        for (i, &hv) in hole.data.iter().enumerate() {
            if hv > 0 {
                let v = out.data[i * 3] as f64;
                diff += (v - truth.data[i * 3] as f64).abs();
                contrast += (!(60.0..=190.0).contains(&v)) as u8 as f64;
                n += 1.0;
            }
        }
        (diff / n, contrast / n)
    }

    #[test]
    fn the_texture_fill_continues_a_pattern_the_smooth_fill_blurs() {
        let (img, truth, hole) = stripes(160, 120, 6, 40);
        let (smooth_diff, smooth_contrast) = score(&fill_hole(&img, &hole), &truth, &hole);
        let textured = fill_hole_texture(&img, &hole);
        let (diff, contrast) = score(&textured, &truth, &hole);
        assert!(contrast > 0.8 && smooth_contrast < 0.2, "the stripes go on: {contrast} against {smooth_contrast}");
        assert!(diff < 25.0 && diff < smooth_diff / 3.0, "in step with the stripes around: {diff} against {smooth_diff}");
        assert_eq!(textured.data[..160 * 3 * 30], img.data[..160 * 3 * 30], "the rest of the picture is untouched");
        assert_eq!(textured, fill_hole_texture(&img, &hole), "the same picture always gets the same fill");
    }

    #[test]
    fn a_picture_larger_than_the_working_size_gets_its_texture_at_full_resolution() {
        // 1600 px wide: searched at 768, copied at full size. Stripes 10 px wide are 4.8 px at the working size.
        let (img, truth, hole) = stripes(1600, 400, 10, 160);
        let (smooth_diff, _) = score(&fill_hole(&img, &hole), &truth, &hole);
        let (diff, contrast) = score(&fill_hole_texture(&img, &hole), &truth, &hole);
        assert!(contrast > 0.6, "sharp stripes, not an enlarged blur: {contrast}");
        // The phase of stripes this fine is found to a working pixel (2 px here), not exactly: closer than a blur.
        assert!(diff < smooth_diff * 0.7, "{diff} against {smooth_diff}");
    }

    #[test]
    fn a_hole_with_nothing_around_to_copy_falls_back_on_the_smooth_fill() {
        let img = Image::filled(20, 20, [10, 20, 30]);
        let mut hole = Mask::filled(20, 20, 255);
        hole.data[0] = 0;
        assert_eq!(fill_hole_texture(&img, &hole), fill_hole(&img, &hole));
    }
}
