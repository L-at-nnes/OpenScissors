//! GrabCut (Rother, Kolmogorov, Blake 2004) with the iterated-GMM energy and a min-cut
//! solved by [`maxflow`](super::maxflow). Colours are RGB; labels follow the usual
//! four-state convention.

use super::maxflow::Graph;
use crate::types::Image;

pub const BGD: u8 = 0;
pub const FGD: u8 = 1;
pub const PR_BGD: u8 = 2;
pub const PR_FGD: u8 = 3;

const K: usize = 5;
const KMEANS_ITERS: usize = 10;

type Color = [f64; 3];

/// Mixture of up to `K` full-covariance Gaussians.
#[derive(Clone)]
struct Gmm {
    coef: [f64; K],
    mean: [Color; K],
    inv: [[[f64; 3]; 3]; K],
    det: [f64; K],
}

fn det3(c: &[[f64; 3]; 3]) -> f64 {
    c[0][0] * (c[1][1] * c[2][2] - c[1][2] * c[2][1]) - c[0][1] * (c[1][0] * c[2][2] - c[1][2] * c[2][0])
        + c[0][2] * (c[1][0] * c[2][1] - c[1][1] * c[2][0])
}

fn inv3(c: &[[f64; 3]; 3], det: f64) -> [[f64; 3]; 3] {
    let d = 1.0 / det;
    [
        [
            (c[1][1] * c[2][2] - c[1][2] * c[2][1]) * d,
            (c[0][2] * c[2][1] - c[0][1] * c[2][2]) * d,
            (c[0][1] * c[1][2] - c[0][2] * c[1][1]) * d,
        ],
        [
            (c[1][2] * c[2][0] - c[1][0] * c[2][2]) * d,
            (c[0][0] * c[2][2] - c[0][2] * c[2][0]) * d,
            (c[0][2] * c[1][0] - c[0][0] * c[1][2]) * d,
        ],
        [
            (c[1][0] * c[2][1] - c[1][1] * c[2][0]) * d,
            (c[0][1] * c[2][0] - c[0][0] * c[2][1]) * d,
            (c[0][0] * c[1][1] - c[0][1] * c[1][0]) * d,
        ],
    ]
}

impl Gmm {
    fn component_pdf(&self, k: usize, c: Color) -> f64 {
        if self.coef[k] <= 0.0 {
            return 0.0;
        }
        let d = [c[0] - self.mean[k][0], c[1] - self.mean[k][1], c[2] - self.mean[k][2]];
        let m = &self.inv[k];
        let mut q = 0.0;
        for i in 0..3 {
            q += d[i] * (m[i][0] * d[0] + m[i][1] * d[1] + m[i][2] * d[2]);
        }
        (1.0 / self.det[k].sqrt()) * (-0.5 * q).exp()
    }

    /// Density of the mixture at `c`.
    fn prob(&self, c: Color) -> f64 {
        (0..K).map(|k| self.coef[k] * self.component_pdf(k, c)).sum()
    }

    /// Most likely component for `c`.
    fn which(&self, c: Color) -> usize {
        let mut best = (0, -1.0);
        for k in 0..K {
            let p = self.component_pdf(k, c);
            if p > best.1 {
                best = (k, p);
            }
        }
        best.0
    }

    /// Fit from samples with their component assignment.
    fn learn(samples: &[Color], comp: &[usize]) -> Self {
        let mut n = [0f64; K];
        let mut sum = [[0f64; 3]; K];
        let mut prod = [[[0f64; 3]; 3]; K];
        for (c, &k) in samples.iter().zip(comp) {
            n[k] += 1.0;
            for i in 0..3 {
                sum[k][i] += c[i];
                for j in 0..3 {
                    prod[k][i][j] += c[i] * c[j];
                }
            }
        }
        let total: f64 = n.iter().sum();
        let mut g = Gmm { coef: [0.0; K], mean: [[0.0; 3]; K], inv: [[[0.0; 3]; 3]; K], det: [1.0; K] };
        for k in 0..K {
            if n[k] == 0.0 {
                continue;
            }
            g.coef[k] = n[k] / total;
            for i in 0..3 {
                g.mean[k][i] = sum[k][i] / n[k];
            }
            let mut cov = [[0f64; 3]; 3];
            for i in 0..3 {
                for j in 0..3 {
                    cov[i][j] = prod[k][i][j] / n[k] - g.mean[k][i] * g.mean[k][j];
                }
            }
            let mut det = det3(&cov);
            if det <= f64::EPSILON {
                // Degenerate (flat colour): add a little variance so the density stays finite.
                for (i, row) in cov.iter_mut().enumerate() {
                    row[i] += 0.01;
                }
                det = det3(&cov);
            }
            g.det[k] = det;
            g.inv[k] = inv3(&cov, det);
        }
        g
    }
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn unit(&mut self) -> f64 {
        self.next() as f64 / (1u64 << 31) as f64
    }
}

fn dist2(a: Color, b: Color) -> f64 {
    (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)
}

/// K-means with k-means++ seeding (fixed seed: deterministic results).
fn kmeans(samples: &[Color]) -> Vec<usize> {
    let n = samples.len();
    if n <= K {
        return (0..n).collect();
    }
    let mut rng = Rng(0x9E3779B97F4A7C15);
    let mut centers: Vec<Color> = vec![samples[(rng.next() as usize) % n]];
    let mut d2: Vec<f64> = samples.iter().map(|&s| dist2(s, centers[0])).collect();
    while centers.len() < K {
        let total: f64 = d2.iter().sum();
        let pick = if total <= 0.0 {
            (rng.next() as usize) % n
        } else {
            let mut t = rng.unit() * total;
            let mut idx = n - 1;
            for (i, &d) in d2.iter().enumerate() {
                if t <= d {
                    idx = i;
                    break;
                }
                t -= d;
            }
            idx
        };
        centers.push(samples[pick]);
        for (i, s) in samples.iter().enumerate() {
            d2[i] = d2[i].min(dist2(*s, samples[pick]));
        }
    }
    let mut labels = vec![0usize; n];
    for _ in 0..KMEANS_ITERS {
        for (i, s) in samples.iter().enumerate() {
            let mut best = (0, f64::MAX);
            for (k, c) in centers.iter().enumerate() {
                let d = dist2(*s, *c);
                if d < best.1 {
                    best = (k, d);
                }
            }
            labels[i] = best.0;
        }
        let mut sum = [[0f64; 3]; K];
        let mut cnt = [0f64; K];
        for (s, &l) in samples.iter().zip(&labels) {
            cnt[l] += 1.0;
            for c in 0..3 {
                sum[l][c] += s[c];
            }
        }
        for k in 0..K {
            if cnt[k] > 0.0 {
                centers[k] = [sum[k][0] / cnt[k], sum[k][1] / cnt[k], sum[k][2] / cnt[k]];
            }
        }
    }
    labels
}

/// Neighbour weights: left, up-left, up, up-right (diagonals scaled by 1/sqrt 2).
fn neighbour_weights(img: &[Color], w: usize, h: usize, gamma: f64) -> [Vec<f64>; 4] {
    let offsets: [(isize, isize); 4] = [(-1, 0), (-1, -1), (0, -1), (1, -1)];
    let mut sum = 0.0;
    let mut count = 0usize;
    for y in 0..h {
        for x in 0..w {
            for (dx, dy) in offsets {
                let (nx, ny) = (x as isize + dx, y as isize + dy);
                if nx >= 0 && ny >= 0 && (nx as usize) < w {
                    sum += dist2(img[y * w + x], img[ny as usize * w + nx as usize]);
                    count += 1;
                }
            }
        }
    }
    let beta = if sum <= f64::EPSILON { 0.0 } else { 1.0 / (2.0 * sum / count as f64) };
    let mut out = [vec![0.0; w * h], vec![0.0; w * h], vec![0.0; w * h], vec![0.0; w * h]];
    for y in 0..h {
        for x in 0..w {
            for (k, (dx, dy)) in offsets.iter().enumerate() {
                let (nx, ny) = (x as isize + dx, y as isize + dy);
                if nx >= 0 && ny >= 0 && (nx as usize) < w {
                    let scale = if k == 0 || k == 2 { 1.0 } else { std::f64::consts::FRAC_1_SQRT_2 };
                    out[k][y * w + x] = gamma * scale * (-beta * dist2(img[y * w + x], img[ny as usize * w + nx as usize])).exp();
                }
            }
        }
    }
    out
}

/// Run `iterations` GrabCut passes over `labels` (initialised from a mask) in place.
/// `gamma` weights the smoothness term (the usual value is 50; larger = smoother).
pub fn grabcut(image: &Image, labels: &mut [u8], iterations: usize, gamma: f64) {
    let (w, h) = (image.w, image.h);
    assert_eq!(labels.len(), w * h);
    let img: Vec<Color> = image.data.chunks_exact(3).map(|p| [p[0] as f64, p[1] as f64, p[2] as f64]).collect();
    let lambda = 9.0 * gamma;
    let weights = neighbour_weights(&img, w, h, gamma);

    // Initial mixtures from the labelled sets.
    let split = |labels: &[u8], fg: bool| -> (Vec<usize>, Vec<Color>) {
        let idx: Vec<usize> = (0..w * h).filter(|&i| (labels[i] == FGD || labels[i] == PR_FGD) == fg).collect();
        let samples = idx.iter().map(|&i| img[i]).collect();
        (idx, samples)
    };
    let (_, bg_s) = split(labels, false);
    let (_, fg_s) = split(labels, true);
    if bg_s.is_empty() || fg_s.is_empty() {
        return;
    }
    let mut bgd = Gmm::learn(&bg_s, &kmeans(&bg_s));
    let mut fgd = Gmm::learn(&fg_s, &kmeans(&fg_s));

    for _ in 0..iterations {
        // Assign each pixel to a component of its own mixture, then refit both.
        let mut comp = vec![0usize; w * h];
        for i in 0..w * h {
            comp[i] = if labels[i] == BGD || labels[i] == PR_BGD { bgd.which(img[i]) } else { fgd.which(img[i]) };
        }
        let gather = |fg: bool| -> (Vec<Color>, Vec<usize>) {
            let mut s = Vec::new();
            let mut c = Vec::new();
            for i in 0..w * h {
                if (labels[i] == FGD || labels[i] == PR_FGD) == fg {
                    s.push(img[i]);
                    c.push(comp[i]);
                }
            }
            (s, c)
        };
        let (bs, bc) = gather(false);
        let (fs, fc) = gather(true);
        if bs.is_empty() || fs.is_empty() {
            return;
        }
        bgd = Gmm::learn(&bs, &bc);
        fgd = Gmm::learn(&fs, &fc);

        // Graph: source = foreground, sink = background.
        let mut g = Graph::new(w * h, w * h * 4);
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                let (from_source, to_sink) = match labels[i] {
                    BGD => (0.0, lambda),
                    FGD => (lambda, 0.0),
                    _ => (-bgd.prob(img[i]).max(f64::MIN_POSITIVE).ln(), -fgd.prob(img[i]).max(f64::MIN_POSITIVE).ln()),
                };
                g.add_terminal(i, from_source, to_sink);
                let neighbours = [
                    (x > 0, i.wrapping_sub(1), 0),
                    (x > 0 && y > 0, i.wrapping_sub(w + 1), 1),
                    (y > 0, i.wrapping_sub(w), 2),
                    (x + 1 < w && y > 0, i.wrapping_sub(w - 1), 3),
                ];
                for (ok, j, k) in neighbours {
                    if ok {
                        g.add_edge(i, j, weights[k][i], weights[k][i]);
                    }
                }
            }
        }
        g.max_flow();
        for i in 0..w * h {
            if labels[i] == PR_BGD || labels[i] == PR_FGD {
                labels[i] = if g.in_source_segment(i) { PR_FGD } else { PR_BGD };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Left half reddish, right half bluish, with deterministic noise.
    fn two_tone(w: usize, h: usize) -> Image {
        let mut img = Image::new(w, h);
        let mut s = 1u32;
        for y in 0..h {
            for x in 0..w {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                let n = ((s >> 24) % 30) as i32 - 15;
                let base: [i32; 3] = if x < w / 2 { [200, 60, 50] } else { [40, 70, 190] };
                for c in 0..3 {
                    img.data[(y * w + x) * 3 + c] = (base[c] + n).clamp(0, 255) as u8;
                }
            }
        }
        img
    }

    #[test]
    fn separates_two_colour_regions_from_seeds() {
        let (w, h) = (40, 30);
        let img = two_tone(w, h);
        let mut labels = vec![PR_BGD; w * h];
        for y in 5..25 {
            for x in 3..9 {
                labels[y * w + x] = FGD; // seed in the red half
                labels[y * w + x + 28] = BGD; // seed in the blue half
            }
        }
        grabcut(&img, &mut labels, 1, 100.0);
        let wrong = (0..w * h).filter(|&i| (labels[i] == FGD || labels[i] == PR_FGD) != (i % w < w / 2)).count();
        assert!(wrong * 100 < w * h, "{wrong} wrong pixels of {}", w * h);
    }

    #[test]
    fn hard_labels_never_change() {
        let (w, h) = (20, 20);
        let img = two_tone(w, h);
        let mut labels = vec![PR_FGD; w * h];
        labels[3] = BGD;
        labels[w * h - 1] = FGD;
        labels[100] = BGD;
        grabcut(&img, &mut labels, 1, 100.0);
        assert_eq!((labels[3], labels[w * h - 1], labels[100]), (BGD, FGD, BGD));
    }

    #[test]
    fn no_op_without_both_classes() {
        let img = two_tone(10, 10);
        let mut labels = vec![PR_FGD; 100];
        grabcut(&img, &mut labels, 1, 100.0);
        assert!(labels.iter().all(|&l| l == PR_FGD));
    }

    #[test]
    fn gmm_density_is_higher_near_its_samples() {
        let samples: Vec<Color> = (0..50).map(|i| [100.0 + (i % 5) as f64, 50.0 + (i % 3) as f64, 20.0 + (i % 7) as f64]).collect();
        let g = Gmm::learn(&samples, &kmeans(&samples));
        assert!(g.prob([102.0, 51.0, 23.0]) > g.prob([10.0, 200.0, 220.0]) * 1000.0);
    }
}
