//! Grayscale morphology with an elliptical structuring element.

use rayon::prelude::*;
use std::collections::VecDeque;

use crate::types::Mask;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MorphOp {
    Erode,
    Dilate,
    Open,
    Close,
}

/// Per-row horizontal extent of a `k` x `k` ellipse: `(lo, hi)` offsets from the anchor,
/// for rows `-r..=r` (anchor at `k / 2`).
fn ellipse_rows(k: usize) -> (isize, Vec<(isize, isize)>) {
    let r = (k / 2) as isize;
    let c = (k / 2) as isize;
    let inv_r2 = if r > 0 { 1.0 / (r * r) as f64 } else { 0.0 };
    let mut rows = Vec::with_capacity(k);
    for i in 0..k as isize {
        let dy = i - r;
        let dx = (c as f64 * (((r * r - dy * dy) as f64) * inv_r2).sqrt()).round() as isize;
        let j1 = (c - dx).max(0);
        let j2 = (c + dx + 1).min(k as isize);
        rows.push((j1 - c, j2 - 1 - c));
    }
    (r, rows)
}

/// Sliding-window extremum over `row` for offsets `lo..=hi`; samples outside are ignored.
fn window_extreme(row: &[u8], lo: isize, hi: isize, max: bool, out: &mut [u8]) {
    let n = row.len() as isize;
    let better = |a: u8, b: u8| if max { a >= b } else { a <= b };
    let mut dq: VecDeque<usize> = VecDeque::new();
    let mut next: isize = 0; // next sample index to push
    for x in 0..n {
        let right = (x + hi).min(n - 1);
        while next <= right {
            while let Some(&back) = dq.back() {
                if better(row[next as usize], row[back]) {
                    dq.pop_back();
                } else {
                    break;
                }
            }
            dq.push_back(next as usize);
            next += 1;
        }
        let left = x + lo;
        while let Some(&front) = dq.front() {
            if (front as isize) < left {
                dq.pop_front();
            } else {
                break;
            }
        }
        out[x as usize] = row[*dq.front().expect("window is never empty")];
    }
}

fn erode_dilate(src: &Mask, k: usize, max: bool) -> Mask {
    if k <= 1 {
        return src.clone();
    }
    let (w, h) = (src.w, src.h);
    let (r, rows) = ellipse_rows(k);
    let mut uniq: Vec<(isize, isize)> = rows.clone();
    uniq.sort_unstable();
    uniq.dedup();
    let mut out = vec![0u8; w * h];
    const BAND: usize = 64;
    out.par_chunks_mut(w * BAND).enumerate().for_each(|(bi, band)| {
        let y0 = bi * BAND;
        let y1 = (y0 + BAND).min(h);
        let src_y0 = y0.saturating_sub(r as usize);
        let src_y1 = (y1 + r as usize).min(h);
        let nrows = src_y1 - src_y0;
        // Horizontal extremum planes, one per distinct window, for the rows this band reads.
        let mut planes: Vec<Vec<u8>> = Vec::with_capacity(uniq.len());
        for &(lo, hi) in &uniq {
            let mut p = vec![0u8; nrows * w];
            for (ry, prow) in p.chunks_mut(w).enumerate() {
                let y = src_y0 + ry;
                window_extreme(&src.data[y * w..(y + 1) * w], lo, hi, max, prow);
            }
            planes.push(p);
        }
        for y in y0..y1 {
            let orow = &mut band[(y - y0) * w..(y - y0 + 1) * w];
            orow.fill(if max { 0 } else { 255 });
            for (i, &(lo, hi)) in rows.iter().enumerate() {
                let yy = y as isize + i as isize - r;
                if yy < 0 || yy >= h as isize {
                    continue;
                }
                let pi = uniq.binary_search(&(lo, hi)).expect("window present");
                let prow = &planes[pi][(yy as usize - src_y0) * w..(yy as usize - src_y0 + 1) * w];
                if max {
                    orow.iter_mut().zip(prow).for_each(|(o, p)| *o = (*o).max(*p));
                } else {
                    orow.iter_mut().zip(prow).for_each(|(o, p)| *o = (*o).min(*p));
                }
            }
        }
    });
    Mask { w, h, data: out }
}

/// Apply `op` with a `k` x `k` elliptical kernel.
pub fn morph_ellipse(mask: &Mask, k: usize, op: MorphOp) -> Mask {
    match op {
        MorphOp::Erode => erode_dilate(mask, k, false),
        MorphOp::Dilate => erode_dilate(mask, k, true),
        MorphOp::Open => erode_dilate(&erode_dilate(mask, k, false), k, true),
        MorphOp::Close => erode_dilate(&erode_dilate(mask, k, true), k, false),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dilating_a_point_gives_an_ellipse() {
        let mut m = Mask::new(21, 21);
        m.data[10 * 21 + 10] = 255;
        let o = morph_ellipse(&m, 7, MorphOp::Dilate);
        assert_eq!(o.get(10, 10), 255);
        assert_eq!(o.get(13, 10), 255);
        assert_eq!(o.get(7, 10), 255);
        assert_eq!(o.get(14, 10), 0);
        assert_eq!(o.get(7, 7), 0); // corner of the 7x7 square is outside the ellipse
    }

    #[test]
    fn closing_after_dilation_roundtrips_a_blob() {
        let mut m = Mask::new(40, 40);
        for y in 10..30 {
            for x in 10..30 {
                m.data[y * 40 + x] = 255;
            }
        }
        let o = morph_ellipse(&morph_ellipse(&m, 5, MorphOp::Dilate), 5, MorphOp::Erode);
        assert_eq!(o, m);
    }

    #[test]
    fn matches_naive_implementation() {
        let (w, h) = (23, 19);
        let mut s = 7u32;
        let data: Vec<u8> = (0..w * h)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                (s >> 24) as u8
            })
            .collect();
        let m = Mask { w, h, data };
        for k in [2usize, 3, 4, 6, 9] {
            let (r, rows) = ellipse_rows(k);
            for (op, max) in [(MorphOp::Dilate, true), (MorphOp::Erode, false)] {
                let got = morph_ellipse(&m, k, op);
                for y in 0..h as isize {
                    for x in 0..w as isize {
                        let mut best: Option<u8> = None;
                        for (i, &(lo, hi)) in rows.iter().enumerate() {
                            let yy = y + i as isize - r;
                            if yy < 0 || yy >= h as isize {
                                continue;
                            }
                            for dx in lo..=hi {
                                let xx = x + dx;
                                if xx < 0 || xx >= w as isize {
                                    continue;
                                }
                                let v = m.data[yy as usize * w + xx as usize];
                                best = Some(match best {
                                    None => v,
                                    Some(b) => {
                                        if max {
                                            b.max(v)
                                        } else {
                                            b.min(v)
                                        }
                                    }
                                });
                            }
                        }
                        assert_eq!(got.data[y as usize * w + x as usize], best.unwrap(), "k={k} {op:?} at {x},{y}");
                    }
                }
            }
        }
    }

    #[test]
    fn open_removes_specks_close_fills_holes() {
        let mut m = Mask::new(30, 30);
        for y in 5..25 {
            for x in 5..25 {
                m.data[y * 30 + x] = 255;
            }
        }
        m.data[2 * 30 + 2] = 255; // speck
        m.data[15 * 30 + 15] = 0; // hole
        assert_eq!(morph_ellipse(&m, 3, MorphOp::Open).get(2, 2), 0);
        assert_eq!(morph_ellipse(&m, 3, MorphOp::Close).get(15, 15), 255);
    }
}
