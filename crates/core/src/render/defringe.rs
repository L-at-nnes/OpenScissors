//! Edge colour clean-up ("defringe").
//!
//! On a soft edge (hair, a feathered or grown outline) the photo still holds a mix of the subject and of the
//! background that was behind it: a white halo on a dark subject once it is placed on a dark background. The
//! pixels that are fully part of the subject are not affected, so their colour is the truth: every
//! semi-transparent pixel takes the average colour of the nearest solid pixels instead of its own.

use rayon::prelude::*;

use crate::types::{Image, Mask};

/// Alpha from which a pixel is certainly the subject (its colour is trusted and never changed).
const SOLID: u8 = 250;

/// Search distances, nearest first: the closest solid pixels give the most faithful colour.
fn rings(radius: usize) -> Vec<usize> {
    let mut v: Vec<usize> = [1, 2, 3, 4, 6, 8, 12, 16, 24].into_iter().filter(|&r| r < radius).collect();
    v.push(radius);
    v
}

/// For every pixel, how far the nearest solid pixel is, counted in square rings (Chebyshev distance), up to `cap`:
/// two passes over the picture. A ring smaller than that distance holds no solid pixel, so it need not be searched.
fn solid_distance(alpha: &Mask, cap: usize) -> Vec<u16> {
    let (w, h) = (alpha.w, alpha.h);
    let cap = cap.min(u16::MAX as usize - 1) as u16;
    let mut d: Vec<u16> = alpha.data.iter().map(|&a| if a >= SOLID { 0 } else { cap }).collect();
    let near = |d: &[u16], x: usize, y: usize, dx: isize, dy: isize| -> u16 {
        let (nx, ny) = (x as isize + dx, y as isize + dy);
        if nx < 0 || ny < 0 || nx >= w as isize || ny >= h as isize {
            cap
        } else {
            d[ny as usize * w + nx as usize].saturating_add(1).min(cap)
        }
    };
    for y in 0..h {
        for x in 0..w {
            let v = [(-1, -1), (0, -1), (1, -1), (-1, 0)].iter().map(|&(dx, dy)| near(&d, x, y, dx, dy)).min().unwrap_or(cap);
            d[y * w + x] = d[y * w + x].min(v);
        }
    }
    for y in (0..h).rev() {
        for x in (0..w).rev() {
            let v = [(1, 1), (0, 1), (-1, 1), (1, 0)].iter().map(|&(dx, dy)| near(&d, x, y, dx, dy)).min().unwrap_or(cap);
            d[y * w + x] = d[y * w + x].min(v);
        }
    }
    d
}

/// Replace the colour of the semi-transparent pixels of `img` (same size as `alpha`) by the average colour of the
/// solid pixels around them, looking no further than `radius` pixels.
pub fn defringe(img: &mut Image, alpha: &Mask, radius: usize) {
    if radius == 0 || (img.w, img.h) != (alpha.w, alpha.h) {
        return;
    }
    let (w, h) = (img.w, img.h);
    let src = img.data.clone();
    let rings = rings(radius);
    // A pixel with no solid pixel within `radius` keeps its colour, and the rings closer than the nearest solid pixel
    // are empty: both are known from the distances without searching (a wide feathered edge was the slow case).
    let dist = solid_distance(alpha, radius + 1);
    img.data.par_chunks_mut(w * 3).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            let a = alpha.data[y * w + x];
            if a == 0 || a >= SOLID {
                continue;
            }
            let nearest = dist[y * w + x] as usize;
            if nearest > radius {
                continue;
            }
            for &r in rings.iter().filter(|&&r| r >= nearest) {
                let (x0, x1) = (x.saturating_sub(r), (x + r).min(w - 1));
                let (y0, y1) = (y.saturating_sub(r), (y + r).min(h - 1));
                let (mut n, mut sum) = (0u32, [0u32; 3]);
                for yy in y0..=y1 {
                    let arow = &alpha.data[yy * w..(yy + 1) * w];
                    for xx in x0..=x1 {
                        if arow[xx] >= SOLID {
                            let p = (yy * w + xx) * 3;
                            n += 1;
                            sum[0] += src[p] as u32;
                            sum[1] += src[p + 1] as u32;
                            sum[2] += src[p + 2] as u32;
                        }
                    }
                }
                // A few samples make a stable average; the last, widest search takes whatever it finds.
                if n >= 3 || (n > 0 && r == radius) {
                    for c in 0..3 {
                        row[x * 3 + c] = ((sum[c] + n / 2) / n) as u8;
                    }
                    break;
                }
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Subject (red) on the left, a one-pixel soft edge whose colour is half red, half blue background.
    fn scene() -> (Image, Mask) {
        let (w, h) = (12usize, 6usize);
        let mut img = Image::filled(w, h, [200, 0, 0]);
        let mut alpha = Mask::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let (a, c) = match x {
                    0..=5 => (255, [200, 0, 0]),
                    6 => (128, [100, 0, 100]),
                    _ => (0, [0, 0, 200]),
                };
                alpha.data[y * w + x] = a;
                img.data[(y * w + x) * 3..(y * w + x) * 3 + 3].copy_from_slice(&c);
            }
        }
        (img, alpha)
    }

    /// The search as it was before the distances: every ring, from the smallest.
    fn reference(img: &mut Image, alpha: &Mask, radius: usize) {
        let (w, h) = (img.w, img.h);
        let src = img.data.clone();
        for y in 0..h {
            for x in 0..w {
                let a = alpha.data[y * w + x];
                if a == 0 || a >= SOLID {
                    continue;
                }
                for &r in &rings(radius) {
                    let (mut n, mut sum) = (0u32, [0u32; 3]);
                    for yy in y.saturating_sub(r)..=(y + r).min(h - 1) {
                        for xx in x.saturating_sub(r)..=(x + r).min(w - 1) {
                            if alpha.data[yy * w + xx] >= SOLID {
                                n += 1;
                                for c in 0..3 {
                                    sum[c] += src[(yy * w + xx) * 3 + c] as u32;
                                }
                            }
                        }
                    }
                    if n >= 3 || (n > 0 && r == radius) {
                        for c in 0..3 {
                            img.data[(y * w + x) * 3 + c] = ((sum[c] + n / 2) / n) as u8;
                        }
                        break;
                    }
                }
            }
        }
    }

    #[test]
    fn skipping_rings_by_distance_changes_nothing() {
        let (w, h) = (61, 47);
        let mut seed = 99u32;
        let mut next = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 24) as u8
        };
        let mut img = Image::new(w, h);
        img.data.iter_mut().for_each(|v| *v = next());
        // A solid blob, a wide soft band around it, scattered solid specks and soft noise.
        let mut alpha = Mask::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let d = ((x as f64 - 20.0).powi(2) + (y as f64 - 22.0).powi(2)).sqrt();
                alpha.data[y * w + x] = if d < 8.0 {
                    255
                } else if d < 30.0 {
                    (255.0 * (30.0 - d) / 22.0) as u8
                } else {
                    0
                };
            }
        }
        for _ in 0..40 {
            let i = (next() as usize * 37 + next() as usize) % (w * h);
            alpha.data[i] = if next() % 2 == 0 { 255 } else { next() };
        }
        for radius in [1, 2, 5, 9, 20] {
            let (mut fast, mut slow) = (img.clone(), img.clone());
            defringe(&mut fast, &alpha, radius);
            reference(&mut slow, &alpha, radius);
            assert!(fast == slow, "radius {radius}");
        }
    }

    #[test]
    fn a_soft_edge_takes_the_colour_of_the_subject() {
        let (mut img, alpha) = scene();
        defringe(&mut img, &alpha, 3);
        let edge = 6 * 3;
        assert_eq!(&img.data[edge..edge + 3], &[200, 0, 0], "the blue of the old background is gone");
    }

    #[test]
    fn solid_and_invisible_pixels_are_left_alone() {
        let (mut img, alpha) = scene();
        let before = img.data.clone();
        defringe(&mut img, &alpha, 3);
        for y in 0..img.h {
            for x in [0, 3, 5, 7, 11] {
                let i = (y * img.w + x) * 3;
                assert_eq!(&img.data[i..i + 3], &before[i..i + 3], "pixel {x},{y}");
            }
        }
    }

    #[test]
    fn nothing_changes_when_no_solid_pixel_is_within_reach_or_the_radius_is_zero() {
        let (img, mut alpha) = scene();
        for v in &mut alpha.data {
            *v = (*v).min(200); // no pixel is solid any more
        }
        let mut a = img.clone();
        defringe(&mut a, &alpha, 5);
        assert_eq!(a.data, img.data);
        let (mut b, alpha) = scene();
        defringe(&mut b, &alpha, 0);
        assert_eq!(b.data, img.data);
    }

    #[test]
    fn the_nearest_solid_colour_wins_over_a_farther_one() {
        let (w, h) = (9usize, 3usize);
        let mut img = Image::filled(w, h, [0, 0, 0]);
        let mut alpha = Mask::new(w, h);
        // Left subject red, right subject green, a soft pixel at x = 2, much nearer to the red one.
        for y in 0..h {
            for x in 0..w {
                let (a, c) = match x {
                    0..=1 => (255, [255, 0, 0]),
                    2 => (100, [90, 90, 90]),
                    8 => (255, [0, 255, 0]),
                    _ => (0, [0, 0, 0]),
                };
                alpha.data[y * w + x] = a;
                img.data[(y * w + x) * 3..(y * w + x) * 3 + 3].copy_from_slice(&c);
            }
        }
        defringe(&mut img, &alpha, 8);
        let i = (w + 2) * 3; // row 1, x = 2
        assert_eq!(&img.data[i..i + 3], &[255, 0, 0]);
    }
}
