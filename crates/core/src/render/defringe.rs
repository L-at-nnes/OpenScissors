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

/// Replace the colour of the semi-transparent pixels of `img` (same size as `alpha`) by the average colour of the
/// solid pixels around them, looking no further than `radius` pixels.
pub fn defringe(img: &mut Image, alpha: &Mask, radius: usize) {
    if radius == 0 || (img.w, img.h) != (alpha.w, alpha.h) {
        return;
    }
    let (w, h) = (img.w, img.h);
    let src = img.data.clone();
    let rings = rings(radius);
    img.data.par_chunks_mut(w * 3).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            let a = alpha.data[y * w + x];
            if a == 0 || a >= SOLID {
                continue;
            }
            for &r in &rings {
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
