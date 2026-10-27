//! Comparing the masks that several models give for one picture: how much of it each keeps, how much two of them agree,
//! and a contact sheet to look at them side by side.

use crate::types::{Image, Mask, Rgba};

/// Mask values above this count as foreground (the threshold the rest of the pipeline uses).
const FOREGROUND: u8 = 127;

/// Share of the picture a mask keeps, 0 to 1.
pub fn coverage(m: &Mask) -> f64 {
    if m.data.is_empty() {
        return 0.0;
    }
    m.data.iter().filter(|&&v| v > FOREGROUND).count() as f64 / m.data.len() as f64
}

/// Intersection over union of the foreground of two masks of the same size: 1 when they agree on every pixel, 0 when
/// they share none. Two empty masks agree.
pub fn iou(a: &Mask, b: &Mask) -> f64 {
    assert_eq!((a.w, a.h), (b.w, b.h), "masks of different sizes cannot be compared");
    let (mut both, mut either) = (0usize, 0usize);
    for (&x, &y) in a.data.iter().zip(&b.data) {
        let (x, y) = (x > FOREGROUND, y > FOREGROUND);
        both += (x && y) as usize;
        either += (x || y) as usize;
    }
    if either == 0 {
        1.0
    } else {
        both as f64 / either as f64
    }
}

/// Height of a tile of the contact sheet.
const TILE_HEIGHT: usize = 480;
/// Space between the tiles, in pixels.
const GAP: usize = 8;
const CHECKER: [[u8; 3]; 2] = [[154, 154, 156], [124, 124, 128]];

/// One row: the picture itself, then its cut-out by each of `results` (in that order) over a grey checker. Every tile is
/// at most [`TILE_HEIGHT`] pixels high, so a big photo gives a sheet that is easy to open.
pub fn contact_sheet(image: &Image, results: &[(String, Mask)]) -> Rgba {
    let scale = (TILE_HEIGHT as f64 / image.h as f64).min(1.0);
    let (tw, th) = (((image.w as f64 * scale).round() as usize).max(1), ((image.h as f64 * scale).round() as usize).max(1));
    let tiles = 1 + results.len();
    let width = tiles * tw + (tiles - 1) * GAP;
    let mut out = Rgba::new(width, th);
    for y in 0..th {
        let sy = (y * image.h / th).min(image.h - 1);
        for x in 0..tw {
            let sx = (x * image.w / tw).min(image.w - 1);
            let rgb = &image.data[(sy * image.w + sx) * 3..(sy * image.w + sx) * 3 + 3];
            for tile in 0..tiles {
                let o = (y * width + tile * (tw + GAP) + x) * 4;
                if tile == 0 {
                    out.data[o..o + 3].copy_from_slice(rgb);
                } else {
                    let a = results[tile - 1].1.data[sy * image.w + sx] as u32;
                    let bg = CHECKER[(x / 12 + y / 12) & 1];
                    for c in 0..3 {
                        out.data[o + c] = ((rgb[c] as u32 * a + bg[c] as u32 * (255 - a) + 127) / 255) as u8;
                    }
                }
                out.data[o + 3] = 255;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn half(w: usize, h: usize, from: usize, to: usize) -> Mask {
        let mut m = Mask::new(w, h);
        for y in 0..h {
            for x in from..to {
                m.data[y * w + x] = 255;
            }
        }
        m
    }

    #[test]
    fn iou_is_one_for_equal_masks_and_zero_for_disjoint_ones() {
        let (a, b, c) = (half(10, 4, 0, 5), half(10, 4, 0, 5), half(10, 4, 5, 10));
        assert_eq!(iou(&a, &b), 1.0);
        assert_eq!(iou(&a, &c), 0.0);
        // Overlap of 2 columns out of 8 covered by either.
        assert!((iou(&half(10, 4, 0, 5), &half(10, 4, 3, 8)) - 2.0 / 8.0).abs() < 1e-12);
        assert_eq!(iou(&Mask::new(3, 3), &Mask::new(3, 3)), 1.0, "two empty masks agree");
    }

    #[test]
    fn coverage_counts_the_foreground() {
        assert_eq!(coverage(&half(10, 4, 0, 3)), 0.3);
        assert_eq!(coverage(&Mask::new(0, 0)), 0.0);
        let mut soft = Mask::new(2, 1);
        soft.data = vec![100, 200];
        assert_eq!(coverage(&soft), 0.5, "only values above 127 count");
    }

    #[test]
    fn the_contact_sheet_lines_up_the_picture_and_each_cut_out() {
        let img = Image::filled(40, 20, [200, 30, 30]);
        let sheet = contact_sheet(&img, &[("a".into(), half(40, 20, 0, 20)), ("b".into(), Mask::new(40, 20))]);
        assert_eq!((sheet.w, sheet.h), (3 * 40 + 2 * GAP, 20));
        let px = |x: usize, y: usize| sheet.data[(y * sheet.w + x) * 4..][..4].to_vec();
        assert_eq!(px(5, 5), [200, 30, 30, 255], "the picture");
        assert_eq!(px(40 + GAP + 5, 5), [200, 30, 30, 255], "kept by the first model");
        let removed = px(40 + GAP + 30, 5);
        assert!(removed[0] < 160 && removed[1] > 100, "removed by the first model: the checker shows {removed:?}");
        let none = px(2 * (40 + GAP) + 5, 5);
        assert!(none[0] < 160 && none[1] > 100, "the second model keeps nothing: {none:?}");
    }

    #[test]
    fn a_tall_picture_gets_tiles_of_a_bounded_height() {
        let img = Image::filled(1000, 2000, [1, 2, 3]);
        let sheet = contact_sheet(&img, &[("a".into(), Mask::filled(1000, 2000, 255))]);
        assert_eq!(sheet.h, TILE_HEIGHT);
        assert_eq!(sheet.w, 2 * 240 + GAP);
    }
}
