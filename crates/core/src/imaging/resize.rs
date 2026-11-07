//! Resampling built on `fast_image_resize` (SIMD, multi-threaded).

use fast_image_resize::images::{Image as FirImage, ImageRef};
use fast_image_resize::{FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer};

use crate::types::{Image, Mask, Rect, Rgba};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Interp {
    Nearest,
    Bilinear,
    Bicubic,
    /// Box filter: area average, the right choice for downscaling.
    Area,
}

fn alg(i: Interp) -> ResizeAlg {
    match i {
        Interp::Nearest => ResizeAlg::Nearest,
        Interp::Bilinear => ResizeAlg::Convolution(FilterType::Bilinear),
        Interp::Bicubic => ResizeAlg::Convolution(FilterType::CatmullRom),
        Interp::Area => ResizeAlg::Convolution(FilterType::Box),
    }
}

/// Resize `src` (`sw` x `sh`, `ch` interleaved channels), optionally reading only `crop`.
/// Output size is `dw` x `dh`. Out-of-image crop areas are clamped to the image.
pub fn resize_raw(src: &[u8], sw: usize, sh: usize, ch: usize, crop: Option<Rect>, dw: usize, dh: usize, interp: Interp) -> Vec<u8> {
    let pixel = match ch {
        1 => PixelType::U8,
        3 => PixelType::U8x3,
        4 => PixelType::U8x4,
        _ => panic!("unsupported channel count {ch}"),
    };
    if dw == 0 || dh == 0 {
        return Vec::new();
    }
    let view = ImageRef::new(sw as u32, sh as u32, src, pixel).expect("source view");
    let mut dst = FirImage::new(dw as u32, dh as u32, pixel);
    let mut opts = ResizeOptions::new().resize_alg(alg(interp));
    if let Some(c) = crop {
        let x0 = c.x0.clamp(0.0, sw as f64);
        let y0 = c.y0.clamp(0.0, sh as f64);
        let x1 = c.x1.clamp(0.0, sw as f64);
        let y1 = c.y1.clamp(0.0, sh as f64);
        if x1 - x0 > 0.0 && y1 - y0 > 0.0 {
            opts = opts.crop(x0, y0, x1 - x0, y1 - y0);
        }
    }
    Resizer::new().resize(&view, &mut dst, &opts).expect("resize");
    dst.into_vec()
}

/// Most pixels a reduced working copy may have, whatever its shape (a panorama would otherwise keep millions).
pub const WORKING_PIXELS_CAP: usize = 1_000_000;

/// Size at which a heavy whole-picture computation (graph cut, matting) runs on a `w` x `h` picture: `None` when the
/// picture has at most `threshold` pixels, or when reducing would not make it smaller; otherwise the short side is
/// brought to `small_side` and the total kept under [`WORKING_PIXELS_CAP`]. A picture is never enlarged.
pub fn working_size(w: usize, h: usize, threshold: usize, small_side: usize) -> Option<(usize, usize)> {
    if w * h <= threshold || w == 0 || h == 0 {
        return None;
    }
    let s = (small_side as f64 / w.min(h) as f64).min((WORKING_PIXELS_CAP as f64 / (w * h) as f64).sqrt());
    if s >= 1.0 {
        return None;
    }
    let size = |f: fn(f64) -> f64| ((f(w as f64 * s) as usize).max(3), (f(h as f64 * s) as usize).max(3));
    let (sw, sh) = size(f64::round);
    // Rounding up both sides can pass the cap by a few pixels: then round down.
    Some(if sw * sh > WORKING_PIXELS_CAP { size(f64::floor) } else { (sw, sh) })
}

pub fn resize_image(src: &Image, dw: usize, dh: usize, interp: Interp) -> Image {
    Image { w: dw, h: dh, data: resize_raw(&src.data, src.w, src.h, 3, None, dw, dh, interp) }
}

pub fn resize_mask(src: &Mask, dw: usize, dh: usize, interp: Interp) -> Mask {
    Mask { w: dw, h: dh, data: resize_raw(&src.data, src.w, src.h, 1, None, dw, dh, interp) }
}

/// Resize straight-alpha RGBA. The resizer weights the colours by their alpha while it averages them (its `use_alpha`
/// option, on by default), so the colour hidden under a transparent pixel (often black) never leaks into the edge of
/// the subject. It works on 16 bits: on 8 bits a nearly transparent pixel (a soft edge, a hair) kept only a few levels
/// of its colour and came back wrong. (The colours used to be weighted here as well, which counted the alpha twice.)
pub fn resize_rgba(src: &Rgba, dw: usize, dh: usize, interp: Interp) -> Rgba {
    if dw == 0 || dh == 0 {
        return Rgba { w: dw, h: dh, data: Vec::new() };
    }
    let wide: Vec<u8> = src.data.iter().flat_map(|&v| (v as u16 * 257).to_ne_bytes()).collect();
    let view = ImageRef::new(src.w as u32, src.h as u32, &wide, PixelType::U16x4).expect("source view");
    let mut dst = FirImage::new(dw as u32, dh as u32, PixelType::U16x4);
    Resizer::new().resize(&view, &mut dst, &ResizeOptions::new().resize_alg(alg(interp))).expect("resize");
    let data = dst.into_vec().chunks_exact(2).map(|b| ((u16::from_ne_bytes([b[0], b[1]]) as u32 + 128) / 257) as u8).collect();
    Rgba { w: dw, h: dh, data }
}

pub fn crop_resize_image(src: &Image, crop: Rect, dw: usize, dh: usize, interp: Interp) -> Image {
    Image { w: dw, h: dh, data: resize_raw(&src.data, src.w, src.h, 3, Some(crop), dw, dh, interp) }
}

pub fn crop_resize_mask(src: &Mask, crop: Rect, dw: usize, dh: usize, interp: Interp) -> Mask {
    Mask { w: dw, h: dh, data: resize_raw(&src.data, src.w, src.h, 1, Some(crop), dw, dh, interp) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_image_stays_constant() {
        let m = Mask::filled(10, 10, 77);
        for i in [Interp::Nearest, Interp::Bilinear, Interp::Bicubic, Interp::Area] {
            let r = resize_mask(&m, 23, 7, i);
            assert!(r.data.iter().all(|&v| v == 77), "{i:?}");
        }
    }

    #[test]
    fn crop_selects_the_requested_region() {
        let mut m = Mask::new(8, 8);
        for y in 0..8 {
            for x in 4..8 {
                m.data[y * 8 + x] = 200;
            }
        }
        let r = crop_resize_mask(&m, Rect::new(4.0, 0.0, 8.0, 8.0), 4, 4, Interp::Area);
        assert!(r.data.iter().all(|&v| v == 200));
        let l = crop_resize_mask(&m, Rect::new(0.0, 0.0, 4.0, 8.0), 4, 4, Interp::Area);
        assert!(l.data.iter().all(|&v| v == 0));
    }

    #[test]
    fn a_nearly_transparent_pixel_keeps_its_colour() {
        // Alpha 3: weighting the colour by its alpha in 8 bits left 2 / 255 of it, and dividing back gave 170.
        let img = Rgba { w: 4, h: 4, data: [200, 100, 50, 3].repeat(16) };
        for interp in [Interp::Area, Interp::Bilinear, Interp::Bicubic] {
            let r = resize_rgba(&img, 7, 3, interp);
            for p in r.data.chunks_exact(4) {
                assert_eq!(p[3], 3, "{interp:?}");
                assert!(p[0].abs_diff(200) <= 2 && p[1].abs_diff(100) <= 2 && p[2].abs_diff(50) <= 2, "{interp:?}: {p:?}");
            }
        }
    }

    #[test]
    fn a_transparent_neighbour_does_not_darken_the_edge() {
        // Left half opaque red, right half fully transparent with black underneath.
        let mut img = Rgba::new(4, 2);
        for y in 0..2 {
            for x in 0..4 {
                let p = (y * 4 + x) * 4;
                img.data[p..p + 4].copy_from_slice(if x < 2 { &[200, 0, 0, 255] } else { &[0, 0, 0, 0] });
            }
        }
        let r = resize_rgba(&img, 3, 1, Interp::Area);
        let middle = &r.data[4..8];
        assert!(middle[3] > 0 && middle[3] < 255, "partly transparent: {middle:?}");
        assert!(middle[0] >= 198, "the colour stays the red of the subject: {middle:?}");
    }

    #[test]
    fn the_working_size_never_enlarges_and_bounds_a_panorama() {
        assert_eq!(working_size(500, 400, 360_000, 600), None, "small enough as it is");
        assert_eq!(working_size(4000, 3000, 360_000, 600), Some((800, 600)), "a photo: short side to 600");
        // A short side under 600 used to be enlarged (1200 x 350 became 2057 x 600).
        assert_eq!(working_size(1200, 350, 360_000, 600), None);
        let (w, h) = working_size(20_000, 1000, 360_000, 600).unwrap();
        assert!(w * h <= WORKING_PIXELS_CAP && (w as f64 / h as f64 - 20.0).abs() < 0.2, "{w} x {h}");
        for (w, h) in [(700, 600), (3000, 200), (100_000, 10), (6000, 3375)] {
            if let Some((sw, sh)) = working_size(w, h, 360_000, 600) {
                assert!(sw <= w && sh <= h && sw * sh <= WORKING_PIXELS_CAP, "{w} x {h} -> {sw} x {sh}");
            }
        }
    }

    #[test]
    fn area_downscale_averages() {
        let mut m = Mask::new(4, 2);
        m.data = vec![0, 100, 0, 100, 0, 100, 0, 100];
        let r = resize_mask(&m, 2, 1, Interp::Area);
        assert_eq!(r.data, vec![50, 50]);
    }
}
