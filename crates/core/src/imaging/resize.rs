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

pub fn resize_image(src: &Image, dw: usize, dh: usize, interp: Interp) -> Image {
    Image { w: dw, h: dh, data: resize_raw(&src.data, src.w, src.h, 3, None, dw, dh, interp) }
}

pub fn resize_mask(src: &Mask, dw: usize, dh: usize, interp: Interp) -> Mask {
    Mask { w: dw, h: dh, data: resize_raw(&src.data, src.w, src.h, 1, None, dw, dh, interp) }
}

/// Resize straight-alpha RGBA. The colours are weighted by their alpha while they are averaged, so the colour hidden
/// under a transparent pixel (often black) never leaks into the edge of the subject.
pub fn resize_rgba(src: &Rgba, dw: usize, dh: usize, interp: Interp) -> Rgba {
    let mut data = src.data.clone();
    for p in data.chunks_exact_mut(4) {
        let a = p[3] as u32;
        for c in &mut p[..3] {
            *c = ((*c as u32 * a + 127) / 255) as u8;
        }
    }
    let mut out = resize_raw(&data, src.w, src.h, 4, None, dw, dh, interp);
    for p in out.chunks_exact_mut(4) {
        let a = p[3] as u32;
        for c in &mut p[..3] {
            *c = (*c as u32 * 255 + a / 2).checked_div(a).map_or(0, |v| v.min(255) as u8);
        }
    }
    Rgba { w: dw, h: dh, data: out }
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
    fn area_downscale_averages() {
        let mut m = Mask::new(4, 2);
        m.data = vec![0, 100, 0, 100, 0, 100, 0, 100];
        let r = resize_mask(&m, 2, 1, Interp::Area);
        assert_eq!(r.data, vec![50, 50]);
    }
}
