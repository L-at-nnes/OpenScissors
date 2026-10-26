//! Rendering: background, shadow, foreground, effects and crop.
//!
//! One code path serves the live preview (a viewport of the image at screen resolution) and
//! the export (the whole image at scale 1), so both always look the same.

pub mod crop;
pub mod defringe;
pub mod filters;
pub mod inpaint;

use rayon::prelude::*;

use crate::imaging::filter::{box_blur_u8, gaussian_blur_u8};
use crate::imaging::morph::{morph_ellipse, MorphOp};
use crate::imaging::resize::{crop_resize_image, crop_resize_mask, resize_raw, resize_rgba, Interp};
use crate::params::{BgMode, Params};
use crate::types::{Image, Image16, Mask, Rect, Rgba, Rgba16};
use filters::{adjust_colors, apply_filter, filter_margin};

#[derive(Clone, Copy)]
pub struct Scene<'a> {
    pub image: &'a Image,
    /// Composed mask before boundary offset and feathering.
    pub alpha: &'a Mask,
    pub params: &'a Params,
    /// Loaded background picture when `background.mode == Image`.
    pub bg_image: Option<&'a Image>,
    /// Original with the subject filled in, when `fill_hole` is on.
    pub filled: Option<&'a Image>,
    /// The canvas that is exported (the crop, in image coordinates, possibly past the picture). A background picture
    /// covers this frame, not the picture. `None` = the whole picture.
    pub frame: Option<Rect>,
}

/// Margin, in output pixels, that the preview gives a tile for the blurs, the boundary offset and the shadow to find
/// their context. It bounds the memory of a zoomed-in tile; the export does not use it.
const PREVIEW_MARGIN_CAP: usize = 512;

fn interp_for(scale: f64) -> Interp {
    if scale < 1.0 {
        Interp::Area
    } else if scale >= 3.0 {
        Interp::Nearest
    } else {
        Interp::Bilinear
    }
}

/// Boundary offset and feathering expressed in output pixels.
fn finish_alpha_scaled(mut a: Mask, offset: i32, feather: u32, scale: f64) -> Mask {
    if offset != 0 {
        let k = ((offset.unsigned_abs() + 2) as f64 * scale).round() as usize;
        if k >= 2 {
            a = morph_ellipse(&a, k, if offset > 0 { MorphOp::Dilate } else { MorphOp::Erode });
        }
    }
    let r = (feather as f64 * scale).round() as usize;
    if r > 0 {
        a.data = box_blur_u8(&a.data, a.w, a.h, 1, r);
    }
    a
}

/// `src` (`ch` interleaved channels, `sw` x `sh`) seen through `exp`, resampled to `dw` x `dh`. The part of `exp`
/// outside the picture is zero, or with `edge` the nearest edge pixel.
#[allow(clippy::too_many_arguments)]
fn sample(src: &[u8], sw: usize, sh: usize, ch: usize, exp: Rect, dw: usize, dh: usize, interp: Interp, edge: bool) -> Vec<u8> {
    let inter = Rect::new(exp.x0.max(0.0), exp.y0.max(0.0), exp.x1.min(sw as f64), exp.y1.min(sh as f64));
    if inter == exp {
        return resize_raw(src, sw, sh, ch, Some(exp), dw, dh, interp);
    }
    let mut out = vec![0u8; dw * dh * ch];
    let (kx, ky) = (dw as f64 / exp.width(), dh as f64 / exp.height());
    let cell = |v: f64, max: usize| (v.round().max(0.0) as usize).min(max);
    let (dx0, dx1) = (cell((inter.x0 - exp.x0) * kx, dw), cell((inter.x1 - exp.x0) * kx, dw));
    let (dy0, dy1) = (cell((inter.y0 - exp.y0) * ky, dh), cell((inter.y1 - exp.y0) * ky, dh));
    if inter.width() <= 0.0 || inter.height() <= 0.0 || dx1 <= dx0 || dy1 <= dy0 {
        if edge && sw > 0 && sh > 0 {
            // Nothing of the picture is in view: every pixel is the edge pixel nearest to the window.
            let px = ((exp.x0 + exp.x1) / 2.0).clamp(0.0, sw as f64 - 1.0) as usize;
            let py = ((exp.y0 + exp.y1) / 2.0).clamp(0.0, sh as f64 - 1.0) as usize;
            let pixel = &src[(py * sw + px) * ch..(py * sw + px + 1) * ch];
            out.chunks_exact_mut(ch).for_each(|o| o.copy_from_slice(pixel));
        }
        return out;
    }
    let part = resize_raw(src, sw, sh, ch, Some(inter), dx1 - dx0, dy1 - dy0, interp);
    for y in dy0..dy1 {
        let row = &part[(y - dy0) * (dx1 - dx0) * ch..(y - dy0 + 1) * (dx1 - dx0) * ch];
        out[(y * dw + dx0) * ch..(y * dw + dx1) * ch].copy_from_slice(row);
    }
    if edge {
        for y in dy0..dy1 {
            let (first, last) = (((y * dw + dx0) * ch), ((y * dw + dx1 - 1) * ch));
            let (left, right) = (out[first..first + ch].to_vec(), out[last..last + ch].to_vec());
            out[y * dw * ch..first].chunks_exact_mut(ch).for_each(|o| o.copy_from_slice(&left));
            out[last + ch..(y + 1) * dw * ch].chunks_exact_mut(ch).for_each(|o| o.copy_from_slice(&right));
        }
        let (top, bottom) = (out[dy0 * dw * ch..(dy0 + 1) * dw * ch].to_vec(), out[(dy1 - 1) * dw * ch..dy1 * dw * ch].to_vec());
        out[..dy0 * dw * ch].chunks_exact_mut(dw * ch).for_each(|o| o.copy_from_slice(&top));
        out[dy1 * dw * ch..].chunks_exact_mut(dw * ch).for_each(|o| o.copy_from_slice(&bottom));
    }
    out
}

fn sample_image(src: &Image, exp: Rect, dw: usize, dh: usize, interp: Interp, edge: bool) -> Image {
    Image { w: dw, h: dh, data: sample(&src.data, src.w, src.h, 3, exp, dw, dh, interp, edge) }
}

fn sample_mask(src: &Mask, exp: Rect, dw: usize, dh: usize, interp: Interp) -> Mask {
    Mask { w: dw, h: dh, data: sample(&src.data, src.w, src.h, 1, exp, dw, dh, interp, false) }
}

/// The gradient of `bg` over `frame` (the canvas that is exported), seen through `exp` at `scale`: the start colour at
/// one end of the frame along the gradient's direction, the end colour at the other.
fn gradient(bg: &crate::params::BackgroundParams, frame: Rect, exp: Rect, scale: f64, ew: usize, eh: usize) -> Image {
    let angle = (bg.gradient_angle as f64).to_radians();
    let (dx, dy) = (angle.sin(), -angle.cos());
    let (cx, cy) = ((frame.x0 + frame.x1) / 2.0, (frame.y0 + frame.y1) / 2.0);
    // Half the extent of the frame along the direction: the corners are exactly at the two ends.
    let half = (frame.width() / 2.0 * dx.abs() + frame.height() / 2.0 * dy.abs()).max(1e-9);
    let mut out = Image::new(ew, eh);
    out.data.par_chunks_mut(ew * 3).enumerate().for_each(|(y, row)| {
        let py = exp.y0 + (y as f64 + 0.5) / scale;
        for x in 0..ew {
            let px = exp.x0 + (x as f64 + 0.5) / scale;
            let t = (((px - cx) * dx + (py - cy) * dy) / (2.0 * half) + 0.5).clamp(0.0, 1.0);
            for c in 0..3 {
                row[x * 3 + c] = (bg.color[c] as f64 * (1.0 - t) + bg.gradient_color[c] as f64 * t).round() as u8;
            }
        }
    });
    out
}

/// Render `region` (image coordinates) into an `out_w` x `out_h` RGBA buffer with straight alpha. Transparent areas
/// keep alpha 0. The region may reach past the picture (a canvas extended by the crop): what is there is the
/// background, with the shadow of the subject if it spills over.
pub fn render_region(scene: &Scene, region: Rect, out_w: usize, out_h: usize) -> Rgba {
    render_region_with(scene, region, out_w, out_h, PREVIEW_MARGIN_CAP)
}

fn render_region_with(scene: &Scene, region: Rect, out_w: usize, out_h: usize, margin_cap: usize) -> Rgba {
    let p = scene.params;
    let (iw, ih) = (scene.image.w as f64, scene.image.h as f64);
    let scale = out_w as f64 / region.width();

    // Border needed around the window so that blurs, offset and shadow have their context.
    let shadow_sigma = p.shadow.blur_radius as f64 * scale / 2.0;
    let margin = (if p.mask.offset != 0 { ((p.mask.offset.unsigned_abs() + 2) as f64 * scale / 2.0).ceil() as usize + 1 } else { 0 })
        + (p.foreground.feather as f64 * scale).ceil() as usize
        + (p.foreground.defringe as f64 * scale).ceil() as usize
        + if p.shadow.enabled {
            (3.0 * shadow_sigma + p.shadow.offset_x.abs().max(p.shadow.offset_y.abs()) as f64 * scale).ceil() as usize
        } else {
            0
        }
        + filter_margin(&p.foreground.filter, scale).max(filter_margin(&p.background.filter, scale));
    let m = margin.min(margin_cap);
    // Towards the inside of the picture the context stops at its edge; past the edge there is nothing to stop it.
    let before = |v: f64| if v < 0.0 { m } else { m.min((v * scale).floor() as usize) };
    let after = |v: f64, size: f64| if v > size { m } else { m.min(((size - v) * scale).floor() as usize) };
    let (pl, pt, pr, pb) = (before(region.x0), before(region.y0), after(region.x1, iw), after(region.y1, ih));
    let (ew, eh) = (out_w + pl + pr, out_h + pt + pb);
    let exp = Rect::new(
        region.x0 - pl as f64 / scale,
        region.y0 - pt as f64 / scale,
        region.x1 + pr as f64 / scale,
        region.y1 + pb as f64 / scale,
    );
    let origin = (-exp.x0 * scale, -exp.y0 * scale);
    let interp = interp_for(scale);

    // Subject mask and colours.
    let a = finish_alpha_scaled(sample_mask(scene.alpha, exp, ew, eh, interp), p.mask.offset, p.foreground.feather, scale);
    let mut fg = sample_image(scene.image, exp, ew, eh, interp, false);
    defringe::defringe(&mut fg, &a, (p.foreground.defringe as f64 * scale).round() as usize);
    adjust_colors(&mut fg, p.foreground.brightness, p.foreground.saturation, p.foreground.warmth);
    apply_filter(&mut fg, &p.foreground.filter, scale, origin);

    // Background layer: colours plus whether it is opaque.
    let (bg, bg_opaque) = match p.background.mode {
        BgMode::Transparent => (Image::new(ew, eh), false),
        BgMode::Color => (Image::filled(ew, eh, p.background.color), true),
        BgMode::Gradient => {
            let frame = scene.frame.unwrap_or(Rect::new(0.0, 0.0, iw, ih));
            (gradient(&p.background, frame, exp, scale, ew, eh), true)
        }
        BgMode::Image => match scene.bg_image {
            Some(pic) => {
                let frame = scene.frame.unwrap_or(Rect::new(0.0, 0.0, iw, ih));
                let sc = (frame.width() / pic.w as f64).max(frame.height() / pic.h as f64);
                let ox = (frame.x0 + frame.x1 - pic.w as f64 * sc) / 2.0;
                let oy = (frame.y0 + frame.y1 - pic.h as f64 * sc) / 2.0;
                let r = Rect::new((exp.x0 - ox) / sc, (exp.y0 - oy) / sc, (exp.x1 - ox) / sc, (exp.y1 - oy) / sc);
                let mut b = sample_image(pic, r, ew, eh, interp_for(scale * sc), true);
                apply_filter(&mut b, &p.background.filter, scale, origin);
                (b, true)
            }
            None => (Image::new(ew, eh), false),
        },
        BgMode::Original => {
            let base = if p.background.fill_hole { scene.filled.unwrap_or(scene.image) } else { scene.image };
            let mut b = sample_image(base, exp, ew, eh, interp, true);
            apply_filter(&mut b, &p.background.filter, scale, origin);
            (b, true)
        }
    };

    // Shadow: blurred subject mask, shifted by the offset.
    let shadow: Option<Vec<u8>> = p.shadow.enabled.then(|| {
        let blurred = gaussian_blur_u8(&a.data, ew, eh, 1, shadow_sigma);
        let k = p.shadow.opacity.min(100);
        blurred.iter().map(|&v| ((v as u32 * k + 50) / 100) as u8).collect()
    });
    let sc = p.shadow.color;
    let shift = ((p.shadow.offset_x as f64 * scale).round() as isize, (p.shadow.offset_y as f64 * scale).round() as isize);

    // Composite: background, shadow, subject; only the requested window is written out.
    let mut out = vec![0u8; out_w * out_h * 4];
    out.par_chunks_mut(out_w * 4).enumerate().for_each(|(y, row)| {
        let ey = y + pt;
        for x in 0..out_w {
            let i = ey * ew + x + pl;
            let mut col = [bg.data[i * 3] as f32, bg.data[i * 3 + 1] as f32, bg.data[i * 3 + 2] as f32];
            let mut alpha = if bg_opaque { 1.0f32 } else { 0.0 };
            if let Some(s) = &shadow {
                // The shadow at this pixel is the blurred mask `shift` further back; nothing comes from outside the window.
                let (sx, sy) = ((x + pl) as isize - shift.0, ey as isize - shift.1);
                let inside = (0..ew as isize).contains(&sx) && (0..eh as isize).contains(&sy);
                let sa = if inside { s[sy as usize * ew + sx as usize] as f32 / 255.0 } else { 0.0 };
                let na = sa + alpha * (1.0 - sa);
                if na > 0.0 {
                    for c in 0..3 {
                        col[c] = (sc[c] as f32 * sa + col[c] * alpha * (1.0 - sa)) / na;
                    }
                }
                alpha = na;
            }
            let fa = a.data[i] as f32 / 255.0;
            let na = fa + alpha * (1.0 - fa);
            if na > 0.0 {
                for c in 0..3 {
                    col[c] = (fg.data[i * 3 + c] as f32 * fa + col[c] * alpha * (1.0 - fa)) / na;
                }
            }
            let o = &mut row[x * 4..x * 4 + 4];
            o[0] = col[0].round().clamp(0.0, 255.0) as u8;
            o[1] = col[1].round().clamp(0.0, 255.0) as u8;
            o[2] = col[2].round().clamp(0.0, 255.0) as u8;
            o[3] = (na * 255.0).round().clamp(0.0, 255.0) as u8;
        }
    });
    Rgba { w: out_w, h: out_h, data: out }
}

const OVERLAY_RGB: [f32; 3] = [61.0, 190.0, 74.0];
const REMOVE_RGB: [f32; 3] = [224.0, 72.0, 64.0];
const HAIR_RGB: [f32; 3] = [240.0, 200.0, 40.0];
const SELECTION_RGB: [f32; 3] = [60.0, 200.0, 230.0];
const OVERLAY_OPACITY: f32 = 0.6;

/// What the left view tints over the source image.
#[derive(Default)]
pub struct SourceLayers<'a> {
    /// AI prediction (green).
    pub predicted: Option<&'a Mask>,
    /// Keep marks (green).
    pub keep: Option<&'a Mask>,
    /// Remove marks (red), graph-cut mode only.
    pub remove: Option<&'a Mask>,
    /// Hair marks (yellow), hair tool only.
    pub hair: Option<&'a Mask>,
    /// Pending smart selection (cyan).
    pub selection: Option<&'a Mask>,
}

/// Left view: the source image with the chosen layers tinted over it.
pub fn render_source_view(image: &Image, layers: &SourceLayers, region: Rect, out_w: usize, out_h: usize) -> Rgba {
    let scale = out_w as f64 / region.width();
    let interp = interp_for(scale);
    let src = crop_resize_image(image, region, out_w, out_h, interp);
    let sample = |m: Option<&Mask>| m.map(|m| crop_resize_mask(m, region, out_w, out_h, interp));
    let tints = [
        (sample(layers.predicted), OVERLAY_RGB),
        (sample(layers.keep), OVERLAY_RGB),
        (sample(layers.remove), REMOVE_RGB),
        (sample(layers.hair), HAIR_RGB),
        (sample(layers.selection), SELECTION_RGB),
    ];
    let mut out = vec![255u8; out_w * out_h * 4];
    out.par_chunks_mut(4).enumerate().for_each(|(i, o)| {
        let mut c = [src.data[i * 3] as f32, src.data[i * 3 + 1] as f32, src.data[i * 3 + 2] as f32];
        for (mask, rgb) in &tints {
            let Some(m) = mask else { continue };
            let a = m.data[i] as f32 / 255.0 * OVERLAY_OPACITY;
            for k in 0..3 {
                c[k] = c[k] * (1.0 - a) + rgb[k] * a;
            }
        }
        o[0] = c[0].round() as u8;
        o[1] = c[1].round() as u8;
        o[2] = c[2].round() as u8;
    });
    Rgba { w: out_w, h: out_h, data: out }
}

/// The crop as whole pixels: first column and row (negative when the canvas is extended past the picture) and size.
/// Both edges are rounded, so a crop inside the picture never reaches a pixel past it.
fn crop_pixels(r: Rect) -> (i64, i64, usize, usize) {
    let (x0, y0) = (r.x0.round() as i64, r.y0.round() as i64);
    let (x1, y1) = (r.x1.round() as i64, r.y1.round() as i64);
    (x0, y0, (x1 - x0).max(1) as usize, (y1 - y0).max(1) as usize)
}

/// Full-resolution result with the crop applied (what gets exported). Only the cropped region is rendered.
pub fn export_rgba(scene: &Scene) -> Rgba {
    let (w, h) = (scene.image.w, scene.image.h);
    let finished = scene
        .params
        .background
        .fit_to_result
        .then(|| finish_alpha_scaled(scene.alpha.clone(), scene.params.mask.offset, scene.params.foreground.feather, 1.0));
    let r = crop::effective_crop(scene.params, finished.as_ref(), (w as f64, h as f64));
    let (x0, y0, cw, ch) = crop_pixels(r);
    let region = Rect::new(x0 as f64, y0 as f64, (x0 + cw as i64) as f64, (y0 + ch as i64) as f64);
    let cropped = render_region_with(&Scene { frame: Some(region), ..*scene }, region, cw, ch, usize::MAX);
    resize_to_output_width(cropped, scene.params.background.output_width as usize)
}

/// Longest side of an exported picture; a very tall crop at a large width cannot ask for more.
const MAX_OUTPUT_SIDE: usize = 20_000;

/// Scale the result to `width` pixels wide (0 = leave it), keeping its proportions.
fn resize_to_output_width(img: Rgba, width: usize) -> Rgba {
    if width == 0 || width == img.w {
        return img;
    }
    let k = (width as f64 / img.w as f64).min(MAX_OUTPUT_SIDE as f64 / img.w.max(img.h) as f64);
    let (dw, dh) = (((img.w as f64 * k).round() as usize).max(1), ((img.h as f64 * k).round() as usize).max(1));
    resize_rgba(&img, dw, dh, if k < 1.0 { Interp::Area } else { Interp::Bicubic })
}

/// Why a 16-bit export is not possible with the current settings, if it is not.
pub fn deep_blocker(params: &Params) -> Option<&'static str> {
    if !matches!(params.background.mode, BgMode::Transparent | BgMode::Color) {
        Some("the background is a picture, a gradient or the original")
    } else if params.background.output_width > 0 {
        Some("a fixed output width is set")
    } else if (params.foreground.brightness, params.foreground.saturation, params.foreground.warmth) != (0, 0, 0) {
        Some("the colour adjustments are on")
    } else if params.foreground.defringe > 0 {
        Some("the edge clean-up is on")
    } else if params.foreground.filter.kind != crate::params::FilterKind::None {
        Some("a foreground effect is active")
    } else if params.shadow.enabled {
        Some("the shadow is on")
    } else {
        None
    }
}

/// Full-resolution, cropped 16-bit result: the 16-bit source cut out with the final mask on a
/// transparent or solid background. `None` when the settings need pixel effects ([`deep_blocker`]).
pub fn export_deep(scene: &Scene, deep: &Image16) -> Option<Rgba16> {
    let p = scene.params;
    if deep_blocker(p).is_some() || (deep.w, deep.h) != (scene.image.w, scene.image.h) {
        return None;
    }
    let (w, h) = (deep.w, deep.h);
    let alpha = finished_alpha(scene.alpha, p);
    let bg: Option<[u32; 3]> = (p.background.mode == BgMode::Color).then(|| p.background.color.map(|c| c as u32 * 257));
    let r = crop::effective_crop(p, p.background.fit_to_result.then_some(&alpha), (w as f64, h as f64));
    let (x0, y0, cw, ch) = crop_pixels(r);
    let mut data = vec![0u16; cw * ch * 4];
    data.par_chunks_mut(cw * 4).enumerate().for_each(|(y, row)| {
        let sy = y0 + y as i64;
        for x in 0..cw {
            let sx = x0 + x as i64;
            if !(0..w as i64).contains(&sx) || !(0..h as i64).contains(&sy) {
                // Canvas extended past the picture: the background colour, or nothing.
                if let Some(b) = bg {
                    row[x * 4..x * 4 + 3].iter_mut().zip(b).for_each(|(o, v)| *o = v as u16);
                    row[x * 4 + 3] = 65535;
                }
                continue;
            }
            let i = sy as usize * w + sx as usize;
            let a = alpha.data[i] as u32 * 257;
            let src = &deep.data[i * 3..i * 3 + 3];
            let o = &mut row[x * 4..x * 4 + 4];
            match bg {
                None => {
                    o[..3].copy_from_slice(src);
                    o[3] = a as u16;
                }
                Some(b) => {
                    for c in 0..3 {
                        o[c] = ((src[c] as u32 * a + b[c] * (65535 - a) + 32767) / 65535) as u16;
                    }
                    o[3] = 65535;
                }
            }
        }
    });
    Some(Rgba16 { w: cw, h: ch, data })
}

/// Alpha after offset and feathering at full resolution (used for the crop box).
pub fn finished_alpha(alpha: &Mask, params: &Params) -> Mask {
    finish_alpha_scaled(alpha.clone(), params.mask.offset, params.foreground.feather, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::{FilterKind, FilterSpec};

    fn scene_parts() -> (Image, Mask) {
        let (w, h) = (80usize, 60usize);
        let mut img = Image::new(w, h);
        for (i, p) in img.data.chunks_exact_mut(3).enumerate() {
            p.copy_from_slice(&[(i % w * 3) as u8, (i / w * 4) as u8, 120]);
        }
        let mut alpha = Mask::new(w, h);
        for y in 20..40 {
            for x in 30..55 {
                alpha.data[y * w + x] = 255;
            }
        }
        (img, alpha)
    }

    fn full(scene: &Scene) -> Rgba {
        render_region(scene, Rect::new(0.0, 0.0, 80.0, 60.0), 80, 60)
    }

    #[test]
    fn transparent_background_gives_mask_as_alpha_and_image_colours() {
        let (img, alpha) = scene_parts();
        let params = Params::default();
        let out = full(&Scene { image: &img, alpha: &alpha, params: &params, bg_image: None, filled: None, frame: None });
        for i in 0..80 * 60 {
            assert_eq!(out.data[i * 4 + 3], alpha.data[i]);
        }
        let i = 30 * 80 + 40;
        assert_eq!(&out.data[i * 4..i * 4 + 3], &img.data[i * 3..i * 3 + 3]);
    }

    #[test]
    fn solid_background_is_opaque_and_visible_outside_the_subject() {
        let (img, alpha) = scene_parts();
        let mut params = Params::default();
        params.background.mode = BgMode::Color;
        params.background.color = [10, 20, 30];
        let out = full(&Scene { image: &img, alpha: &alpha, params: &params, bg_image: None, filled: None, frame: None });
        assert!(out.data.chunks_exact(4).all(|p| p[3] == 255));
        assert_eq!(&out.data[..3], &[10, 20, 30]);
    }

    #[test]
    fn original_background_with_filter_and_image_background_cover() {
        let (img, alpha) = scene_parts();
        let mut params = Params::default();
        params.background.mode = BgMode::Original;
        params.background.filter = FilterSpec { kind: FilterKind::Negative, ..Default::default() };
        let out = full(&Scene { image: &img, alpha: &alpha, params: &params, bg_image: None, filled: None, frame: None });
        assert_eq!(out.data[0], 255 - img.data[0]); // background negated, subject (inside mask) not
        let i = 30 * 80 + 40;
        assert_eq!(out.data[i * 4], img.data[i * 3]);

        let pic = Image::filled(40, 40, [9, 9, 9]);
        params.background.mode = BgMode::Image;
        params.background.filter = FilterSpec::default();
        let out = full(&Scene { image: &img, alpha: &alpha, params: &params, bg_image: Some(&pic), filled: None, frame: None });
        assert_eq!(&out.data[..4], &[9, 9, 9, 255]);
    }

    #[test]
    fn shadow_appears_around_the_subject_only_when_enabled() {
        let (img, alpha) = scene_parts();
        let mut params = Params::default();
        let outside = (30 * 80 + 27) * 4 + 3; // 3 px left of the subject
        let off = full(&Scene { image: &img, alpha: &alpha, params: &params, bg_image: None, filled: None, frame: None });
        assert_eq!(off.data[outside], 0);
        params.shadow.enabled = true;
        let on = full(&Scene { image: &img, alpha: &alpha, params: &params, bg_image: None, filled: None, frame: None });
        assert!(on.data[outside] > 0 && (on.data[outside] as u32) < 255 * 75 / 100 + 1);
        assert_eq!(&on.data[outside - 3..outside], &[0, 0, 0]); // shadow colour
    }

    #[test]
    fn the_shadow_follows_its_offset() {
        let (img, alpha) = scene_parts(); // the subject spans x 30..55, y 20..40
        let mut params = Params::default();
        params.shadow.enabled = true;
        let at = |p: &Params, x: usize, y: usize| {
            full(&Scene { image: &img, alpha: &alpha, params: p, bg_image: None, filled: None, frame: None }).data[(y * 80 + x) * 4 + 3]
        };
        // 5 px right of the subject and 5 px left of it, in the middle of its height.
        let (right, left) = ((60, 30), (25, 30));
        assert!(at(&params, left.0, left.1) >= at(&params, right.0, right.1) - 2, "centred shadow: both sides alike");
        params.shadow.offset_x = 10;
        assert!(at(&params, right.0, right.1) > at(&params, left.0, left.1) + 30, "pushed to the right");
        params.shadow.offset_x = 0;
        params.shadow.offset_y = 10;
        let (below, above) = ((42, 45), (42, 15));
        assert!(at(&params, below.0, below.1) > at(&params, above.0, above.1) + 30, "pushed down");
    }

    #[test]
    fn viewport_tile_matches_the_full_render() {
        let (img, alpha) = scene_parts();
        let mut params = Params::default();
        params.background.mode = BgMode::Color;
        params.foreground.feather = 2;
        params.foreground.defringe = 3;
        params.foreground.brightness = 20;
        params.foreground.warmth = -30;
        params.mask.offset = 2;
        params.shadow.enabled = true;
        params.shadow.blur_radius = 6;
        params.shadow.offset_x = 4;
        params.shadow.offset_y = -3;
        params.foreground.filter = FilterSpec { kind: FilterKind::Blur, radius: 4, ..Default::default() };
        let scene = Scene { image: &img, alpha: &alpha, params: &params, bg_image: None, filled: None, frame: None };
        let whole = full(&scene);
        let tile = render_region(&scene, Rect::new(20.0, 10.0, 60.0, 50.0), 40, 40);
        let mut worst = 0i32;
        for y in 0..40 {
            for x in 0..40 {
                for c in 0..4 {
                    let a = tile.data[(y * 40 + x) * 4 + c] as i32;
                    let b = whole.data[((y + 10) * 80 + x + 20) * 4 + c] as i32;
                    worst = worst.max((a - b).abs());
                }
            }
        }
        assert!(worst <= 3, "tile differs from the full render by {worst}");
    }

    #[test]
    fn the_export_of_a_crop_equals_that_crop_of_the_whole_render() {
        let (img, alpha) = scene_parts();
        let mut params = Params::default();
        params.background.mode = BgMode::Color;
        params.foreground.feather = 3;
        params.mask.offset = 2;
        params.shadow.enabled = true;
        params.shadow.blur_radius = 10;
        params.shadow.offset_x = 6;
        params.shadow.offset_y = 4;
        params.foreground.filter = FilterSpec { kind: FilterKind::Blur, radius: 5, ..Default::default() };
        let whole = full(&Scene { image: &img, alpha: &alpha, params: &params, bg_image: None, filled: None, frame: None });
        params.crop = Some(Rect::new(22.0, 12.0, 62.0, 46.0));
        let out = export_rgba(&Scene { image: &img, alpha: &alpha, params: &params, bg_image: None, filled: None, frame: None });
        assert_eq!((out.w, out.h), (40, 34));
        let worst = (0..34)
            .flat_map(|y| (0..40).flat_map(move |x| (0..4).map(move |c| (x, y, c))))
            .map(|(x, y, c)| (out.data[(y * 40 + x) * 4 + c] as i32 - whole.data[((y + 12) * 80 + x + 22) * 4 + c] as i32).abs())
            .max()
            .unwrap();
        assert!(worst <= 2, "the cropped export differs from the whole render by {worst}");
    }

    #[test]
    fn a_region_past_the_picture_shows_the_background_and_the_shadow_spilling_over() {
        let (img, mut alpha) = scene_parts();
        // The subject touches the left edge of the picture.
        alpha.data.iter_mut().for_each(|v| *v = 0);
        for y in 20..40 {
            for x in 0..10 {
                alpha.data[y * 80 + x] = 255;
            }
        }
        let region = Rect::new(-20.0, -10.0, 100.0, 70.0);
        let mut params = Params::default();
        let at = |o: &Rgba, x: i64, y: i64| {
            let i = (((y + 10) * 120 + x + 20) * 4) as usize;
            [o.data[i], o.data[i + 1], o.data[i + 2], o.data[i + 3]]
        };
        let scene = |p: &Params| {
            render_region(&Scene { image: &img, alpha: &alpha, params: p, bg_image: None, filled: None, frame: None }, region, 120, 80)
        };

        // Transparent: nothing outside the picture, the picture itself as usual.
        let t = scene(&params);
        assert_eq!(at(&t, -15, 5)[3], 0);
        assert_eq!(at(&t, 5, 30)[3], 255);
        assert_eq!(&at(&t, 5, 30)[..3], &img.data[(30 * 80 + 5) * 3..(30 * 80 + 5) * 3 + 3]);

        // A colour fills the extra canvas.
        params.background.mode = BgMode::Color;
        params.background.color = [10, 20, 30];
        let c = scene(&params);
        assert_eq!(at(&c, -15, 5), [10, 20, 30, 255]);
        assert_eq!(at(&c, 90, 65), [10, 20, 30, 255]);

        // The original is continued with its edge pixels.
        params.background.mode = BgMode::Original;
        let o = scene(&params);
        assert_eq!(&at(&o, -15, 5)[..3], &img.data[5 * 80 * 3..5 * 80 * 3 + 3], "left of the picture: its first column");
        assert_eq!(&at(&o, 95, 65)[..3], &img.data[(59 * 80 + 79) * 3..(59 * 80 + 79) * 3 + 3], "past the corner: the corner pixel");

        // The shadow of a subject at the edge spills past it.
        params.background.mode = BgMode::Transparent;
        params.shadow.enabled = true;
        params.shadow.blur_radius = 12;
        let sh = scene(&params);
        assert!(at(&sh, -4, 30)[3] > 0 && at(&sh, -4, 30)[3] < 255, "{:?}", at(&sh, -4, 30));
        assert_eq!(at(&sh, -19, 0)[3], 0);
    }

    #[test]
    fn a_gradient_runs_from_one_colour_to_the_other_across_the_exported_frame() {
        let (img, alpha) = scene_parts(); // 80 x 60
        let mut params = Params::default();
        params.background.mode = BgMode::Gradient;
        params.background.color = [0, 0, 0];
        params.background.gradient_color = [200, 100, 0];
        let export = |p: &Params| export_rgba(&Scene { image: &img, alpha: &alpha, params: p, bg_image: None, filled: None, frame: None });
        // 90 degrees: from the left edge to the right edge.
        params.background.gradient_angle = 90;
        let out = export(&params);
        let at = |x: usize, y: usize| out.data[(y * 80 + x) * 4..][..4].to_vec();
        let (left, right) = (at(0, 5), at(79, 5));
        assert!(left[0] <= 3 && left[1] <= 2 && right[0] >= 197 && right[1] >= 98, "the ends are the two colours: {left:?} {right:?}");
        let middle = at(10 + 20, 55); // below the subject, a quarter of the way: about a quarter of the end colour
        assert!(middle[0].abs_diff(75) <= 3 && middle[1].abs_diff(37) <= 3 && middle[3] == 255, "{middle:?}");
        assert_eq!(at(10, 2)[0], at(10, 57)[0], "the colour does not change along the other axis");
        // 180 degrees: from the top to the bottom.
        params.background.gradient_angle = 180;
        let out = export(&params);
        assert!(out.data[(2 * 80 + 5) * 4] < 10 && out.data[(57 * 80 + 5) * 4] > 190);
        // The gradient follows the exported frame, not the picture.
        params.crop = Some(Rect::new(40.0, 0.0, 80.0, 60.0));
        params.background.gradient_angle = 90;
        let out = export(&params);
        assert!(out.data[(5 * 40) * 4] < 10 && out.data[(5 * 40 + 39) * 4] > 190, "start and end are the edges of the crop");
        assert!(deep_blocker(&params).is_some());
    }

    #[test]
    fn a_background_picture_covers_the_exported_frame_not_the_whole_picture() {
        let (img, alpha) = scene_parts(); // 80 x 60, the subject spans x 30..55, y 20..40
        let mut pic = Image::filled(40, 20, [200, 0, 0]);
        for y in 0..20 {
            for x in 20..40 {
                pic.data[(y * 40 + x) * 3..(y * 40 + x) * 3 + 3].copy_from_slice(&[0, 0, 200]);
            }
        }
        let mut params = Params::default();
        params.background.mode = BgMode::Image;
        let export =
            |p: &Params| export_rgba(&Scene { image: &img, alpha: &alpha, params: p, bg_image: Some(&pic), filled: None, frame: None });
        // Uncropped, the picture is stretched over the whole 80 x 60 picture: red on the left half, blue on the right.
        let whole = export(&params);
        assert_eq!((&whole.data[(5 * 80 + 5) * 4..][..3], &whole.data[(5 * 80 + 75) * 4..][..3]), (&[200, 0, 0][..], &[0, 0, 200][..]));
        // Cropped to the top-left 40 x 30 corner, the picture is scaled to that frame: both halves are visible in it.
        params.crop = Some(Rect::new(0.0, 0.0, 40.0, 30.0));
        let corner = export(&params);
        assert_eq!((corner.w, corner.h), (40, 30));
        assert_eq!(&corner.data[(5 * 40 + 5) * 4..][..3], &[200, 0, 0]);
        assert_eq!(&corner.data[(5 * 40 + 35) * 4..][..3], &[0, 0, 200], "the right half of the picture reaches the frame");
    }

    #[test]
    fn a_crop_past_the_picture_exports_a_bigger_canvas_in_8_and_16_bits() {
        let (img, alpha) = scene_parts(); // 80 x 60
        let deep = Image16 { w: 80, h: 60, data: img.data.iter().map(|&v| v as u16 * 257).collect() };
        let mut params = Params { crop: Some(Rect::new(-10.0, -5.0, 90.0, 65.0)), ..Params::default() };
        params.background.mode = BgMode::Color;
        params.background.color = [10, 20, 30];
        let scene = Scene { image: &img, alpha: &alpha, params: &params, bg_image: None, filled: None, frame: None };
        let out = export_rgba(&scene);
        assert_eq!((out.w, out.h), (100, 70));
        assert_eq!(&out.data[..4], &[10, 20, 30, 255], "the added margin is the background colour");
        let inside = ((35 + 5) * 100 + 40 + 10) * 4; // picture pixel (40, 35) is inside the subject
        assert_eq!(&out.data[inside..inside + 3], &img.data[(35 * 80 + 40) * 3..(35 * 80 + 40) * 3 + 3]);

        let d = export_deep(&scene, &deep).unwrap();
        assert_eq!((d.w, d.h), (100, 70));
        assert_eq!(&d.data[..4], &[10 * 257, 20 * 257, 30 * 257, 65535]);
        assert_eq!(d.data[inside..inside + 3], deep.data[(35 * 80 + 40) * 3..(35 * 80 + 40) * 3 + 3]);

        // Transparent background: the margin is transparent.
        params.background.mode = BgMode::Transparent;
        let scene = Scene { image: &img, alpha: &alpha, params: &params, bg_image: None, filled: None, frame: None };
        assert_eq!(export_rgba(&scene).data[3], 0);
        assert_eq!(export_deep(&scene, &deep).unwrap().data[3], 0);
    }

    #[test]
    fn downscaled_preview_has_requested_size_and_similar_mean() {
        let (img, alpha) = scene_parts();
        let mut params = Params::default();
        params.background.mode = BgMode::Color;
        let scene = Scene { image: &img, alpha: &alpha, params: &params, bg_image: None, filled: None, frame: None };
        let small = render_region(&scene, Rect::new(0.0, 0.0, 80.0, 60.0), 40, 30);
        assert_eq!((small.w, small.h, small.data.len()), (40, 30, 40 * 30 * 4));
    }

    #[test]
    fn source_view_tints_marked_and_predicted_pixels() {
        let (img, alpha) = scene_parts();
        let marks = Mask::new(80, 60);
        let region = Rect::new(0.0, 0.0, 80.0, 60.0);
        let v =
            render_source_view(&img, &SourceLayers { predicted: Some(&alpha), keep: Some(&marks), ..Default::default() }, region, 80, 60);
        let inside = (30 * 80 + 40) * 4;
        assert_eq!(&v.data[0..3], &img.data[0..3]);
        assert_ne!(&v.data[inside..inside + 3], &img.data[(30 * 80 + 40) * 3..(30 * 80 + 40) * 3 + 3]);
        assert_eq!(v.data[inside + 3], 255);
        // Remove marks are tinted red, hair marks yellow.
        let full = Mask::filled(80, 60, 255);
        let r = render_source_view(&img, &SourceLayers { remove: Some(&full), ..Default::default() }, region, 80, 60);
        assert!(r.data[0] > img.data[0] && r.data[1] < 200);
        let y = render_source_view(&img, &SourceLayers { hair: Some(&full), ..Default::default() }, region, 80, 60);
        assert!(y.data[0] > img.data[0] && y.data[1] > img.data[1]);
    }

    #[test]
    fn the_edge_cleanup_removes_a_halo_in_the_export_and_blocks_the_16_bit_path() {
        let (mut img, mut alpha) = scene_parts();
        img.data.chunks_exact_mut(3).for_each(|p| p.copy_from_slice(&[200, 0, 0]));
        let halo = 30 * 80 + 29;
        img.data[halo * 3..halo * 3 + 3].copy_from_slice(&[255, 255, 255]);
        alpha.data[halo] = 128;
        let mut params = Params::default();
        let off = export_rgba(&Scene { image: &img, alpha: &alpha, params: &params, bg_image: None, filled: None, frame: None });
        assert_eq!(&off.data[halo * 4..halo * 4 + 3], &[255, 255, 255]);
        assert!(deep_blocker(&params).is_none());
        params.foreground.defringe = 2;
        let on = export_rgba(&Scene { image: &img, alpha: &alpha, params: &params, bg_image: None, filled: None, frame: None });
        assert_eq!(&on.data[halo * 4..halo * 4 + 3], &[200, 0, 0]);
        assert_eq!(on.data[halo * 4 + 3], 128, "the transparency is untouched");
        assert!(deep_blocker(&params).is_some());
    }

    #[test]
    fn export_scales_to_the_output_width_and_a_preset_gives_the_exact_size() {
        let (img, alpha) = scene_parts();
        let mut params = Params::default();
        let size = |p: &Params| {
            let o = export_rgba(&Scene { image: &img, alpha: &alpha, params: p, bg_image: None, filled: None, frame: None });
            (o.w, o.h, o.data.len())
        };
        assert_eq!(size(&params), (80, 60, 80 * 60 * 4));
        params.background.output_width = 40;
        assert_eq!(size(&params), (40, 30, 40 * 30 * 4));
        params.background.output_width = 160;
        assert_eq!(size(&params), (160, 120, 160 * 120 * 4));
        crate::params::PRESETS[0].apply(&mut params);
        assert_eq!(size(&params), (1000, 1000, 1000 * 1000 * 4));
        assert!(deep_blocker(&params).is_some());
    }

    #[test]
    fn colour_adjustments_change_the_subject_and_never_the_background() {
        let (img, alpha) = scene_parts();
        let mut params = Params::default();
        params.background.mode = BgMode::Color;
        params.background.color = [10, 20, 30];
        let inside = (30 * 80 + 40) * 4;
        let render = |p: &Params| export_rgba(&Scene { image: &img, alpha: &alpha, params: p, bg_image: None, filled: None, frame: None });
        let plain = render(&params);
        params.foreground.brightness = 50;
        params.foreground.saturation = -100;
        let adjusted = render(&params);
        assert_eq!(&adjusted.data[..4], &plain.data[..4], "the background keeps its colour");
        assert_ne!(&adjusted.data[inside..inside + 3], &plain.data[inside..inside + 3]);
        let px = &adjusted.data[inside..inside + 3];
        assert!(px[0].abs_diff(px[1]) <= 1 && px[1].abs_diff(px[2]) <= 1, "fully desaturated: {px:?}");
        assert!(deep_blocker(&params).is_some());
    }

    #[test]
    fn export_applies_the_crop() {
        let (img, alpha) = scene_parts();
        let mut params = Params { crop: Some(Rect::new(10.0, 5.0, 50.0, 45.0)), ..Params::default() };
        let out = export_rgba(&Scene { image: &img, alpha: &alpha, params: &params, bg_image: None, filled: None, frame: None });
        assert_eq!((out.w, out.h), (40, 40));
        params.crop = None;
        params.background.fit_to_result = true;
        params.background.padding = 0;
        let out = export_rgba(&Scene { image: &img, alpha: &alpha, params: &params, bg_image: None, filled: None, frame: None });
        assert_eq!((out.w, out.h), (25, 20));
    }
}
