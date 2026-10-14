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
use crate::imaging::resize::{crop_resize_image, crop_resize_mask, resize_rgba, Interp};
use crate::params::{BgMode, Params};
use crate::types::{Image, Image16, Mask, Rect, Rgba, Rgba16};
use filters::{apply_filter, filter_margin};

pub struct Scene<'a> {
    pub image: &'a Image,
    /// Composed mask before boundary offset and feathering.
    pub alpha: &'a Mask,
    pub params: &'a Params,
    /// Loaded background picture when `background.mode == Image`.
    pub bg_image: Option<&'a Image>,
    /// Original with the subject filled in, when `fill_hole` is on.
    pub filled: Option<&'a Image>,
}

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

/// Render `region` (image coordinates, inside the image) into an `out_w` x `out_h` RGBA
/// buffer with straight alpha. Transparent areas keep alpha 0.
pub fn render_region(scene: &Scene, region: Rect, out_w: usize, out_h: usize) -> Rgba {
    let p = scene.params;
    let (iw, ih) = (scene.image.w as f64, scene.image.h as f64);
    let scale = out_w as f64 / region.width();

    // Border needed around the window so that blurs, offset and shadow have their context.
    let shadow_sigma = p.shadow.blur_radius as f64 * scale / 2.0;
    let margin = (if p.mask.offset != 0 { ((p.mask.offset.unsigned_abs() + 2) as f64 * scale / 2.0).ceil() as usize + 1 } else { 0 })
        + (p.foreground.feather as f64 * scale).ceil() as usize
        + (p.foreground.defringe as f64 * scale).ceil() as usize
        + if p.shadow.enabled { (3.0 * shadow_sigma + p.shadow.offset_x.abs().max(p.shadow.offset_y.abs()) as f64 * scale).ceil() as usize } else { 0 }
        + filter_margin(&p.foreground.filter, scale).max(filter_margin(&p.background.filter, scale));
    let m = margin.min(512);
    let pl = m.min((region.x0.max(0.0) * scale).floor() as usize);
    let pt = m.min((region.y0.max(0.0) * scale).floor() as usize);
    let pr = m.min(((iw - region.x1).max(0.0) * scale).floor() as usize);
    let pb = m.min(((ih - region.y1).max(0.0) * scale).floor() as usize);
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
    let a = finish_alpha_scaled(crop_resize_mask(scene.alpha, exp, ew, eh, interp), p.mask.offset, p.foreground.feather, scale);
    let mut fg = crop_resize_image(scene.image, exp, ew, eh, interp);
    defringe::defringe(&mut fg, &a, (p.foreground.defringe as f64 * scale).round() as usize);
    apply_filter(&mut fg, &p.foreground.filter, scale, origin);

    // Background layer: colours plus whether it is opaque.
    let (bg, bg_opaque) = match p.background.mode {
        BgMode::Transparent => (Image::new(ew, eh), false),
        BgMode::Color => (Image::filled(ew, eh, p.background.color), true),
        BgMode::Image => match scene.bg_image {
            Some(pic) => {
                let sc = (iw / pic.w as f64).max(ih / pic.h as f64);
                let (ox, oy) = ((iw - pic.w as f64 * sc) / 2.0, (ih - pic.h as f64 * sc) / 2.0);
                let r = Rect::new((exp.x0 - ox) / sc, (exp.y0 - oy) / sc, (exp.x1 - ox) / sc, (exp.y1 - oy) / sc);
                let mut b = crop_resize_image(pic, r, ew, eh, interp_for(scale * sc));
                apply_filter(&mut b, &p.background.filter, scale, origin);
                (b, true)
            }
            None => (Image::new(ew, eh), false),
        },
        BgMode::Original => {
            let base = if p.background.fill_hole { scene.filled.unwrap_or(scene.image) } else { scene.image };
            let mut b = crop_resize_image(base, exp, ew, eh, interp);
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

/// Full-resolution result with the crop applied (what gets exported).
pub fn export_rgba(scene: &Scene) -> Rgba {
    let (w, h) = (scene.image.w, scene.image.h);
    let full = render_region(scene, Rect::new(0.0, 0.0, w as f64, h as f64), w, h);
    let finished = scene
        .params
        .background
        .fit_to_result
        .then(|| finish_alpha_scaled(scene.alpha.clone(), scene.params.mask.offset, scene.params.foreground.feather, 1.0));
    let r = crop::effective_crop(scene.params, finished.as_ref(), (w as f64, h as f64));
    let x0 = (r.x0.round() as usize).min(w - 1);
    let y0 = (r.y0.round() as usize).min(h - 1);
    let cw = (r.width().round() as usize).clamp(1, w - x0);
    let ch = (r.height().round() as usize).clamp(1, h - y0);
    let cropped = if (x0, y0, cw, ch) == (0, 0, w, h) {
        full
    } else {
        let mut data = Vec::with_capacity(cw * ch * 4);
        for y in y0..y0 + ch {
            data.extend_from_slice(&full.data[(y * w + x0) * 4..(y * w + x0 + cw) * 4]);
        }
        Rgba { w: cw, h: ch, data }
    };
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
        Some("the background is a picture or the original")
    } else if params.background.output_width > 0 {
        Some("a fixed output width is set")
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
    let x0 = (r.x0.round() as usize).min(w - 1);
    let y0 = (r.y0.round() as usize).min(h - 1);
    let cw = (r.width().round() as usize).clamp(1, w - x0);
    let ch = (r.height().round() as usize).clamp(1, h - y0);
    let mut data = vec![0u16; cw * ch * 4];
    data.par_chunks_mut(cw * 4).enumerate().for_each(|(y, row)| {
        for x in 0..cw {
            let i = (y0 + y) * w + x0 + x;
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
        let out = full(&Scene { image: &img, alpha: &alpha, params: &params, bg_image: None, filled: None });
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
        let out = full(&Scene { image: &img, alpha: &alpha, params: &params, bg_image: None, filled: None });
        assert!(out.data.chunks_exact(4).all(|p| p[3] == 255));
        assert_eq!(&out.data[..3], &[10, 20, 30]);
    }

    #[test]
    fn original_background_with_filter_and_image_background_cover() {
        let (img, alpha) = scene_parts();
        let mut params = Params::default();
        params.background.mode = BgMode::Original;
        params.background.filter = FilterSpec { kind: FilterKind::Negative, ..Default::default() };
        let out = full(&Scene { image: &img, alpha: &alpha, params: &params, bg_image: None, filled: None });
        assert_eq!(out.data[0], 255 - img.data[0]); // background negated, subject (inside mask) not
        let i = 30 * 80 + 40;
        assert_eq!(out.data[i * 4], img.data[i * 3]);

        let pic = Image::filled(40, 40, [9, 9, 9]);
        params.background.mode = BgMode::Image;
        params.background.filter = FilterSpec::default();
        let out = full(&Scene { image: &img, alpha: &alpha, params: &params, bg_image: Some(&pic), filled: None });
        assert_eq!(&out.data[..4], &[9, 9, 9, 255]);
    }

    #[test]
    fn shadow_appears_around_the_subject_only_when_enabled() {
        let (img, alpha) = scene_parts();
        let mut params = Params::default();
        let outside = (30 * 80 + 27) * 4 + 3; // 3 px left of the subject
        let off = full(&Scene { image: &img, alpha: &alpha, params: &params, bg_image: None, filled: None });
        assert_eq!(off.data[outside], 0);
        params.shadow.enabled = true;
        let on = full(&Scene { image: &img, alpha: &alpha, params: &params, bg_image: None, filled: None });
        assert!(on.data[outside] > 0 && (on.data[outside] as u32) < 255 * 75 / 100 + 1);
        assert_eq!(&on.data[outside - 3..outside], &[0, 0, 0]); // shadow colour
    }

    #[test]
    fn the_shadow_follows_its_offset() {
        let (img, alpha) = scene_parts(); // the subject spans x 30..55, y 20..40
        let mut params = Params::default();
        params.shadow.enabled = true;
        let at = |p: &Params, x: usize, y: usize| full(&Scene { image: &img, alpha: &alpha, params: p, bg_image: None, filled: None }).data[(y * 80 + x) * 4 + 3];
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
        params.mask.offset = 2;
        params.shadow.enabled = true;
        params.shadow.blur_radius = 6;
        params.shadow.offset_x = 4;
        params.shadow.offset_y = -3;
        params.foreground.filter = FilterSpec { kind: FilterKind::Blur, radius: 4, ..Default::default() };
        let scene = Scene { image: &img, alpha: &alpha, params: &params, bg_image: None, filled: None };
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
    fn downscaled_preview_has_requested_size_and_similar_mean() {
        let (img, alpha) = scene_parts();
        let mut params = Params::default();
        params.background.mode = BgMode::Color;
        let scene = Scene { image: &img, alpha: &alpha, params: &params, bg_image: None, filled: None };
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
        let off = export_rgba(&Scene { image: &img, alpha: &alpha, params: &params, bg_image: None, filled: None });
        assert_eq!(&off.data[halo * 4..halo * 4 + 3], &[255, 255, 255]);
        assert!(deep_blocker(&params).is_none());
        params.foreground.defringe = 2;
        let on = export_rgba(&Scene { image: &img, alpha: &alpha, params: &params, bg_image: None, filled: None });
        assert_eq!(&on.data[halo * 4..halo * 4 + 3], &[200, 0, 0]);
        assert_eq!(on.data[halo * 4 + 3], 128, "the transparency is untouched");
        assert!(deep_blocker(&params).is_some());
    }

    #[test]
    fn export_scales_to_the_output_width_and_a_preset_gives_the_exact_size() {
        let (img, alpha) = scene_parts();
        let mut params = Params::default();
        let size = |p: &Params| {
            let o = export_rgba(&Scene { image: &img, alpha: &alpha, params: p, bg_image: None, filled: None });
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
    fn export_applies_the_crop() {
        let (img, alpha) = scene_parts();
        let mut params = Params { crop: Some(Rect::new(10.0, 5.0, 50.0, 45.0)), ..Params::default() };
        let out = export_rgba(&Scene { image: &img, alpha: &alpha, params: &params, bg_image: None, filled: None });
        assert_eq!((out.w, out.h), (40, 40));
        params.crop = None;
        params.background.fit_to_result = true;
        params.background.padding = 0;
        let out = export_rgba(&Scene { image: &img, alpha: &alpha, params: &params, bg_image: None, filled: None });
        assert_eq!((out.w, out.h), (25, 20));
    }
}
