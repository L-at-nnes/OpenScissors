//! Image file input/output.

use std::io::Cursor;
use std::path::Path;

use image::codecs::jpeg::JpegEncoder;
use image::{DynamicImage, ImageDecoder, ImageEncoder, ImageReader};
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::types::{Image, Image16, Mask, Rgba, Rgba16};

/// File extensions that can be opened (the formats the `image` crate is built with). The first frame of a GIF is read.
pub const IMAGE_EXTENSIONS: &[&str] = &["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp", "gif"];

/// Largest picture ever accepted, in pixels (500 megapixels). It only guards against a header that lies about its
/// size; what the machine can really hold is decided by [`check_dimensions`] from the memory budget.
pub const MAX_PIXELS: u64 = 500_000_000;
/// Memory one open picture costs per pixel, in bytes: the working copy, the masks, the rendering buffers and the
/// copies the document keeps. Measured at about 20 on a 24-megapixel edit and export (docs/PERF.md), plus a margin
/// for the decoder's own buffers.
pub const OPEN_BYTES_PER_PIXEL: u64 = 24;

/// Memory in MB that a picture of `w` x `h` pixels needs once opened.
pub fn open_cost_mb(w: u64, h: u64) -> u64 {
    (w * h * OPEN_BYTES_PER_PIXEL) >> 20
}

/// The decision of [`check_dimensions`], from the numbers it looks at: the memory budget of the application and the
/// memory still free, in MB (`None` when the platform does not tell).
fn open_verdict(w: u64, h: u64, budget_mb: Option<u64>, free_mb: Option<u64>) -> Result<()> {
    if w == 0 || h == 0 {
        return Err(Error::msg("the picture is empty"));
    }
    if w.saturating_mul(h) > MAX_PIXELS {
        return Err(Error::msg(format!(
            "the picture is {w} x {h} pixels; the application accepts up to {} megapixels",
            MAX_PIXELS / 1_000_000
        )));
    }
    let need = open_cost_mb(w, h);
    if let Some(budget) = budget_mb.filter(|&b| need > b) {
        return Err(Error::msg(format!(
            "a {w} x {h} px picture needs about {need} MB, more than the {budget} MB the application may use (the memory limit in the settings can be raised)"
        )));
    }
    if let Some(free) = free_mb.filter(|&f| need > f) {
        return Err(Error::msg(format!(
            "not enough free memory for a {w} x {h} px picture: about {need} MB are needed and {free} MB are free; close other applications"
        )));
    }
    Ok(())
}

/// Refuse a picture that would not fit, from its size alone and before anything is allocated for it: too big to be
/// believable, over the memory budget of the application, or more than the memory that is free.
pub fn check_dimensions(w: u64, h: u64) -> Result<()> {
    open_verdict(w, h, crate::sysmem::budget_mb(), crate::sysmem::available_memory_mb())
}

/// An opened image: the 8-bit working copy, plus the 16-bit samples when the file has them.
pub struct Loaded {
    pub image: Image,
    pub deep: Option<Image16>,
    /// The transparency of a picture that is already cut out (a PNG, WebP or TIFF with an alpha channel that is not
    /// fully opaque). `None` for a picture with no transparency.
    pub alpha: Option<Mask>,
    /// The file carried a colour profile other than sRGB and the pixels were converted to sRGB.
    pub converted: bool,
}

/// The colour profile `icc` when it describes an RGB space other than sRGB (the only case that needs a conversion).
fn foreign_profile(icc: &[u8]) -> Option<moxcms::ColorProfile> {
    let profile = moxcms::ColorProfile::new_from_slice(icc).ok()?;
    let is_srgb = format!("{:?}", profile.description).contains("sRGB");
    (profile.color_space == moxcms::DataColorSpace::Rgb && !is_srgb).then_some(profile)
}

/// Samples converted per call: the transform reads and writes separate buffers, so the picture goes through in pieces.
const CONVERT_CHUNK: usize = 3 << 20;

/// Convert RGB samples (3 per pixel) from `profile` to sRGB, in place. False if the profile cannot be used.
fn to_srgb<T: Copy + Default + Send + Sync>(data: &mut [T], transform: &dyn moxcms::TransformExecutor<T>) -> bool {
    let mut scratch = vec![T::default(); CONVERT_CHUNK.min(data.len())];
    for chunk in data.chunks_mut(CONVERT_CHUNK) {
        let tmp = &mut scratch[..chunk.len()];
        tmp.copy_from_slice(chunk);
        if transform.transform(tmp, chunk).is_err() {
            return false;
        }
    }
    true
}

/// The sRGB colour profile, as written into the exports of a picture that was colour-converted.
pub fn srgb_profile() -> &'static [u8] {
    static ICC: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    ICC.get_or_init(|| moxcms::ColorProfile::new_srgb().encode().unwrap_or_default())
}

/// Read an image as 8-bit RGB. EXIF orientation is applied; alpha is dropped.
/// Write `bytes` to `path` so that a crash or a full disk never leaves a half-written file: the data goes to a
/// temporary file in the same folder, is flushed to disk, then replaces the target in one step.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    write_atomic_with(path, |out| std::io::Write::write_all(out, bytes))
}

/// Like [`write_atomic`], for data that is produced as it is written (a big project is not built in memory first).
pub fn write_atomic_with(
    path: &Path,
    write: impl FnOnce(&mut std::io::BufWriter<std::fs::File>) -> std::io::Result<()>,
) -> std::io::Result<()> {
    use std::io::Write;
    let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let tmp = path.with_file_name(format!(".{name}.{}.tmp", std::process::id()));
    let result = (|| {
        let mut out = std::io::BufWriter::new(std::fs::File::create(&tmp)?);
        write(&mut out)?;
        out.flush()?;
        out.get_ref().sync_all()?;
        drop(out);
        std::fs::rename(&tmp, path)
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    result
}

pub fn load_image(path: &Path) -> Result<Image> {
    Ok(load_image_deep(path)?.image)
}

/// Like [`load_image`], and keeps the 16-bit samples of 16-bit PNG and TIFF files.
pub fn load_image_deep(path: &Path) -> Result<Loaded> {
    let mut reader = ImageReader::open(path)?.with_guessed_format()?;
    // The size is checked from the header ([`check_dimensions`]); the decoder's own default allocation limit is lifted
    // so that a large 16-bit picture that fits the budget can be opened.
    let mut limits = image::Limits::default();
    limits.max_alloc = None;
    reader.limits(limits);
    let mut decoder = reader.into_decoder()?;
    let (w, h) = decoder.dimensions();
    check_dimensions(w as u64, h as u64)?;
    let orientation = decoder.orientation()?;
    let profile = decoder.icc_profile().ok().flatten().as_deref().and_then(foreign_profile);
    let mut img = DynamicImage::from_decoder(decoder)?;
    img.apply_orientation(orientation);
    let alpha = cut_out_alpha(&img);
    let is_deep = matches!(
        img,
        DynamicImage::ImageRgb16(_) | DynamicImage::ImageRgba16(_) | DynamicImage::ImageLuma16(_) | DynamicImage::ImageLumaA16(_)
    );
    let mut deep = is_deep.then(|| {
        let d = img.to_rgb16();
        Image16 { w: d.width() as usize, h: d.height() as usize, data: d.into_raw() }
    });
    let mut rgb = img.to_rgb8();
    drop(img);
    let mut converted = false;
    if let Some(profile) = &profile {
        let srgb = moxcms::ColorProfile::new_srgb();
        let options = moxcms::TransformOptions::default();
        if let Ok(t) = profile.create_transform_8bit(moxcms::Layout::Rgb, &srgb, moxcms::Layout::Rgb, options) {
            converted = to_srgb(&mut rgb, t.as_ref());
        }
        if let (Some(d), Ok(t)) = (&mut deep, profile.create_transform_16bit(moxcms::Layout::Rgb, &srgb, moxcms::Layout::Rgb, options)) {
            to_srgb(&mut d.data, t.as_ref());
        }
    }
    Ok(Loaded { image: Image { w: rgb.width() as usize, h: rgb.height() as usize, data: rgb.into_raw() }, deep, alpha, converted })
}

/// The alpha channel of `img` when it really makes the picture transparent somewhere.
fn cut_out_alpha(img: &DynamicImage) -> Option<Mask> {
    if !img.color().has_alpha() {
        return None;
    }
    let rgba = img.to_rgba8();
    let data: Vec<u8> = rgba.as_raw().chunks_exact(4).map(|p| p[3]).collect();
    data.iter().any(|&a| a < 255).then(|| Mask { w: rgba.width() as usize, h: rgba.height() as usize, data })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Format {
    Png,
    Jpeg,
    Webp,
    /// Written only: the build has an AVIF encoder, no decoder.
    Avif,
    Bmp,
    Tiff,
}

impl Format {
    pub fn from_path(path: &Path) -> Option<Self> {
        match path.extension()?.to_str()?.to_ascii_lowercase().as_str() {
            "png" => Some(Self::Png),
            "jpg" | "jpeg" => Some(Self::Jpeg),
            "webp" => Some(Self::Webp),
            "avif" => Some(Self::Avif),
            "bmp" => Some(Self::Bmp),
            "tif" | "tiff" => Some(Self::Tiff),
            _ => None,
        }
    }
    pub fn supports_alpha(self) -> bool {
        !matches!(self, Self::Jpeg | Self::Bmp)
    }
    pub fn extension(self) -> &'static str {
        match self {
            Self::Png => "png",
            Self::Jpeg => "jpg",
            Self::Webp => "webp",
            Self::Avif => "avif",
            Self::Bmp => "bmp",
            Self::Tiff => "tiff",
        }
    }
}

/// Composite straight-alpha RGBA over a flat colour.
pub fn flatten(rgba: &Rgba, bg: [u8; 3]) -> Image {
    let mut out = Vec::with_capacity(rgba.w * rgba.h * 3);
    for p in rgba.data.chunks_exact(4) {
        let a = p[3] as u32;
        for c in 0..3 {
            out.push(((p[c] as u32 * a + bg[c] as u32 * (255 - a) + 127) / 255) as u8);
        }
    }
    Image { w: rgba.w, h: rgba.h, data: out }
}

/// AVIF encoder effort, 1 (slowest, smallest) to 10: 6 keeps a 12-megapixel export within a few seconds.
const AVIF_SPEED: u8 = 6;

/// Save `rgba` in `format`. Formats without alpha are flattened on `flatten_color`.
/// `quality` (10-100) applies to JPEG, WebP and AVIF; WebP at 100 is lossless (and keeps the colour profile tag).
/// The file is encoded in memory first and written with [`write_atomic`], so a failure never leaves a truncated file.
pub fn save_rgba(path: &Path, rgba: &Rgba, format: Format, quality: u8, flatten_color: [u8; 3]) -> Result<()> {
    save_rgba_tagged(path, rgba, format, quality, flatten_color, false)
}

/// Like [`save_rgba`]; with `srgb_tag` the sRGB colour profile is embedded (formats that can hold one), which tells
/// colour-managed software that the colours of a picture converted at opening ([`Loaded::converted`]) are sRGB.
pub fn save_rgba_tagged(path: &Path, rgba: &Rgba, format: Format, quality: u8, flatten_color: [u8; 3], srgb_tag: bool) -> Result<()> {
    let (w, h) = (rgba.w as u32, rgba.h as u32);
    let mut out = Cursor::new(Vec::new());
    // Encoders that cannot hold a profile say so; the picture is then written without one.
    let tag = |e: &mut dyn ImageEncoder| {
        if srgb_tag {
            let _ = e.set_icc_profile(srgb_profile().to_vec());
        }
    };
    if format.supports_alpha() {
        let buf = image::RgbaImage::from_raw(w, h, rgba.data.clone()).ok_or_else(|| Error::msg("bad RGBA buffer"))?;
        let color = image::ExtendedColorType::Rgba8;
        match format {
            Format::Png => {
                let mut e = image::codecs::png::PngEncoder::new(&mut out);
                tag(&mut e);
                e.write_image(buf.as_raw(), w, h, color)?
            }
            Format::Webp if quality >= 100 => {
                let mut e = image::codecs::webp::WebPEncoder::new_lossless(&mut out);
                tag(&mut e);
                e.write_image(buf.as_raw(), w, h, color)?
            }
            Format::Webp => {
                // libwebp: the colours are compressed with loss, the transparency is kept exact.
                let lossy = webp::Encoder::from_rgba(buf.as_raw(), w, h).encode(quality.max(10) as f32);
                std::io::Write::write_all(&mut out, &lossy)?
            }
            Format::Avif => {
                let mut e = image::codecs::avif::AvifEncoder::new_with_speed_quality(&mut out, AVIF_SPEED, quality.clamp(10, 100));
                tag(&mut e);
                e.write_image(buf.as_raw(), w, h, color)?
            }
            _ => {
                let mut e = image::codecs::tiff::TiffEncoder::new(&mut out);
                tag(&mut e);
                e.write_image(buf.as_raw(), w, h, color)?
            }
        }
    } else {
        let flat = flatten(rgba, flatten_color);
        let buf = image::RgbImage::from_raw(w, h, flat.data).ok_or_else(|| Error::msg("bad RGB buffer"))?;
        match format {
            Format::Jpeg => {
                let mut e = JpegEncoder::new_with_quality(&mut out, quality.clamp(10, 100));
                tag(&mut e);
                e.write_image(buf.as_raw(), w, h, image::ExtendedColorType::Rgb8)?
            }
            _ => DynamicImage::ImageRgb8(buf).write_to(&mut out, image::ImageFormat::Bmp)?,
        }
    }
    write_atomic(path, &out.into_inner())?;
    Ok(())
}

/// Save 16-bit RGBA as PNG or TIFF (atomically, like [`save_rgba`]).
pub fn save_rgba16(path: &Path, rgba: &Rgba16, format: Format) -> Result<()> {
    save_rgba16_tagged(path, rgba, format, false)
}

/// Like [`save_rgba16`], embedding the sRGB profile with `srgb_tag` (see [`save_rgba_tagged`]).
pub fn save_rgba16_tagged(path: &Path, rgba: &Rgba16, format: Format, srgb_tag: bool) -> Result<()> {
    let (w, h) = (rgba.w as u32, rgba.h as u32);
    let bytes: Vec<u8> = rgba.data.iter().flat_map(|v| v.to_ne_bytes()).collect();
    let mut out = Cursor::new(Vec::new());
    let tag = |e: &mut dyn ImageEncoder| {
        if srgb_tag {
            let _ = e.set_icc_profile(srgb_profile().to_vec());
        }
    };
    match format {
        Format::Png => {
            let mut e = image::codecs::png::PngEncoder::new(&mut out);
            tag(&mut e);
            e.write_image(&bytes, w, h, image::ExtendedColorType::Rgba16)?
        }
        Format::Tiff => {
            let mut e = image::codecs::tiff::TiffEncoder::new(&mut out);
            tag(&mut e);
            e.write_image(&bytes, w, h, image::ExtendedColorType::Rgba16)?
        }
        _ => return Err(Error::msg("16-bit output needs PNG or TIFF")),
    }
    write_atomic(path, &out.into_inner())?;
    Ok(())
}

pub fn save_mask_png(path: &Path, mask: &crate::types::Mask) -> Result<()> {
    let buf = image::GrayImage::from_raw(mask.w as u32, mask.h as u32, mask.data.clone()).ok_or_else(|| Error::msg("bad mask buffer"))?;
    buf.save_with_format(path, image::ImageFormat::Png)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn png_roundtrip_keeps_alpha_and_jpeg_flattens() {
        let dir = std::env::temp_dir().join(format!("osc-io-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut rgba = Rgba::new(4, 2);
        for (i, p) in rgba.data.chunks_exact_mut(4).enumerate() {
            p.copy_from_slice(&[200, 100, 50, if i % 2 == 0 { 255 } else { 0 }]);
        }
        let png = dir.join("a.png");
        save_rgba(&png, &rgba, Format::Png, 95, [255; 3]).unwrap();
        let back = image::open(&png).unwrap().to_rgba8();
        assert_eq!(back.get_pixel(0, 0).0, [200, 100, 50, 255]);
        assert_eq!(back.get_pixel(1, 0).0[3], 0);
        let jpg = dir.join("a.jpg");
        save_rgba(&jpg, &rgba, Format::Jpeg, 95, [255; 3]).unwrap();
        let j = load_image(&jpg).unwrap();
        assert!(j.data[3] > 240 && j.data[4] > 240); // transparent pixel became ~white
        std::fs::remove_dir_all(dir).ok();
    }

    fn png_with_profile(path: &Path, pixels: &[[u8; 3]], profile: Option<Vec<u8>>) {
        let mut out = Vec::new();
        let mut e = image::codecs::png::PngEncoder::new(&mut out);
        if let Some(p) = profile {
            e.set_icc_profile(p).unwrap();
        }
        let raw: Vec<u8> = pixels.iter().flatten().copied().collect();
        e.write_image(&raw, pixels.len() as u32, 1, image::ExtendedColorType::Rgb8).unwrap();
        std::fs::write(path, out).unwrap();
    }

    #[test]
    fn colours_are_converted_from_the_embedded_profile_to_srgb() {
        let dir = std::env::temp_dir().join(format!("osc-icc-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let pixels = [[128, 128, 128], [200, 100, 50]];
        let p3 = moxcms::ColorProfile::new_display_p3().encode().unwrap();
        let srgb = moxcms::ColorProfile::new_srgb().encode().unwrap();

        png_with_profile(&dir.join("p3.png"), &pixels, Some(p3));
        let loaded = load_image_deep(&dir.join("p3.png")).unwrap();
        assert!(loaded.converted);
        let (grey, colour) = (&loaded.image.data[..3], &loaded.image.data[3..6]);
        assert!(grey.iter().all(|&v| v.abs_diff(128) <= 2), "a neutral grey stays grey: {grey:?}");
        assert!(colour[0] > 205 && colour[1] < 95, "a Display P3 colour is more saturated in sRGB: {colour:?}");

        // No profile, or an sRGB one: the pixels are what the file says.
        png_with_profile(&dir.join("none.png"), &pixels, None);
        png_with_profile(&dir.join("srgb.png"), &pixels, Some(srgb));
        for name in ["none.png", "srgb.png"] {
            let l = load_image_deep(&dir.join(name)).unwrap();
            assert!(!l.converted, "{name}");
            assert_eq!(l.image.data, vec![128, 128, 128, 200, 100, 50], "{name}");
        }
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn a_picture_that_is_already_cut_out_keeps_its_transparency() {
        let dir = std::env::temp_dir().join(format!("osc-alpha-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut rgba = Rgba::new(4, 2);
        for (i, p) in rgba.data.chunks_exact_mut(4).enumerate() {
            p.copy_from_slice(&[10, 20, 30, [0, 60, 200, 255][i % 4]]);
        }
        save_rgba(&dir.join("cut.png"), &rgba, Format::Png, 95, [255; 3]).unwrap();
        let loaded = load_image_deep(&dir.join("cut.png")).unwrap();
        let alpha = loaded.alpha.expect("the transparency is kept");
        assert_eq!((alpha.w, alpha.h), (4, 2));
        assert_eq!(alpha.data, vec![0, 60, 200, 255, 0, 60, 200, 255]);
        assert_eq!(&loaded.image.data[3..6], &[10, 20, 30], "the colours are still there");
        // An alpha channel with nothing transparent in it is not a cut-out.
        rgba.data.chunks_exact_mut(4).for_each(|p| p[3] = 255);
        save_rgba(&dir.join("opaque.png"), &rgba, Format::Png, 95, [255; 3]).unwrap();
        assert!(load_image_deep(&dir.join("opaque.png")).unwrap().alpha.is_none());
        save_rgba(&dir.join("plain.jpg"), &rgba, Format::Jpeg, 90, [255; 3]).unwrap();
        assert!(load_image_deep(&dir.join("plain.jpg")).unwrap().alpha.is_none());
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn exports_can_carry_the_srgb_profile() {
        let dir = std::env::temp_dir().join(format!("osc-tag-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let rgba = Rgba::new(3, 3);
        let profile_of = |name: &str| {
            let mut reader = ImageReader::open(dir.join(name)).unwrap().with_guessed_format().unwrap();
            reader.no_limits();
            reader.into_decoder().unwrap().icc_profile().unwrap()
        };
        for (name, format) in [("t.png", Format::Png), ("t.jpg", Format::Jpeg), ("t.webp", Format::Webp)] {
            save_rgba_tagged(&dir.join(name), &rgba, format, 100, [255; 3], true).unwrap(); // 100: WebP stays lossless
            assert_eq!(profile_of(name).as_deref(), Some(srgb_profile()), "{name}");
        }
        // The TIFF decoder of the `image` crate does not hand the profile back: look for it in the file.
        save_rgba_tagged(&dir.join("t.tiff"), &rgba, Format::Tiff, 90, [255; 3], true).unwrap();
        let tiff = std::fs::read(dir.join("t.tiff")).unwrap();
        assert!(tiff.windows(srgb_profile().len()).any(|w| w == srgb_profile()), "the profile is stored in the TIFF");
        save_rgba(&dir.join("plain.png"), &rgba, Format::Png, 90, [255; 3]).unwrap();
        assert!(profile_of("plain.png").is_none());
        let deep = Rgba16 { w: 2, h: 2, data: vec![1000; 16] };
        save_rgba16_tagged(&dir.join("d.png"), &deep, Format::Png, true).unwrap();
        assert_eq!(profile_of("d.png").as_deref(), Some(srgb_profile()));
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn a_gif_is_opened_and_listed_with_the_other_pictures() {
        let dir = std::env::temp_dir().join(format!("osc-gif-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("a.gif");
        let mut frame = image::RgbaImage::new(6, 4);
        frame.pixels_mut().for_each(|p| *p = image::Rgba([250, 10, 10, 255]));
        image::codecs::gif::GifEncoder::new(std::fs::File::create(&file).unwrap()).encode_frame(image::Frame::new(frame)).unwrap();
        let img = load_image(&file).unwrap();
        assert_eq!((img.w, img.h), (6, 4));
        assert!(img.data[0] > 200 && img.data[1] < 60, "the colour survives the palette");
        assert!(IMAGE_EXTENSIONS.contains(&"gif"));
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn webp_is_lossy_below_100_and_lossless_at_100_and_avif_is_written() {
        let dir = std::env::temp_dir().join(format!("osc-lossy-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        // A noisy picture with a transparent half: lossy files are smaller, the transparency stays exact.
        let mut rgba = Rgba::new(64, 64);
        let mut seed = 7u32;
        for (i, p) in rgba.data.chunks_exact_mut(4).enumerate() {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            p.copy_from_slice(&[(seed >> 24) as u8, (seed >> 16) as u8, (seed >> 8) as u8, if i % 64 < 32 { 255 } else { 0 }]);
        }
        let size = |name: &str, q: u8| {
            save_rgba(&dir.join(name), &rgba, Format::from_path(Path::new(name)).unwrap(), q, [255; 3]).unwrap();
            std::fs::metadata(dir.join(name)).unwrap().len()
        };
        let (lossless, lossy) = (size("a.webp", 100), size("b.webp", 60));
        assert!(lossy < lossless, "{lossy} < {lossless}");
        let back = image::open(dir.join("b.webp")).unwrap().to_rgba8();
        assert_eq!((back.get_pixel(0, 0).0[3], back.get_pixel(40, 0).0[3]), (255, 0), "the transparency is kept exact");
        let exact = image::open(dir.join("a.webp")).unwrap().to_rgba8();
        assert_eq!(exact.get_pixel(3, 3).0[..3], rgba.data[(3 * 64 + 3) * 4..][..3], "100 is lossless");
        assert!(size("c.avif", 60) > 0);
        let head = std::fs::read(dir.join("c.avif")).unwrap();
        assert_eq!(&head[4..12], b"ftypavif", "an AVIF container");
        assert_eq!(Format::from_path(Path::new("x.AVIF")), Some(Format::Avif));
        assert!(Format::Avif.supports_alpha() && !IMAGE_EXTENSIONS.contains(&"avif"), "written, not read");
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn format_detection() {
        assert_eq!(Format::from_path(Path::new("x.JPEG")), Some(Format::Jpeg));
        assert_eq!(Format::from_path(Path::new("x.txt")), None);
        assert!(!Format::Jpeg.supports_alpha() && Format::Png.supports_alpha());
    }

    #[test]
    fn atomic_write_replaces_the_file_and_leaves_nothing_behind() {
        let dir = std::env::temp_dir().join(format!("osc-atomic-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("a.json");
        write_atomic(&file, b"first").unwrap();
        write_atomic(&file, b"second").unwrap();
        assert_eq!(std::fs::read(&file).unwrap(), b"second");
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 1, "no temporary file is left");
        // A folder that does not exist is an error, not a silent loss.
        assert!(write_atomic(&dir.join("missing").join("b.json"), b"x").is_err());
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn a_failed_export_keeps_the_previous_file_and_leaves_no_temporary_file() {
        let dir = std::env::temp_dir().join(format!("osc-save-atomic-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("out.png");
        std::fs::write(&file, b"previous export").unwrap();
        // A buffer whose length does not match its size cannot be encoded.
        let broken = Rgba { w: 4, h: 4, data: vec![0; 3] };
        assert!(save_rgba(&file, &broken, Format::Png, 95, [255; 3]).is_err());
        assert!(save_rgba(&file, &broken, Format::Jpeg, 95, [255; 3]).is_err());
        assert_eq!(std::fs::read(&file).unwrap(), b"previous export", "the old file is untouched");
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 1, "no temporary file is left");
        // A good export replaces it, in every format that is written.
        let ok = Rgba::new(3, 3);
        for (name, f) in
            [("a.png", Format::Png), ("a.webp", Format::Webp), ("a.tiff", Format::Tiff), ("a.jpg", Format::Jpeg), ("a.bmp", Format::Bmp)]
        {
            save_rgba(&dir.join(name), &ok, f, 90, [255; 3]).unwrap();
            assert_eq!(load_image(&dir.join(name)).unwrap().w, 3, "{name}");
        }
        let deep = Rgba16 { w: 2, h: 2, data: vec![40_000; 16] };
        save_rgba16(&dir.join("d.png"), &deep, Format::Png).unwrap();
        assert!(load_image_deep(&dir.join("d.png")).unwrap().deep.is_some());
        assert!(save_rgba16(&dir.join("d.bmp"), &deep, Format::Bmp).is_err());
        assert!(!dir.join("d.bmp").exists());
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn a_picture_is_accepted_only_if_it_fits_the_memory() {
        let gb = 1024;
        // 8K on a machine that allows 2 GB and has 6 GB free: about 760 MB.
        assert!(open_verdict(7680, 4320, Some(2 * gb), Some(6 * gb)).is_ok());
        // 100 megapixels need about 2.3 GB: over a 2 GB budget, fine under a 4 GB one.
        let over = open_verdict(10_000, 10_000, Some(2 * gb), Some(32 * gb)).unwrap_err().to_string();
        assert!(over.contains("2288 MB") && over.contains("2048 MB"), "{over}");
        assert!(open_verdict(10_000, 10_000, Some(4 * gb), Some(32 * gb)).is_ok());
        // Enough budget but not enough free memory.
        let busy = open_verdict(7680, 4320, Some(8 * gb), Some(300)).unwrap_err().to_string();
        assert!(busy.contains("free memory"), "{busy}");
        // Nothing known about the machine: only the absolute limit applies.
        assert!(open_verdict(20_000, 20_000, None, None).is_ok());
        assert!(open_verdict(30_000, 30_000, None, None).is_err(), "900 megapixels is not believable");
        assert!(open_verdict(0, 5, None, None).is_err());
        assert_eq!(open_cost_mb(1000, 1000), 22);
    }

    #[test]
    fn a_header_that_claims_a_huge_picture_is_refused_before_decoding() {
        // A PNG that only claims 30 000 x 30 000 pixels: its header is enough to be refused, nothing is allocated.
        fn chunk(kind: &[u8; 4], data: &[u8]) -> Vec<u8> {
            let mut crc = 0xFFFF_FFFFu32;
            for &b in kind.iter().chain(data) {
                crc ^= b as u32;
                for _ in 0..8 {
                    crc = if crc & 1 == 1 { (crc >> 1) ^ 0xEDB8_8320 } else { crc >> 1 };
                }
            }
            let mut v = (data.len() as u32).to_be_bytes().to_vec();
            v.extend_from_slice(kind);
            v.extend_from_slice(data);
            v.extend_from_slice(&(!crc).to_be_bytes());
            v
        }
        let mut ihdr = Vec::new();
        ihdr.extend_from_slice(&30_000u32.to_be_bytes());
        ihdr.extend_from_slice(&30_000u32.to_be_bytes());
        ihdr.extend_from_slice(&[8, 2, 0, 0, 0]); // 8 bits, RGB
        let mut png = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
        png.extend(chunk(b"IHDR", &ihdr));
        png.extend(chunk(b"IDAT", &[0x78, 0x9C, 0x03, 0x00, 0x00, 0x00, 0x00, 0x01]));
        png.extend(chunk(b"IEND", &[]));
        let file = std::env::temp_dir().join(format!("osc-huge-{}.png", std::process::id()));
        std::fs::write(&file, png).unwrap();
        let err = load_image_deep(&file).err().expect("refused").to_string();
        std::fs::remove_file(&file).ok();
        assert!(err.contains("30000 x 30000") && err.contains("megapixels"), "{err}");
    }
}
