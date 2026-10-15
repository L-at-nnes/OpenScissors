//! Image file input/output.

use std::io::BufWriter;
use std::path::Path;

use image::codecs::jpeg::JpegEncoder;
use image::{DynamicImage, ImageDecoder, ImageEncoder, ImageReader};
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::types::{Image, Image16, Rgba, Rgba16};

pub const IMAGE_EXTENSIONS: &[&str] = &["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"];

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
}

/// Read an image as 8-bit RGB. EXIF orientation is applied; alpha is dropped.
/// Write `bytes` to `path` so that a crash or a full disk never leaves a half-written file: the data goes to a
/// temporary file in the same folder, is flushed to disk, then replaces the target in one step.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    use std::io::Write;
    let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let tmp = path.with_file_name(format!(".{name}.{}.tmp", std::process::id()));
    let result = (|| {
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(bytes)?;
        f.sync_all()?;
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
    let mut img = DynamicImage::from_decoder(decoder)?;
    img.apply_orientation(orientation);
    let is_deep = matches!(
        img,
        DynamicImage::ImageRgb16(_) | DynamicImage::ImageRgba16(_) | DynamicImage::ImageLuma16(_) | DynamicImage::ImageLumaA16(_)
    );
    let deep = is_deep.then(|| {
        let d = img.to_rgb16();
        Image16 { w: d.width() as usize, h: d.height() as usize, data: d.into_raw() }
    });
    let rgb = img.to_rgb8();
    Ok(Loaded { image: Image { w: rgb.width() as usize, h: rgb.height() as usize, data: rgb.into_raw() }, deep })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Format {
    Png,
    Jpeg,
    Webp,
    Bmp,
    Tiff,
}

impl Format {
    pub fn from_path(path: &Path) -> Option<Self> {
        match path.extension()?.to_str()?.to_ascii_lowercase().as_str() {
            "png" => Some(Self::Png),
            "jpg" | "jpeg" => Some(Self::Jpeg),
            "webp" => Some(Self::Webp),
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

/// Save `rgba` in `format`. Formats without alpha are flattened on `flatten_color`.
/// `quality` (10-100) only applies to JPEG; WebP is written lossless.
pub fn save_rgba(path: &Path, rgba: &Rgba, format: Format, quality: u8, flatten_color: [u8; 3]) -> Result<()> {
    let (w, h) = (rgba.w as u32, rgba.h as u32);
    if format.supports_alpha() {
        let buf = image::RgbaImage::from_raw(w, h, rgba.data.clone()).ok_or_else(|| Error::msg("bad RGBA buffer"))?;
        match format {
            Format::Png => {
                let f = BufWriter::new(std::fs::File::create(path)?);
                image::codecs::png::PngEncoder::new(f).write_image(buf.as_raw(), w, h, image::ExtendedColorType::Rgba8)?;
            }
            Format::Webp => {
                let f = BufWriter::new(std::fs::File::create(path)?);
                image::codecs::webp::WebPEncoder::new_lossless(f).write_image(buf.as_raw(), w, h, image::ExtendedColorType::Rgba8)?;
            }
            _ => DynamicImage::ImageRgba8(buf).save_with_format(path, image::ImageFormat::Tiff)?,
        }
    } else {
        let flat = flatten(rgba, flatten_color);
        let buf = image::RgbImage::from_raw(w, h, flat.data).ok_or_else(|| Error::msg("bad RGB buffer"))?;
        match format {
            Format::Jpeg => {
                let f = BufWriter::new(std::fs::File::create(path)?);
                JpegEncoder::new_with_quality(f, quality.clamp(10, 100)).write_image(buf.as_raw(), w, h, image::ExtendedColorType::Rgb8)?;
            }
            _ => DynamicImage::ImageRgb8(buf).save_with_format(path, image::ImageFormat::Bmp)?,
        }
    }
    Ok(())
}

/// Save 16-bit RGBA as PNG or TIFF.
pub fn save_rgba16(path: &Path, rgba: &Rgba16, format: Format) -> Result<()> {
    let (w, h) = (rgba.w as u32, rgba.h as u32);
    let bytes: Vec<u8> = rgba.data.iter().flat_map(|v| v.to_ne_bytes()).collect();
    let file = BufWriter::new(std::fs::File::create(path)?);
    match format {
        Format::Png => image::codecs::png::PngEncoder::new(file).write_image(&bytes, w, h, image::ExtendedColorType::Rgba16)?,
        Format::Tiff => image::codecs::tiff::TiffEncoder::new(file).write_image(&bytes, w, h, image::ExtendedColorType::Rgba16)?,
        _ => return Err(Error::msg("16-bit output needs PNG or TIFF")),
    }
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
