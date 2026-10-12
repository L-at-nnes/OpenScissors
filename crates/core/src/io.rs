//! Image file input/output.

use std::io::BufWriter;
use std::path::Path;

use image::codecs::jpeg::JpegEncoder;
use image::{DynamicImage, ImageDecoder, ImageEncoder, ImageReader};
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::types::{Image, Image16, Rgba, Rgba16};

pub const IMAGE_EXTENSIONS: &[&str] = &["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"];

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
    let mut decoder = ImageReader::open(path)?.with_guessed_format()?.into_decoder()?;
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
}
