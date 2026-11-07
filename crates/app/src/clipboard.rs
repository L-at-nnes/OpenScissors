//! The system clipboard: paste an image to open it, copy the result.
//!
//! One `Clipboard` lives for the whole run: on Linux (X11) the data we put there is served by the process that owns
//! it, so dropping the handle would empty the clipboard.

use std::borrow::Cow;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

use arboard::{Clipboard, Error, ImageData};
use openscissors_core::sync::LockExt;
use openscissors_core::types::{Image, Mask, Rgba};

/// What a paste found.
pub enum Pasted {
    /// A picture, with its transparency when it has some (a picture that is already cut out).
    Image(Image, Option<Mask>),
    /// Files copied in a file manager.
    Files(Vec<PathBuf>),
}

pub enum ClipboardError {
    /// Nothing usable on the clipboard.
    Empty,
    /// The system refused (the text is the system's own).
    System(String),
}

impl From<Error> for ClipboardError {
    fn from(e: Error) -> Self {
        match e {
            Error::ContentNotAvailable => Self::Empty,
            other => Self::System(other.to_string()),
        }
    }
}

fn shared() -> &'static Mutex<Option<Clipboard>> {
    static CLIPBOARD: OnceLock<Mutex<Option<Clipboard>>> = OnceLock::new();
    CLIPBOARD.get_or_init(|| Mutex::new(None))
}

fn with<R>(f: impl FnOnce(&mut Clipboard) -> Result<R, Error>) -> Result<R, ClipboardError> {
    let mut slot = shared().guard();
    if slot.is_none() {
        *slot = Some(Clipboard::new()?);
    }
    Ok(f(slot.as_mut().expect("just created"))?)
}

/// The picture on the clipboard (its transparency kept, as when a cut-out file is opened); or the files copied in a
/// file manager.
pub fn read() -> Result<Pasted, ClipboardError> {
    enum Found {
        Rgba(usize, usize, Vec<u8>),
        Files(Vec<PathBuf>),
    }
    let found = with(|cb| match cb.get_image() {
        Ok(img) => Ok(Found::Rgba(img.width, img.height, img.bytes.into_owned())),
        Err(Error::ContentNotAvailable) => cb.get().file_list().map(Found::Files),
        Err(e) => Err(e),
    })?;
    match found {
        Found::Rgba(w, h, bytes) => from_rgba(w, h, &bytes).map(|(img, alpha)| Pasted::Image(img, alpha)).map_err(ClipboardError::System),
        Found::Files(files) => Ok(Pasted::Files(files)),
    }
}

/// Put `img` (straight alpha) on the clipboard.
pub fn write(img: &Rgba) -> Result<(), ClipboardError> {
    with(|cb| cb.set_image(ImageData { width: img.w, height: img.h, bytes: Cow::Borrowed(&img.data) }))
}

/// An RGBA buffer (as the clipboard hands it out) as a picture and, when it is transparent somewhere, its alpha.
fn from_rgba(w: usize, h: usize, bytes: &[u8]) -> Result<(Image, Option<Mask>), String> {
    openscissors_core::io::check_dimensions(w as u64, h as u64).map_err(|e| e.to_string())?;
    if bytes.len() != w * h * 4 {
        return Err("the clipboard picture has an unexpected size".into());
    }
    let pixels = bytes.as_chunks::<4>().0;
    let rgb = pixels.iter().flat_map(|p| [p[0], p[1], p[2]]).collect();
    let alpha: Vec<u8> = pixels.iter().map(|p| p[3]).collect();
    let transparent = alpha.iter().any(|&a| a < 255);
    Ok((Image::from_raw(w, h, rgb), transparent.then(|| Mask::from_raw(w, h, alpha))))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_pasted_cut_out_keeps_its_transparency_and_an_opaque_picture_has_none() {
        // One opaque red pixel, one fully transparent pixel, one half transparent black pixel.
        let (img, alpha) = from_rgba(3, 1, &[200, 0, 0, 255, 10, 20, 30, 0, 0, 0, 0, 128]).unwrap();
        assert_eq!((img.w, img.h), (3, 1));
        assert_eq!(img.data, vec![200, 0, 0, 10, 20, 30, 0, 0, 0], "the colours as they are, as for an opened file");
        assert_eq!(alpha.unwrap().data, vec![255, 0, 128]);
        let (_, none) = from_rgba(1, 1, &[1, 2, 3, 255]).unwrap();
        assert!(none.is_none(), "an opaque picture is not a cut-out");
    }

    #[test]
    fn a_picture_that_does_not_match_its_size_is_refused() {
        assert!(from_rgba(2, 2, &[0; 15]).is_err());
        assert!(from_rgba(0, 4, &[]).is_err());
    }

    /// Needs a desktop session and replaces what is on the clipboard: `cargo test -p openscissors -- --ignored clipboard`.
    #[test]
    #[ignore]
    fn the_result_makes_the_round_trip_through_the_system_clipboard() {
        let img = Rgba { w: 2, h: 1, data: vec![200, 0, 0, 255, 0, 0, 200, 255] };
        assert!(write(&img).is_ok());
        match read() {
            Ok(Pasted::Image(back, _)) => assert_eq!(back.data, vec![200, 0, 0, 0, 0, 200]),
            _ => panic!("the picture did not come back"),
        }
    }
}
