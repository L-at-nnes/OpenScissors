//! The system clipboard: paste an image to open it, copy the result.
//!
//! One `Clipboard` lives for the whole run: on Linux (X11) the data we put there is served by the process that owns
//! it, so dropping the handle would empty the clipboard.

use std::borrow::Cow;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

use arboard::{Clipboard, Error, ImageData};
use openscissors_core::io::flatten;
use openscissors_core::sync::LockExt;
use openscissors_core::types::{Image, Rgba};

/// What a paste found.
pub enum Pasted {
    Image(Image),
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

/// The picture on the clipboard, drawn over white where it is transparent; or the files copied in a file manager.
pub fn read() -> Result<Pasted, ClipboardError> {
    with(|cb| match cb.get_image() {
        Ok(img) => Ok(Pasted::Image(from_rgba(img.width, img.height, &img.bytes).map_err(|_| Error::ConversionFailure)?)),
        Err(Error::ContentNotAvailable) => cb.get().file_list().map(Pasted::Files),
        Err(e) => Err(e),
    })
}

/// Put `img` (straight alpha) on the clipboard.
pub fn write(img: &Rgba) -> Result<(), ClipboardError> {
    with(|cb| cb.set_image(ImageData { width: img.w, height: img.h, bytes: Cow::Borrowed(&img.data) }))
}

/// An RGBA buffer (as the clipboard hands it out) as an opaque image.
fn from_rgba(w: usize, h: usize, bytes: &[u8]) -> Result<Image, String> {
    if w == 0 || h == 0 || bytes.len() != w * h * 4 {
        return Err("the clipboard picture has an unexpected size".into());
    }
    Ok(flatten(&Rgba { w, h, data: bytes.to_vec() }, [255, 255, 255]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_pasted_picture_is_drawn_over_white() {
        // One opaque red pixel, one fully transparent pixel, one half transparent black pixel.
        let img = from_rgba(3, 1, &[200, 0, 0, 255, 10, 20, 30, 0, 0, 0, 0, 128]).unwrap();
        assert_eq!((img.w, img.h), (3, 1));
        assert_eq!(&img.data[0..3], &[200, 0, 0]);
        assert_eq!(&img.data[3..6], &[255, 255, 255], "transparent becomes white, not the hidden colour");
        assert!(img.data[6] > 100 && img.data[6] < 155, "half transparent black is mid grey: {}", img.data[6]);
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
            Ok(Pasted::Image(back)) => assert_eq!(back.data, vec![200, 0, 0, 0, 0, 200]),
            _ => panic!("the picture did not come back"),
        }
    }
}
