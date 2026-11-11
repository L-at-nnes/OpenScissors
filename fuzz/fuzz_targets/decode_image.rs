//! Any bytes as a picture (PNG, JPEG, WebP, TIFF, BMP, GIF, with EXIF and colour profiles): an error, never a crash.
#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = openscissors_core::io::load_image_bytes(data);
});
