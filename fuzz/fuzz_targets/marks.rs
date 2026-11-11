//! Any text as a marks file (`.psm`) for a picture of the size given by the first bytes: an error, never a crash.
#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() < 2 {
        return;
    }
    let (w, h) = (usize::from(data[0]) + 1, usize::from(data[1]) + 1);
    if let Ok(text) = std::str::from_utf8(&data[2..]) {
        let _ = openscissors_core::project::marks_from_json(text, w, h);
    }
});
