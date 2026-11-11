//! Any bytes as an `.osproj` project (JSON with pictures in base64): an error, never a crash. The loader reads a
//! file, so each input is written to one first.
#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let path = std::env::temp_dir().join(format!("osc-fuzz-{}.osproj", std::process::id()));
    if std::fs::write(&path, data).is_ok() {
        let _ = openscissors_core::project::load_project(&path);
    }
});
