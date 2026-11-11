//! Any bytes as `settings.json` (a file edited by hand): the readable values are kept, the rest reset, never a crash.
#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let dir = std::env::temp_dir().join(format!("osc-fuzz-settings-{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("settings.json");
    if std::fs::write(&path, data).is_ok() {
        let mut s = openscissors_core::settings::Settings::load(&path);
        s.sanitize();
    }
});
