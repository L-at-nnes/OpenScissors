use std::fmt::Write as _;
use std::path::{Path, PathBuf};

/// Escape a string for a gettext catalogue.
fn esc(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n")
}

fn write_po(dir: &Path, entries: &[(String, String)]) {
    let mut po = String::from("msgid \"\"\nmsgstr \"Content-Type: text/plain; charset=UTF-8\\n\"\n\n");
    for (k, v) in entries {
        let _ = write!(po, "msgid \"{}\"\nmsgstr \"{}\"\n\n", esc(k), esc(v));
    }
    let dir = dir.join("LC_MESSAGES");
    std::fs::create_dir_all(&dir).expect("po dir");
    std::fs::write(dir.join("openscissors.po"), po).expect("write po");
}

/// Writes one gettext catalogue per language from `locales/*.json` (keys are the English strings),
/// so the `.slint` files translate with `@tr("...")` from the same source as the Rust code.
fn write_catalogues(out: &Path) {
    let locales = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("manifest dir")).join("locales");
    println!("cargo:rerun-if-changed={}", locales.display());
    let mut english: Vec<(String, String)> = Vec::new();
    for entry in std::fs::read_dir(&locales).expect("locales dir").flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        let code = path.file_stem().and_then(|s| s.to_str()).expect("locale code").to_string();
        let map: serde_json::Map<String, serde_json::Value> =
            serde_json::from_str(&std::fs::read_to_string(&path).expect("read locale")).expect("valid json");
        let entries: Vec<(String, String)> = map.iter().map(|(k, v)| (k.clone(), v.as_str().unwrap_or(k).to_string())).collect();
        write_po(&out.join(&code), &entries);
        if english.is_empty() {
            english = map.keys().map(|k| (k.clone(), k.clone())).collect();
        }
    }
    // English: every string is its own translation.
    write_po(&out.join("en"), &english);
}

fn main() {
    let out = PathBuf::from(std::env::var("OUT_DIR").expect("out dir")).join("po");
    write_catalogues(&out);
    let config = slint_build::CompilerConfiguration::new()
        .with_bundled_translations(&out)
        .with_default_translation_context(slint_build::DefaultTranslationContext::None);
    slint_build::compile_with_config("ui/main.slint", config).expect("slint build");

    #[cfg(windows)]
    {
        let mut res = winresource::WindowsResource::new();
        res.set_icon("assets/icon.ico");
        res.set("ProductName", "OpenScissors");
        res.set("FileDescription", "OpenScissors");
        let _ = res.compile();
    }
}
