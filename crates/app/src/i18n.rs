//! Interface translations. Keys are the English source strings; a missing key shows English.
//! The `.slint` files use `@tr("key")` (compiled from the same JSON files by the build script);
//! Rust code uses `t("key")` and `tf("key", &[("name", value)])`, always with a string literal:
//! a test scans the sources for them.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::OnceLock;

pub const LOCALES: &[(&str, &str)] = &[
    ("en", "English"),
    ("fr", "Français"),
    ("de", "Deutsch"),
    ("es", "Español"),
    ("it", "Italiano"),
    ("pt-BR", "Português (Brasil)"),
    ("ru", "Русский"),
    ("zh", "简体中文"),
    ("ja", "日本語"),
];

const SOURCES: &[(&str, &str)] = &[
    ("fr", include_str!("../locales/fr.json")),
    ("de", include_str!("../locales/de.json")),
    ("es", include_str!("../locales/es.json")),
    ("it", include_str!("../locales/it.json")),
    ("pt-BR", include_str!("../locales/pt-BR.json")),
    ("ru", include_str!("../locales/ru.json")),
    ("zh", include_str!("../locales/zh.json")),
    ("ja", include_str!("../locales/ja.json")),
];

static CURRENT: AtomicUsize = AtomicUsize::new(0);

fn dictionaries() -> &'static HashMap<&'static str, HashMap<String, String>> {
    static D: OnceLock<HashMap<&'static str, HashMap<String, String>>> = OnceLock::new();
    D.get_or_init(|| SOURCES.iter().map(|(code, text)| (*code, serde_json::from_str(text).unwrap_or_default())).collect())
}

/// Best supported locale for a language tag such as "fr-CA" or "pt-PT".
pub fn resolve(tag: &str) -> &'static str {
    let tag = tag.replace('_', "-");
    if let Some((code, _)) = LOCALES.iter().find(|(c, _)| c.eq_ignore_ascii_case(&tag)) {
        return code;
    }
    // Otherwise the same language from another country: pt-PT gives pt-BR, zh-TW gives zh.
    let base = |t: &str| t.split('-').next().unwrap_or("").to_ascii_lowercase();
    LOCALES.iter().find(|(c, _)| base(c) == base(&tag)).map(|(c, _)| *c).unwrap_or("en")
}

/// Index in [`LOCALES`] of the locale for a setting ("auto" follows the system), and its code.
pub fn apply(setting: &str) -> &'static str {
    let tag = if setting == "auto" { sys_locale::get_locale().unwrap_or_else(|| "en".into()) } else { setting.to_string() };
    let code = resolve(&tag);
    let idx = LOCALES.iter().position(|(c, _)| *c == code).unwrap_or(0);
    CURRENT.store(idx, Ordering::Relaxed);
    code
}

/// The font of the interface for the language `code`: Chinese and Japanese each need a font of their own (the
/// fallback of the system font lacks the characters of one or the other).
pub fn font_family(code: &str) -> &'static str {
    match code {
        "zh" if cfg!(windows) => "Microsoft YaHei UI",
        "zh" if cfg!(target_os = "macos") => "PingFang SC",
        "zh" => "Noto Sans CJK SC",
        "ja" if cfg!(windows) => "Yu Gothic UI",
        "ja" if cfg!(target_os = "macos") => "Hiragino Sans",
        "ja" => "Noto Sans CJK JP",
        _ => "Segoe UI",
    }
}

pub fn code() -> &'static str {
    LOCALES[CURRENT.load(Ordering::Relaxed)].0
}

/// Translate `key`.
pub fn t(key: &str) -> String {
    t_in(code(), key)
}

/// [`t`] in the language `lang`.
fn t_in(lang: &str, key: &str) -> String {
    dictionaries().get(lang).and_then(|d| d.get(key)).cloned().unwrap_or_else(|| key.to_string())
}

/// Messages that come from the core, in English, with `{name}` for their variable parts. They reach the interface as
/// error texts and are translated by [`tr_message`].
pub const CORE_MESSAGES: &[&str] = &[
    // opening, memory
    "the picture is empty",
    "the picture is {w} x {h} pixels; the application accepts up to {max} megapixels",
    "a {w} x {h} px picture needs about {need} MB, more than the {budget} MB the application may use (the memory limit in the settings can be raised)",
    "not enough free memory for a {w} x {h} px picture: about {need} MB are needed and {free} MB are free; close other applications",
    // export
    "this export needs about {need} MB more, over the {budget} MB the application may use (the memory limit in the settings can be raised, or the output made smaller)",
    "not enough free memory: this export needs about {need} MB and {free} MB are free; close other applications",
    "WebP cannot hold a {w} x {h} px picture (at most {max} px per side): choose PNG, TIFF or AVIF, or an output width",
    "the background is a picture, a gradient or the original",
    "a fixed output width is set",
    "the colour adjustments are on",
    "the edge clean-up is on",
    "a foreground effect is active",
    "the shadow is on",
    "the reflection is on",
    // batch
    "The source folder does not exist.",
    "The source and destination folders must be different.",
    "The source folder contains no supported image (jpg, png, bmp, tif, webp, gif, jxl).",
    r#"The suffix cannot hold \ / : * ? " < > | (it is added to the file names)."#,
    // export
    "A largest file size works with JPEG, WebP and AVIF.",
    "A PSD file holds at most {max} px per side; this picture is {w} x {h} px.",
    "The result does not hold in {kb} KB even at the lowest quality: choose a smaller output size.",
    // projects and marks
    "project file is too large",
    "this is not an OpenScissors project",
    "unsupported image encoding '{encoding}'",
    "corrupt image data: {error}",
    "the stored prediction does not match the image size",
    "the stored 16-bit picture does not match the image size",
    "This project was made by a newer version (format {version}); some settings may be ignored.",
    "unsupported marks file version",
    "these marks were made for an image of a different size",
    // models and downloads
    "GPU acceleration is not available on this platform yet",
    "refusing to delete the folder of model '{id}'",
    "file '{file}' of model '{id}' has no pinned sha256",
    "model URL must use HTTPS",
    "model '{id}' names a folder or a file outside the models folder",
    "the server announces {total} bytes for '{file}', the catalogue says {size}",
    "nothing was received for {seconds} seconds; the download can be resumed",
    "the connection was lost",
    "'{file}' is larger than the {size} bytes the catalogue says",
    "the connection closed after {done} of {total} bytes; the download can be resumed",
    "give at least one point or box",
    "too many prompts",
    // plug-ins
    "the file to process does not exist",
    "this plug-in handles: {extensions}",
    "the plug-in did not finish within {seconds} seconds and was stopped",
    "no plugin.json",
    "plugin.json: {error}",
    "the plug-in id may only use a-z, 0-9, '-' and '_'",
    "the command must be a file inside the plug-in folder",
    "command file '{file}' not found",
    "the command resolves outside the plug-in folder",
    // the kinds of errors (`openscissors_core::Error`), around a message of their own
    "I/O error: {error}",
    "image error: {error}",
    "download failed: {error}",
    "checksum mismatch for model '{id}'",
    "model runtime error: {error}",
    "JSON error: {error}",
];

/// One piece of a message template: text as it is, or a `{name}` variable.
enum Piece<'a> {
    Text(&'a str),
    Var(&'a str),
}

/// The pieces of `template`. A `{` that does not open a `{name}` (letters, digits, `_`) is text.
fn pieces(template: &str) -> Vec<Piece<'_>> {
    let mut out = Vec::new();
    let (mut start, mut i) = (0, 0);
    let bytes = template.as_bytes();
    while i < bytes.len() {
        if bytes[i] == b'{' {
            let len = template[i + 1..].find(|c: char| !(c.is_ascii_alphanumeric() || c == '_')).unwrap_or(0);
            if len > 0 && template[i + 1 + len..].starts_with('}') {
                if start < i {
                    out.push(Piece::Text(&template[start..i]));
                }
                out.push(Piece::Var(&template[i + 1..i + 1 + len]));
                i += len + 2;
                start = i;
                continue;
            }
        }
        i += 1;
    }
    if start < template.len() {
        out.push(Piece::Text(&template[start..]));
    }
    out
}

/// The values of the variables of `template` in `msg`, when `msg` is that template filled in.
fn match_template(template: &str, msg: &str) -> Option<Vec<(String, String)>> {
    let pieces = pieces(template);
    let mut rest = msg;
    let mut vars = Vec::new();
    for (i, piece) in pieces.iter().enumerate() {
        match piece {
            Piece::Text(t) => rest = rest.strip_prefix(t)?,
            Piece::Var(name) => {
                let end = match pieces.get(i + 1) {
                    Some(Piece::Text(next)) => rest.find(next)?,
                    Some(Piece::Var(_)) => return None,
                    None => rest.len(),
                };
                vars.push((name.to_string(), rest[..end].to_string()));
                rest = &rest[end..];
            }
        }
    }
    rest.is_empty().then_some(vars)
}

/// Translate a message of the core (see [`CORE_MESSAGES`]): the template it fills in is translated and the variable
/// parts are put back, each translated too when it is itself a message ("I/O error: …"). Anything else is returned
/// as it is (a message of the system is already in its language).
pub fn tr_message(msg: &str) -> String {
    tr_message_in(code(), msg)
}

/// [`tr_message`] in the language `lang`.
fn tr_message_in(lang: &str, msg: &str) -> String {
    for template in CORE_MESSAGES {
        if let Some(vars) = match_template(template, msg) {
            let vars: Vec<(String, String)> = vars.into_iter().map(|(k, v)| (k, tr_message_in(lang, &v))).collect();
            let refs: Vec<(&str, &str)> = vars.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
            return fill(&t_in(lang, template), &refs);
        }
    }
    msg.to_string()
}

/// Translate `key`, replacing `{name}` placeholders.
pub fn tf(key: &str, vars: &[(&str, &str)]) -> String {
    fill(&t(key), vars)
}

/// `text` with its `{name}` placeholders replaced by `vars`, in one pass: a value is never searched again, so a file
/// named `{error}.png` stays as it is.
fn fill(text: &str, vars: &[(&str, &str)]) -> String {
    let mut out = String::with_capacity(text.len());
    for piece in pieces(text) {
        match piece {
            Piece::Text(t) => out.push_str(t),
            Piece::Var(name) => match vars.iter().find(|(k, _)| *k == name) {
                Some((_, v)) => out.push_str(v),
                None => {
                    out.push('{');
                    out.push_str(name);
                    out.push('}');
                }
            },
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;
    use std::path::Path;

    /// String literals given to `t(`, `tf(` or `@tr(` in the sources.
    fn used_keys() -> BTreeSet<String> {
        fn files(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
            for e in std::fs::read_dir(dir).unwrap().flatten() {
                let p = e.path();
                if p.is_dir() {
                    files(&p, out);
                } else if matches!(p.extension().and_then(|x| x.to_str()), Some("rs" | "slint")) {
                    out.push(p);
                }
            }
        }
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let mut paths = Vec::new();
        files(&root.join("src"), &mut paths);
        files(&root.join("ui"), &mut paths);
        let mut keys = BTreeSet::new();
        for p in paths {
            if p.ends_with("i18n.rs") {
                continue;
            }
            let text = std::fs::read_to_string(&p).unwrap();
            for marker in ["t(\"", "tf(\"", "@tr(\""] {
                let mut rest = text.as_str();
                while let Some(i) = rest.find(marker) {
                    // `t(` must start a call, not end another identifier (`set(`, `format(` ...).
                    let before = rest[..i].chars().last();
                    rest = &rest[i + marker.len()..];
                    if marker == "t(\"" && before.is_some_and(|c| c.is_alphanumeric() || c == '_') {
                        continue;
                    }
                    let mut key = String::new();
                    let mut chars = rest.chars();
                    while let Some(c) = chars.next() {
                        match c {
                            '\\' => match chars.next() {
                                Some('n') => key.push('\n'),
                                Some(o) => key.push(o),
                                None => break,
                            },
                            '"' => break,
                            c => key.push(c),
                        }
                    }
                    keys.insert(key);
                }
            }
        }
        // Catalogue descriptions are translated at run time with `t(&description)`, core messages with `tr_message`.
        for spec in openscissors_core::model::builtin_catalog() {
            keys.insert(spec.description);
        }
        keys.extend(CORE_MESSAGES.iter().map(|m| m.to_string()));
        // The steps of the history panel are named by the core and translated with `t(label)`.
        keys.extend(openscissors_core::document::HISTORY_LABELS.iter().map(|l| l.to_string()));
        keys
    }

    fn placeholders(s: &str) -> Vec<String> {
        let mut out = Vec::new();
        let mut rest = s;
        while let Some(i) = rest.find('{') {
            if let Some(j) = rest[i..].find('}') {
                out.push(rest[i + 1..i + j].to_string());
                rest = &rest[i + j + 1..];
            } else {
                break;
            }
        }
        out.sort();
        out
    }

    #[test]
    fn the_sources_use_a_sensible_number_of_keys() {
        assert!(used_keys().len() > 100);
    }

    #[test]
    fn every_language_has_every_key_used_and_nothing_unused() {
        let keys = used_keys();
        for (code, text) in SOURCES {
            let dict: HashMap<String, String> = serde_json::from_str(text).unwrap();
            let missing: Vec<_> = keys.iter().filter(|k| !dict.contains_key(*k)).collect();
            let unused: Vec<_> = dict.keys().filter(|k| !keys.contains(*k)).collect();
            assert!(missing.is_empty(), "{code} misses {missing:?}");
            assert!(unused.is_empty(), "{code} has unused {unused:?}");
        }
    }

    #[test]
    fn translations_keep_placeholders_and_shortcuts_and_really_translate() {
        let keys = used_keys();
        for (code, text) in SOURCES {
            let dict: HashMap<String, String> = serde_json::from_str(text).unwrap();
            let mut same = 0;
            for k in &keys {
                let v = &dict[k];
                assert_eq!(placeholders(v), placeholders(k), "{code}: {k}");
                assert!(!v.trim().is_empty(), "{code}: {k}");
                for part in k.split("Ctrl+").skip(1) {
                    let shortcut: String = "Ctrl+".to_string() + &part.chars().take(1).collect::<String>();
                    assert!(v.contains(&shortcut), "{code}: {k}");
                }
                if v == k {
                    same += 1;
                }
            }
            assert!((same as f64) / (keys.len() as f64) < 0.2, "{code} looks untranslated");
        }
    }

    #[test]
    fn messages_of_the_core_are_translated_with_their_values() {
        // In French, without changing the language of the process (other tests read English texts meanwhile).
        let fr = |m: &str| tr_message_in("fr", m);
        let t = |k: &str| t_in("fr", k);
        // Real messages from the core.
        let missing =
            openscissors_core::batch::validate(Path::new("/no/such/folder"), &std::env::temp_dir(), false).unwrap_err().to_string();
        assert_eq!(fr(&missing), t("The source folder does not exist."));
        assert_ne!(fr(&missing), missing);
        let marks = openscissors_core::project::marks_from_json(r#"{"version":1}"#, 1, 1).unwrap_err().to_string();
        assert_ne!(fr(&marks), marks);
        let empty = openscissors_core::io::check_dimensions(0, 5).unwrap_err().to_string();
        assert_ne!(fr(&empty), empty);
        // Values are put back, and a message inside another is translated too.
        let big = "the picture is 30000 x 30000 pixels; the application accepts up to 500 megapixels";
        let got = fr(big);
        assert!(got.contains("30000") && got.contains("500") && got != big, "{got}");
        let io = openscissors_core::io::load_image(Path::new("/no/such/picture.png")).unwrap_err().to_string();
        assert!(io.starts_with("I/O error: ") && !fr(&io).starts_with("I/O error: "), "{io} -> {}", fr(&io));
        let wrapped = fr("download failed: the connection was lost");
        assert_eq!(wrapped, fill(&t("download failed: {error}"), &[("error", &t("the connection was lost"))]));
        // A message that is not one of them stays as it is.
        assert_eq!(fr("Accès refusé. (os error 5)"), "Accès refusé. (os error 5)");
        assert_eq!(fr("unbalanced '{' in args"), "unbalanced '{' in args");
    }

    #[test]
    fn a_value_that_looks_like_a_placeholder_is_left_as_it_is() {
        let text = fill("Cannot read file {file}: {error}", &[("file", "{error}.png"), ("error", "denied")]);
        assert_eq!(text, "Cannot read file {error}.png: denied");
        assert_eq!(fill("{a} and {a}, {b} {unknown}", &[("a", "1"), ("b", "2")]), "1 and 1, 2 {unknown}");
    }

    #[test]
    fn templates_match_only_what_they_describe() {
        assert_eq!(
            match_template("'{file}' is larger than the {size} bytes", "'a.onnx' is larger than the 12 bytes"),
            Some(vec![("file".into(), "a.onnx".into()), ("size".into(), "12".into())])
        );
        assert_eq!(match_template("the picture is empty", "the picture is empty!"), None);
        assert_eq!(match_template("x {a} y", "x 1 z"), None);
    }

    #[test]
    fn resolves_language_tags() {
        assert_eq!(resolve("fr-CA"), "fr");
        assert_eq!(resolve("de_AT"), "de");
        assert_eq!(resolve("ru-RU"), "ru");
        assert_eq!(resolve("pt-PT"), "pt-BR", "the same language from another country");
        assert_eq!((resolve("pt-br"), resolve("es-MX"), resolve("it-IT")), ("pt-BR", "es", "it"));
        assert_eq!((resolve("zh-CN"), resolve("zh-Hans-CN"), resolve("ja-JP")), ("zh", "zh", "ja"));
        assert_eq!(resolve("ko-KR"), "en", "not translated: English");
        assert_eq!(resolve("EN-us"), "en");
    }
}
