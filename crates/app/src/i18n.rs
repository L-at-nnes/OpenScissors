//! Interface translations. Keys are the English source strings; a missing key shows English.
//! The `.slint` files use `@tr("key")` (compiled from the same JSON files by the build script);
//! Rust code uses `t("key")` and `tf("key", &[("name", value)])`, always with a string literal:
//! a test scans the sources for them.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::OnceLock;

pub const LOCALES: &[(&str, &str)] = &[("en", "English"), ("fr", "Français"), ("de", "Deutsch"), ("ru", "Русский")];

const SOURCES: &[(&str, &str)] =
    &[("fr", include_str!("../locales/fr.json")), ("de", include_str!("../locales/de.json")), ("ru", include_str!("../locales/ru.json"))];

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
    let base = tag.split('-').next().unwrap_or("").to_ascii_lowercase();
    LOCALES.iter().find(|(c, _)| c.eq_ignore_ascii_case(&base)).map(|(c, _)| *c).unwrap_or("en")
}

/// Index in [`LOCALES`] of the locale for a setting ("auto" follows the system), and its code.
pub fn apply(setting: &str) -> &'static str {
    let tag = if setting == "auto" { sys_locale::get_locale().unwrap_or_else(|| "en".into()) } else { setting.to_string() };
    let code = resolve(&tag);
    let idx = LOCALES.iter().position(|(c, _)| *c == code).unwrap_or(0);
    CURRENT.store(idx, Ordering::Relaxed);
    code
}

pub fn code() -> &'static str {
    LOCALES[CURRENT.load(Ordering::Relaxed)].0
}

/// Translate `key`.
pub fn t(key: &str) -> String {
    dictionaries().get(code()).and_then(|d| d.get(key)).cloned().unwrap_or_else(|| key.to_string())
}

/// Translate `key`, replacing `{name}` placeholders.
pub fn tf(key: &str, vars: &[(&str, &str)]) -> String {
    let mut text = t(key);
    for (k, v) in vars {
        text = text.replace(&format!("{{{k}}}"), v);
    }
    text
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
        // Catalogue descriptions are translated at run time with `t(&description)`.
        for spec in openscissors_core::model::builtin_catalog() {
            keys.insert(spec.description);
        }
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
    fn resolves_language_tags() {
        assert_eq!(resolve("fr-CA"), "fr");
        assert_eq!(resolve("de_AT"), "de");
        assert_eq!(resolve("ru-RU"), "ru");
        assert_eq!(resolve("pt-PT"), "en"); // not translated: English
        assert_eq!(resolve("zh-CN"), "en");
        assert_eq!(resolve("EN-us"), "en");
    }
}
