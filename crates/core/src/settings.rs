//! Persisted user settings (`data/settings.json`), shared by the app and the command line.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::model::Accelerator;
use crate::params::Params;

/// Where the window was, in physical pixels, so it comes back at the same place and size.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WindowState {
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
    pub maximized: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    /// The model the user picked with "Use". Empty until then: the lightest one on disk is used.
    /// (Older versions stored an `active_model` that was also set automatically; it is ignored.)
    pub chosen_model: Option<String>,
    /// "Save Current as Defaults" target; used for new documents and for batch processing.
    pub defaults: Params,
    pub marker_size: u32,
    /// "Group by Color Similarity" stays on between sessions.
    pub group_by_color: bool,
    pub jpeg_quality: u8,
    /// Write 16-bit PNG/TIFF when the source is 16-bit and the settings allow it.
    pub export_16bit: bool,
    /// Ids of the plug-ins the user turned on (they never run by themselves).
    pub plugins_enabled: Vec<String>,
    /// Share of the physical memory the application may use: 25 (default) or 50.
    pub ram_percent: u8,
    /// Keep the memory arena on for speed (uses more than twice the memory).
    pub fast_memory: bool,
    /// Where models run: "cpu" (default), "gpu" or "auto".
    pub accelerator: Accelerator,
    /// Interface language: "en" (default), another code such as "fr", or "auto" (follow the system).
    pub locale: String,
    pub last_dir: Option<String>,
    pub batch_src: Option<String>,
    pub batch_dst: Option<String>,
    pub recent: Vec<String>,
    pub window: Option<WindowState>,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            chosen_model: None,
            defaults: Params::default(),
            marker_size: 30,
            group_by_color: false,
            jpeg_quality: 95,
            export_16bit: true,
            plugins_enabled: Vec::new(),
            ram_percent: 25,
            fast_memory: false,
            accelerator: Accelerator::Cpu,
            locale: "en".into(),
            last_dir: None,
            batch_src: None,
            batch_dst: None,
            recent: Vec::new(),
            window: None,
        }
    }
}

impl Settings {
    /// Read the settings file. A missing file gives the defaults. A value of the wrong type only resets that one
    /// field (the rest of the file is kept); when anything had to be dropped, the original file is first copied to
    /// `settings.json.bak` so nothing the user typed is lost for good.
    pub fn load(path: &Path) -> Self {
        let Ok(text) = std::fs::read_to_string(path) else { return Self::default().sanitized() };
        let (mut s, clean) = Self::from_json(&text);
        if !clean {
            let mut bak = path.as_os_str().to_owned();
            bak.push(".bak");
            match std::fs::copy(path, &bak) {
                Ok(_) => eprintln!("OpenScissors: some settings were invalid; the original file is kept in {}", Path::new(&bak).display()),
                Err(e) => eprintln!("OpenScissors: some settings were invalid and the backup failed: {e}"),
            }
        }
        s.sanitize();
        s
    }

    /// Parse `text` field by field, nested ones included. Returns the settings and whether every value was accepted.
    fn from_json(text: &str) -> (Self, bool) {
        let Ok(serde_json::Value::Object(file)) = serde_json::from_str(text) else { return (Self::default(), false) };
        let mut root = serde_json::to_value(Self::default()).expect("settings serialize");
        let mut clean = true;
        let mut leaves = Vec::new();
        collect_leaves(&root, &serde_json::Value::Object(file), &mut Vec::new(), &mut leaves);
        for (path, value) in leaves {
            let mut candidate = root.clone();
            let slot = path.iter().fold(&mut candidate, |v, k| &mut v[k.as_str()]);
            *slot = value;
            if serde_json::from_value::<Self>(candidate.clone()).is_ok() {
                root = candidate;
            } else {
                clean = false;
            }
        }
        (serde_json::from_value(root).unwrap_or_default(), clean)
    }

    fn sanitized(mut self) -> Self {
        self.sanitize();
        self
    }

    /// A settings file can be edited by hand: bring every value back into the range the interface allows.
    pub fn sanitize(&mut self) {
        self.marker_size = self.marker_size.clamp(1, 300);
        self.jpeg_quality = self.jpeg_quality.clamp(10, 100);
        if !matches!(self.ram_percent, 25 | 50) {
            self.ram_percent = 25;
        }
        self.defaults.sanitize();
        self.recent.truncate(10);
    }

    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let text = serde_json::to_string_pretty(self).map_err(std::io::Error::other)?;
        crate::io::write_atomic(path, text.as_bytes())
    }

    pub fn push_recent(&mut self, path: &str) {
        self.recent.retain(|p| p != path);
        self.recent.insert(0, path.to_string());
        self.recent.truncate(10);
    }
}

/// The values of `file` that can be applied one by one over `base`: it descends into the objects both have, and
/// stops at anything else (a number, a list, an unknown key is a single value).
fn collect_leaves(
    base: &serde_json::Value,
    file: &serde_json::Value,
    path: &mut Vec<String>,
    out: &mut Vec<(Vec<String>, serde_json::Value)>,
) {
    let (serde_json::Value::Object(b), serde_json::Value::Object(f)) = (base, file) else { return };
    for (key, value) in f {
        let Some(base_value) = b.get(key) else { continue };
        path.push(key.clone());
        if base_value.is_object() && value.is_object() {
            collect_leaves(base_value, value, path, out);
        } else {
            out.push((path.clone(), value.clone()));
        }
        path.pop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_and_tolerance_to_missing_or_broken_files() {
        let dir = std::env::temp_dir().join(format!("osc-settings-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("settings.json");
        assert_eq!(Settings::load(&file).marker_size, 30, "missing file gives defaults");
        let mut s = Settings { locale: "fr".into(), ..Settings::default() };
        s.defaults.shadow.enabled = true;
        s.push_recent("a.png");
        s.push_recent("b.png");
        s.push_recent("a.png");
        s.save(&file).unwrap();
        assert!(s.save(&dir.join("missing").join("s.json")).is_err(), "a failed save is reported");
        let back = Settings::load(&file);
        assert_eq!(back.locale, "fr");
        assert!(back.defaults.shadow.enabled);
        assert_eq!(back.recent, vec!["a.png", "b.png"]);
        std::fs::write(&file, "{ not json").unwrap();
        assert_eq!(Settings::load(&file).locale, "en", "broken file gives defaults");
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn out_of_range_values_are_brought_back_when_loading() {
        let dir = std::env::temp_dir().join(format!("osc-settings-range-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("settings.json");
        std::fs::write(&file, r#"{"marker_size":99999,"jpeg_quality":0,"ram_percent":77,"defaults":{"shadow":{"opacity":900}}}"#).unwrap();
        let s = Settings::load(&file);
        assert_eq!((s.marker_size, s.jpeg_quality, s.ram_percent), (300, 10, 25));
        assert_eq!(s.defaults.shadow.opacity, 100);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn one_wrong_value_only_resets_that_field_and_the_file_is_backed_up() {
        let dir = std::env::temp_dir().join(format!("osc-settings-bak-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("settings.json");
        let text = r#"{"locale":"fr","marker_size":"big","jpeg_quality":80,"recent":["a.png"],"unknown":1,
            "defaults":{"shadow":{"enabled":true,"opacity":"x"},"background":7}}"#;
        std::fs::write(&file, text).unwrap();
        let s = Settings::load(&file);
        assert_eq!((s.locale.as_str(), s.jpeg_quality, s.recent.len()), ("fr", 80, 1));
        assert_eq!(s.marker_size, 30, "the wrong value falls back to its default");
        assert!(s.defaults.shadow.enabled, "a good value next to a bad one is kept");
        assert_eq!(s.defaults.shadow.opacity, Params::default().shadow.opacity);
        assert_eq!(std::fs::read_to_string(dir.join("settings.json.bak")).unwrap(), text, "the original is saved");
        // A clean file leaves no backup, a file that is not JSON at all is also saved.
        let clean = dir.join("clean.json");
        std::fs::write(&clean, r#"{"locale":"de"}"#).unwrap();
        assert_eq!(Settings::load(&clean).locale, "de");
        assert!(!dir.join("clean.json.bak").exists());
        std::fs::write(&clean, "{ not json").unwrap();
        assert_eq!(Settings::load(&clean).locale, "en");
        assert_eq!(std::fs::read_to_string(dir.join("clean.json.bak")).unwrap(), "{ not json");
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn recent_list_is_capped() {
        let mut s = Settings::default();
        for i in 0..20 {
            s.push_recent(&format!("{i}.png"));
        }
        assert_eq!(s.recent.len(), 10);
        assert_eq!(s.recent[0], "19.png");
    }
}
