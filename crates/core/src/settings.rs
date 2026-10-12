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
    pub fn load(path: &Path) -> Self {
        std::fs::read_to_string(path).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_default()
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
    fn recent_list_is_capped() {
        let mut s = Settings::default();
        for i in 0..20 {
            s.push_recent(&format!("{i}.png"));
        }
        assert_eq!(s.recent.len(), 10);
        assert_eq!(s.recent[0], "19.png");
    }
}
