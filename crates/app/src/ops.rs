//! Document operations: open, save, export, projects, marks, properties, undo.

use openscissors_core::sync::LockExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use openscissors_core::document::Document;
use openscissors_core::io::{self, Format};
use openscissors_core::mask::Stroke;
use openscissors_core::params::{Params, ASPECT_RATIOS, OUTPUT_WIDTHS, PRESETS};
use openscissors_core::project::{self, ModelRef};
use openscissors_core::types::Rect;
use serde_json::{json, Value};

use crate::ctl::{Ctl, Prediction, Tool, KIND_KEYS, MODE_KEYS, R_ALL, R_MAIN, R_MENU, R_VIEW};
use crate::dialogs::{self, Answer, IMAGE_EXT};
use crate::i18n::{t, tf};

/// Reasons the exporter gives for a 16-bit source saved as 8-bit, as translatable keys.
fn why_8bit(reason: &str) -> String {
    match reason {
        "the background is a picture or the original" => t("the background is a picture or the original"),
        "a foreground effect is active" => t("a foreground effect is active"),
        "the shadow is on" => t("the shadow is on"),
        other => other.to_string(),
    }
}

const SAVE_FILTERS: &[(&str, &[&str])] =
    &[("PNG (transparency)", &["png"]), ("WebP", &["webp"]), ("JPEG", &["jpg", "jpeg"]), ("TIFF", &["tif", "tiff"]), ("BMP", &["bmp"])];

enum Exported {
    Bits8(openscissors_core::types::Rgba, [u8; 3]),
    Bits16(openscissors_core::types::Rgba16),
}

fn stem(name: Option<&str>, fallback: &str) -> String {
    name.map(|n| n.rsplit_once('.').map(|(s, _)| s).unwrap_or(n).to_string()).unwrap_or_else(|| fallback.to_string())
}

impl Ctl {
    /// Write the settings; a failure is shown, never silent (the next start would otherwise forget them).
    pub fn save_settings(self: &Arc<Self>) {
        if let Err(e) = self.st.save_settings() {
            self.say(tf("Cannot save the settings: {error}", &[("error", &e.to_string())]), true);
        }
    }

    pub fn doc_name(&self) -> Option<String> {
        self.ux.guard().doc.as_ref().and_then(|d| d.name.clone())
    }

    pub fn is_modified(&self) -> bool {
        self.ux.guard().doc.as_ref().is_some_and(|d| d.modified)
    }

    pub fn last_dir(&self) -> Option<String> {
        self.st.settings.guard().last_dir.clone()
    }

    /// Ask what to do with unsaved changes. `true` when it is fine to go on.
    pub fn confirm_discard_sync(self: &Arc<Self>) -> bool {
        if !self.is_modified() {
            return true;
        }
        match dialogs::confirm_discard() {
            Answer::Save => self.save_sync(true),
            Answer::Discard => true,
            Answer::Cancel => false,
        }
    }

    // -- opening

    pub fn open_dialog(self: &Arc<Self>) {
        let me = self.clone();
        std::thread::spawn(move || {
            if !me.confirm_discard_sync() {
                return;
            }
            let mut exts: Vec<&str> = IMAGE_EXT.to_vec();
            exts.push("osproj");
            if let Some(p) = dialogs::open_file(me.last_dir().as_deref(), &t("Open image"), &[("Images and projects", &exts)]) {
                me.open_path_sync(&p);
            }
        });
    }

    /// Open from a path (file dialog, recent list, drag and drop), asking about unsaved changes first.
    pub fn open_path(self: &Arc<Self>, path: PathBuf) {
        let me = self.clone();
        std::thread::spawn(move || {
            if me.confirm_discard_sync() {
                me.open_path_sync(&path);
            }
        });
    }

    pub fn open_path_sync(self: &Arc<Self>, path: &Path) {
        let is_project = path.extension().is_some_and(|e| e.eq_ignore_ascii_case("osproj"));
        let result = if is_project { self.open_project_sync(path) } else { self.open_image_sync(path) };
        match result {
            Ok(Some(warning)) => self.say(warning, true),
            Ok(None) => {
                let name = self.doc_name().unwrap_or_else(|| path.display().to_string());
                self.say(tf("Opened {name}", &[("name", &name)]), false);
            }
            Err(e) => self.say(e, true),
        }
    }

    fn after_open(self: &Arc<Self>, predicted: bool) {
        {
            let mut ux = self.ux.guard();
            ux.prompts.clear();
            ux.stroke = None;
            ux.live_crop = None;
            ux.tool = Tool::Move; // the brush is chosen on purpose, never the pointer's default
            ux.last_export = None;
            ux.prediction = if predicted { Prediction::Idle } else { Prediction::Running };
        }
        self.refresh_doc();
        self.fit();
        self.refresh(R_ALL);
        self.maybe_predict();
        // The model loads on the first image, so say what it is doing now.
        let (ready, loading, error) = {
            let s = self.st.model_status.guard();
            (s.ready, s.loading, s.error.is_some())
        };
        if !predicted && !ready && !loading {
            self.ux.guard().prediction = if error { Prediction::Failed } else { Prediction::NoModel };
        }
        self.refresh(R_MAIN | R_VIEW);
    }

    fn open_image_sync(self: &Arc<Self>, path: &Path) -> Result<Option<String>, String> {
        let loaded = io::load_image_deep(path).map_err(|e| format!("Cannot read file {}: {e}", file_label(path)))?;
        let id = self.st.next_doc_id.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let (mut defaults, group) = {
            let s = self.st.settings.guard();
            (s.defaults.clone(), s.group_by_color)
        };
        defaults.mask.graphcut = group;
        let doc = Document::new(id, loaded.image, Some(path.to_path_buf())).with_deep(loaded.deep).with_params(defaults);
        *self.st.doc.guard() = Some(doc);
        *self.st.sam_cache.guard() = None;
        *self.st.predicted_doc.guard() = None;
        {
            let mut s = self.st.settings.guard();
            s.push_recent(&path.display().to_string());
            s.last_dir = path.parent().map(|d| d.display().to_string());
        }
        self.save_settings();
        self.after_open(false);
        Ok(None)
    }

    fn open_project_sync(self: &Arc<Self>, path: &Path) -> Result<Option<String>, String> {
        let loaded = project::load_project(path).map_err(|e| format!("Cannot open project: {e}"))?;
        let id = self.st.next_doc_id.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let image_path = loaded.orig_path.map(PathBuf::from);
        let predicted = loaded.predicted.is_some();
        let doc = Document::restore(id, loaded.image, image_path, loaded.predicted, loaded.strokes, loaded.params);
        *self.st.doc.guard() = Some(doc);
        *self.st.predicted_doc.guard() = None;
        {
            let mut s = self.st.settings.guard();
            s.push_recent(&path.display().to_string());
        }
        self.save_settings();
        self.ux.guard().marker = loaded.marker_size;
        self.after_open(predicted);
        self.kick_alpha();
        Ok(loaded.warning)
    }

    // -- saving

    pub fn save(self: &Arc<Self>) {
        let me = self.clone();
        std::thread::spawn(move || {
            me.save_sync(false);
        });
    }

    pub fn save_as(self: &Arc<Self>) {
        let me = self.clone();
        std::thread::spawn(move || {
            me.save_sync(true);
        });
    }

    /// Export the result: to the last file unless `ask`; returns whether a file was written.
    pub fn save_sync(self: &Arc<Self>, ask: bool) -> bool {
        let (name, last) = {
            let ux = self.ux.guard();
            (ux.doc.as_ref().and_then(|d| d.name.clone()), ux.last_export.clone())
        };
        if self.ux.guard().doc.is_none() {
            return false;
        }
        let path = match (ask, last) {
            (false, Some(p)) => PathBuf::from(p),
            _ => {
                let file = format!("{}.png", stem(name.as_deref(), "cutout"));
                match dialogs::save_file(self.last_dir().as_deref(), &t("Save as…"), &file, SAVE_FILTERS) {
                    Some(p) => p,
                    None => return false,
                }
            }
        };
        match self.export_to(&path) {
            Ok(()) => true,
            Err(e) => {
                self.say(e, true);
                false
            }
        }
    }

    pub fn export_to(self: &Arc<Self>, target: &Path) -> Result<(), String> {
        let format = Format::from_path(target).ok_or("unsupported file type; use png, jpg, webp, bmp or tiff")?;
        let (quality, want16) = {
            let s = self.st.settings.guard();
            (s.jpeg_quality, s.export_16bit && matches!(format, Format::Png | Format::Tiff))
        };
        let (exported, why) = {
            let mut g = self.st.doc.guard();
            let d = g.as_mut().ok_or("no image is open")?;
            if let Some(deep) = want16.then(|| d.export_deep()).flatten() {
                (Exported::Bits16(deep), None)
            } else {
                let flatten = if matches!(d.params().background.mode, openscissors_core::params::BgMode::Color) {
                    d.params().background.color
                } else {
                    [255, 255, 255]
                };
                let why = if want16 { d.deep_blocker() } else { None };
                (Exported::Bits8(d.export(), flatten), why)
            }
        };
        let bits16 = matches!(exported, Exported::Bits16(_));
        match exported {
            Exported::Bits16(img) => io::save_rgba16(target, &img, format),
            Exported::Bits8(img, flatten) => io::save_rgba(target, &img, format, quality, flatten),
        }
        .map_err(|e| e.to_string())?;
        self.st.settings.guard().last_dir = target.parent().map(|d| d.display().to_string());
        self.save_settings();
        let (transparent, name) = {
            let mut g = self.st.doc.guard();
            if let Some(d) = g.as_mut() {
                d.mark_clean();
            }
            (
                g.as_ref().is_some_and(|d| matches!(d.params().background.mode, openscissors_core::params::BgMode::Transparent)),
                file_label(target),
            )
        };
        {
            let mut ux = self.ux.guard();
            ux.last_export = Some(target.display().to_string());
            ux.last_saved = Some(target.display().to_string());
        }
        self.refresh_doc();
        self.refresh(R_MENU);
        let is_jpeg = matches!(format, Format::Jpeg);
        let msg = if is_jpeg && transparent {
            tf("Saved {name}. JPEG has no transparency: the background was filled with white.", &[("name", &name)])
        } else if bits16 {
            tf("Saved {name} (16-bit)", &[("name", &name)])
        } else if let Some(r) = why {
            tf("Saved {name} as 8-bit: {reason}", &[("name", &name), ("reason", &why_8bit(r))])
        } else {
            tf("Saved {name}", &[("name", &name)])
        };
        self.say(msg, false);
        Ok(())
    }

    pub fn save_project_dialog(self: &Arc<Self>) {
        let me = self.clone();
        std::thread::spawn(move || {
            let name = me.doc_name();
            let file = format!("{}.osproj", stem(name.as_deref(), "project"));
            let Some(mut target) =
                dialogs::save_file(me.last_dir().as_deref(), &t("Save project…"), &file, &[("OpenScissors project", &["osproj"])])
            else {
                return;
            };
            if !target.extension().is_some_and(|e| e.eq_ignore_ascii_case("osproj")) {
                target.set_extension("osproj");
            }
            let marker = me.st.settings.guard().marker_size;
            let model = me.active_model_ref();
            let result = {
                let mut g = me.st.doc.guard();
                match g.as_mut() {
                    Some(d) => project::save_project(d, model, marker, &target).map(|_| d.mark_clean()).map_err(|e| e.to_string()),
                    None => Err("no image is open".to_string()),
                }
            };
            match result {
                Ok(()) => {
                    me.refresh_doc();
                    me.say(tf("Saved {name}", &[("name", &file_label(&target))]), false);
                }
                Err(e) => me.say(e, true),
            }
        });
    }

    fn active_model_ref(&self) -> Option<ModelRef> {
        let id = self.st.model_status.guard().active.clone()?;
        let sha256 = self.st.models().into_iter().find(|s| s.id == id).map(|s| s.sha256).unwrap_or_default();
        Some(ModelRef { id, sha256 })
    }

    pub fn load_marks_dialog(self: &Arc<Self>) {
        let me = self.clone();
        std::thread::spawn(move || {
            let Some(p) = dialogs::open_file(me.last_dir().as_deref(), &t("Load marks…"), &[("Marks", &["psm"])]) else { return };
            let result = std::fs::read_to_string(&p).map_err(|e| e.to_string()).and_then(|text| {
                let mut g = me.st.doc.guard();
                let d = g.as_mut().ok_or("no image is open")?;
                let strokes = project::marks_from_json(&text, d.image.w, d.image.h).map_err(|e| e.to_string())?;
                d.load_marks(strokes);
                Ok::<_, String>(())
            });
            match result {
                Ok(()) => {
                    me.refresh_doc();
                    me.kick_alpha();
                    me.say(t("Marks loaded"), false);
                }
                Err(e) => me.say(e, true),
            }
        });
    }

    pub fn save_marks_dialog(self: &Arc<Self>) {
        let me = self.clone();
        std::thread::spawn(move || {
            let file = format!("{}.psm", stem(me.doc_name().as_deref(), "marks"));
            let Some(mut target) = dialogs::save_file(me.last_dir().as_deref(), &t("Save marks…"), &file, &[("Marks", &["psm"])]) else {
                return;
            };
            if !target.extension().is_some_and(|e| e.eq_ignore_ascii_case("psm")) {
                target.set_extension("psm");
            }
            let result = {
                let g = me.st.doc.guard();
                match g.as_ref() {
                    Some(d) => {
                        std::fs::write(&target, project::marks_to_json(d.strokes(), d.image.w, d.image.h)).map_err(|e| e.to_string())
                    }
                    None => Err("no image is open".to_string()),
                }
            };
            match result {
                Ok(()) => me.say(tf("Saved {name}", &[("name", &file_label(&target))]), false),
                Err(e) => me.say(e, true),
            }
        });
    }

    // -- editing

    pub fn undo(self: &Arc<Self>) {
        if let Some(d) = self.st.doc.guard().as_mut() {
            d.undo();
        }
        self.refresh_doc();
        self.kick_alpha();
    }

    pub fn redo(self: &Arc<Self>) {
        if let Some(d) = self.st.doc.guard().as_mut() {
            d.redo();
        }
        self.refresh_doc();
        self.kick_alpha();
    }

    pub fn add_stroke(self: &Arc<Self>, tool: Tool, width: u32, points: Vec<(f64, f64)>) {
        let Some(kind) = tool.stroke_kind() else { return };
        if points.is_empty() {
            return;
        }
        let stroke = Stroke { kind, width, points: points.into_iter().map(|(x, y)| [x as f32, y as f32]).collect() };
        if let Some(d) = self.st.doc.guard().as_mut() {
            d.add_stroke(stroke);
        }
        self.refresh_doc();
        self.kick_alpha();
    }

    /// Change one property of the result (JSON path, as the panel names it).
    pub fn set_param(self: &Arc<Self>, path: &str, value: Value) {
        let result = {
            let mut g = self.st.doc.guard();
            match g.as_mut() {
                Some(d) => d.set_param(path, value).map_err(|e| e.to_string()),
                None => return,
            }
        };
        if let Err(e) = result {
            self.say(e, true);
        }
        self.refresh_doc();
        self.kick_alpha();
    }

    pub fn set_index(self: &Arc<Self>, path: &str, i: i32) {
        let i = i.max(0) as usize;
        let value = if path.ends_with(".kind") {
            KIND_KEYS.get(i).map(|k| json!(k))
        } else if path == "background.mode" {
            MODE_KEYS.get(i).map(|k| json!(k))
        } else if path == "background.aspect_ratio" {
            ASPECT_RATIOS.get(i).map(|(k, _)| json!(k))
        } else if path == "background.output_width" {
            OUTPUT_WIDTHS.get(i).map(|w| json!(w))
        } else {
            None
        };
        if let Some(v) = value {
            self.set_param(path, v);
        }
    }

    /// Apply preset `i` of the list (0 is "Custom": nothing to do) as one undoable change.
    pub fn apply_preset(self: &Arc<Self>, i: i32) {
        let Some(preset) = usize::try_from(i - 1).ok().and_then(|n| PRESETS.get(n)) else { return };
        {
            let mut g = self.st.doc.guard();
            let Some(d) = g.as_mut() else { return };
            let mut p = d.params().clone();
            preset.apply(&mut p);
            d.set_params(p);
        }
        self.refresh_doc();
        self.kick_alpha();
    }

    pub fn set_crop(self: &Arc<Self>, rect: Option<Rect>) {
        self.set_param("crop", rect.map(|r| json!({"x0": r.x0, "y0": r.y0, "x1": r.x1, "y1": r.y1})).unwrap_or(Value::Null));
    }

    pub fn pick_bg_image(self: &Arc<Self>) {
        let me = self.clone();
        std::thread::spawn(move || {
            if let Some(p) = dialogs::open_file(me.last_dir().as_deref(), &t("Choose image…"), &[("Images", IMAGE_EXT)]) {
                me.set_param("background.image_path", json!(p.display().to_string()));
                me.set_param("background.mode", json!("image"));
            }
        });
    }

    /// Reset one tab of the right panel (`tab`: 0 foreground, 1 background, 2 shadow).
    pub fn reset_tab(self: &Arc<Self>, tab: i32, factory: bool) {
        let source = if factory { Params::default() } else { self.st.settings.guard().defaults.clone() };
        {
            let mut g = self.st.doc.guard();
            let Some(d) = g.as_mut() else { return };
            let mut p = d.params().clone();
            match tab {
                0 => {
                    p.mask = source.mask;
                    p.foreground = source.foreground;
                }
                1 => {
                    let picture = p.background.image_path.take();
                    p.background = source.background;
                    p.background.image_path = picture; // keep the chosen picture so the user can switch back
                    p.crop = None;
                }
                _ => p.shadow = source.shadow,
            }
            d.set_params(p);
        }
        self.refresh_doc();
        self.kick_alpha();
    }

    pub fn save_defaults(self: &Arc<Self>) {
        let params = self.st.doc.guard().as_ref().map(|d| d.params().clone());
        if let Some(mut params) = params {
            params.crop = None;
            params.background.image_path = None;
            self.st.settings.guard().defaults = params;
            self.save_settings();
            self.say(t("Current settings saved as defaults"), false);
        }
    }

    pub fn toggle_graphcut(self: &Arc<Self>) {
        let on = {
            let ux = self.ux.guard();
            let Some(d) = &ux.doc else { return };
            !d.params.mask.graphcut
        };
        self.set_param("mask.graphcut", json!(on));
        {
            let mut ux = self.ux.guard();
            if !on && ux.tool == Tool::Remove {
                ux.tool = Tool::Keep;
            }
        }
        self.st.settings.guard().group_by_color = on;
        self.save_settings();
        self.refresh(R_MAIN);
    }

    pub fn set_tool(self: &Arc<Self>, tool: Tool) {
        let cancel = {
            let mut ux = self.ux.guard();
            if tool == Tool::Remove && !ux.doc.as_ref().is_some_and(|d| d.params.mask.graphcut) {
                return; // the red tool only exists in "group by color" mode
            }
            let cancel = ux.tool == Tool::Select && tool != Tool::Select;
            ux.tool = tool;
            ux.render_key = None; // the hair marks show on the left view in the hair tools
            cancel
        };
        if cancel {
            self.smart_cancel();
        }
        self.refresh(R_MAIN | R_VIEW);
    }

    pub fn set_marker(self: &Arc<Self>, n: i32) {
        let n = n.clamp(1, 300) as u32;
        {
            let mut ux = self.ux.guard();
            ux.marker = n;
        }
        self.st.settings.guard().marker_size = n;
        self.save_settings();
        self.refresh(R_MAIN | R_VIEW);
    }

    // -- batch

    pub fn pick_folder(self: &Arc<Self>, source: bool) {
        let me = self.clone();
        std::thread::spawn(move || {
            let start = {
                let s = me.st.settings.guard();
                (if source { s.batch_src.clone() } else { s.batch_dst.clone() }).or_else(|| s.last_dir.clone())
            };
            let title = if source { t("Source folder") } else { t("Destination folder") };
            if let Some(p) = dialogs::pick_folder(start.as_deref(), &title) {
                {
                    let mut s = me.st.settings.guard();
                    let v = Some(p.display().to_string());
                    if source {
                        s.batch_src = v;
                    } else {
                        s.batch_dst = v;
                    }
                }
                me.save_settings();
                me.refresh(crate::ctl::R_BATCH);
            }
        });
    }

    pub fn run_batch(self: &Arc<Self>) {
        use openscissors_core::batch::{self, BatchOptions};
        let Some(model) = self.st.model.guard().clone() else {
            self.say(t("No AI model is ready. Choose and download a model first."), true);
            return;
        };
        let (src, dst, defaults, quality) = {
            let s = self.st.settings.guard();
            (s.batch_src.clone().unwrap_or_default(), s.batch_dst.clone().unwrap_or_default(), s.defaults.clone(), s.jpeg_quality)
        };
        // The look of the image being edited, when asked for and there is one; otherwise the saved defaults.
        let open_params = if self.ux.guard().batch.settings == 1 { self.st.doc.guard().as_ref().map(|d| d.params().clone()) } else { None };
        let params = open_params.unwrap_or(defaults);
        let format = if self.ux.guard().batch.format == 1 { Format::Jpeg } else { Format::Png };
        let (src, dst) = (PathBuf::from(src), PathBuf::from(dst));
        let files = match batch::validate(&src, &dst) {
            Ok(f) => f,
            Err(e) => return self.say(e.to_string(), true),
        };
        let cancel = Arc::new(std::sync::atomic::AtomicBool::new(false));
        {
            let mut running = self.st.batch.guard();
            if running.is_some() {
                return self.say("A batch is already running.", true);
            }
            *running = Some(cancel.clone());
        }
        {
            let mut ux = self.ux.guard();
            ux.batch.running = true;
            ux.batch.done = 0;
            ux.batch.total = files.len();
            ux.batch.report.clear();
            ux.batch.file = None;
        }
        self.refresh(crate::ctl::R_BATCH);
        let me = self.clone();
        std::thread::spawn(move || {
            let opts = BatchOptions { dst, format, quality, params };
            let predict = |img: &openscissors_core::types::Image| model.predict(img, true);
            let report = batch::run_batch(
                &files,
                &opts,
                &predict,
                &mut |p| {
                    {
                        let mut ux = me.ux.guard();
                        ux.batch.done = p.done;
                        ux.batch.total = p.total;
                        ux.batch.file = p.current.and_then(|c| c.file_name()).map(|n| n.to_string_lossy().into_owned());
                    }
                    me.refresh(crate::ctl::R_BATCH);
                },
                &cancel,
            );
            *me.st.batch.guard() = None;
            let mut text = format!(
                "{}. {}",
                if report.cancelled { t("Cancelled") } else { t("Done") },
                tf("{n} processed, {m} failed.", &[("n", &report.processed.to_string()), ("m", &report.failed.len().to_string())])
            );
            for f in report.failed.iter().take(5) {
                text.push_str(&format!("\n{} {}", f.file, f.error));
            }
            {
                let mut ux = me.ux.guard();
                ux.batch.running = false;
                ux.batch.file = None;
                ux.batch.report = text;
            }
            me.refresh(crate::ctl::R_BATCH);
        });
    }

    pub fn stop_batch(&self) {
        if let Some(flag) = self.st.batch.guard().as_ref() {
            flag.store(true, std::sync::atomic::Ordering::Relaxed);
        }
    }

    // -- plug-ins

    pub fn plugin_toggle(self: &Arc<Self>, index: usize, enabled: bool) {
        let me = self.clone();
        std::thread::spawn(move || {
            let found = openscissors_core::plugin::discover(&me.st.paths.plugins);
            let Some(Ok(p)) = found.into_iter().nth(index).map(|d| d.plugin) else { return };
            if enabled {
                let ok = dialogs::confirm(
                    &tf("Turn on {name}?", &[("name", &p.manifest.name)]),
                    &t("A plug-in is a program you installed. It runs only when you click Run, on the file you saved."),
                    t("Turn on"),
                    t("Cancel"),
                );
                if !ok {
                    me.refresh(R_MENU);
                    return;
                }
            }
            {
                let mut s = me.st.settings.guard();
                s.plugins_enabled.retain(|x| x != &p.manifest.id);
                if enabled {
                    s.plugins_enabled.push(p.manifest.id.clone());
                }
            }
            me.save_settings();
            me.refresh(R_MENU);
        });
    }

    pub fn plugin_run(self: &Arc<Self>, index: usize) {
        let me = self.clone();
        std::thread::spawn(move || {
            let Some(file) = me.ux.guard().last_saved.clone() else { return };
            let found = openscissors_core::plugin::discover(&me.st.paths.plugins);
            let Some(Ok(p)) = found.into_iter().nth(index).map(|d| d.plugin) else { return };
            if !me.st.settings.guard().plugins_enabled.contains(&p.manifest.id) {
                return me.say("This plug-in is not enabled.", true);
            }
            let name = p.manifest.name.clone();
            me.say(tf("Running {name}…", &[("name", &name)]), false);
            match openscissors_core::plugin::run(&p, Path::new(&file), openscissors_core::plugin::RUN_TIMEOUT) {
                Ok(r) => {
                    let last = r.output.lines().rfind(|l| !l.trim().is_empty()).unwrap_or("").to_string();
                    if r.success {
                        me.say(
                            if last.is_empty() {
                                tf("{name} finished", &[("name", &name)])
                            } else {
                                tf("{name}: {output}", &[("name", &name), ("output", &last)])
                            },
                            false,
                        );
                    } else {
                        me.say(tf("{name} failed: {output}", &[("name", &name), ("output", &last)]), true);
                    }
                }
                Err(e) => me.say(e.to_string(), true),
            }
        });
    }

    // -- settings

    pub fn set_language(self: &Arc<Self>, index: usize) {
        let setting = if index == 0 {
            "auto".to_string()
        } else {
            crate::i18n::LOCALES.get(index - 1).map(|(c, _)| c.to_string()).unwrap_or_else(|| "auto".into())
        };
        let code = crate::i18n::apply(&setting);
        let _ = slint::select_bundled_translation(code);
        self.st.settings.guard().locale = setting;
        self.save_settings();
        self.refresh(R_ALL);
    }

    pub fn clear_recent(self: &Arc<Self>) {
        self.st.settings.guard().recent.clear();
        self.save_settings();
        self.refresh(R_MENU);
    }

    pub fn set_jpeg_quality(self: &Arc<Self>, q: i32) {
        self.st.settings.guard().jpeg_quality = q.clamp(10, 100) as u8;
        self.save_settings();
    }

    pub fn set_export16(self: &Arc<Self>, on: bool) {
        self.st.settings.guard().export_16bit = on;
        self.save_settings();
    }

    pub fn set_drop_active(self: &Arc<Self>, on: bool) {
        self.ux.guard().drop_active = on;
        self.refresh(R_MAIN);
    }
}

pub fn file_label(path: &Path) -> String {
    path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| path.display().to_string())
}
