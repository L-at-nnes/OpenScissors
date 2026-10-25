//! Document model: source image, AI prediction, strokes and layer properties,
//! with a single undo stack. The prediction itself is an undoable command.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::Serialize;
use serde_json::Value;

use crate::error::{Error, Result};
use crate::imaging::morph::{morph_ellipse, MorphOp};
use crate::io::load_image;
use crate::mask::compose::{apply_stroke, compose_alpha};
use crate::mask::graphcut::{graphcut_alpha, GraphcutOptions};
use crate::mask::matting::hair_matting;
use crate::mask::strokes::{erased_mask, paint_stroke, rasterize, Stroke, StrokeKind};
use crate::params::{BgMode, Params};
use crate::render::{self, crop, inpaint, Scene, SourceLayers};
use crate::types::{Image, Image16, Mask, Rect, Rgba, Rgba16};

/// Consecutive edits of the same property closer than this collapse into one undo step.
const MERGE_WINDOW: Duration = Duration::from_millis(800);

/// Most memory the undo history may hold. Each AI prediction or smart selection keeps a full-size mask; on a large
/// photo a long session would otherwise grow without limit. The oldest steps are forgotten first.
const UNDO_LIMIT_BYTES: usize = 384 << 20;

enum Command {
    AddStroke(Stroke),
    SetPrediction {
        old: Option<Arc<Mask>>,
        new: Option<Arc<Mask>>,
    },
    SetParam {
        path: String,
        old: Value,
        new: Value,
        at: Instant,
    },
    SetParams {
        old: Box<Params>,
        new: Box<Params>,
    },
    /// Replace every stroke (loading a marks file).
    SetStrokes {
        old: Vec<Stroke>,
        new: Vec<Stroke>,
    },
}

/// What the UI needs to know after every change.
#[derive(Clone, Debug, Serialize)]
pub struct DocState {
    pub id: u64,
    pub rev: u64,
    pub width: usize,
    pub height: usize,
    pub name: Option<String>,
    pub can_undo: bool,
    pub can_redo: bool,
    pub modified: bool,
    pub has_prediction: bool,
    /// The mask is being recomputed in the background (graph-cut or hair matting); the views
    /// show a provisional mask until `rev` changes.
    pub computing: bool,
    pub params: Params,
    /// Crop rectangle in effect (manual, ratio-constrained or fitted to the result).
    pub crop: Rect,
    /// The background picture the parameters ask for could not be read (the path); nothing is drawn for it and the
    /// load is not retried until the path changes.
    pub bg_error: Option<String>,
}

pub struct Document {
    pub id: u64,
    pub image: Arc<Image>,
    /// The 16-bit samples when the source file had them (see `export_deep`).
    deep: Option<Arc<Image16>>,
    path: Option<PathBuf>,
    predicted: Option<Arc<Mask>>,
    strokes: Vec<Stroke>,
    params: Params,
    undo: Vec<Command>,
    undo_limit: usize,
    index: usize,
    clean: Option<usize>,
    /// Bumped on every visible change; the UI uses it to know when to re-fetch pixels.
    rev: u64,
    /// Bumped only when the composed mask changes.
    mask_rev: u64,
    alpha: Option<Mask>,
    marks: Option<Mask>,
    /// Remove marks (255), shown in red in graph-cut mode.
    bg_marks: Option<Mask>,
    /// Hair marks (255) with hair erasers removing them.
    hair_marks: Option<Mask>,
    /// Pending smart selection (not part of the document until applied; not undoable).
    selection: Option<Mask>,
    /// `mask_rev` for which `alpha` is the exact result (not a provisional one).
    alpha_exact: u64,
    job_running: bool,
    bg: Option<(String, Image)>,
    /// Path of a background picture that failed to load, so that `prepare()` does not try again at every call.
    bg_failed: Option<String>,
    filled: Option<(FillKey, Image)>,
    /// The canvas of the result (the crop) for `rev`, kept while a background picture has to cover it.
    frame: Option<(u64, Rect)>,
}

type FillKey = (u64, i32, u32);

/// Rough size in bytes of a JSON value, without printing it.
fn json_size(v: &Value) -> usize {
    match v {
        Value::String(s) => s.len(),
        Value::Array(a) => a.iter().map(|x| 8 + json_size(x)).sum(),
        Value::Object(o) => o.iter().map(|(k, x)| 8 + k.len() + json_size(x)).sum(),
        _ => 8,
    }
}

fn set_json(root: &mut Value, path: &str, v: Value) -> Result<Value> {
    let ptr = format!("/{}", path.replace('.', "/"));
    let slot = root.pointer_mut(&ptr).ok_or_else(|| Error::msg(format!("unknown property '{path}'")))?;
    Ok(std::mem::replace(slot, v))
}

impl Document {
    pub fn new(id: u64, image: Image, path: Option<PathBuf>) -> Self {
        Self {
            id,
            image: Arc::new(image),
            deep: None,
            path,
            predicted: None,
            strokes: Vec::new(),
            params: Params::default(),
            undo: Vec::new(),
            undo_limit: UNDO_LIMIT_BYTES,
            index: 0,
            clean: Some(0),
            rev: 0,
            mask_rev: 0,
            alpha: None,
            marks: None,
            bg_marks: None,
            hair_marks: None,
            selection: None,
            alpha_exact: u64::MAX,
            job_running: false,
            bg: None,
            bg_failed: None,
            filled: None,
            frame: None,
        }
    }

    /// Rebuild a document from saved parts (opening a project). The undo history starts empty
    /// and the document counts as unmodified.
    pub fn restore(id: u64, image: Image, path: Option<PathBuf>, predicted: Option<Mask>, strokes: Vec<Stroke>, params: Params) -> Self {
        let mut doc = Self::new(id, image, path).with_params(params);
        doc.predicted = predicted.map(Arc::new);
        doc.strokes = strokes;
        doc
    }

    /// Attach the 16-bit version of the source image (same size as the 8-bit one).
    pub fn with_deep(mut self, deep: Option<Image16>) -> Self {
        self.deep = deep.filter(|d| (d.w, d.h) == (self.image.w, self.image.h)).map(Arc::new);
        self
    }

    pub fn has_deep_source(&self) -> bool {
        self.deep.is_some()
    }

    pub fn with_params(mut self, params: Params) -> Self {
        self.params = params;
        self
    }

    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }
    pub fn params(&self) -> &Params {
        &self.params
    }
    pub fn strokes(&self) -> &[Stroke] {
        &self.strokes
    }
    /// Show (or clear) the pending smart selection. It is a preview: it does not change the
    /// document, the undo stack or the modified flag.
    pub fn set_selection(&mut self, selection: Option<Mask>) {
        self.selection = selection;
        self.rev += 1;
    }

    pub fn has_selection(&self) -> bool {
        self.selection.is_some()
    }

    /// Fold the pending selection into the AI prediction as one undoable step. Returns false
    /// when there is no selection.
    pub fn apply_selection(&mut self, mode: SelectionMode) -> bool {
        let Some(sel) = self.selection.take() else { return false };
        let (w, h) = (self.image.w, self.image.h);
        let base = self.predicted.as_deref().cloned().unwrap_or_else(|| Mask::new(w, h));
        let data = match mode {
            SelectionMode::Replace => sel.data,
            SelectionMode::Add => base.data.iter().zip(&sel.data).map(|(&b, &s)| b.max(s)).collect(),
            SelectionMode::Subtract => {
                base.data.iter().zip(&sel.data).map(|(&b, &s)| ((b as u32 * (255 - s as u32) + 127) / 255) as u8).collect()
            }
        };
        self.set_prediction(Some(Mask { w, h, data }));
        true
    }

    pub fn predicted(&self) -> Option<&Mask> {
        self.predicted.as_deref()
    }
    pub fn has_prediction(&self) -> bool {
        self.predicted.is_some()
    }
    pub fn rev(&self) -> u64 {
        self.rev
    }
    pub fn can_undo(&self) -> bool {
        self.index > 0
    }
    pub fn can_redo(&self) -> bool {
        self.index < self.undo.len()
    }
    pub fn is_clean(&self) -> bool {
        self.clean == Some(self.index)
    }
    pub fn mark_clean(&mut self) {
        self.clean = Some(self.index);
        self.rev += 1;
    }

    // -- commands

    fn push(&mut self, cmd: Command) {
        self.undo.truncate(self.index);
        if matches!(self.clean, Some(c) if c > self.index) {
            self.clean = None; // the saved state lived in the discarded redo branch
        }
        self.apply(&cmd, true);
        self.undo.push(cmd);
        self.index += 1;
        self.trim_history();
    }

    /// Memory held by the undo history. A mask shared by two steps (the "new" of one is the "old" of the next) counts once.
    fn history_bytes(&self) -> usize {
        let mut seen: std::collections::HashSet<*const Mask> = std::collections::HashSet::new();
        let mut mask = |m: &Option<Arc<Mask>>| match m {
            Some(m) if seen.insert(Arc::as_ptr(m)) => m.data.len(),
            _ => 0,
        };
        let strokes = |v: &[Stroke]| v.iter().map(|s| 32 + s.points.len() * 8).sum::<usize>();
        let params = std::mem::size_of::<Params>() + 256;
        self.undo
            .iter()
            .map(|c| match c {
                Command::AddStroke(s) => strokes(std::slice::from_ref(s)),
                Command::SetPrediction { old, new } => mask(old) + mask(new),
                Command::SetParam { path, old, new, .. } => path.len() + 2 * std::mem::size_of::<Value>() + json_size(old) + json_size(new),
                Command::SetParams { .. } => 2 * params,
                Command::SetStrokes { old, new } => strokes(old) + strokes(new),
            })
            .sum()
    }

    /// Forget the oldest steps while the history is over its memory limit. The newest step always stays, so the
    /// last edit can still be undone.
    fn trim_history(&mut self) {
        while self.undo.len() > 1 && self.history_bytes() > self.undo_limit {
            self.undo.remove(0);
            self.index = self.index.saturating_sub(1);
            // The saved state is the one before step `c`; once step 0 is gone, state 0 cannot be reached again.
            self.clean = match self.clean {
                Some(c) if c > 0 => Some(c - 1),
                _ => None,
            };
        }
    }

    pub fn add_stroke(&mut self, stroke: Stroke) {
        self.push(Command::AddStroke(stroke));
    }

    /// Replace all strokes with the ones of a marks file, as one undoable step.
    pub fn load_marks(&mut self, strokes: Vec<Stroke>) {
        let old = self.strokes.clone();
        self.push(Command::SetStrokes { old, new: strokes });
    }

    pub fn set_prediction(&mut self, mask: Option<Mask>) {
        let old = self.predicted.clone();
        self.push(Command::SetPrediction { old, new: mask.map(Arc::new) });
    }

    /// Set one property, addressed by a dotted path such as `background.mode`.
    pub fn set_param(&mut self, path: &str, value: Value) -> Result<()> {
        if path == "background.image_path" {
            // Choosing the picture again is how a failed load is retried.
            self.bg_failed = None;
        }
        let mut root = serde_json::to_value(&self.params)?;
        let old = set_json(&mut root, path, value.clone())?;
        if old == value {
            return Ok(());
        }
        let mut next: Params = serde_json::from_value(root).map_err(|e| Error::msg(format!("invalid value for '{path}': {e}")))?;
        next.sanitize();
        if path == "background.image_path" {
            self.load_bg(next.background.image_path.clone())?;
        }
        let now = Instant::now();
        let top = self.index.checked_sub(1);
        let mergeable = self.index == self.undo.len() && self.clean != Some(self.index);
        if let (true, Some(i)) = (mergeable, top) {
            if let Command::SetParam { path: p, new, at, .. } = &mut self.undo[i] {
                if p == path && now.duration_since(*at) < MERGE_WINDOW {
                    *new = value;
                    *at = now;
                    self.replace_params(next);
                    return Ok(());
                }
            }
        }
        self.push(Command::SetParam { path: path.to_string(), old, new: value, at: now });
        Ok(())
    }

    /// Replace every property (e.g. reset to defaults).
    pub fn set_params(&mut self, params: Params) {
        let mut new = params;
        new.sanitize();
        if new == self.params {
            return;
        }
        let old = Box::new(self.params.clone());
        self.push(Command::SetParams { old, new: Box::new(new) });
    }

    pub fn undo(&mut self) {
        if self.index > 0 {
            self.index -= 1;
            let cmd = std::mem::replace(
                &mut self.undo[self.index],
                Command::AddStroke(Stroke { kind: StrokeKind::Keep, width: 1, points: vec![] }),
            );
            self.apply(&cmd, false);
            self.undo[self.index] = cmd;
        }
    }

    pub fn redo(&mut self) {
        if self.index < self.undo.len() {
            let cmd = std::mem::replace(
                &mut self.undo[self.index],
                Command::AddStroke(Stroke { kind: StrokeKind::Keep, width: 1, points: vec![] }),
            );
            self.apply(&cmd, true);
            self.undo[self.index] = cmd;
            self.index += 1;
        }
    }

    fn apply(&mut self, cmd: &Command, forward: bool) {
        match cmd {
            Command::AddStroke(s) => {
                if forward {
                    self.strokes.push(s.clone());
                    // Provisional (or exact, in simple mode) mask: paint the stroke right away.
                    if let Some(a) = &mut self.alpha {
                        apply_stroke(a, s);
                    }
                    for (marks, kinds) in [
                        (&mut self.marks, [StrokeKind::Keep, StrokeKind::Erase]),
                        (&mut self.bg_marks, [StrokeKind::Remove, StrokeKind::Erase]),
                        (&mut self.hair_marks, [StrokeKind::Hair, StrokeKind::HairErase]),
                    ] {
                        if let (Some(m), true) = (marks, kinds.contains(&s.kind)) {
                            paint_stroke(m, s, s.kind.value());
                        }
                    }
                } else {
                    self.strokes.pop();
                    self.alpha = None;
                    self.marks = None;
                    self.bg_marks = None;
                    self.hair_marks = None;
                }
                self.touch(true);
            }
            Command::SetPrediction { old, new } => {
                self.predicted = if forward { new.clone() } else { old.clone() };
                self.alpha = None;
                self.touch(true);
            }
            Command::SetParam { path, old, new, .. } => {
                let mut root = serde_json::to_value(&self.params).expect("params serialise");
                let v = if forward { new } else { old };
                if set_json(&mut root, path, v.clone()).is_ok() {
                    if let Ok(mut p) = serde_json::from_value::<Params>(root) {
                        p.sanitize();
                        self.replace_params(p);
                        return;
                    }
                }
                self.touch(false);
            }
            Command::SetStrokes { old, new } => {
                self.strokes = if forward { new.clone() } else { old.clone() };
                self.alpha = None;
                self.marks = None;
                self.bg_marks = None;
                self.hair_marks = None;
                self.touch(true);
            }
            Command::SetParams { old, new } => {
                self.replace_params(if forward { (**new).clone() } else { (**old).clone() });
            }
        }
    }

    /// Install new properties. Switching graph-cut mode (or the feather, under hair marks) changes the composed mask.
    fn replace_params(&mut self, p: Params) {
        // The hair matting reads the feather radius too: with hair marks, changing it changes the mask.
        let hair = self.strokes.iter().any(|s| matches!(s.kind, StrokeKind::Hair | StrokeKind::HairErase));
        let mask_changed = p.mask.graphcut != self.params.mask.graphcut || (hair && p.foreground.feather != self.params.foreground.feather);
        self.params = p;
        if mask_changed {
            self.alpha = None;
        }
        self.touch(mask_changed);
    }

    fn touch(&mut self, mask_changed: bool) {
        self.rev += 1;
        if mask_changed {
            self.mask_rev += 1;
            // In simple mode the incrementally updated mask is always exact.
            if !self.heavy() && self.alpha.is_some() {
                self.alpha_exact = self.mask_rev;
            }
        }
    }

    // -- mask computation

    /// Graph-cut and hair matting take long enough to run off the interaction path.
    fn heavy(&self) -> bool {
        is_heavy(&self.params, &self.strokes)
    }

    fn alpha_is_exact(&self) -> bool {
        self.alpha.is_some() && self.alpha_exact == self.mask_rev
    }

    /// True while a provisional mask is shown and an exact one is still to come.
    pub fn computing(&self) -> bool {
        self.heavy() && !self.alpha_is_exact()
    }

    /// Snapshot for a background computation of the exact mask, if one is needed and none is
    /// running. Run it with [`AlphaJob::run`] on any thread, then hand the result to
    /// [`Document::finish_alpha_job`].
    pub fn take_alpha_job(&mut self) -> Option<AlphaJob> {
        if self.job_running || !self.computing() {
            return None;
        }
        self.job_running = true;
        Some(AlphaJob {
            rev: self.mask_rev,
            image: self.image.clone(),
            predicted: self.predicted.clone(),
            strokes: self.strokes.clone(),
            params: self.params.clone(),
        })
    }

    /// Install the result of a job. Returns true when it was still current (the views must
    /// then refresh); a stale result is dropped and the caller should ask for a new job.
    pub fn finish_alpha_job(&mut self, rev: u64, mask: Mask) -> bool {
        self.job_running = false;
        if rev != self.mask_rev {
            return false;
        }
        self.alpha = Some(mask);
        self.alpha_exact = rev;
        self.rev += 1;
        true
    }

    /// The exact mask, computed now if necessary (export, tests). Blocks on heavy modes.
    pub fn alpha_blocking(&mut self) -> &Mask {
        self.prepare();
        if self.computing() {
            let exact = compute_alpha(&self.image, self.predicted.as_deref(), &self.strokes, &self.params);
            self.alpha = Some(exact);
            self.alpha_exact = self.mask_rev;
            self.rev += 1;
        }
        self.alpha.as_ref().expect("alpha ready")
    }

    // -- derived data

    fn load_bg(&mut self, path: Option<String>) -> Result<()> {
        match path {
            Some(p) if !matches!(&self.bg, Some((cur, _)) if *cur == p) => {
                let img = load_image(Path::new(&p))?;
                self.bg = Some((p, img));
            }
            Some(_) => {}
            None => self.bg = None,
        }
        Ok(())
    }

    /// Bring caches (mask, marks, background picture, filled original) up to date.
    pub fn prepare(&mut self) {
        let (w, h) = (self.image.w, self.image.h);
        if self.alpha.is_none() {
            if self.heavy() {
                // Immediate, provisional: the plain paint result. The exact one comes from a job.
                self.alpha = Some(compose_alpha(self.predicted.as_deref(), &self.strokes, w, h));
            } else {
                self.alpha = Some(compute_alpha(&self.image, self.predicted.as_deref(), &self.strokes, &self.params));
                self.alpha_exact = self.mask_rev;
            }
        }
        if self.marks.is_none() {
            self.marks = Some(rasterize(&self.strokes, w, h, &[StrokeKind::Keep, StrokeKind::Erase]));
        }
        if self.bg_marks.is_none() {
            self.bg_marks = Some(rasterize(&self.strokes, w, h, &[StrokeKind::Remove, StrokeKind::Erase]));
        }
        if self.hair_marks.is_none() {
            self.hair_marks = Some(rasterize(&self.strokes, w, h, &[StrokeKind::Hair, StrokeKind::HairErase]));
        }
        let wanted = self.params.background.image_path.clone();
        if self.params.background.mode == BgMode::Image && self.bg.as_ref().map(|(p, _)| p) != wanted.as_ref() && self.bg_failed != wanted {
            // Undo/redo can change the path without going through set_param. A failure is remembered (and shown by
            // `state()`) instead of being retried at every call.
            if self.load_bg(wanted.clone()).is_err() {
                self.bg = None;
                self.bg_failed = wanted;
            }
        }
        let bgp = &self.params.background;
        if bgp.mode == BgMode::Original && bgp.fill_hole {
            let key: FillKey = (self.mask_rev, self.params.mask.offset, self.params.foreground.feather);
            if self.filled.as_ref().map(|(k, _)| *k) != Some(key) {
                let finished = render::finished_alpha(self.alpha.as_ref().expect("alpha ready"), &self.params);
                let hole = morph_ellipse(&finished, 7, MorphOp::Dilate);
                self.filled = Some((key, inpaint::fill_hole(&self.image, &hole)));
            }
        }
        if self.params.background.mode == BgMode::Image && self.frame.as_ref().map(|(r, _)| *r) != Some(self.rev) {
            let alpha = self.alpha.as_ref().expect("alpha ready");
            let finished = self.params.background.fit_to_result.then(|| render::finished_alpha(alpha, &self.params));
            self.frame = Some((self.rev, crop::effective_crop(&self.params, finished.as_ref(), (w as f64, h as f64))));
        }
    }

    fn scene(&self) -> Scene<'_> {
        Scene {
            image: &self.image,
            alpha: self.alpha.as_ref().expect("prepare() must run first"),
            params: &self.params,
            bg_image: self.bg.as_ref().map(|(_, i)| i),
            filled: self.filled.as_ref().map(|(_, i)| i),
            frame: self.frame.filter(|_| self.params.background.mode == BgMode::Image).map(|(_, r)| r),
        }
    }

    pub fn state(&mut self) -> DocState {
        self.prepare();
        let (w, h) = (self.image.w as f64, self.image.h as f64);
        let finished =
            self.params.background.fit_to_result.then(|| render::finished_alpha(self.alpha.as_ref().expect("alpha ready"), &self.params));
        DocState {
            id: self.id,
            rev: self.rev,
            width: self.image.w,
            height: self.image.h,
            name: self.path.as_ref().and_then(|p| p.file_name()).map(|n| n.to_string_lossy().into_owned()),
            can_undo: self.can_undo(),
            can_redo: self.can_redo(),
            modified: !self.is_clean(),
            has_prediction: self.predicted.is_some(),
            computing: self.computing(),
            params: self.params.clone(),
            crop: crop::effective_crop(&self.params, finished.as_ref(), (w, h)),
            bg_error: self.bg_failed.clone().filter(|p| self.params.background.image_path.as_ref() == Some(p)),
        }
    }

    /// Left view: source plus prediction and marks. With `hair_mode` only the hair marks are
    /// shown (yellow), as while the hair tool is active.
    pub fn render_source(&mut self, region: Rect, out_w: usize, out_h: usize, hair_mode: bool) -> Rgba {
        self.prepare();
        let layers = SourceLayers {
            predicted: self.predicted.as_deref(),
            keep: (!hair_mode).then(|| self.marks.as_ref().expect("marks ready")),
            remove: (!hair_mode && self.params.mask.graphcut).then(|| self.bg_marks.as_ref().expect("marks ready")),
            hair: hair_mode.then(|| self.hair_marks.as_ref().expect("marks ready")),
            selection: self.selection.as_ref(),
        };
        render::render_source_view(&self.image, &layers, region, out_w, out_h)
    }

    /// Right view: the composed result.
    pub fn render_result(&mut self, region: Rect, out_w: usize, out_h: usize) -> Rgba {
        self.prepare();
        render::render_region(&self.scene(), region, out_w, out_h)
    }

    /// Full-resolution, cropped result.
    pub fn export(&mut self) -> Rgba {
        self.alpha_blocking();
        self.prepare();
        render::export_rgba(&self.scene())
    }

    /// Everything [`Document::export`] needs, detached from the document: the document can be unlocked while the (long)
    /// rendering runs.
    pub fn export_job(&mut self) -> ExportJob {
        self.alpha_blocking();
        self.prepare();
        let scene = self.scene();
        ExportJob {
            image: self.image.clone(),
            alpha: scene.alpha.clone(),
            params: self.params.clone(),
            bg_image: scene.bg_image.cloned(),
            filled: scene.filled.cloned(),
        }
    }

    /// 16-bit result, when the source is 16-bit and the settings allow it.
    pub fn export_deep(&mut self) -> Option<Rgba16> {
        let deep = self.deep.clone()?;
        self.alpha_blocking();
        self.prepare();
        render::export_deep(&self.scene(), &deep)
    }

    /// For a 16-bit source: why a 16-bit export is not possible now. `None` when it is.
    pub fn deep_blocker(&self) -> Option<&'static str> {
        self.deep.as_ref().and_then(|_| render::deep_blocker(&self.params))
    }

    /// The mask currently shown (before offset and feathering), full resolution. In graph-cut
    /// and hair modes it may be provisional while `computing()`; use `alpha_blocking` for the exact one.
    pub fn alpha(&mut self) -> &Mask {
        self.prepare();
        self.alpha.as_ref().expect("alpha ready")
    }
}

/// How a smart selection is combined with the AI prediction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SelectionMode {
    Replace,
    Add,
    Subtract,
}

/// A full-resolution export prepared by [`Document::export_job`].
pub struct ExportJob {
    image: Arc<Image>,
    alpha: Mask,
    params: Params,
    bg_image: Option<Image>,
    filled: Option<Image>,
}

impl ExportJob {
    pub fn run(&self) -> Rgba {
        let scene = Scene {
            image: &self.image,
            alpha: &self.alpha,
            params: &self.params,
            bg_image: self.bg_image.as_ref(),
            filled: self.filled.as_ref(),
            frame: None,
        };
        render::export_rgba(&scene)
    }
}

/// A background computation of the exact mask, detached from the document.
pub struct AlphaJob {
    rev: u64,
    image: Arc<Image>,
    predicted: Option<Arc<Mask>>,
    strokes: Vec<Stroke>,
    params: Params,
}

impl AlphaJob {
    pub fn rev(&self) -> u64 {
        self.rev
    }

    pub fn run(&self) -> Mask {
        compute_alpha(&self.image, self.predicted.as_deref(), &self.strokes, &self.params)
    }
}

fn is_heavy(params: &Params, strokes: &[Stroke]) -> bool {
    params.mask.graphcut || strokes.iter().any(|s| matches!(s.kind, StrokeKind::Hair | StrokeKind::HairErase))
}

/// Exact mask: prediction and strokes (simple paint or GrabCut), then hair matting.
fn compute_alpha(image: &Image, predicted: Option<&Mask>, strokes: &[Stroke], params: &Params) -> Mask {
    let (w, h) = (image.w, image.h);
    let mut alpha = None;
    if params.mask.graphcut {
        let fg = rasterize(strokes, w, h, &[StrokeKind::Keep, StrokeKind::Erase]);
        let bg = rasterize(strokes, w, h, &[StrokeKind::Remove, StrokeKind::Erase]);
        // Erasers also push the prediction back to "probably background".
        let prior = predicted.map(|p| {
            let mut m = p.clone();
            for s in strokes.iter().filter(|s| s.kind == StrokeKind::Erase) {
                paint_stroke(&mut m, s, 0);
            }
            m
        });
        // Missing a class (no green or no red mark yet): keep the simple result.
        alpha = graphcut_alpha(image, prior.as_ref(), &fg, &bg, &GraphcutOptions::default());
        // The eraser is as certain as a mark: the colour model must not bring back what it wiped out.
        if let Some(a) = &mut alpha {
            let erased = erased_mask(strokes, w, h);
            a.data.iter_mut().zip(&erased.data).for_each(|(v, &e)| {
                if e > 127 {
                    *v = 0;
                }
            });
        }
    }
    let alpha = alpha.unwrap_or_else(|| compose_alpha(predicted, strokes, w, h));
    let hair = rasterize(strokes, w, h, &[StrokeKind::Hair, StrokeKind::HairErase]);
    if hair.is_empty_mask() {
        alpha
    } else {
        hair_matting(image, &alpha, &hair, params.foreground.feather)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn doc() -> Document {
        Document::new(1, Image::filled(50, 40, [100, 100, 100]), None)
    }

    fn keep(x: f32, y: f32, w: u32) -> Stroke {
        Stroke { kind: StrokeKind::Keep, width: w, points: vec![[x, y]] }
    }

    #[test]
    fn the_undo_history_forgets_its_oldest_steps_when_it_gets_too_big() {
        let mut d = Document::new(1, Image::filled(1000, 1000, [90, 90, 90]), None);
        d.undo_limit = 3 << 20; // room for about three 1 MB masks
        for level in 1..=10u8 {
            d.set_prediction(Some(Mask::filled(1000, 1000, level * 20)));
        }
        assert!(d.undo.len() < 10 && d.undo.len() >= 2, "{} steps kept", d.undo.len());
        assert!(d.history_bytes() <= 3 << 20 || d.undo.len() == 1);
        assert!(!d.is_clean(), "the opened state is no longer reachable");
        // What is left still works, back to the oldest kept step and forward again.
        let kept = d.undo.len();
        for _ in 0..kept {
            d.undo();
        }
        assert!(!d.can_undo());
        for _ in 0..kept {
            d.redo();
        }
        assert_eq!(d.predicted().unwrap().data[0], 200);
    }

    #[test]
    fn small_edits_are_never_trimmed_and_a_saved_state_survives_in_range() {
        let mut d = doc();
        for i in 0..200 {
            d.add_stroke(keep(i as f32 % 40.0 + 0.5, 10.5, 3));
        }
        assert_eq!(d.undo.len(), 200, "strokes are small");
        d.mark_clean();
        d.undo_limit = 1; // everything is over the limit: only the newest step stays
        d.add_stroke(keep(5.5, 5.5, 3));
        assert_eq!(d.undo.len(), 1);
        assert!(!d.is_clean() && d.can_undo());
        d.undo();
        assert!(d.is_clean(), "going back one step reaches the state that was saved");
    }

    #[test]
    fn stroke_undo_redo_and_clean_state() {
        let mut d = doc();
        assert!(d.is_clean() && !d.can_undo());
        d.add_stroke(keep(10.5, 10.5, 5));
        assert_eq!(d.alpha().get(10, 10), 255);
        assert!(!d.is_clean() && d.can_undo());
        d.undo();
        assert!(d.alpha().is_empty_mask() && d.is_clean());
        d.redo();
        assert_eq!(d.alpha().get(10, 10), 255);
    }

    #[test]
    fn prediction_is_an_undoable_command() {
        let mut d = doc();
        d.set_prediction(Some(Mask::filled(50, 40, 128)));
        assert!(d.alpha().data.iter().all(|&v| v == 128) && d.has_prediction());
        d.undo();
        assert!(!d.has_prediction() && d.alpha().is_empty_mask());
    }

    #[test]
    fn incremental_alpha_equals_full_recompute() {
        let mut d = doc();
        d.set_prediction(Some(Mask::filled(50, 40, 90)));
        d.add_stroke(keep(10.5, 10.5, 7));
        let _ = d.alpha(); // fill the cache, then add on top of it
        d.add_stroke(Stroke { kind: StrokeKind::Erase, width: 9, points: vec![[20.5, 15.5]] });
        let incremental = d.alpha().clone();
        d.alpha = None;
        assert_eq!(&incremental, d.alpha());
    }

    #[test]
    fn new_command_after_undo_drops_redo_and_saved_state() {
        let mut d = doc();
        d.add_stroke(keep(1.0, 1.0, 3));
        d.mark_clean();
        d.undo();
        d.add_stroke(keep(2.0, 2.0, 3));
        assert!(!d.can_redo() && !d.is_clean());
        d.undo();
        assert!(!d.is_clean(), "saved state was discarded");
    }

    #[test]
    fn set_param_validates_clamps_and_undoes() {
        let mut d = doc();
        d.set_param("shadow.enabled", json!(true)).unwrap();
        d.set_param("shadow.opacity", json!(500)).unwrap();
        assert_eq!(d.params().shadow.opacity, 100); // clamped
        assert!(d.set_param("shadow.nope", json!(1)).is_err());
        assert!(d.set_param("shadow.opacity", json!("high")).is_err());
        d.undo();
        d.undo();
        assert!(!d.params().shadow.enabled);
        d.redo();
        assert!(d.params().shadow.enabled);
    }

    #[test]
    fn rapid_edits_of_one_property_merge_into_one_undo_step() {
        let mut d = doc();
        for v in 1..=20 {
            d.set_param("foreground.feather", json!(v)).unwrap();
        }
        assert_eq!(d.params().foreground.feather, 20);
        d.undo();
        assert_eq!(d.params().foreground.feather, 0);
        assert!(!d.can_undo());
    }

    #[test]
    fn different_properties_do_not_merge_and_saved_state_is_not_rewritten() {
        let mut d = doc();
        d.set_param("shadow.opacity", json!(10)).unwrap();
        d.set_param("shadow.blur_radius", json!(20)).unwrap();
        d.undo();
        assert_eq!(d.params().shadow.opacity, 10);
        d.mark_clean();
        d.set_param("shadow.opacity", json!(30)).unwrap();
        assert!(!d.is_clean());
        d.undo();
        assert!(d.is_clean() && d.params().shadow.opacity == 10);
    }

    #[test]
    fn state_reports_crop_and_flags() {
        let mut d = doc();
        d.set_prediction(Some(Mask::new(50, 40)));
        let s = d.state();
        assert_eq!((s.width, s.height), (50, 40));
        assert!(s.can_undo && s.modified && s.has_prediction);
        assert_eq!(s.crop, Rect::new(0.0, 0.0, 50.0, 40.0));
        d.set_param("background.aspect_ratio", json!("1:1")).unwrap();
        let c = d.state().crop;
        assert!((c.width() - c.height()).abs() < 1e-9);
    }

    #[test]
    fn fill_hole_original_background_renders() {
        let mut d = Document::new(2, Image::filled(50, 40, [10, 20, 30]), None);
        d.add_stroke(keep(25.5, 20.5, 10));
        d.set_param("background.mode", json!("original")).unwrap();
        d.set_param("background.fill_hole", json!(true)).unwrap();
        let out = d.render_result(Rect::new(0.0, 0.0, 50.0, 40.0), 50, 40);
        assert_eq!(&out.data[(20 * 50 + 25) * 4..(20 * 50 + 25) * 4 + 4], &[10, 20, 30, 255]);
        let e = d.export();
        assert_eq!((e.w, e.h), (50, 40));
    }

    #[test]
    fn set_params_resets_everything_in_one_step() {
        let mut d = doc();
        d.set_param("shadow.enabled", json!(true)).unwrap();
        d.set_param("background.mode", json!("color")).unwrap();
        d.set_params(Params::default());
        assert_eq!(d.params(), &Params::default());
        d.undo();
        assert!(d.params().shadow.enabled);
    }

    fn disc_scene(w: usize, h: usize) -> Image {
        let mut img = Image::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let inside = (x as f64 - w as f64 / 2.0).powi(2) + (y as f64 - h as f64 / 2.0).powi(2) < (h as f64 / 3.0).powi(2);
                let rgb: [u8; 3] = if inside { [210, 60, 50] } else { [40, 80, 180] };
                img.data[(y * w + x) * 3..(y * w + x) * 3 + 3].copy_from_slice(&rgb);
            }
        }
        img
    }

    #[test]
    fn graphcut_mode_uses_green_and_red_strokes_and_falls_back_without_both() {
        let (w, h) = (80, 60);
        let mut d = Document::new(3, disc_scene(w, h), None);
        d.set_param("mask.graphcut", json!(true)).unwrap();
        // Only a green stroke: nothing to separate yet, simple paint is shown.
        d.add_stroke(Stroke { kind: StrokeKind::Keep, width: 5, points: vec![[40.5, 30.5]] });
        let simple_area = d.alpha().data.iter().filter(|&&v| v > 127).count();
        assert!(simple_area < 40, "{simple_area}");
        // Add a red stroke on the background: the whole disc is found.
        d.add_stroke(Stroke { kind: StrokeKind::Remove, width: 5, points: vec![[5.5, 5.5]] });
        let area = d.alpha_blocking().data.iter().filter(|&&v| v > 127).count();
        let disc = (std::f64::consts::PI * (h as f64 / 3.0).powi(2)) as usize;
        assert!((area as f64 - disc as f64).abs() < disc as f64 * 0.08, "{area} vs {disc}");
        // Red marks are only visible in graph-cut mode.
        let on = d.render_source(Rect::new(0.0, 0.0, w as f64, h as f64), w, h, false);
        d.set_param("mask.graphcut", json!(false)).unwrap();
        let off = d.render_source(Rect::new(0.0, 0.0, w as f64, h as f64), w, h, false);
        let px = (5 * w + 5) * 4;
        assert_ne!(&on.data[px..px + 3], &off.data[px..px + 3]);
        // Back to simple mode: only the painted strokes remain.
        assert!(d.alpha().data.iter().filter(|&&v| v > 127).count() < 40);
    }

    #[test]
    fn graphcut_undo_and_eraser() {
        let (w, h) = (80, 60);
        let mut d = Document::new(4, disc_scene(w, h), None);
        d.set_param("mask.graphcut", json!(true)).unwrap();
        d.add_stroke(Stroke { kind: StrokeKind::Keep, width: 5, points: vec![[40.5, 30.5]] });
        d.add_stroke(Stroke { kind: StrokeKind::Remove, width: 5, points: vec![[5.5, 5.5]] });
        let full = d.alpha_blocking().clone();
        assert_eq!(full.get(40, 30), 255);
        // Erasing the red mark removes the constraint: the pure-colour cut no longer has a background seed.
        d.add_stroke(Stroke { kind: StrokeKind::Erase, width: 9, points: vec![[5.5, 5.5]] });
        assert!(d.alpha_blocking().data.iter().filter(|&&v| v > 127).count() < 200);
        d.undo();
        assert_eq!(d.alpha_blocking(), &full);
        d.undo();
        d.undo();
        assert!(d.alpha().is_empty_mask());
    }

    #[test]
    fn an_export_job_renders_the_same_picture_without_the_document() {
        let mut d = doc();
        d.set_prediction(Some(Mask::filled(50, 40, 255)));
        d.set_param("background.mode", json!("color")).unwrap();
        d.set_param("crop", json!({ "x0": 5.0, "y0": 5.0, "x1": 35.0, "y1": 30.0 })).unwrap();
        let expected = d.export();
        let job = d.export_job();
        d.set_param("background.color", json!([1, 2, 3])).unwrap(); // later edits do not leak into the job
        let out = job.run();
        assert_eq!((out.w, out.h), (30, 25));
        assert!(out == expected);
    }

    #[test]
    fn graphcut_mode_keeps_what_the_eraser_removed_removed() {
        let (w, h) = (80, 60);
        let mut d = Document::new(6, disc_scene(w, h), None);
        d.set_param("mask.graphcut", json!(true)).unwrap();
        d.add_stroke(Stroke { kind: StrokeKind::Keep, width: 5, points: vec![[40.5, 30.5]] });
        d.add_stroke(Stroke { kind: StrokeKind::Remove, width: 5, points: vec![[5.5, 5.5]] });
        assert_eq!(d.alpha_blocking().get(52, 30), 255, "the whole disc is found");
        // The eraser goes over a part of the disc that has no mark: that part is removed, as in simple mode.
        d.add_stroke(Stroke { kind: StrokeKind::Erase, width: 9, points: vec![[52.5, 30.5]] });
        let a = d.alpha_blocking().clone();
        assert_eq!((a.get(52, 30), a.get(50, 28), a.get(54, 32)), (0, 0, 0), "erased pixels are transparent");
        assert_eq!(a.get(40, 30), 255, "the rest of the disc stays");
        // A later green mark over the erased part brings it back (the last stroke wins, like in simple mode).
        d.add_stroke(Stroke { kind: StrokeKind::Keep, width: 3, points: vec![[52.5, 30.5]] });
        assert_eq!(d.alpha_blocking().get(52, 30), 255);
        // An eraser that goes over a green mark removes the mark and the pixels under it.
        d.add_stroke(Stroke { kind: StrokeKind::Erase, width: 9, points: vec![[40.5, 30.5]] });
        assert_eq!(d.alpha_blocking().get(40, 30), 0);
    }

    #[test]
    fn background_job_flow_provisional_then_exact_and_stale_results_are_dropped() {
        let (w, h) = (80, 60);
        let mut d = Document::new(5, disc_scene(w, h), None);
        assert!(d.take_alpha_job().is_none(), "simple mode never needs a job");
        d.set_param("mask.graphcut", json!(true)).unwrap();
        d.add_stroke(Stroke { kind: StrokeKind::Keep, width: 5, points: vec![[40.5, 30.5]] });
        d.add_stroke(Stroke { kind: StrokeKind::Remove, width: 5, points: vec![[5.5, 5.5]] });
        assert!(d.computing() && d.state().computing);
        // Provisional mask is available immediately and is just the painted strokes.
        assert!(d.alpha().data.iter().filter(|&&v| v > 127).count() < 40);

        let job = d.take_alpha_job().expect("a job is needed");
        assert!(d.take_alpha_job().is_none(), "only one job at a time");
        let exact = job.run();
        let rev_before = d.rev();
        assert!(d.finish_alpha_job(job.rev(), exact.clone()));
        assert!(!d.computing() && d.rev() > rev_before, "views must see a new revision");
        assert_eq!(d.alpha(), &exact);

        // An edit while a job runs makes its result stale.
        d.add_stroke(Stroke { kind: StrokeKind::Keep, width: 5, points: vec![[60.5, 30.5]] });
        let job = d.take_alpha_job().unwrap();
        d.add_stroke(Stroke { kind: StrokeKind::Remove, width: 5, points: vec![[70.5, 50.5]] });
        assert!(!d.finish_alpha_job(job.rev(), job.run()), "stale");
        assert!(d.computing());
        let job = d.take_alpha_job().expect("a fresh job is possible after a stale one");
        assert!(d.finish_alpha_job(job.rev(), job.run()));
        assert!(!d.computing());
    }

    #[test]
    fn export_waits_for_the_exact_mask() {
        let (w, h) = (80, 60);
        let mut d = Document::new(6, disc_scene(w, h), None);
        d.set_param("mask.graphcut", json!(true)).unwrap();
        d.add_stroke(Stroke { kind: StrokeKind::Keep, width: 5, points: vec![[40.5, 30.5]] });
        d.add_stroke(Stroke { kind: StrokeKind::Remove, width: 5, points: vec![[5.5, 5.5]] });
        let out = d.export();
        let opaque = out.data.chunks_exact(4).filter(|p| p[3] > 127).count();
        let disc = (std::f64::consts::PI * (h as f64 / 3.0).powi(2)) as usize;
        assert!((opaque as f64 - disc as f64).abs() < disc as f64 * 0.08, "{opaque} vs {disc}");
        assert!(!d.computing());
    }

    #[test]
    fn hair_strokes_soften_the_mask_in_the_background_and_show_only_in_hair_mode() {
        let (w, h) = (60usize, 30usize);
        // Colour ramp from background to foreground over columns 20..40.
        let mut img = Image::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let a = ((x as f64 - 20.0) / 20.0).clamp(0.0, 1.0);
                for (c, (f, b)) in [(230.0, 20.0), (200.0, 40.0), (60.0, 150.0)].into_iter().enumerate() {
                    img.data[(y * w + x) * 3 + c] = (a * f + (1.0 - a) * b) as u8;
                }
            }
        }
        let mut d = Document::new(7, img, None);
        let mut pred = Mask::new(w, h);
        for y in 0..h {
            for x in 30..w {
                pred.data[y * w + x] = 255;
            }
        }
        d.set_prediction(Some(pred));
        assert!(!d.computing());
        // A tall stroke over the whole transition band, like painting along a hair edge.
        d.add_stroke(Stroke { kind: StrokeKind::Hair, width: 24, points: vec![[30.0, -5.0], [30.0, 35.0]] });
        assert!(d.computing(), "hair matting is a background job");
        let job = d.take_alpha_job().unwrap();
        assert!(d.finish_alpha_job(job.rev(), job.run()));
        let soft = (0..w * h).filter(|&i| d.alpha().data[i] > 20 && d.alpha().data[i] < 235).count();
        assert!(soft > 50, "{soft} soft pixels");

        let region = Rect::new(0.0, 0.0, w as f64, h as f64);
        let normal = d.render_source(region, w, h, false);
        let hair = d.render_source(region, w, h, true);
        let px = (15 * w + 30) * 4;
        assert_ne!(&normal.data[px..px + 3], &hair.data[px..px + 3]);
        // Undoing the hair stroke returns to the exact simple mask without any job.
        d.undo();
        assert!(!d.computing());
        assert_eq!(d.alpha().data[15 * w + 25], 0);
    }

    #[test]
    fn changing_the_feather_recomputes_the_hair_matting() {
        let (w, h) = (60usize, 30usize);
        let mut img = Image::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let a = ((x as f64 - 20.0) / 20.0).clamp(0.0, 1.0);
                for (c, (f, b)) in [(230.0, 20.0), (200.0, 40.0), (60.0, 150.0)].into_iter().enumerate() {
                    img.data[(y * w + x) * 3 + c] = (a * f + (1.0 - a) * b) as u8;
                }
            }
        }
        let mut d = Document::new(8, img, None);
        let mut pred = Mask::new(w, h);
        for y in 0..h {
            for x in 30..w {
                pred.data[y * w + x] = 255;
            }
        }
        d.set_prediction(Some(pred));
        d.add_stroke(Stroke { kind: StrokeKind::Hair, width: 24, points: vec![[30.0, -5.0], [30.0, 35.0]] });
        let first = d.alpha_blocking().clone();
        assert!(!d.computing());

        d.set_param("foreground.feather", json!(8)).unwrap();
        assert!(d.computing(), "the cached matting is stale: a new job is needed");
        let second = d.alpha_blocking().clone();
        assert_ne!(first.data, second.data, "the feather changes the matting");
    }

    #[test]
    fn loading_marks_replaces_the_strokes_as_one_undoable_step() {
        let mut d = doc();
        d.add_stroke(keep(5.5, 5.5, 5));
        let loaded = vec![
            Stroke { kind: StrokeKind::Keep, width: 9, points: vec![[30.5, 20.5]] },
            Stroke { kind: StrokeKind::Remove, width: 5, points: vec![[2.5, 2.5]] },
        ];
        d.load_marks(loaded.clone());
        assert_eq!(d.strokes(), loaded.as_slice());
        assert_eq!(d.alpha().get(30, 20), 255);
        assert_eq!(d.alpha().get(5, 5), 0, "the old stroke is gone");
        d.undo();
        assert_eq!(d.strokes().len(), 1);
        assert_eq!(d.alpha().get(5, 5), 255);
        d.redo();
        assert_eq!(d.strokes(), loaded.as_slice());
    }

    #[test]
    fn smart_selection_previews_then_folds_into_the_prediction() {
        let mut d = doc();
        let mut pred = Mask::new(50, 40);
        for y in 0..40 {
            for x in 0..20 {
                pred.data[y * 50 + x] = 255;
            }
        }
        d.set_prediction(Some(pred));
        d.mark_clean();
        let mut sel = Mask::new(50, 40);
        for y in 0..40 {
            for x in 10..30 {
                sel.data[y * 50 + x] = 255;
            }
        }
        // A preview changes nothing in the document.
        let rev = d.rev();
        d.set_selection(Some(sel.clone()));
        assert!(d.has_selection() && d.is_clean() && d.rev() > rev);
        assert_eq!(d.alpha().get(25, 5), 0);
        let region = Rect::new(0.0, 0.0, 50.0, 40.0);
        let shown = d.render_source(region, 50, 40, false);
        let plain = {
            let mut e = doc();
            e.render_source(region, 50, 40, false)
        };
        assert_ne!(
            &shown.data[(5 * 50 + 25) * 4..(5 * 50 + 25) * 4 + 3],
            &plain.data[(5 * 50 + 25) * 4..(5 * 50 + 25) * 4 + 3],
            "cyan tint"
        );

        // Add: union with the prediction.
        assert!(d.apply_selection(SelectionMode::Add));
        assert!(!d.has_selection() && !d.is_clean());
        assert_eq!((d.alpha().get(5, 5), d.alpha().get(25, 5), d.alpha().get(40, 5)), (255, 255, 0));
        d.undo();
        assert_eq!((d.alpha().get(5, 5), d.alpha().get(25, 5)), (255, 0));
        assert!(d.is_clean(), "undo returns to the saved state");

        // Subtract removes the selected part; Replace swaps the prediction.
        d.set_selection(Some(sel.clone()));
        d.apply_selection(SelectionMode::Subtract);
        assert_eq!((d.alpha().get(5, 5), d.alpha().get(15, 5)), (255, 0));
        d.set_selection(Some(sel));
        d.apply_selection(SelectionMode::Replace);
        assert_eq!((d.alpha().get(5, 5), d.alpha().get(25, 5)), (0, 255));
        assert!(!d.apply_selection(SelectionMode::Add), "nothing pending");
    }

    #[test]
    fn a_missing_background_picture_is_reported_once_and_not_retried() {
        let dir = std::env::temp_dir().join(format!("osc-bg-missing-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("bg.png");
        let mut d = doc();
        let mut p = d.params().clone();
        p.background.mode = BgMode::Image;
        p.background.image_path = Some(file.display().to_string());
        d.set_params(p);
        assert_eq!(d.state().bg_error.as_deref(), Some(file.display().to_string().as_str()));
        // The file shows up afterwards: the failure was remembered, so no new attempt is made.
        crate::io::save_rgba(&file, &Rgba::new(2, 2), crate::io::Format::Png, 95, [0; 3]).unwrap();
        assert!(d.state().bg_error.is_some() && d.bg.is_none());
        // Choosing the picture again loads it and clears the error.
        d.set_param("background.image_path", json!(file.display().to_string())).unwrap();
        assert!(d.state().bg_error.is_none() && d.bg.is_some());
        std::fs::remove_dir_all(dir).ok();
    }
}
