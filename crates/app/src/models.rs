//! Models and background computation: memory limits, loading, prediction, the exact mask jobs,
//! downloads and smart select.

use openscissors_core::sync::LockExt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use openscissors_core::document::{AlphaJob, SelectionMode};
use openscissors_core::model::{Accelerator, ModelKind, ModelSpec, Predictor, Prompt, Sam2};
use openscissors_core::sysmem;

use crate::ctl::{Ctl, Prediction, R_BATCH, R_MAIN, R_MODELS, R_VIEW};
use crate::i18n::{t, tf};
use crate::state::{AppState, ModelStatus, LOCAL_MODEL_ID};

/// Refuse to load a model the machine has no room for: a failed allocation would end the whole
/// application without a message.
pub fn check_memory(spec: &ModelSpec, fast: bool) -> Result<(), String> {
    let need = model_need_mb(spec, fast);
    if let Some(budget) = sysmem::budget_mb() {
        if need > budget {
            return Err(format!(
                "{} needs about {} MB, more than the {} MB ({}% of the memory) the application may use. Choose a lighter model.",
                spec.name,
                need,
                budget,
                sysmem::budget_percent()
            ));
        }
    }
    match sysmem::available_memory_mb() {
        Some(free) if need > 0 && free < need => Err(format!(
            "Not enough free memory to run {}: about {} MB are needed and {} MB are free. Close other applications{}.",
            spec.name,
            need,
            free,
            if fast { ", turn off \"Use more memory for speed\"" } else { " or choose a lighter model" }
        )),
        _ => Ok(()),
    }
}

/// Peak memory of `spec` in MB (0 when not measured); the speed mode keeps ONNX Runtime's arena, about 2.3 times more.
pub fn model_need_mb(spec: &ModelSpec, fast: bool) -> u64 {
    (spec.ram_mb as f64 * if fast { 2.3 } else { 1.0 }) as u64
}

/// Whether `spec` fits in the memory budget of the application.
pub fn fits_budget(spec: &ModelSpec, fast: bool) -> bool {
    sysmem::budget_mb().is_none_or(|b| model_need_mb(spec, fast) <= b)
}

/// The model to start with: the one the user chose, else the lightest one on disk that fits the memory budget
/// (the basic model is enough most of the time; heavier ones are opt-in).
pub fn pick_default_model(state: &AppState) -> Option<String> {
    let wanted = state.settings.guard().chosen_model.clone();
    let mut ready: Vec<_> = state
        .models()
        .into_iter()
        .filter(|s| s.kind == ModelKind::Segmenter && state.is_ready(s) && (s.id == LOCAL_MODEL_ID || fits_budget(s, false)))
        .collect();
    if let Some(id) = wanted.filter(|id| ready.iter().any(|s| s.id == *id)) {
        return Some(id);
    }
    // Models without a measured figure (0) come last.
    ready.sort_by_key(|s| if s.ram_mb == 0 { u32::MAX } else { s.ram_mb });
    ready.first().map(|s| s.id.clone())
}

/// Working memory of the prediction around the network, in MB: the image is resized and guided-filtered
/// in bands, so what scales with the image is the decoded pixels, the masks and the copies the document keeps.
pub fn pipeline_mb(w: usize, h: usize) -> u64 {
    (w as u64 * h as u64 * 10) >> 20
}

/// Refuse a prediction that would take the application over its memory budget (an 8K photo with a heavy model).
fn check_pipeline(model: &Predictor, w: usize, h: usize) -> Result<(), String> {
    let (Some(budget), Some(now)) = (sysmem::budget_mb(), sysmem::process_memory_mb()) else { return Ok(()) };
    let need = now + pipeline_mb(w, h) + model.spec().ram_mb as u64;
    if need > budget {
        return Err(format!(
            "A {w} x {h} px image needs about {need} MB with {}, more than the {budget} MB ({}% of the memory) the application may use. Choose a lighter model.",
            model.spec().name,
            sysmem::budget_percent()
        ));
    }
    Ok(())
}

type PendingJob = (u64, AlphaJob);

fn take_job(state: &AppState) -> Option<PendingJob> {
    let mut g = state.doc.guard();
    g.as_mut().and_then(|d| d.take_alpha_job().map(|j| (d.id, j)))
}

/// Run `first`, install its result, and keep going while newer edits ask for another job.
/// `on_applied(doc_id)` is called each time an up-to-date mask was installed.
pub fn run_alpha_jobs(state: &AppState, first: PendingJob, on_applied: impl Fn(u64)) {
    let mut current = Some(first);
    while let Some((doc_id, job)) = current.take() {
        let mask = job.run();
        let (applied, next) = {
            let mut g = state.doc.guard();
            match g.as_mut() {
                Some(d) if d.id == doc_id => (d.finish_alpha_job(job.rev(), mask), d.take_alpha_job().map(|j| (d.id, j))),
                _ => (false, None),
            }
        };
        if applied {
            on_applied(doc_id);
        }
        current = next;
    }
}

const SAM_MISSING: &str = "SAM2_MISSING";

fn ensure_sam(state: &AppState) -> Result<Arc<Sam2>, String> {
    if let Some(s) = state.sam.guard().clone() {
        return Ok(s);
    }
    let spec = state.registry.available().into_iter().find(|s| s.kind == ModelKind::Sam2).ok_or(SAM_MISSING)?;
    if !state.is_ready(&spec) {
        return Err(SAM_MISSING.into());
    }
    let sam = Arc::new(Sam2::load(&state.registry.model_dir(&spec)).map_err(|e| e.to_string())?);
    *state.sam.guard() = Some(sam.clone());
    Ok(sam)
}

impl Ctl {
    // -- exact mask in the background

    /// Start computing the exact mask (graph-cut, hair matting) off the interaction path when the
    /// document needs it. The views keep showing a provisional mask until the exact one is ready.
    pub fn kick_alpha(self: &Arc<Self>) {
        let Some(first) = take_job(&self.st) else { return };
        let me = self.clone();
        std::thread::spawn(move || {
            run_alpha_jobs(&me.st, first, |_| me.refresh_doc());
        });
    }

    // -- loading a model

    /// Load `id` on a background thread and make it the active model.
    pub fn load_model_async(self: &Arc<Self>, id: String) {
        *self.st.model_status.guard() = ModelStatus { active: Some(id.clone()), loading: true, ready: false, error: None, backend: None };
        self.refresh(R_MAIN | R_MODELS);
        let me = self.clone();
        std::thread::spawn(move || {
            let state = &me.st;
            let result = (|| -> Result<Predictor, String> {
                let spec = state.models().into_iter().find(|s| s.id == id).ok_or("unknown model")?;
                let path = state.model_path(&spec);
                if !path.is_file() {
                    return Err("model file is missing".into());
                }
                let (accel, fast) = {
                    let s = state.settings.guard();
                    // The speed mode is dropped when it would not fit the memory budget.
                    (s.accelerator, s.fast_memory && fits_budget(&spec, true))
                };
                check_memory(&spec, fast)?;
                let p = Predictor::load_with_memory(&spec, &path, accel, fast).map_err(|e| e.to_string())?;
                p.warmup().map_err(|e| e.to_string())?;
                Ok(p)
            })();
            match result {
                Ok(p) => {
                    let backend = Some(p.backend().to_string());
                    *state.model.guard() = Some(Arc::new(p));
                    *state.model_status.guard() = ModelStatus { active: Some(id), loading: false, ready: true, error: None, backend };
                }
                Err(e) => {
                    *state.model.guard() = None;
                    *state.model_status.guard() =
                        ModelStatus { active: Some(id), loading: false, ready: false, error: Some(e), backend: None };
                }
            }
            me.refresh(R_MAIN | R_MODELS | R_BATCH);
            me.maybe_predict();
        });
    }

    pub fn use_model(self: &Arc<Self>, id: &str) {
        let Some(spec) = self.st.models().into_iter().find(|s| s.id == id) else { return };
        if spec.kind != ModelKind::Segmenter {
            return self.say("this model is used by Smart select, not as the cut-out model", true);
        }
        if !self.st.is_ready(&spec) {
            return self.say("download this model first", true);
        }
        self.st.settings.guard().chosen_model = Some(id.to_string());
        self.save_settings();
        *self.st.predicted_doc.guard() = None;
        self.ux.guard().repredict = true; // the open image is predicted again with it
        self.load_model_async(id.to_string());
    }

    pub fn download_model(self: &Arc<Self>, id: &str) {
        let Some(spec) = self.st.models().into_iter().find(|s| s.id == id) else { return };
        let cancel = Arc::new(AtomicBool::new(false));
        {
            let mut d = self.st.downloads.guard();
            if d.contains_key(id) {
                return;
            }
            d.insert(id.to_string(), cancel.clone());
        }
        self.ux.guard().downloads.insert(id.to_string(), (0, 0));
        self.refresh(R_MODELS);
        let me = self.clone();
        std::thread::spawn(move || {
            let mut last = std::time::Instant::now();
            let mut progress = |done: u64, total: u64| {
                if last.elapsed().as_millis() > 100 || done == total {
                    last = std::time::Instant::now();
                    me.ux.guard().downloads.insert(spec.id.clone(), (done, total));
                    me.refresh(R_MODELS);
                }
            };
            let result = me.st.registry.download(&spec, &mut progress, &cancel);
            me.st.downloads.guard().remove(&spec.id);
            me.ux.guard().downloads.remove(&spec.id);
            match &result {
                Ok(_) => me.say(t("Model downloaded"), false),
                Err(e) if e.to_string() != "cancelled" => me.say(tf("Download failed: {error}", &[("error", &e.to_string())]), true),
                Err(_) => {}
            }
            me.refresh(R_MODELS | R_MAIN);
            // Nothing is activated here: the model is picked when an image needs it (the user's choice,
            // otherwise the lightest on disk), so downloading a heavy model never makes the application heavy.
            if result.is_ok() && spec.kind == ModelKind::Segmenter {
                me.maybe_predict();
            }
        });
    }

    pub fn cancel_download(&self, id: &str) {
        if let Some(flag) = self.st.downloads.guard().get(id) {
            flag.store(true, Ordering::Relaxed);
        }
    }

    pub fn delete_model(self: &Arc<Self>, id: &str) {
        let Some(spec) = self.st.registry.get(id) else { return };
        let was_active = self.st.model_status.guard().active.as_deref() == Some(id);
        if was_active {
            *self.st.model.guard() = None;
            *self.st.model_status.guard() = ModelStatus::default();
            self.st.settings.guard().chosen_model = None;
            self.save_settings();
        }
        if let Err(e) = self.st.registry.delete(&spec) {
            self.say(e.to_string(), true);
        }
        self.refresh(R_MAIN | R_MODELS | R_BATCH);
    }

    pub fn set_accelerator(self: &Arc<Self>, index: i32) {
        self.st.settings.guard().accelerator = match index {
            1 => Accelerator::Gpu,
            2 => Accelerator::Auto,
            _ => Accelerator::Cpu,
        };
        self.save_settings();
        self.reload_active();
    }

    pub fn set_fast(self: &Arc<Self>, on: bool) {
        self.st.settings.guard().fast_memory = on;
        self.save_settings();
        self.reload_active();
    }

    pub fn set_half(self: &Arc<Self>, on: bool) {
        let p = if on { 50 } else { 25 };
        self.st.settings.guard().ram_percent = p;
        sysmem::set_budget_percent(p as u64);
        self.save_settings();
        // A model refused for lack of room gets another try.
        if !self.st.model_status.guard().ready {
            self.reload_active();
        } else {
            self.refresh(R_MODELS);
        }
    }

    /// Run the active model again with the new settings.
    fn reload_active(self: &Arc<Self>) {
        let active = self.st.model_status.guard().active.clone();
        if let Some(id) = active {
            self.load_model_async(id);
        } else {
            self.refresh(R_MODELS);
        }
    }

    // -- prediction

    /// Start predicting the open image when a model is ready and it has not been tried yet.
    pub fn maybe_predict(self: &Arc<Self>) {
        let Some(model) = self.st.model.guard().clone() else {
            let (loading, error, started) = {
                let st = self.st.model_status.guard();
                (st.loading, st.error.clone(), st.active.is_some())
            };
            if loading {
                return; // load_model_async calls us again when the model is ready
            }
            if self.st.doc.guard().is_none() {
                return;
            }
            if started {
                let message = error.unwrap_or_else(|| "The model could not be loaded".into());
                self.prediction_failed(message);
            } else if let Some(id) = pick_default_model(&self.st) {
                // First image of the session: only now does the model take any memory.
                self.ux.guard().prediction = Prediction::Running;
                self.load_model_async(id);
            } else {
                self.ux.guard().prediction = Prediction::NoModel;
                self.refresh(R_MAIN);
            }
            return;
        };
        let (id, image) = {
            let g = self.st.doc.guard();
            let Some(doc) = g.as_ref() else { return };
            if doc.has_prediction() && !std::mem::take(&mut self.ux.guard().repredict) {
                return;
            }
            (doc.id, doc.image.clone())
        };
        if let Err(message) = check_pipeline(&model, image.w, image.h) {
            return self.prediction_failed(message);
        }
        {
            let mut running = self.st.predicted_doc.guard();
            if *running == Some(id) {
                return;
            }
            *running = Some(id);
        }
        self.ux.guard().prediction = Prediction::Running;
        self.refresh(R_MAIN);
        let me = self.clone();
        std::thread::spawn(move || {
            let result = model.predict(&image, true);
            match result {
                Ok(mask) => {
                    let applied = {
                        let mut g = me.st.doc.guard();
                        match g.as_mut() {
                            Some(doc) if doc.id == id => {
                                // The proposal is not an edit of the person: it must not ask to save on closing.
                                let was_clean = doc.is_clean();
                                doc.set_prediction(Some(mask));
                                if was_clean {
                                    doc.mark_clean();
                                }
                                true
                            }
                            _ => false,
                        }
                    };
                    if applied {
                        me.ux.guard().prediction = Prediction::Idle;
                        me.kick_alpha();
                        me.refresh_doc();
                    }
                }
                Err(e) => me.prediction_failed(e.to_string()),
            }
        });
    }

    fn prediction_failed(self: &Arc<Self>, message: String) {
        self.ux.guard().prediction = Prediction::Failed;
        self.say(tf("Prediction failed: {message}", &[("message", &message)]), true);
    }

    // -- smart select (SAM 2)

    pub fn smart_add(self: &Arc<Self>, prompt: Prompt) {
        self.ux.guard().prompts.push(prompt);
        self.smart_run();
    }

    /// Remove the last prompt (Backspace); the selection follows.
    pub fn smart_undo_prompt(self: &Arc<Self>) {
        let empty = {
            let mut ux = self.ux.guard();
            if ux.prompts.is_empty() {
                return;
            }
            ux.prompts.pop();
            ux.prompts.is_empty()
        };
        if empty {
            self.smart_cancel();
        } else {
            self.smart_run();
        }
    }

    fn smart_run(self: &Arc<Self>) {
        let (doc_id, image) = {
            let g = self.st.doc.guard();
            let Some(d) = g.as_ref() else { return };
            (d.id, d.image.clone())
        };
        let prompts = {
            let mut ux = self.ux.guard();
            ux.smart_busy = true;
            ux.prompts.clone()
        };
        self.say(t("Analysing the image…"), false);
        self.refresh(R_MAIN | R_VIEW);
        let me = self.clone();
        std::thread::spawn(move || {
            let result = (|| -> Result<_, String> {
                let sam = ensure_sam(&me.st)?;
                let cached = me.st.sam_cache.guard().as_ref().filter(|(id, _)| *id == doc_id).map(|(_, e)| e.clone());
                let emb = match cached {
                    Some(e) => e,
                    None => Arc::new(sam.embed(&image).map_err(|e| e.to_string())?),
                };
                let mask = sam.select(&emb, &prompts, &image).map_err(|e| e.to_string())?;
                *me.st.sam_cache.guard() = Some((doc_id, emb));
                Ok(mask)
            })();
            me.ux.guard().smart_busy = false;
            match result {
                Ok(mask) => {
                    if let Some(d) = me.st.doc.guard().as_mut().filter(|d| d.id == doc_id) {
                        d.set_selection(Some(mask));
                    }
                    me.say("", false);
                    me.refresh_doc();
                }
                Err(e) => {
                    me.ux.guard().prompts.clear();
                    if e.contains(SAM_MISSING) {
                        me.say(t("Smart select needs the SAM 2 model. Download it in AI models."), true);
                        me.with_ui(|me, ui| {
                            use slint::ComponentHandle;
                            ui.global::<crate::Backend>().set_popover(1);
                            me.refresh(R_MODELS);
                        });
                    } else {
                        me.say(e, true);
                    }
                    me.refresh(R_MAIN | R_VIEW);
                }
            }
        });
    }

    pub fn smart_apply(self: &Arc<Self>) {
        let mode = {
            let mut ux = self.ux.guard();
            if ux.prompts.is_empty() {
                return;
            }
            ux.prompts.clear();
            ux.smart_mode
        };
        if let Some(d) = self.st.doc.guard().as_mut() {
            d.apply_selection(mode);
        }
        self.refresh_doc();
        self.kick_alpha();
    }

    pub fn smart_cancel(self: &Arc<Self>) {
        {
            let mut ux = self.ux.guard();
            if ux.prompts.is_empty() {
                return;
            }
            ux.prompts.clear();
        }
        if let Some(d) = self.st.doc.guard().as_mut() {
            d.set_selection(None);
        }
        self.refresh_doc();
    }

    pub fn set_smart_mode(self: &Arc<Self>, index: i32) {
        self.ux.guard().smart_mode = match index {
            1 => SelectionMode::Add,
            2 => SelectionMode::Subtract,
            _ => SelectionMode::Replace,
        };
        self.refresh(R_MAIN);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use openscissors_core::document::Document;
    use openscissors_core::mask::{Stroke, StrokeKind};
    use openscissors_core::types::Image;
    use std::cell::RefCell;

    fn disc_doc(id: u64) -> Document {
        let mut img = Image::new(80, 60);
        for y in 0..60 {
            for x in 0..80 {
                let inside = (x as f64 - 40.0).powi(2) + (y as f64 - 30.0).powi(2) < 400.0;
                img.data[(y * 80 + x) * 3..(y * 80 + x) * 3 + 3].copy_from_slice(if inside { &[210, 60, 50] } else { &[40, 80, 180] });
            }
        }
        let mut doc = Document::new(id, img, None);
        doc.set_param("mask.graphcut", serde_json::json!(true)).unwrap();
        doc.add_stroke(Stroke { kind: StrokeKind::Keep, width: 5, points: vec![[40.5, 30.5]] });
        doc.add_stroke(Stroke { kind: StrokeKind::Remove, width: 5, points: vec![[5.5, 5.5]] });
        doc
    }

    fn state_with(doc: Document) -> AppState {
        let home = std::env::temp_dir().join(format!("osc-kick-{}", std::process::id()));
        std::env::set_var("OPENSCISSORS_HOME", &home);
        let state = AppState::new();
        *state.doc.guard() = Some(doc);
        state
    }

    #[test]
    fn the_exact_mask_arrives_and_a_newer_edit_gets_its_own_job() {
        let state = state_with(disc_doc(1));
        let first = take_job(&state).expect("a graph-cut document needs a job");
        // An edit during the job: the result of the first one must not be shown as final.
        state.doc.guard().as_mut().unwrap().add_stroke(Stroke { kind: StrokeKind::Keep, width: 5, points: vec![[42.5, 30.5]] });
        let applied = RefCell::new(Vec::new());
        run_alpha_jobs(&state, first, |id| applied.borrow_mut().push(id));
        assert!(!state.doc.guard().as_mut().unwrap().computing(), "all jobs done");
        assert!(!applied.borrow().is_empty());
    }

    #[test]
    fn a_result_for_a_closed_document_is_ignored() {
        let state = state_with(disc_doc(1));
        let first = take_job(&state).unwrap();
        *state.doc.guard() = Some(disc_doc(2)); // another image was opened
        let applied = RefCell::new(0);
        run_alpha_jobs(&state, first, |_| *applied.borrow_mut() += 1);
        assert_eq!(*applied.borrow(), 0);
        assert!(state.doc.guard().as_ref().unwrap().computing(), "the new document still needs its own job");
    }

    fn spec(ram_mb: u32) -> ModelSpec {
        let mut s = openscissors_core::model::builtin_catalog().remove(0);
        s.ram_mb = ram_mb;
        s
    }

    #[test]
    fn a_model_over_the_budget_is_refused_with_advice_and_an_unmeasured_one_is_allowed() {
        let err = check_memory(&spec(u32::MAX), false).unwrap_err();
        assert!(err.contains("25% of the memory") && err.contains("lighter model"), "{err}");
        assert!(!fits_budget(&spec(u32::MAX), false));
        assert!(check_memory(&spec(0), true).is_ok(), "unknown need: do not block");
        assert!(check_memory(&spec(1), false).is_ok(), "1 MB always fits");
    }

    #[test]
    fn the_speed_mode_needs_more_than_twice_the_lean_memory() {
        assert!(model_need_mb(&spec(1000), true) > 2 * model_need_mb(&spec(1000), false));
        assert_eq!(model_need_mb(&spec(0), true), 0);
    }

    #[test]
    fn the_pipeline_estimate_grows_with_the_image() {
        assert!(pipeline_mb(7680, 4320) > 10 * pipeline_mb(1920, 1080) / 2);
        assert!(pipeline_mb(7680, 4320) < 1000, "an 8K photo costs well under 1 GB around the network");
    }
}
