//! Models and background computation: memory limits, loading, prediction, the exact mask jobs,
//! downloads and smart select.

use openscissors_core::sync::LockExt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use openscissors_core::document::{AlphaJob, SelectionMode};
use openscissors_core::model::{Accelerator, Embeddings, ModelKind, ModelSpec, Predictor, Prompt, Sam2};
use openscissors_core::sysmem;
use openscissors_core::types::{Image, Mask};

use crate::ctl::{Ctl, Prediction, R_BATCH, R_MAIN, R_MODELS, R_VIEW};
use crate::i18n::{t, tf, tr_message};
use crate::state::{AppState, ModelStatus, LOCAL_MODEL_ID};

/// Refuse to load a model the machine has no room for: a failed allocation would end the whole
/// application without a message.
pub fn check_memory(spec: &ModelSpec, fast: bool) -> Result<(), String> {
    let need = model_need_mb(spec, fast);
    if let Some(budget) = sysmem::budget_mb() {
        if need > budget {
            let (need, budget, percent) = (need.to_string(), budget.to_string(), sysmem::budget_percent().to_string());
            let vars = [("model", spec.name.as_str()), ("need", need.as_str()), ("budget", budget.as_str()), ("percent", percent.as_str())];
            return Err(tf("{model} needs about {need} MB, more than the {budget} MB ({percent}% of the memory) the application may use. Choose a lighter model.", &vars));
        }
    }
    match sysmem::available_memory_mb() {
        Some(free) if need > 0 && free < need => {
            let (need, free) = (need.to_string(), free.to_string());
            let vars = [("model", spec.name.as_str()), ("need", need.as_str()), ("free", free.as_str())];
            Err(if fast {
                tf("Not enough free memory to run {model}: about {need} MB are needed and {free} MB are free. Close other applications or turn off \"Use more memory for speed\".", &vars)
            } else {
                tf("Not enough free memory to run {model}: about {need} MB are needed and {free} MB are free. Close other applications or choose a lighter model.", &vars)
            })
        }
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
    let ready: Vec<_> = state
        .models()
        .into_iter()
        .filter(|s| s.kind == ModelKind::Segmenter && state.is_ready(s) && (s.id == LOCAL_MODEL_ID || fits_budget(s, false)))
        .collect();
    openscissors_core::model::default_model(&ready, wanted.as_deref()).map(|s| s.id.clone())
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
        let (w, h, need, budget) = (w.to_string(), h.to_string(), need.to_string(), budget.to_string());
        let percent = sysmem::budget_percent().to_string();
        let vars = [
            ("w", w.as_str()),
            ("h", h.as_str()),
            ("need", need.as_str()),
            ("model", model.spec().name.as_str()),
            ("budget", budget.as_str()),
            ("percent", percent.as_str()),
        ];
        return Err(tf("A {w} x {h} px image needs about {need} MB with {model}, more than the {budget} MB ({percent}% of the memory) the application may use. Choose a lighter model.", &vars));
    }
    Ok(())
}

/// One model of a comparison, for the panel.
#[derive(Clone)]
pub struct CompareResult {
    /// Empty when there is nothing to use: the model did not run, or it is the one in use.
    pub id: String,
    pub name: String,
    pub detail: String,
    /// The cut-out over a checker, 240 px high at most (RGBA).
    pub thumb: Option<openscissors_core::types::Rgba>,
}

/// The cut-out of `image` by `mask` over a grey checker, small: the right tile of a one-model contact sheet.
fn cutout_thumb(image: &Image, mask: &Mask) -> openscissors_core::types::Rgba {
    let sheet = openscissors_core::compare::contact_sheet(image, &[(String::new(), mask.clone())]);
    let tile = (sheet.w - 8) / 2; // two tiles and the gap between them
    let mut out = openscissors_core::types::Rgba::new(tile, sheet.h);
    for y in 0..sheet.h {
        let from = (y * sheet.w + sheet.w - tile) * 4;
        out.data[y * tile * 4..(y + 1) * tile * 4].copy_from_slice(&sheet.data[from..from + tile * 4]);
    }
    out
}

type PendingJob = (u64, AlphaJob);

fn take_job(state: &AppState) -> Option<PendingJob> {
    let mut g = state.doc.guard();
    g.as_mut().and_then(|d| d.take_alpha_job().map(|j| (d.id, j)))
}

/// Run `first`, install its result, and keep going while newer edits ask for another job.
/// `on_applied(doc_id)` is called each time an up-to-date mask was installed.
/// A job that panics (a bug) is abandoned: the document keeps its provisional mask instead of waiting for ever.
pub fn run_alpha_jobs(state: &AppState, first: PendingJob, on_applied: impl Fn(u64)) {
    run_alpha_jobs_with(state, first, AlphaJob::run, on_applied);
}

fn run_alpha_jobs_with(state: &AppState, first: PendingJob, run: impl Fn(&AlphaJob) -> Mask, on_applied: impl Fn(u64)) {
    let mut current = Some(first);
    while let Some((doc_id, job)) = current.take() {
        let mask = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(&job)));
        let (applied, next) = {
            let mut g = state.doc.guard();
            match (g.as_mut(), mask) {
                (Some(d), Ok(mask)) if d.id == doc_id => (d.finish_alpha_job(job.rev(), mask), d.take_alpha_job().map(|j| (d.id, j))),
                (Some(d), Err(_)) if d.id == doc_id => {
                    d.abandon_alpha_job(job.rev());
                    (true, None)
                }
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

/// SAM 2, loaded on first use. The slot stays locked while it loads: two quick clicks load it once.
fn ensure_sam(state: &AppState) -> Result<Arc<Sam2>, String> {
    let mut slot = state.sam.guard();
    if let Some(s) = slot.clone() {
        return Ok(s);
    }
    let spec = state.registry.available().into_iter().find(|s| s.kind == ModelKind::Sam2).ok_or(SAM_MISSING)?;
    if !state.is_ready(&spec) {
        return Err(SAM_MISSING.into());
    }
    let sam = Arc::new(Sam2::load(&state.registry.model_dir(&spec)).map_err(|e| e.to_string())?);
    *slot = Some(sam.clone());
    Ok(sam)
}

/// The encoder output for document `doc_id`, computed once: the cache stays locked while it is computed, so a second
/// click waits for it instead of running the encoder again.
fn embeddings(state: &AppState, sam: &Sam2, key: (u64, u64), image: &Image) -> Result<Arc<Embeddings>, String> {
    let mut cache = state.sam_cache.guard();
    if let Some((_, e)) = cache.as_ref().filter(|(k, _)| *k == key) {
        return Ok(e.clone());
    }
    let e = Arc::new(sam.embed(image).map_err(|e| e.to_string())?);
    *cache = Some((key, e.clone()));
    Ok(e)
}

impl Ctl {
    // -- exact mask in the background

    /// Start computing the exact mask (graph-cut, hair matting) off the interaction path when the
    /// document needs it. The views keep showing a provisional mask until the exact one is ready.
    pub fn kick_alpha(self: &Arc<Self>) {
        let Some(first) = take_job(&self.st) else { return };
        let me = self.clone();
        self.spawn(move || {
            run_alpha_jobs(&me.st, first, |_| me.refresh_doc());
        });
    }

    /// Compute the textured fill of the original behind the subject off the interface thread when the document still
    /// shows the smooth one. One job at a time; when it ends the state is read again, which starts the next job if
    /// the mask changed in the meantime. A job that panics (a bug) leaves the smooth fill as the result.
    pub fn kick_fill(self: &Arc<Self>) {
        let taken = self.st.doc.guard().as_mut().and_then(|d| d.take_fill_job().map(|j| (d.id, j)));
        let Some((doc_id, job)) = taken else { return };
        let me = self.clone();
        self.spawn(move || {
            let filled = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| job.run()));
            if let Some(d) = me.st.doc.guard().as_mut().filter(|d| d.id == doc_id) {
                match filled {
                    Ok(img) => {
                        d.finish_fill_job(job, img);
                    }
                    Err(_) => d.abandon_fill_job(job),
                }
            }
            me.refresh_doc();
        });
    }

    // -- comparing the models

    /// Run every downloaded cut-out model on the open picture, one after the other (one in memory at a time; those that
    /// would not fit the memory budget are listed, not run), and show each cut-out with what it kept, how long it took
    /// and how much it agrees with the model in use.
    pub fn compare_models(self: &Arc<Self>) {
        let Some(image) = self.st.doc.guard().as_ref().map(|d| d.image.clone()) else { return };
        let specs: Vec<ModelSpec> =
            self.st.models().into_iter().filter(|s| s.kind == ModelKind::Segmenter && self.st.is_ready(s)).collect();
        if specs.len() < 2 || self.ux.guard().comparing {
            return;
        }
        let active = self.st.model.guard().as_ref().map(|m| m.spec().id.clone());
        {
            let mut ux = self.ux.guard();
            ux.comparing = true;
            ux.compare.clear();
        }
        self.refresh(R_MODELS);
        let me = self.clone();
        let owner = self.clone();
        self.spawn_or(
            move || {
                let accelerator = me.st.settings.guard().accelerator;
                let mut masks: Vec<(ModelSpec, Option<(Mask, u128)>)> = Vec::new();
                for spec in &specs {
                    me.say(tf("Running {name}…", &[("name", &spec.name)]), false);
                    let result = fits_budget(spec, false)
                        .then(|| {
                            let started = std::time::Instant::now();
                            let p = Predictor::load_with(spec, &me.st.model_path(spec), accelerator).ok()?;
                            check_pipeline(&p, image.w, image.h).ok()?;
                            let mask = p.predict(&image, true).ok()?;
                            Some((mask, started.elapsed().as_millis()))
                        })
                        .flatten();
                    masks.push((spec.clone(), result));
                }
                let reference =
                    masks.iter().find(|(s, r)| Some(&s.id) == active.as_ref() && r.is_some()).map(|(s, r)| (s.name.clone(), r.clone()));
                let rows: Vec<CompareResult> = masks
                    .iter()
                    .map(|(spec, result)| match result {
                        Some((mask, ms)) => {
                            let kept = format!("{:.0}", openscissors_core::compare::coverage(mask) * 100.0);
                            let mut detail = tf("{kept}% kept, {ms} ms", &[("kept", &kept), ("ms", &ms.to_string())]);
                            if let Some((name, Some((m, _)))) = &reference {
                                if Some(&spec.id) != active.as_ref() {
                                    let agree = format!("{:.0}", openscissors_core::compare::iou(mask, m) * 100.0);
                                    detail.push_str(&format!("\n{}", tf("{agree}% like {name}", &[("agree", &agree), ("name", name)])));
                                }
                            }
                            {
                                // The model in use needs no "Use" button.
                                let id = if Some(&spec.id) == active.as_ref() { String::new() } else { spec.id.clone() };
                                CompareResult { id, name: spec.name.clone(), detail, thumb: Some(cutout_thumb(&image, mask)) }
                            }
                        }
                        None => CompareResult {
                            id: String::new(),
                            name: spec.name.clone(),
                            detail: t("Not run: too heavy for the memory budget, or it could not load."),
                            thumb: None,
                        },
                    })
                    .collect();
                {
                    let mut ux = me.ux.guard();
                    ux.compare = rows;
                    ux.comparing = false;
                }
                me.say("", false);
                me.refresh(R_MODELS);
            },
            move || owner.ux.guard().comparing = false,
        );
    }

    // -- loading a model

    /// Load `id` on a background thread and make it the active model. Loads run one at a time; when several are asked
    /// for in a row (clicking "Use" on two models, changing the processor setting) only the latest one is carried out.
    pub fn load_model_async(self: &Arc<Self>, id: String) {
        let gen = self.st.model_gen.fetch_add(1, Ordering::SeqCst) + 1;
        // The model in use is let go now: two models are never held at once (a running prediction keeps its own).
        *self.st.model.guard() = None;
        *self.st.model_status.guard() = ModelStatus { active: Some(id.clone()), loading: true, ready: false, error: None, backend: None };
        self.refresh(R_MAIN | R_MODELS);
        let me = self.clone();
        let st = self.st.clone();
        let failed_id = id.clone();
        let reset = move || {
            let error = Some(t("The model could not be loaded"));
            *st.model_status.guard() = ModelStatus { active: Some(failed_id), loading: false, ready: false, error, backend: None };
        };
        self.spawn_or(
            move || {
                let state = &me.st;
                let _one_at_a_time = state.model_load.guard();
                let latest = || state.model_gen.load(Ordering::SeqCst) == gen;
                if !latest() {
                    return; // a newer load was asked for meanwhile
                }
                let result = (|| -> Result<Predictor, String> {
                    let spec = state.models().into_iter().find(|s| s.id == id).ok_or_else(|| t("Unknown model."))?;
                    let path = state.model_path(&spec);
                    if !path.is_file() {
                        return Err(t("The model file is missing."));
                    }
                    let (accel, fast) = {
                        let s = state.settings.guard();
                        // The speed mode is dropped when it would not fit the memory budget.
                        (s.accelerator, s.fast_memory && fits_budget(&spec, true))
                    };
                    check_memory(&spec, fast)?;
                    let p = Predictor::load_with_memory(&spec, &path, accel, fast).map_err(|e| e.to_string())?;
                    // A GPU session was already run once by the loader (that run is how it is checked).
                    if p.backend() == "CPU" {
                        p.warmup().map_err(|e| e.to_string())?;
                    }
                    Ok(p)
                })();
                if !latest() {
                    return; // asked for another model while this one loaded: it is dropped here
                }
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
            },
            reset,
        );
    }

    pub fn use_model(self: &Arc<Self>, id: &str) {
        let Some(spec) = self.st.models().into_iter().find(|s| s.id == id) else { return };
        if spec.kind != ModelKind::Segmenter {
            return self.say(t("This model is used by Smart select, not as the cut-out model."), true);
        }
        if !self.st.is_ready(&spec) {
            return self.say(t("Download this model first."), true);
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
        let owner = self.clone();
        let failed_id = id.to_string();
        let reset = move || {
            owner.st.downloads.guard().remove(&failed_id);
            owner.ux.guard().downloads.remove(&failed_id);
        };
        self.spawn_or(
            move || {
                let mut last = std::time::Instant::now();
                let mut progress = |done: u64, total: u64| {
                    if last.elapsed().as_millis() > 100 || done == total {
                        last = std::time::Instant::now();
                        me.ux.guard().downloads.insert(spec.id.clone(), (done, total));
                        me.refresh(R_MODELS);
                    }
                };
                let result = me.st.registry.download(&spec, &mut progress, &cancel);
                me.st.rescan_models();
                me.st.downloads.guard().remove(&spec.id);
                me.ux.guard().downloads.remove(&spec.id);
                match &result {
                    Ok(_) => me.say(t("Model downloaded"), false),
                    Err(openscissors_core::Error::Cancelled) => {}
                    // The kind of error is already said by "Download failed": only its own message follows.
                    Err(openscissors_core::Error::Download(m)) => {
                        me.say(tf("Download failed: {error}", &[("error", &tr_message(m))]), true)
                    }
                    Err(e) => me.say(tf("Download failed: {error}", &[("error", &tr_message(&e.to_string()))]), true),
                }
                me.refresh(R_MODELS | R_MAIN);
                // Nothing is activated here: the model is picked when an image needs it (the user's choice,
                // otherwise the lightest on disk), so downloading a heavy model never makes the application heavy.
                if result.is_ok() && spec.kind == ModelKind::Segmenter {
                    me.maybe_predict();
                }
            },
            reset,
        );
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
            self.say(tr_message(&e.to_string()), true);
        }
        self.st.rescan_models();
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
        // A picture that came with its own mask (a cut-out, a project) needs no model, so none is loaded for it.
        if self.st.doc.guard().as_ref().is_some_and(|d| d.has_prediction()) && !self.ux.guard().repredict {
            return;
        }
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
                let message = error.unwrap_or_else(|| t("The model could not be loaded"));
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
        let gen = self.st.predict_gen.fetch_add(1, Ordering::SeqCst) + 1;
        self.ux.guard().prediction = Prediction::Running;
        self.refresh(R_MAIN);
        let me = self.clone();
        let owner = self.clone();
        let reset = move || {
            *owner.st.predicted_doc.guard() = None;
            owner.ux.guard().prediction = Prediction::Failed;
        };
        self.spawn_or(
            move || {
                let result = model.predict(&image, true);
                // Another model was chosen while this one ran: its prediction is the one that counts.
                let current = || me.st.predict_gen.load(Ordering::SeqCst) == gen;
                if !current() {
                    return;
                }
                match result {
                    Ok(mask) => {
                        let applied = {
                            let mut g = me.st.doc.guard();
                            match g.as_mut() {
                                Some(doc) if doc.id == id && current() => {
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
                    Err(e) => {
                        // Not "done" for this document: the next occasion (another model, more memory) tries again.
                        let mut running = me.st.predicted_doc.guard();
                        if *running == Some(id) {
                            *running = None;
                        }
                        drop(running);
                        me.prediction_failed(e.to_string());
                    }
                }
            },
            reset,
        );
    }

    fn prediction_failed(self: &Arc<Self>, message: String) {
        self.ux.guard().prediction = Prediction::Failed;
        self.say(tf("Prediction failed: {message}", &[("message", &tr_message(&message))]), true);
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
        let (doc_id, geometry, image) = {
            let g = self.st.doc.guard();
            let Some(d) = g.as_ref() else { return };
            (d.id, d.geometry(), d.image.clone())
        };
        let prompts = {
            let mut ux = self.ux.guard();
            ux.smart_busy = true;
            ux.prompts.clone()
        };
        let gen = self.st.smart_gen.fetch_add(1, Ordering::SeqCst) + 1;
        self.say(t("Analysing the image…"), false);
        self.refresh(R_MAIN | R_VIEW);
        let me = self.clone();
        let owner = self.clone();
        let reset = move || {
            let mut ux = owner.ux.guard();
            ux.smart_busy = false;
            ux.prompts.clear();
        };
        self.spawn_or(
            move || {
                let result = (|| -> Result<_, String> {
                    let sam = ensure_sam(&me.st)?;
                    let emb = embeddings(&me.st, &sam, (doc_id, geometry), &image)?;
                    let found = sam.select_candidates(&emb, &prompts, &image).map_err(|e| e.to_string())?;
                    let mask = openscissors_core::model::sam::refine(&image, &found[0]);
                    Ok((mask, found))
                })();
                // A later click started another run: its selection (with more points) is the one to show.
                if me.st.smart_gen.load(Ordering::SeqCst) != gen {
                    return;
                }
                me.ux.guard().smart_busy = false;
                match result {
                    Ok((mask, found)) => {
                        if let Some(d) = me.st.doc.guard().as_mut().filter(|d| d.id == doc_id && d.geometry() == geometry) {
                            d.set_selection(Some(mask));
                        }
                        let total = found.len();
                        {
                            let mut ux = me.ux.guard();
                            ux.smart_candidates = Arc::new(found);
                            ux.smart_choice = 0;
                        }
                        me.say_candidate(0, total);
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
                            me.say(tr_message(&e), true);
                        }
                        me.refresh(R_MAIN | R_VIEW);
                    }
                }
            },
            reset,
        );
    }

    /// Show the next candidate of the smart selection (Tab): the decoder proposes a part, a whole and something in
    /// between for the same points.
    pub fn smart_next_candidate(self: &Arc<Self>) {
        let (found, choice) = {
            let mut ux = self.ux.guard();
            if ux.prompts.is_empty() || ux.smart_busy || ux.smart_candidates.len() < 2 {
                return;
            }
            ux.smart_choice = (ux.smart_choice + 1) % ux.smart_candidates.len();
            (ux.smart_candidates.clone(), ux.smart_choice)
        };
        let (doc_id, geometry, image) = {
            let g = self.st.doc.guard();
            let Some(d) = g.as_ref() else { return };
            (d.id, d.geometry(), d.image.clone())
        };
        let gen = self.st.smart_gen.load(Ordering::SeqCst);
        let me = self.clone();
        self.spawn(move || {
            let mask = openscissors_core::model::sam::refine(&image, &found[choice]);
            // Points added meanwhile start a new run, whose candidates are the ones to show.
            if me.st.smart_gen.load(Ordering::SeqCst) != gen || me.ux.guard().smart_choice != choice {
                return;
            }
            if let Some(d) = me.st.doc.guard().as_mut().filter(|d| d.id == doc_id && d.geometry() == geometry) {
                d.set_selection(Some(mask));
            }
            me.say_candidate(choice, found.len());
            me.refresh_doc();
        });
    }

    /// Say which candidate is shown, when there is a choice.
    fn say_candidate(self: &Arc<Self>, choice: usize, total: usize) {
        if total > 1 {
            let (n, total) = ((choice + 1).to_string(), total.to_string());
            self.say(tf("Candidate {n} of {total}. Tab shows the next one, Enter applies it.", &[("n", &n), ("total", &total)]), false);
        } else {
            self.say("", false);
        }
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
        let state = AppState::at(&home);
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
    fn a_job_that_panics_is_abandoned_and_the_document_stops_waiting() {
        let state = state_with(disc_doc(1));
        let first = take_job(&state).unwrap();
        let applied = RefCell::new(0);
        run_alpha_jobs_with(&state, first, |_| panic!("a bug in the mask code"), |_| *applied.borrow_mut() += 1);
        let mut g = state.doc.guard();
        let d = g.as_mut().unwrap();
        assert!(!d.computing(), "no endless 'computing'");
        assert_eq!(*applied.borrow(), 1, "the views are told to refresh");
        d.add_stroke(Stroke { kind: StrokeKind::Keep, width: 5, points: vec![[42.5, 30.5]] });
        assert!(d.take_alpha_job().is_some(), "the next edit starts a job again");
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
