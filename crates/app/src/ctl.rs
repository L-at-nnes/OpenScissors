//! The controller: owns the interface state that is not in the document, turns events into
//! document operations and pushes the result into the Slint properties.

use openscissors_core::sync::LockExt;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use openscissors_core::document::{DocState, SelectionMode};
use openscissors_core::mask::StrokeKind;
use openscissors_core::model::{ModelKind, Prompt};
use openscissors_core::params::{BgMode, FilterKind, FilterSpec, Params, ASPECT_RATIOS, OUTPUT_WIDTHS, PRESETS};
use openscissors_core::types::Rect;
use slint::{ComponentHandle, ModelRc, SharedPixelBuffer, SharedString, VecModel};

use crate::cropmath::{same_rect, Handle};
use crate::i18n::{self, t, tf};
use crate::render::{Job, Output, Renderer};
use crate::state::AppState;
use crate::viewport::{device_grid, fit_view, to_screen, visible_region, View};
use crate::{Backend, Box2, Dot, FilterV, MainWindow, ModelRow, ParamsV, PluginRow};

pub const R_VIEW: u32 = 1;
pub const R_MAIN: u32 = 2;
pub const R_MODELS: u32 = 4;
pub const R_MENU: u32 = 8;
pub const R_BATCH: u32 = 16;
pub const R_ALL: u32 = 31;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tool {
    Keep,
    Remove,
    Erase,
    Select,
    Hair,
    HairErase,
    Move,
}

impl Tool {
    pub fn as_str(self) -> &'static str {
        match self {
            Tool::Keep => "keep",
            Tool::Remove => "remove",
            Tool::Erase => "erase",
            Tool::Select => "select",
            Tool::Hair => "hair",
            Tool::HairErase => "hair_erase",
            Tool::Move => "move",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "keep" => Tool::Keep,
            "remove" => Tool::Remove,
            "erase" => Tool::Erase,
            "select" => Tool::Select,
            "hair" => Tool::Hair,
            "hair_erase" => Tool::HairErase,
            "move" => Tool::Move,
            _ => return None,
        })
    }

    /// The stroke a brush of this tool leaves, `None` for the tools that do not paint.
    pub fn stroke_kind(self) -> Option<StrokeKind> {
        match self {
            Tool::Keep => Some(StrokeKind::Keep),
            Tool::Remove => Some(StrokeKind::Remove),
            Tool::Erase => Some(StrokeKind::Erase),
            Tool::Hair => Some(StrokeKind::Hair),
            Tool::HairErase => Some(StrokeKind::HairErase),
            Tool::Select | Tool::Move => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Prediction {
    Idle,
    Running,
    Failed,
    NoModel,
}

pub struct SmartDrag {
    pub from: (f64, f64),
    pub to: (f64, f64),
    pub button: i32,
    pub alt: bool,
}

pub struct CropDrag {
    pub handle: Handle,
    pub start: (f64, f64),
    pub from: Rect,
}

#[derive(Default)]
pub struct BatchUx {
    pub running: bool,
    pub done: usize,
    pub total: usize,
    pub file: Option<String>,
    pub report: String,
    pub format: i32,
    /// 0 = the saved defaults, 1 = the settings of the open image.
    pub settings: i32,
}

/// What the interface needs besides the document itself.
pub struct Ux {
    pub doc: Option<DocState>,
    pub tool: Tool,
    pub marker: u32,
    pub view: View,
    pub pane: (f64, f64),
    pub dpr: f64,
    pub space: bool,
    pub prompts: Vec<Prompt>,
    pub smart_mode: SelectionMode,
    pub smart_busy: bool,
    pub stroke: Option<Vec<(f64, f64)>>,
    pub pan_from: Option<(f64, f64)>,
    pub drag: Option<SmartDrag>,
    pub hover: Option<(i32, f64, f64)>,
    pub crop_drag: Option<CropDrag>,
    pub live_crop: Option<Rect>,
    pub prediction: Prediction,
    pub last_export: Option<String>,
    pub last_saved: Option<String>,
    pub downloads: HashMap<String, (u64, u64)>,
    pub batch: BatchUx,
    pub status: (String, bool),
    pub drop_active: bool,
    pub render_key: Option<String>,
    /// A new model was chosen: predict the open image again with it.
    pub repredict: bool,
}

pub struct Ctl {
    pub st: Arc<AppState>,
    pub ui: slint::Weak<MainWindow>,
    pub ux: Mutex<Ux>,
    renderer: Renderer,
    pending: AtomicU32,
    status_epoch: AtomicU64,
    /// Id of the open document (0 = none), read by the render delivery to drop stale images.
    current_doc: Arc<AtomicU64>,
}

pub const KIND_KEYS: [&str; 6] = ["none", "blur", "colorize", "grayscale", "negative", "pixelate"];
pub const MODE_KEYS: [&str; 4] = ["transparent", "color", "image", "original"];

fn kind_index(k: FilterKind) -> i32 {
    match k {
        FilterKind::None => 0,
        FilterKind::Blur => 1,
        FilterKind::Colorize => 2,
        FilterKind::Grayscale => 3,
        FilterKind::Negative => 4,
        FilterKind::Pixelate => 5,
    }
}

fn mode_index(m: BgMode) -> i32 {
    match m {
        BgMode::Transparent => 0,
        BgMode::Color => 1,
        BgMode::Image => 2,
        BgMode::Original => 3,
    }
}

fn rgb(c: [u8; 3]) -> slint::Color {
    slint::Color::from_rgb_u8(c[0], c[1], c[2])
}

fn filter_v(f: &FilterSpec) -> FilterV {
    FilterV { kind: kind_index(f.kind), radius: f.radius as i32, strength: f.strength as i32, color: rgb(f.color) }
}

pub fn params_v(p: &Params) -> ParamsV {
    ParamsV {
        mask_offset: p.mask.offset,
        graphcut: p.mask.graphcut,
        feather: p.foreground.feather as i32,
        defringe: p.foreground.defringe as i32,
        fg_filter: filter_v(&p.foreground.filter),
        bg_mode: mode_index(p.background.mode),
        bg_color: rgb(p.background.color),
        bg_image: SharedString::from(p.background.image_path.as_deref().and_then(|s| s.rsplit(['/', '\\']).next()).unwrap_or("")),
        fill_hole: p.background.fill_hole,
        ratio: ASPECT_RATIOS.iter().position(|(k, _)| *k == p.background.aspect_ratio).unwrap_or(0) as i32,
        fit: p.background.fit_to_result,
        padding: p.background.padding as i32,
        bg_filter: filter_v(&p.background.filter),
        sh_on: p.shadow.enabled,
        sh_opacity: p.shadow.opacity as i32,
        sh_blur: p.shadow.blur_radius as i32,
        sh_color: rgb(p.shadow.color),
        sh_dx: p.shadow.offset_x,
        sh_dy: p.shadow.offset_y,
        has_crop: p.crop.is_some(),
        preset: PRESETS.iter().position(|q| q.matches(p)).map_or(0, |i| i as i32 + 1),
        out_width: OUTPUT_WIDTHS.iter().position(|&w| w == p.background.output_width).unwrap_or(0) as i32,
    }
}

fn strings(v: impl IntoIterator<Item = String>) -> ModelRc<SharedString> {
    ModelRc::new(VecModel::from(v.into_iter().map(SharedString::from).collect::<Vec<_>>()))
}

fn file_name(path: &str) -> String {
    path.rsplit(['/', '\\']).next().unwrap_or(path).to_string()
}

impl Ctl {
    pub fn new(ui: &MainWindow) -> Arc<Self> {
        let st = Arc::new(AppState::new());
        let locale = st.settings.guard().locale.clone();
        let code = i18n::apply(&locale);
        let _ = slint::select_bundled_translation(code);
        let marker = st.settings.guard().marker_size;
        let current_doc = Arc::new(AtomicU64::new(0));
        let weak = ui.as_weak();
        let alive = current_doc.clone();
        let renderer = Renderer::new(st.clone(), move |out: Output| {
            if alive.load(Ordering::Relaxed) != out.job.doc {
                return;
            }
            let _ = weak.upgrade_in_event_loop(move |ui| {
                let buf = SharedPixelBuffer::<slint::Rgba8Pixel>::clone_from_slice(&out.pixels, out.job.w as u32, out.job.h as u32);
                let img = slint::Image::from_rgba8(buf);
                let r = out.job.region;
                let region = Box2 { x0: r.x0 as f32, y0: r.y0 as f32, x1: r.x1 as f32, y1: r.y1 as f32 };
                let b = ui.global::<Backend>();
                if out.job.side == 0 {
                    b.set_src_img(img);
                    b.set_src_region(region);
                } else {
                    b.set_res_img(img);
                    b.set_res_region(region);
                }
            });
        });
        Arc::new(Self {
            st,
            ui: ui.as_weak(),
            ux: Mutex::new(Ux {
                doc: None,
                tool: Tool::Move,
                marker,
                view: View::default(),
                pane: (0.0, 0.0),
                dpr: 1.0,
                space: false,
                prompts: Vec::new(),
                smart_mode: SelectionMode::Replace,
                smart_busy: false,
                stroke: None,
                pan_from: None,
                drag: None,
                hover: None,
                crop_drag: None,
                live_crop: None,
                prediction: Prediction::Idle,
                last_export: None,
                last_saved: None,
                downloads: HashMap::new(),
                batch: BatchUx::default(),
                status: (String::new(), false),
                drop_active: false,
                render_key: None,
                repredict: false,
            }),
            renderer,
            pending: AtomicU32::new(0),
            status_epoch: AtomicU64::new(0),
            current_doc,
        })
    }

    // -- refreshing the interface

    /// Ask for the given parts of the interface to be rebuilt on the interface thread (coalesced).
    pub fn refresh(self: &Arc<Self>, bits: u32) {
        if self.pending.fetch_or(bits, Ordering::AcqRel) == 0 {
            let me = self.clone();
            let _ = self.ui.upgrade_in_event_loop(move |ui| {
                let bits = me.pending.swap(0, Ordering::AcqRel);
                me.sync(&ui, bits);
            });
        }
    }

    pub fn with_ui(self: &Arc<Self>, f: impl FnOnce(&Arc<Ctl>, &MainWindow) + Send + 'static) {
        let me = self.clone();
        let _ = self.ui.upgrade_in_event_loop(move |ui| f(&me, &ui));
    }

    /// Show `text` in the status bar; informational messages fade after a few seconds.
    pub fn say(self: &Arc<Self>, text: impl Into<String>, error: bool) {
        let text = text.into();
        let epoch = self.status_epoch.fetch_add(1, Ordering::Relaxed) + 1;
        let shown_for = std::time::Duration::from_secs(if error { 10 } else { 4 });
        let fades = !text.is_empty();
        self.ux.guard().status = (text, error);
        self.refresh(R_MAIN);
        if fades {
            // The timer lives on the interface thread: no thread per message.
            self.with_ui(move |me, _| {
                let me = me.clone();
                slint::Timer::single_shot(shown_for, move || {
                    if me.status_epoch.load(Ordering::Relaxed) == epoch {
                        me.ux.guard().status = (String::new(), false);
                        me.refresh(R_MAIN);
                    }
                });
            });
        }
    }

    pub fn sync(self: &Arc<Self>, ui: &MainWindow, bits: u32) {
        {
            let mut ux = self.ux.guard();
            let dpr = ui.window().scale_factor() as f64;
            if (dpr - ux.dpr).abs() > 1e-6 {
                ux.dpr = dpr;
                ux.render_key = None;
            }
        }
        if bits & R_MAIN != 0 {
            self.sync_main(ui);
        }
        if bits & R_MODELS != 0 {
            self.sync_models(ui);
        }
        if bits & R_MENU != 0 {
            self.sync_menu(ui);
        }
        if bits & R_BATCH != 0 {
            self.sync_batch(ui);
        }
        if bits & (R_VIEW | R_MAIN) != 0 {
            self.sync_view(ui);
        }
    }

    fn sync_main(self: &Arc<Self>, ui: &MainWindow) {
        let b = ui.global::<Backend>();
        let (status, active_name) = {
            let s = self.st.model_status.guard().clone();
            let name = s.active.as_ref().and_then(|id| self.st.models().into_iter().find(|m| &m.id == id)).map(|m| m.name);
            (s, name)
        };
        let any_ready = self.st.models().iter().any(|m| m.kind == ModelKind::Segmenter && self.st.is_ready(m));
        let ux = self.ux.guard();
        let doc = ux.doc.as_ref();
        b.set_has_doc(doc.is_some());
        let title = match doc {
            Some(d) => format!("{}{} - OpenScissors", if d.modified { "• " } else { "" }, d.name.as_deref().unwrap_or("Untitled")),
            None => "OpenScissors".to_string(),
        };
        b.set_doc_title(title.into());
        b.set_can_undo(doc.is_some_and(|d| d.can_undo));
        b.set_can_redo(doc.is_some_and(|d| d.can_redo));
        if let Some(d) = doc {
            b.set_p(params_v(&d.params));
        }
        b.set_ratio_options(strings(ASPECT_RATIOS.iter().map(|(k, _)| if *k == "free" { t("Unconstrained") } else { k.to_string() })));
        b.set_preset_options(strings([
            t("Custom"),
            t("Marketplace 1000 px"),
            t("Marketplace 2000 px"),
            t("Thumbnail 500 px"),
            t("Banner 16:9"),
        ]));
        b.set_width_options(strings(OUTPUT_WIDTHS.iter().map(|&w| if w == 0 { t("Original size") } else { format!("{w} px") })));
        b.set_tool(ux.tool.as_str().into());
        b.set_marker_size(ux.marker as i32);
        b.set_smart_count(ux.prompts.len() as i32);
        b.set_smart_mode(match ux.smart_mode {
            SelectionMode::Replace => 0,
            SelectionMode::Add => 1,
            SelectionMode::Subtract => 2,
        });
        b.set_smart_busy(ux.smart_busy);
        b.set_no_model(doc.is_some() && !status.ready && !status.loading);
        b.set_busy(ux.prediction == Prediction::Running || doc.is_some_and(|d| d.computing));
        b.set_drop_active(ux.drop_active);
        let (chip, tone) = if status.loading {
            (t("Loading model…"), 0)
        } else if status.ready {
            (active_name.unwrap_or_else(|| t("Model ready")), 1)
        } else if status.error.is_some() {
            (t("Model error"), 2)
        } else if any_ready {
            (t("Loads with the first image"), 3)
        } else {
            (t("No model"), 2)
        };
        b.set_chip_text(chip.into());
        b.set_chip_tone(tone);
        b.set_status_text(ux.status.0.clone().into());
        b.set_status_error(ux.status.1);
        let pred = if ux.prediction == Prediction::Running {
            t("Predicting mask…")
        } else if doc.is_some_and(|d| d.computing) {
            t("Computing mask…")
        } else if ux.prediction == Prediction::NoModel && doc.is_some() {
            t("No prediction (no model)")
        } else if ux.prediction == Prediction::Failed {
            t("Prediction failed")
        } else {
            String::new()
        };
        b.set_pred_text(pred.into());
        b.set_size_text(doc.map(|d| format!("{} × {} px", d.width, d.height)).unwrap_or_default().into());
    }

    fn sync_models(self: &Arc<Self>, ui: &MainWindow) {
        let b = ui.global::<Backend>();
        let status = self.st.model_status.guard().clone();
        let downloads = self.ux.guard().downloads.clone();
        let rows: Vec<ModelRow> = self
            .st
            .models()
            .into_iter()
            .map(|m| {
                let custom = m.url.is_empty() && m.files.is_empty();
                let is_sam = m.kind == ModelKind::Sam2;
                let dl = downloads.get(&m.id).copied();
                let size = m.total_size();
                let mut meta: Vec<String> = Vec::new();
                if size > 0 {
                    meta.push(if size >= 1_000_000_000 {
                        format!("{:.2} GB", size as f64 / 1e9)
                    } else {
                        format!("{} MB", (size as f64 / 1e6).round())
                    });
                }
                if m.ram_mb > 0 && !is_sam {
                    meta.push(format!("RAM {:.1} GB", m.ram_mb as f64 / 1024.0));
                }
                meta.push(m.license.clone());
                let description = t(&m.description);
                let badge = if m.recommended {
                    t("Recommended")
                } else if is_sam {
                    t("Smart select")
                } else {
                    String::new()
                };
                let active = status.active.as_deref() == Some(&m.id);
                ModelRow {
                    ready: self.st.is_ready(&m),
                    active,
                    loading: active && status.loading,
                    downloading: dl.is_some(),
                    progress: dl.and_then(|(d, t)| (d * 100).checked_div(t)).map(|p| p.min(100) as i32).unwrap_or(0),
                    deletable: !custom,
                    is_sam,
                    id: m.id.into(),
                    name: m.name.into(),
                    description: description.into(),
                    meta: meta.join(" · ").into(),
                    badge: badge.into(),
                }
            })
            .collect();
        b.set_models(ModelRc::new(VecModel::from(rows)));
        b.set_model_error(status.error.clone().unwrap_or_default().into());
        let s = self.st.settings.guard();
        b.set_accel_index(match s.accelerator {
            openscissors_core::model::Accelerator::Cpu => 0,
            openscissors_core::model::Accelerator::Gpu => 1,
            openscissors_core::model::Accelerator::Auto => 2,
        });
        b.set_fast(s.fast_memory);
        b.set_half(s.ram_percent == 50);
        b.set_backend_text(if status.ready {
            status.backend.as_deref().map(|be| tf("Running on {backend}", &[("backend", be)])).unwrap_or_default().into()
        } else {
            SharedString::new()
        });
        b.set_models_dir(self.st.paths.models.display().to_string().into());
    }

    fn sync_menu(self: &Arc<Self>, ui: &MainWindow) {
        let b = ui.global::<Backend>();
        let s = self.st.settings.guard().clone();
        b.set_jpeg_quality(s.jpeg_quality as i32);
        b.set_export16(s.export_16bit);
        let mut langs = vec![t("System default")];
        langs.extend(i18n::LOCALES.iter().map(|(_, n)| n.to_string()));
        b.set_lang_options(strings(langs));
        b.set_lang_index(if s.locale == "auto" {
            0
        } else {
            i18n::LOCALES.iter().position(|(c, _)| *c == i18n::resolve(&s.locale)).map(|i| i as i32 + 1).unwrap_or(0)
        });
        b.set_recent(strings(s.recent.iter().take(5).map(|r| file_name(r))));
        let plugins = self.plugin_rows();
        b.set_plugins(ModelRc::new(VecModel::from(plugins)));
        b.set_can_run_plugin(self.ux.guard().last_saved.is_some());
        b.set_about_version(env!("CARGO_PKG_VERSION").into());
        b.set_root_path(self.st.paths.root.display().to_string().into());
    }

    fn sync_batch(self: &Arc<Self>, ui: &MainWindow) {
        let b = ui.global::<Backend>();
        let (src, dst) = {
            let s = self.st.settings.guard();
            (s.batch_src.clone().unwrap_or_default(), s.batch_dst.clone().unwrap_or_default())
        };
        let ready = self.st.model_status.guard().ready;
        let ux = self.ux.guard();
        b.set_batch_src(src.clone().into());
        b.set_batch_dst(dst.clone().into());
        b.set_batch_format(ux.batch.format);
        b.set_batch_settings(ux.batch.settings);
        b.set_batch_running(ux.batch.running);
        b.set_batch_done(ux.batch.done as i32);
        b.set_batch_total(ux.batch.total as i32);
        b.set_batch_file(ux.batch.file.clone().unwrap_or_default().into());
        b.set_batch_report(ux.batch.report.clone().into());
        b.set_batch_ready(ready);
        b.set_batch_can_run(ready && !src.is_empty() && !dst.is_empty() && !ux.batch.running);
    }

    fn plugin_rows(&self) -> Vec<PluginRow> {
        let enabled = self.st.settings.guard().plugins_enabled.clone();
        openscissors_core::plugin::discover(&self.st.paths.plugins)
            .into_iter()
            .map(|d| match d.plugin {
                Ok(p) => PluginRow {
                    enabled: enabled.contains(&p.manifest.id),
                    name: p.manifest.name.into(),
                    description: p.manifest.description.into(),
                    error: SharedString::new(),
                },
                Err(e) => PluginRow { name: d.folder.into(), description: SharedString::new(), enabled: false, error: e.into() },
            })
            .collect()
    }

    /// Pan, zoom and everything drawn over the views; also asks for new pixels when they changed.
    pub fn sync_view(self: &Arc<Self>, ui: &MainWindow) {
        let b = ui.global::<Backend>();
        let mut ux = self.ux.guard();
        let v = ux.view;
        b.set_view_scale(v.scale as f32);
        b.set_view_tx(v.tx as f32);
        b.set_view_ty(v.ty as f32);
        b.set_cursor(self.cursor(&ux));

        // Mark being drawn.
        let (path, color, width) = match (&ux.stroke, ux.tool) {
            (Some(points), tool) if !points.is_empty() => {
                let mut d = String::new();
                for (i, (x, y)) in points.iter().enumerate() {
                    let (sx, sy) = to_screen(&v, *x, *y);
                    d.push_str(&format!("{}{:.1} {:.1} ", if i == 0 { "M" } else { "L" }, sx, sy));
                }
                if points.len() == 1 {
                    let (sx, sy) = to_screen(&v, points[0].0, points[0].1);
                    d.push_str(&format!("L{:.1} {:.1}", sx + 0.01, sy));
                }
                let color = match tool {
                    Tool::Keep => slint::Color::from_argb_u8(140, 80, 210, 120),
                    Tool::Remove => slint::Color::from_argb_u8(140, 230, 80, 70),
                    Tool::Hair => slint::Color::from_argb_u8(140, 240, 200, 40),
                    _ => slint::Color::from_argb_u8(115, 255, 255, 255),
                };
                (d, color, ux.marker as f32 * v.scale as f32)
            }
            _ => (String::new(), slint::Color::from_argb_u8(0, 0, 0, 0), 1.0),
        };
        b.set_stroke_path(path.into());
        b.set_stroke_color(color);
        b.set_stroke_width(width);

        let dots: Vec<Dot> = ux
            .prompts
            .iter()
            .filter_map(|p| match p {
                Prompt::Point { x, y, keep } => Some(Dot { x: *x, y: *y, keep: *keep }),
                Prompt::Box { .. } => None,
            })
            .collect();
        let boxes: Vec<Box2> = ux
            .prompts
            .iter()
            .filter_map(|p| match p {
                Prompt::Box { x0, y0, x1, y1 } => Some(Box2 { x0: *x0, y0: *y0, x1: *x1, y1: *y1 }),
                Prompt::Point { .. } => None,
            })
            .collect();
        b.set_dots(ModelRc::new(VecModel::from(dots)));
        b.set_boxes(ModelRc::new(VecModel::from(boxes)));
        match &ux.drag {
            Some(d) if (d.to.0 - d.from.0).hypot(d.to.1 - d.from.1) > 6.0 => {
                b.set_drag_visible(true);
                b.set_drag_box(Box2 { x0: d.from.0 as f32, y0: d.from.1 as f32, x1: d.to.0 as f32, y1: d.to.1 as f32 });
            }
            _ => b.set_drag_visible(false),
        }
        let drawing = self.drawing(&ux);
        match ux.hover {
            Some((0, x, y)) if drawing && ux.pan_from.is_none() => {
                b.set_brush_visible(true);
                b.set_brush_x(x as f32);
                b.set_brush_y(y as f32);
                b.set_brush_d(ux.marker as f32 * v.scale as f32);
            }
            _ => b.set_brush_visible(false),
        }

        // Crop rectangle.
        if let Some(doc) = &ux.doc {
            let r = ux.live_crop.unwrap_or(doc.crop);
            let full = same_rect(&doc.crop, &Rect::new(0.0, 0.0, doc.width as f64, doc.height as f64), 0.5);
            b.set_crop_visible(true);
            b.set_crop_full(full && ux.live_crop.is_none());
            b.set_crop_locked(doc.params.background.fit_to_result);
            b.set_crop(Box2 { x0: r.x0 as f32, y0: r.y0 as f32, x1: r.x1 as f32, y1: r.y1 as f32 });
            b.set_crop_label(format!("{} × {}", (r.x1 - r.x0).round(), (r.y1 - r.y0).round()).into());
        } else {
            b.set_crop_visible(false);
        }

        // Pixels.
        let Some(doc) = ux.doc.clone() else {
            self.current_doc.store(0, Ordering::Relaxed);
            b.set_src_img(slint::Image::default());
            b.set_res_img(slint::Image::default());
            ux.render_key = None;
            return;
        };
        self.current_doc.store(doc.id, Ordering::Relaxed);
        let hair = matches!(ux.tool, Tool::Hair | Tool::HairErase);
        let key = format!("{}:{}:{:.5}:{:.3}:{:.3}:{:?}:{}:{}", doc.id, doc.rev, v.scale, v.tx, v.ty, ux.pane, hair, ux.dpr);
        if ux.render_key.as_deref() == Some(&key) {
            return;
        }
        ux.render_key = Some(key);
        let img = (doc.width as f64, doc.height as f64);
        let Some(region) = visible_region(&v, img, ux.pane) else {
            b.set_src_img(slint::Image::default());
            b.set_res_img(slint::Image::default());
            return;
        };
        let (w, h, grid) = device_grid(&v, &region, ux.dpr);
        let origin = (((v.tx + grid.x0 * v.scale) * ux.dpr).round() as i64, ((v.ty + grid.y0 * v.scale) * ux.dpr).round() as i64);
        let cell = (8.0 * ux.dpr).round().max(1.0) as usize;
        for side in 0..2 {
            self.renderer.submit(Job { side, doc: doc.id, region: grid, w, h, hair: hair && side == 0, origin, cell });
        }
    }

    pub fn drawing(&self, ux: &Ux) -> bool {
        ux.doc.is_some() && !ux.space && ux.tool.stroke_kind().is_some()
    }

    pub fn selecting(&self, ux: &Ux) -> bool {
        ux.doc.is_some() && !ux.space && ux.tool == Tool::Select
    }

    fn cursor(&self, ux: &Ux) -> i32 {
        if ux.doc.is_none() {
            return 0;
        }
        if ux.pan_from.is_some() {
            return 3;
        }
        if let Some(d) = &ux.crop_drag {
            return d.handle.cursor();
        }
        if let (Some((1, x, y)), Some(doc)) = (ux.hover, &ux.doc) {
            let r = ux.live_crop.unwrap_or(doc.crop);
            let (sx0, sy0) = to_screen(&ux.view, r.x0, r.y0);
            let (sx1, sy1) = to_screen(&ux.view, r.x1, r.y1);
            let full = same_rect(&doc.crop, &Rect::new(0.0, 0.0, doc.width as f64, doc.height as f64), 0.5);
            let locked = doc.params.background.fit_to_result;
            if let Some(h) = crate::cropmath::hit_test(sx0, sy0, sx1, sy1, x, y, !locked, !locked && !full) {
                return h.cursor();
            }
        }
        if self.drawing(ux) {
            return 4;
        }
        if self.selecting(ux) {
            return 1;
        }
        2
    }

    // -- helpers shared by the operations

    pub fn fit(&self) {
        let mut ux = self.ux.guard();
        if let Some(d) = &ux.doc {
            if ux.pane.0 > 0.0 {
                ux.view = fit_view((d.width as f64, d.height as f64), ux.pane, 24.0);
            }
        }
    }

    /// Read the document state into the interface state.
    pub fn refresh_doc(self: &Arc<Self>) {
        let st = self.st.doc.guard().as_mut().map(|d| d.state());
        self.ux.guard().doc = st;
        self.refresh(R_MAIN | R_VIEW);
    }
}
