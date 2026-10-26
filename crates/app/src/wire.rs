//! Connects the interface callbacks to the controller, and the background tickers.

use openscissors_core::sync::LockExt;
use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;
use std::time::{Duration, Instant};

use openscissors_core::sysmem;
use serde_json::json;
use slint::ComponentHandle;

use crate::ctl::{Ctl, Tool, R_ALL, R_BATCH, R_MAIN, R_MENU, R_MODELS};
use crate::window::Tracker;
use crate::{Backend, MainWindow};

macro_rules! cb {
    ($b:ident, $ctl:ident, $on:ident, |$c:ident $(, $a:ident)*| $body:expr) => {{
        let $c = $ctl.clone();
        $b.$on(move |$($a),*| $body);
    }};
}

pub fn wire(ui: &MainWindow, ctl: &Arc<Ctl>) {
    crate::dialogs::init(ui);
    let b = ui.global::<Backend>();

    cb!(b, ctl, on_open_image, |c| c.open_dialog());
    cb!(b, ctl, on_save, |c| c.save());
    cb!(b, ctl, on_save_as, |c| c.save_as());
    cb!(b, ctl, on_undo, |c| c.undo());
    cb!(b, ctl, on_redo, |c| c.redo());
    cb!(b, ctl, on_zoom_in, |c| c.zoom_in());
    cb!(b, ctl, on_zoom_out, |c| c.zoom_out());
    cb!(b, ctl, on_zoom_fit, |c| c.zoom_fit());
    cb!(b, ctl, on_zoom_actual, |c| c.zoom_actual());
    cb!(b, ctl, on_toggle_graphcut, |c| c.toggle_graphcut());
    cb!(b, ctl, on_set_tool, |c, name| {
        if let Some(t) = Tool::parse(&name) {
            c.set_tool(t);
        }
    });
    cb!(b, ctl, on_set_marker, |c, n| c.set_marker(n));
    cb!(b, ctl, on_key_down, |c, text, ctrl, shift, alt| c.key_down(&text, ctrl, shift, alt));
    cb!(b, ctl, on_key_up, |c, text| c.key_up(&text));
    cb!(b, ctl, on_popover_changed, |c, which| match which {
        1 => c.refresh(R_MODELS),
        2 => c.refresh(R_MENU),
        3 => c.refresh(R_BATCH),
        4 => c.refresh(R_MENU),
        5 => c.refresh(R_MENU | R_MODELS),
        _ => {}
    });

    cb!(b, ctl, on_pane_resized, |c, w, h| c.pane_resized(w as f64, h as f64));
    cb!(b, ctl, on_pane_down, |c, side, button, x, y, alt| c.pane_down(side, button, x as f64, y as f64, alt));
    cb!(b, ctl, on_pane_up, |c, side, button, x, y| c.pane_up(side, button, x as f64, y as f64));
    cb!(b, ctl, on_pane_move, |c, side, x, y| c.pane_move(side, x as f64, y as f64));
    cb!(b, ctl, on_pane_leave, |c, side| c.pane_leave(side));
    cb!(b, ctl, on_pane_wheel, |c, side, x, y, dx, dy, ctrl, shift| c
        .pane_wheel(side, x as f64, y as f64, dx as f64, dy as f64, ctrl, shift));

    cb!(b, ctl, on_set_int, |c, path, v| c.set_param(&path, json!(v)));
    cb!(b, ctl, on_set_bool, |c, path, v| c.set_param(&path, json!(v)));
    cb!(b, ctl, on_set_index, |c, path, i| c.set_index(&path, i));
    cb!(b, ctl, on_set_rgb, |c, path, r, g, bl| c.set_param(&path, json!([r.clamp(0, 255), g.clamp(0, 255), bl.clamp(0, 255)])));
    cb!(b, ctl, on_pick_color, |c, path| c.pick_color(path.to_string()));
    cb!(b, ctl, on_apply_preset, |c, i| c.apply_preset(i));
    cb!(b, ctl, on_copy_result, |c| c.copy_result());
    cb!(b, ctl, on_pick_bg_image, |c| c.pick_bg_image());
    cb!(b, ctl, on_reset_crop, |c| c.set_crop(None));
    cb!(b, ctl, on_reset, |c, tab, factory| c.reset_tab(tab, factory));
    cb!(b, ctl, on_save_defaults, |c| c.save_defaults());
    cb!(b, ctl, on_smart_apply, |c| c.smart_apply());
    cb!(b, ctl, on_smart_cancel, |c| c.smart_cancel());
    cb!(b, ctl, on_set_smart_mode, |c, i| c.set_smart_mode(i));

    cb!(b, ctl, on_use_model, |c, id| c.use_model(&id));
    cb!(b, ctl, on_download_model, |c, id| c.download_model(&id));
    cb!(b, ctl, on_cancel_download, |c, id| c.cancel_download(&id));
    cb!(b, ctl, on_delete_model, |c, id| c.delete_model(&id));
    cb!(b, ctl, on_set_accelerator, |c, i| c.set_accelerator(i));
    cb!(b, ctl, on_set_fast, |c, on| c.set_fast(on));
    cb!(b, ctl, on_set_half, |c, on| c.set_half(on));
    cb!(b, ctl, on_save_project, |c| c.save_project_dialog());
    cb!(b, ctl, on_load_marks, |c| c.load_marks_dialog());
    cb!(b, ctl, on_save_marks, |c| c.save_marks_dialog());
    cb!(b, ctl, on_open_recent, |c, i| {
        let path = c.st.settings.guard().recent.get(i as usize).cloned();
        if let Some(p) = path {
            c.open_path(PathBuf::from(p));
        }
    });
    cb!(b, ctl, on_clear_recent, |c| c.clear_recent());
    cb!(b, ctl, on_set_jpeg_quality, |c, q| c.set_jpeg_quality(q));
    cb!(b, ctl, on_set_export16, |c, on| c.set_export16(on));
    cb!(b, ctl, on_set_language, |c, i| c.set_language(i.max(0) as usize));
    cb!(b, ctl, on_plugin_toggle, |c, i, on| c.plugin_toggle(i as usize, on));
    cb!(b, ctl, on_plugin_run, |c, i| c.plugin_run(i as usize));
    cb!(b, ctl, on_pick_folder, |c, source| c.pick_folder(source));
    cb!(b, ctl, on_set_batch_format, |c, i| {
        c.ux.guard().batch.format = i;
        c.refresh(R_BATCH);
    });
    cb!(b, ctl, on_set_batch_settings, |c, i| {
        c.ux.guard().batch.settings = i;
        c.refresh(R_BATCH);
    });
    cb!(b, ctl, on_run_batch, |c| c.run_batch());
    cb!(b, ctl, on_stop_batch, |c| c.stop_batch());

    let tracker = Rc::new(RefCell::new(Tracker::new(ctl.st.clone())));

    // Closing the window asks about unsaved changes first.
    let c = ctl.clone();
    let weak = ui.as_weak();
    let t = tracker.clone();
    ui.window().on_close_requested(move || {
        if let Some(ui) = weak.upgrade() {
            t.borrow_mut().finish(&ui);
        }
        if !c.is_modified() {
            return slint::CloseRequestResponse::HideWindow;
        }
        // The question is a dialog of the interface: it cannot be answered while this thread waits.
        let c = c.clone();
        std::thread::spawn(move || {
            if c.confirm_discard_sync() {
                let _ = slint::invoke_from_event_loop(|| {
                    let _ = slint::quit_event_loop();
                });
            }
        });
        slint::CloseRequestResponse::KeepWindowShown
    });

    // Files dropped on the window open like the Open dialog (unsaved changes are asked about first); a folder starts a batch.
    {
        use slint::winit_030::{winit::event::WindowEvent, EventResult, WinitWindowAccessor};
        let c = ctl.clone();
        ui.window().on_winit_window_event(move |_, event| {
            match event {
                WindowEvent::HoveredFile(_) => c.set_drop_active(true),
                WindowEvent::HoveredFileCancelled => c.set_drop_active(false),
                WindowEvent::DroppedFile(path) => {
                    c.set_drop_active(false);
                    let ok = path.extension().and_then(|e| e.to_str()).is_some_and(|e| {
                        let e = e.to_ascii_lowercase();
                        e == "osproj" || crate::dialogs::IMAGE_EXT.contains(&e.as_str())
                    });
                    if ok {
                        c.open_path(path.clone());
                    } else if path.is_dir() {
                        c.batch_from_folder(path.clone());
                    }
                }
                _ => {}
            }
            EventResult::Propagate
        });
    }

    // Resource meter, once a second.
    let timer = slint::Timer::default();
    let weak = ui.as_weak();
    let mut last: Option<(Instant, u64)> = None;
    timer.start(slint::TimerMode::Repeated, Duration::from_secs(1), move || {
        let Some(ui) = weak.upgrade() else { return };
        tracker.borrow_mut().tick(&ui);
        let now = Instant::now();
        let cpu = sysmem::process_cpu_ms().unwrap_or(0);
        let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1) as f64;
        let pct = match last {
            Some((t, c0)) if now > t => {
                (cpu.saturating_sub(c0) as f64 / (now.duration_since(t).as_secs_f64() * 1000.0 * cores) * 100.0).clamp(0.0, 100.0)
            }
            _ => 0.0,
        };
        last = Some((now, cpu));
        let ram = sysmem::process_memory_mb().unwrap_or(0);
        let budget = sysmem::budget_mb().unwrap_or(0);
        let gb = |mb: u64| if mb >= 1024 { format!("{:.1} GB", mb as f64 / 1024.0) } else { format!("{mb} MB") };
        let text =
            format!("RAM {}{} · CPU {} %", gb(ram), if budget > 0 { format!(" / {}", gb(budget)) } else { String::new() }, pct.round());
        let b = ui.global::<Backend>();
        b.set_usage_text(text.into());
        b.set_usage_near(budget > 0 && ram as f64 > budget as f64 * 0.85);
    });
    std::mem::forget(timer); // lives as long as the application

    ctl.refresh(R_ALL | R_MAIN);
}
