//! Dialogs. Questions are drawn by the interface (same look as the rest of the application); the
//! file choosers are the ones of the system, which people already know and which see their own
//! folders, drives and shortcuts.
//!
//! Every call blocks the calling thread until the person answers: call them from a worker thread,
//! never from the interface thread (which has to keep running to show or to block the window).

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Sender};
use std::sync::{Mutex, OnceLock};

use rfd::FileDialog;
use slint::{ComponentHandle, SharedString};

use crate::i18n::t;
use crate::{Backend, MainWindow};

pub const IMAGE_EXT: &[&str] = &["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"];

static UI: OnceLock<slint::Weak<MainWindow>> = OnceLock::new();
static QUESTION: Mutex<Option<Sender<usize>>> = Mutex::new(None);
static SYSTEM_DIALOG: AtomicBool = AtomicBool::new(false);

/// Whether a dialog is on screen (the shortcuts are off meanwhile).
pub fn is_open() -> bool {
    SYSTEM_DIALOG.load(Ordering::Relaxed) || QUESTION.lock().expect("question lock").is_some()
}

/// Connect the buttons of the questions to the interface; call once, on the interface thread.
pub fn init(ui: &MainWindow) {
    let _ = UI.set(ui.as_weak());
    ui.global::<Backend>().on_dlg_answer(|i| {
        if let Some(tx) = QUESTION.lock().expect("question lock").take() {
            if let Some(ui) = UI.get().and_then(|w| w.upgrade()) {
                ui.global::<Backend>().set_dlg_open(false);
            }
            let _ = tx.send(i.max(0) as usize);
        }
    });
}

/// Run a system dialog while the window is blocked (a click on it would otherwise start a second
/// action behind the dialog). A second request while one is open is refused.
fn system<R>(f: impl FnOnce() -> Option<R>) -> Option<R> {
    if SYSTEM_DIALOG.swap(true, Ordering::AcqRel) {
        return None;
    }
    let weak = UI.get().cloned();
    let block = |on: bool| {
        if let Some(w) = &weak {
            let _ = w.upgrade_in_event_loop(move |ui| ui.global::<Backend>().set_blocked(on));
        }
    };
    block(true);
    let result = f();
    block(false);
    SYSTEM_DIALOG.store(false, Ordering::Release);
    result
}

fn base(dir: Option<&str>) -> FileDialog {
    let d = FileDialog::new();
    match dir {
        Some(dir) if std::path::Path::new(dir).is_dir() => d.set_directory(dir),
        _ => d,
    }
}

pub fn open_file(dir: Option<&str>, title: &str, filters: &[(&str, &[&str])]) -> Option<PathBuf> {
    system(|| {
        let mut d = base(dir).set_title(title);
        for (name, exts) in filters {
            d = d.add_filter(*name, exts);
        }
        d.pick_file()
    })
}

pub fn save_file(dir: Option<&str>, title: &str, file_name: &str, filters: &[(&str, &[&str])]) -> Option<PathBuf> {
    system(|| {
        let mut d = base(dir).set_title(title).set_file_name(file_name);
        for (name, exts) in filters {
            d = d.add_filter(*name, exts);
        }
        d.save_file()
    })
}

pub fn pick_folder(dir: Option<&str>, title: &str) -> Option<PathBuf> {
    system(|| base(dir).set_title(title).pick_folder())
}

/// A question with up to three buttons; returns the index of the one pressed. `cancel` is the index
/// answered by Escape. The first button is the default one (Enter).
pub fn question(title: &str, text: &str, buttons: &[String], cancel: usize) -> usize {
    let (tx, rx) = channel();
    let Some(weak) = UI.get().cloned() else { return cancel };
    {
        let mut slot = QUESTION.lock().expect("question lock");
        if slot.is_some() {
            return cancel; // one question at a time
        }
        *slot = Some(tx);
    }
    let (title, text, buttons) = (title.to_string(), text.to_string(), buttons.to_vec());
    let shown = weak.upgrade_in_event_loop(move |ui| {
        let g = ui.global::<Backend>();
        g.set_dlg_title(title.into());
        g.set_dlg_text(text.into());
        let b = |i: usize| SharedString::from(buttons.get(i).map(String::as_str).unwrap_or(""));
        g.set_dlg_b1(b(0));
        g.set_dlg_b2(b(1));
        g.set_dlg_b3(b(2));
        g.set_dlg_cancel(cancel as i32);
        g.set_dlg_open(true);
    });
    if shown.is_err() {
        *QUESTION.lock().expect("question lock") = None;
        return cancel;
    }
    rx.recv().unwrap_or(cancel)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Answer {
    Save,
    Discard,
    Cancel,
}

/// "The image has been modified" with Save / Don't save / Cancel.
pub fn confirm_discard() -> Answer {
    let buttons = [t("Save"), t("Don't save"), t("Cancel")];
    match question("OpenScissors", &t("The image has been modified. Do you want to save your changes?"), &buttons, 2) {
        0 => Answer::Save,
        1 => Answer::Discard,
        _ => Answer::Cancel,
    }
}

/// An OK / Cancel question.
pub fn confirm(title: &str, text: &str, ok: String, cancel: String) -> bool {
    question(title, text, &[ok, cancel], 1) == 0
}
