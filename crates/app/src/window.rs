//! Remembers where the window was (position, size, maximized) and puts it back at the next start.

use openscissors_core::sync::LockExt;
use std::sync::Arc;

use openscissors_core::settings::WindowState;
use slint::winit_030::winit::dpi::PhysicalPosition;
use slint::winit_030::WinitWindowAccessor;
use slint::{ComponentHandle, PhysicalSize};

use crate::state::AppState;
use crate::MainWindow;

/// Smallest and largest sizes taken from the settings; anything else is a damaged file.
const MIN_SIDE: u32 = 400;
const MAX_SIDE: u32 = 20_000;

/// Size and position from the previous run, set before the window is shown.
pub fn restore(ui: &MainWindow, saved: Option<WindowState>) {
    let Some(ws) = saved else { return };
    if !(MIN_SIDE..=MAX_SIDE).contains(&ws.width) || !(MIN_SIDE..=MAX_SIDE).contains(&ws.height) {
        return;
    }
    let w = ui.window();
    w.set_size(PhysicalSize::new(ws.width, ws.height));
    w.set_position(slint::PhysicalPosition::new(ws.x, ws.y));
}

/// After the window is shown: maximize it again if it was, and bring it back on a screen when the
/// monitor it was on is gone.
pub fn settle(ui: &MainWindow, saved: Option<WindowState>) {
    let w = ui.window();
    if saved.is_some_and(|s| s.maximized) {
        w.set_maximized(true);
    }
    w.with_winit_window(|win| {
        let (Ok(pos), size) = (win.outer_position(), win.outer_size()) else { return };
        let (cx, cy) = (pos.x + size.width as i32 / 2, pos.y + 30);
        let visible = win.available_monitors().any(|m| {
            let (p, s) = (m.position(), m.size());
            cx >= p.x && cx < p.x + s.width as i32 && cy >= p.y && cy < p.y + s.height as i32
        });
        if !visible {
            let origin = win.primary_monitor().map(|m| m.position()).unwrap_or_default();
            win.set_outer_position(PhysicalPosition::new(origin.x + 60, origin.y + 60));
        }
    });
}

/// Current geometry, `None` while minimized (the system reports a position far off screen).
fn capture(ui: &MainWindow) -> Option<WindowState> {
    let w = ui.window();
    if w.is_minimized() {
        return None;
    }
    let (p, s) = (w.position(), w.size());
    (p.x > -30_000 && p.y > -30_000).then_some(WindowState {
        x: p.x,
        y: p.y,
        width: s.width,
        height: s.height,
        maximized: w.is_maximized(),
    })
}

/// Watches the window and writes its geometry to the settings once it stops changing.
pub struct Tracker {
    state: Arc<AppState>,
    /// Last geometry seen while not maximized: what a maximized window goes back to.
    normal: Option<WindowState>,
    seen: Option<WindowState>,
}

impl Tracker {
    pub fn new(state: Arc<AppState>) -> Self {
        let saved = state.settings.guard().window;
        Self { state, normal: saved.filter(|s| !s.maximized), seen: saved }
    }

    fn geometry(&mut self, ui: &MainWindow) -> Option<WindowState> {
        let now = capture(ui)?;
        if !now.maximized {
            self.normal = Some(now);
        }
        Some(match self.normal {
            Some(n) if now.maximized => WindowState { maximized: true, ..n },
            _ => now,
        })
    }

    fn store(&self, ws: WindowState) {
        let changed = {
            let mut s = self.state.settings.guard();
            let changed = s.window != Some(ws);
            s.window = Some(ws);
            changed
        };
        if changed {
            let _ = self.state.save_settings(); // the window position is not worth an error message
        }
    }

    /// Called every second: saves when two readings in a row agree (the window is not being dragged).
    pub fn tick(&mut self, ui: &MainWindow) {
        let Some(now) = self.geometry(ui) else { return };
        if self.seen == Some(now) {
            self.store(now);
        }
        self.seen = Some(now);
    }

    /// Called when the window closes.
    pub fn finish(&mut self, ui: &MainWindow) {
        if let Some(now) = self.geometry(ui) {
            self.store(now);
        }
    }
}
