//! Pointer and keyboard handling for the two views.

use openscissors_core::sync::LockExt;
use std::sync::Arc;

use openscissors_core::model::Prompt;
use openscissors_core::types::Rect;
use slint::platform::Key;

use crate::cropmath::{drag_rect, hit_test, same_rect, Handle};
use crate::ctl::canvas_of;
use crate::ctl::{CropDrag, Ctl, SmartDrag, Tool, R_MAIN, R_VIEW};
use crate::viewport::{fit_area, pan_by, to_image, to_screen, zoom_at, ZOOM_STEP};

fn ratio_of(key: &str) -> Option<(u32, u32)> {
    openscissors_core::params::parse_ratio(key)
}

impl Ctl {
    pub fn pane_resized(self: &Arc<Self>, w: f64, h: f64) {
        {
            let mut ux = self.ux.guard();
            if (ux.pane.0 - w).abs() < 0.5 && (ux.pane.1 - h).abs() < 0.5 {
                return;
            }
            ux.pane = (w, h);
            if ux.view.fit {
                if let Some(d) = &ux.doc {
                    ux.view = fit_area(canvas_of(d), (w, h), 24.0);
                }
            }
        }
        self.refresh(R_VIEW);
    }

    pub fn zoom_by(self: &Arc<Self>, factor: f64, at: Option<(f64, f64)>) {
        {
            let mut ux = self.ux.guard();
            if ux.doc.is_none() {
                return;
            }
            let (px, py) = at.unwrap_or((ux.pane.0 / 2.0, ux.pane.1 / 2.0));
            ux.view = zoom_at(ux.view, factor, px, py);
        }
        self.refresh(R_VIEW);
    }

    pub fn zoom_in(self: &Arc<Self>) {
        self.zoom_by(ZOOM_STEP, None);
    }

    pub fn zoom_out(self: &Arc<Self>) {
        self.zoom_by(1.0 / ZOOM_STEP, None);
    }

    pub fn zoom_actual(self: &Arc<Self>) {
        let scale = self.ux.guard().view.scale;
        self.zoom_by(1.0 / scale, None);
    }

    pub fn zoom_fit(self: &Arc<Self>) {
        self.fit();
        self.refresh(R_VIEW);
    }

    pub fn pan(self: &Arc<Self>, dx: f64, dy: f64) {
        {
            let mut ux = self.ux.guard();
            if ux.doc.is_none() {
                return;
            }
            ux.view = pan_by(ux.view, dx, dy);
        }
        self.refresh(R_VIEW);
    }

    /// Screen rectangle of the crop (live or current) and whether it can be resized or moved.
    fn crop_screen(&self, ux: &crate::ctl::Ux) -> Option<(f64, f64, f64, f64, bool, bool)> {
        let d = ux.doc.as_ref()?;
        let r = ux.live_crop.unwrap_or(d.crop);
        let (x0, y0) = to_screen(&ux.view, r.x0, r.y0);
        let (x1, y1) = to_screen(&ux.view, r.x1, r.y1);
        let full = same_rect(&d.crop, &Rect::new(0.0, 0.0, d.width as f64, d.height as f64), 0.5);
        let locked = d.params.background.fit_to_result;
        Some((x0, y0, x1, y1, !locked, !locked && !full))
    }

    pub fn pane_down(self: &Arc<Self>, side: i32, button: i32, x: f64, y: f64, alt: bool) {
        let mut ux = self.ux.guard();
        if ux.doc.is_none() {
            return;
        }
        if button == 0 && ux.pick.is_some() {
            let (ix, iy) = to_image(&ux.view, x, y);
            let path = ux.pick.take().expect("pick");
            drop(ux);
            let pixel = self.st.doc.guard().as_ref().and_then(|d| {
                let (px, py) = (ix.floor(), iy.floor());
                let inside = px >= 0.0 && py >= 0.0 && (px as usize) < d.image.w && (py as usize) < d.image.h;
                inside.then(|| {
                    let i = (py as usize * d.image.w + px as usize) * 3;
                    [d.image.data[i], d.image.data[i + 1], d.image.data[i + 2]]
                })
            });
            match pixel {
                Some(rgb) => {
                    self.say("", false);
                    self.set_param(&path, serde_json::json!(rgb));
                }
                None => self.ux.guard().pick = Some(path), // outside the picture: keep waiting
            }
            self.refresh(R_VIEW);
            return;
        }
        if side == 1 && button == 0 {
            if let Some((x0, y0, x1, y1, resize, mv)) = self.crop_screen(&ux) {
                if let Some(handle) = hit_test(x0, y0, x1, y1, x, y, resize, mv) {
                    let from = ux.doc.as_ref().expect("doc").crop;
                    ux.crop_drag = Some(CropDrag { handle, start: (x, y), from });
                    drop(ux);
                    self.refresh(R_VIEW);
                    return;
                }
            }
        }
        let selecting = side == 0 && self.selecting(&ux);
        let drawing = side == 0 && self.drawing(&ux);
        if selecting && (button == 0 || button == 2) {
            ux.drag = Some(SmartDrag { from: (x, y), to: (x, y), button, alt });
        } else if button == 1 || (button == 0 && !drawing) {
            ux.pan_from = Some((x, y));
        } else if button == 0 && drawing {
            let (ix, iy) = to_image(&ux.view, x, y);
            ux.stroke = Some(vec![(ix, iy)]);
        } else {
            return;
        }
        drop(ux);
        self.refresh(R_VIEW);
    }

    pub fn pane_move(self: &Arc<Self>, side: i32, x: f64, y: f64) {
        {
            let mut ux = self.ux.guard();
            ux.hover = Some((side, x, y));
            if let Some(from) = ux.pan_from {
                ux.view = pan_by(ux.view, x - from.0, y - from.1);
                ux.pan_from = Some((x, y));
            } else if let Some(d) = &mut ux.drag {
                d.to = (x, y);
            } else if ux.crop_drag.is_some() {
                let (scale, bounds, ratio) = {
                    let doc = ux.doc.as_ref().expect("doc");
                    (ux.view.scale, (doc.width as f64, doc.height as f64), ratio_of(&doc.params.background.aspect_ratio))
                };
                let cd = ux.crop_drag.as_ref().expect("crop drag");
                let (dx, dy) = ((x - cd.start.0) / scale, (y - cd.start.1) / scale);
                let live = drag_rect(cd.from, cd.handle, dx, dy, bounds, ratio);
                ux.live_crop = Some(live);
            } else if ux.stroke.is_some() {
                let (ix, iy) = to_image(&ux.view, x, y);
                let scale = ux.view.scale;
                if let Some(points) = &mut ux.stroke {
                    let last = points[points.len() - 1];
                    if (ix - last.0).hypot(iy - last.1) * scale >= 1.5 {
                        points.push((ix, iy));
                    }
                }
            }
        }
        self.refresh(R_VIEW);
    }

    pub fn pane_leave(self: &Arc<Self>, _side: i32) {
        self.ux.guard().hover = None;
        self.refresh(R_VIEW);
    }

    pub fn pane_up(self: &Arc<Self>, _side: i32, _button: i32, _x: f64, _y: f64) {
        enum Action {
            None,
            Crop(Option<Rect>),
            Prompt(Prompt),
            Stroke(Tool, u32, Vec<(f64, f64)>),
        }
        let action = {
            let mut ux = self.ux.guard();
            ux.pan_from = None;
            let mut action = Action::None;
            if let Some(cd) = ux.crop_drag.take() {
                let live = ux.live_crop.take();
                if let (Some(r), Some(doc)) = (live, ux.doc.as_ref()) {
                    if !same_rect(&r, &cd.from, 0.01) {
                        let ratio = ratio_of(&doc.params.background.aspect_ratio);
                        let full = same_rect(&r, &Rect::new(0.0, 0.0, doc.width as f64, doc.height as f64), 0.5) && ratio.is_none();
                        action = Action::Crop(if full { None } else { Some(r) });
                    }
                }
            }
            if let Some(d) = ux.drag.take() {
                let (ax, ay) = to_image(&ux.view, d.from.0, d.from.1);
                let (bx, by) = to_image(&ux.view, d.to.0, d.to.1);
                if let Some(doc) = &ux.doc {
                    let inside = |x: f64, y: f64| x >= 0.0 && y >= 0.0 && x <= doc.width as f64 && y <= doc.height as f64;
                    if (d.to.0 - d.from.0).hypot(d.to.1 - d.from.1) > 6.0 {
                        action = Action::Prompt(Prompt::Box {
                            x0: ax.min(bx) as f32,
                            y0: ay.min(by) as f32,
                            x1: ax.max(bx) as f32,
                            y1: ay.max(by) as f32,
                        });
                    } else if inside(ax, ay) {
                        action = Action::Prompt(Prompt::Point { x: ax as f32, y: ay as f32, keep: d.button == 0 && !d.alt });
                    }
                }
            }
            if let Some(points) = ux.stroke.take() {
                action = Action::Stroke(ux.tool, ux.marker, points);
            }
            action
        };
        match action {
            Action::None => self.refresh(R_VIEW),
            Action::Crop(r) => self.set_crop(r),
            Action::Prompt(p) => self.smart_add(p),
            Action::Stroke(tool, width, points) => self.add_stroke(tool, width, points),
        }
        self.refresh(R_VIEW | R_MAIN);
    }

    /// Wheel: Ctrl zooms around the pointer, Shift scrolls sideways, otherwise pan.
    #[allow(clippy::too_many_arguments)]
    pub fn pane_wheel(self: &Arc<Self>, _side: i32, x: f64, y: f64, dx: f64, dy: f64, ctrl: bool, shift: bool) {
        if ctrl {
            self.zoom_by((dy * 0.0025).exp(), Some((x, y)));
        } else if shift {
            self.pan(dy, 0.0);
        } else {
            self.pan(dx, dy);
        }
    }

    pub fn space(self: &Arc<Self>, down: bool) {
        {
            let mut ux = self.ux.guard();
            if ux.space == down {
                return;
            }
            ux.space = down;
        }
        self.refresh(R_VIEW);
    }

    /// Keyboard shortcuts. `text` is what Slint reports for the key (special keys are private-use characters).
    pub fn key_down(self: &Arc<Self>, text: &str, ctrl: bool, shift: bool, alt: bool) {
        if crate::dialogs::is_open() {
            return;
        }
        let mut chars = text.chars();
        let Some(c) = chars.next() else { return };
        if chars.next().is_some() {
            return;
        }
        let special = |k: Key| c == char::from(k);
        if special(Key::Escape) {
            if self.ux.guard().pick.take().is_some() {
                self.say("", false);
                self.refresh(R_VIEW);
            }
            self.close_popovers();
            self.smart_cancel();
            return;
        }
        let has_prompts = !self.ux.guard().prompts.is_empty();
        if has_prompts {
            if c == '\n' || c == '\r' {
                return self.smart_apply();
            }
            if special(Key::Backspace) || c == '\u{8}' {
                return self.smart_undo_prompt();
            }
        }
        if ctrl {
            // Some platforms report Ctrl+letter as a control code.
            let k = if c.is_ascii_control() && (c as u8) < 27 { ((c as u8) + 96) as char } else { c.to_ascii_lowercase() };
            match k {
                'o' => self.open_dialog(),
                'v' => self.paste(),
                'c' => self.copy_result(),
                's' => {
                    if shift {
                        self.save_as()
                    } else {
                        self.save()
                    }
                }
                'z' => {
                    if shift {
                        self.redo()
                    } else {
                        self.undo()
                    }
                }
                'y' => self.redo(),
                '0' => self.zoom_actual(),
                '9' => self.zoom_fit(),
                '=' | '+' => self.zoom_in(),
                '-' => self.zoom_out(),
                _ => {}
            }
            return;
        }
        if alt {
            // Alt + arrows move the crop by one image pixel (ten with Shift).
            let step = if shift { 10.0 } else { 1.0 };
            let (dx, dy) = match c {
                c if c == char::from(Key::LeftArrow) => (-step, 0.0),
                c if c == char::from(Key::RightArrow) => (step, 0.0),
                c if c == char::from(Key::UpArrow) => (0.0, -step),
                c if c == char::from(Key::DownArrow) => (0.0, step),
                _ => (0.0, 0.0),
            };
            if (dx, dy) != (0.0, 0.0) {
                return self.move_crop(dx, dy);
            }
        }
        let step = if shift { 200.0 } else { 40.0 };
        if special(Key::LeftArrow) {
            return self.pan(step, 0.0);
        }
        if special(Key::RightArrow) {
            return self.pan(-step, 0.0);
        }
        if special(Key::UpArrow) {
            return self.pan(0.0, step);
        }
        if special(Key::DownArrow) {
            return self.pan(0.0, -step);
        }
        match c {
            ' ' => self.space(true),
            'f' | 'F' => self.set_tool(Tool::Keep),
            'b' | 'B' => self.set_tool(Tool::Remove),
            's' | 'S' => self.set_tool(Tool::Select),
            'h' | 'H' => self.set_tool(if shift { Tool::HairErase } else { Tool::Hair }),
            'e' | 'E' => self.set_tool(Tool::Erase),
            'm' | 'M' => self.set_tool(Tool::Move),
            '[' => self.step_marker(-5),
            ']' => self.step_marker(5),
            _ => {}
        }
    }

    /// Move the crop by (dx, dy) image pixels, as the pointer does: not when it follows the result, nor when it is the whole
    /// image. It may leave the picture (the canvas is extended there).
    fn move_crop(self: &Arc<Self>, dx: f64, dy: f64) {
        let moved = {
            let ux = self.ux.guard();
            let Some(d) = ux.doc.as_ref() else { return };
            let full = same_rect(&d.crop, &Rect::new(0.0, 0.0, d.width as f64, d.height as f64), 0.5);
            if d.params.background.fit_to_result || full {
                return;
            }
            drag_rect(d.crop, Handle::Move, dx, dy, (d.width as f64, d.height as f64), None)
        };
        self.set_crop(Some(moved));
    }

    pub fn key_up(self: &Arc<Self>, text: &str) {
        if text == " " {
            self.space(false);
        }
    }

    fn step_marker(self: &Arc<Self>, delta: i32) {
        let n = self.ux.guard().marker as i32 + delta;
        self.set_marker(n);
    }

    pub fn close_popovers(self: &Arc<Self>) {
        self.with_ui(|_, ui| {
            use slint::ComponentHandle;
            ui.global::<crate::Backend>().set_popover(0);
        });
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn control_codes_map_back_to_letters() {
        let k = |c: char| if c.is_ascii_control() && (c as u8) < 27 { ((c as u8) + 96) as char } else { c.to_ascii_lowercase() };
        assert_eq!(k('\u{f}'), 'o');
        assert_eq!(k('\u{13}'), 's');
        assert_eq!(k('Z'), 'z');
    }
}
