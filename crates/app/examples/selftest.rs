//! Drives the real window in-process and saves snapshots of it, to look at the interface and to
//! check the interactions without touching the screen or the mouse of the person at the desk.
//!
//! `cargo run -p openscissors --example selftest -- <image> <out-dir> [scenario]`

use std::path::PathBuf;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use openscissors::ctl::{Ctl, Tool, R_ALL};
use openscissors::{wire, Backend, MainWindow};
use openscissors_core::io::{self, Format};
use openscissors_core::types::Rgba;
use slint::ComponentHandle;

struct Driver {
    ctl: std::sync::Arc<Ctl>,
    ui: slint::Weak<MainWindow>,
    out: PathBuf,
}

impl Driver {
    /// Run `f` on the interface thread and wait for it.
    fn ui<R: Send + 'static>(&self, f: impl FnOnce(&std::sync::Arc<Ctl>, &MainWindow) -> R + Send + 'static) -> R {
        let (tx, rx) = mpsc::channel();
        let ctl = self.ctl.clone();
        self.ui
            .upgrade_in_event_loop(move |ui| {
                let _ = tx.send(f(&ctl, &ui));
            })
            .expect("event loop");
        rx.recv_timeout(Duration::from_secs(30)).expect("interface thread answered")
    }

    fn settle(&self, ms: u64) {
        std::thread::sleep(Duration::from_millis(ms));
        self.ui(|_, _| ());
        std::thread::sleep(Duration::from_millis(60));
    }

    fn shot(&self, name: &str) {
        self.settle(250);
        let path = self.out.join(format!("{name}.png"));
        let snap = self.ui(|_, ui| ui.window().take_snapshot());
        match snap {
            Ok(buf) => {
                let img = Rgba { w: buf.width() as usize, h: buf.height() as usize, data: buf.as_bytes().to_vec() };
                io::save_rgba(&path, &img, Format::Png, 95, [255, 255, 255]).expect("save snapshot");
                println!("saved {}", path.display());
            }
            Err(e) => println!("snapshot failed: {e}"),
        }
    }

    /// A real left click at a logical position of the window.
    fn click(&self, x: f32, y: f32) {
        use slint::platform::{PointerEventButton, WindowEvent};
        self.ui(move |_, ui| {
            let pos = slint::LogicalPosition::new(x, y);
            let w = ui.window();
            let _ = w.dispatch_event_with_result(WindowEvent::PointerMoved { position: pos });
            let _ = w.dispatch_event_with_result(WindowEvent::PointerPressed { position: pos, button: PointerEventButton::Left });
            let _ = w.dispatch_event_with_result(WindowEvent::PointerReleased { position: pos, button: PointerEventButton::Left });
        });
        self.settle(250);
    }

    fn wait_prediction(&self) {
        let t = Instant::now();
        loop {
            let done = self.ctl.st.doc.lock().unwrap().as_ref().is_some_and(|d| d.has_prediction());
            if done || t.elapsed() > Duration::from_secs(60) {
                break;
            }
            std::thread::sleep(Duration::from_millis(200));
        }
        self.settle(800);
    }

    fn drag(&self, side: i32, from: (f64, f64), to: (f64, f64)) {
        let (c, f, t) = (self.ctl.clone(), from, to);
        self.ui(move |_, _| {
            c.pane_move(side, f.0, f.1);
            c.pane_down(side, 0, f.0, f.1, false);
            for i in 1..=12 {
                let k = i as f64 / 12.0;
                c.pane_move(side, f.0 + (t.0 - f.0) * k, f.1 + (t.1 - f.1) * k);
            }
            c.pane_up(side, 0, t.0, t.1);
        });
        self.settle(300);
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let image = PathBuf::from(&args[1]);
    let out = PathBuf::from(&args[2]);
    let scenario = args.get(3).cloned().unwrap_or_else(|| "tour".into());
    std::fs::create_dir_all(&out).expect("out dir");

    let ui = MainWindow::new().expect("window");
    let ctl = Ctl::new(&ui);
    wire::wire(&ui, &ctl);
    let (w, h) = std::env::var("OSC_SIZE")
        .ok()
        .and_then(|v| v.split_once('x').map(|(a, b)| (a.parse().unwrap_or(1320.0), b.parse().unwrap_or(820.0))))
        .unwrap_or((1320.0, 820.0));
    if std::env::var("OSC_SIZE").is_ok() || std::env::var("OSC_KEEP_WINDOW").is_err() {
        ui.window().set_size(slint::LogicalSize::new(w, h));
    } else {
        let saved = ctl.st.settings.lock().unwrap().window;
        openscissors::window::restore(&ui, saved);
    }
    if let Ok(lang) = std::env::var("OSC_LANG") {
        let idx = openscissors::i18n::LOCALES.iter().position(|(c, _)| *c == lang).map(|i| i + 1).unwrap_or(1);
        ctl.set_language(idx);
    }
    ui.show().expect("show");

    let driver = Driver { ctl: ctl.clone(), ui: ui.as_weak(), out };
    std::thread::spawn(move || {
        let d = driver;
        d.settle(600);
        d.shot("01-empty");
        let c = d.ctl.clone();
        c.open_path_sync(&image);
        d.wait_prediction();
        d.shot("02-opened");

        if scenario == "tour" || scenario == "all" {
            // Brush stroke on the original.
            d.ui(|c, _| c.set_tool(Tool::Keep));
            d.drag(0, (300.0, 300.0), (420.0, 360.0));
            d.shot("03-stroke");
            d.ui(|c, _| c.set_tool(Tool::Erase));
            d.drag(0, (250.0, 320.0), (300.0, 330.0));
            d.ui(|c, _| c.undo());
            d.shot("04-undo");
            for (tab, name) in [(1, "05-background"), (2, "06-shadow")] {
                d.ui(move |_, ui| ui.global::<Backend>().set_tab(tab));
                d.shot(name);
            }
            // Background colour and shadow.
            d.ui(|c, _| {
                c.set_param("background.mode", serde_json::json!("color"));
                c.set_param("background.color", serde_json::json!([40, 120, 200]));
                c.set_param("shadow.enabled", serde_json::json!(true));
            });
            d.ui(|_, ui| ui.global::<Backend>().set_tab(1));
            d.shot("07-color-bg");
            d.ui(|c, _| c.set_param("background.aspect_ratio", serde_json::json!("16:9")));
            d.shot("08-ratio");
            d.ui(|_, ui| ui.global::<Backend>().set_tab(0));
            d.ui(|c, _| c.toggle_graphcut());
            d.shot("09-graphcut");
            d.ui(|c, _| c.set_tool(Tool::Select));
            d.shot("10-select-tool");
            for (p, name) in [(1, "11-models"), (2, "12-menu"), (3, "13-batch")] {
                d.ui(move |c, ui| {
                    ui.global::<Backend>().set_popover(p);
                    c.refresh(R_ALL);
                });
                d.shot(name);
            }
            d.ui(|_, ui| ui.global::<Backend>().set_popover(0));
            d.ui(|c, _| c.zoom_actual());
            d.shot("14-zoom100");
        }
        if scenario == "keys" || scenario == "all" {
            use slint::platform::{Key, WindowEvent};
            let press = |text: &str| {
                let t: slint::SharedString = text.into();
                d.ui(move |_, ui| {
                    let _ = ui.window().dispatch_event_with_result(WindowEvent::KeyPressed { text: t.clone() });
                    let _ = ui.window().dispatch_event_with_result(WindowEvent::KeyReleased { text: t });
                });
                d.settle(150);
            };
            let tool = |d: &Driver| d.ctl.ux.lock().unwrap().tool.as_str();
            press("h");
            println!("after h: {}", tool(&d));
            press("e");
            println!("after e: {}", tool(&d));
            press("]");
            println!("marker after ]: {}", d.ctl.ux.lock().unwrap().marker);
            // Ctrl+Z via the Control key then z
            let before = d.ctl.ux.lock().unwrap().doc.as_ref().map(|x| x.can_undo);
            d.ui(|_, ui| {
                let _ = ui.window().dispatch_event_with_result(WindowEvent::KeyPressed { text: Key::Control.into() });
                let _ = ui.window().dispatch_event_with_result(WindowEvent::KeyPressed { text: "z".into() });
                let _ = ui.window().dispatch_event_with_result(WindowEvent::KeyReleased { text: "z".into() });
                let _ = ui.window().dispatch_event_with_result(WindowEvent::KeyReleased { text: Key::Control.into() });
            });
            d.settle(300);
            let after = d.ctl.ux.lock().unwrap().doc.as_ref().map(|x| x.can_undo);
            println!("can_undo before {before:?} after ctrl+z {after:?}");
            // Wheel: ctrl + scroll zooms, plain scroll pans.
            let s0 = d.ctl.ux.lock().unwrap().view;
            d.ui(|_, ui| {
                let _ = ui
                    .window()
                    .dispatch_event_with_result(WindowEvent::PointerMoved { position: slint::LogicalPosition::new(300.0, 300.0) });
                let _ = ui.window().dispatch_event_with_result(WindowEvent::KeyPressed { text: Key::Control.into() });
                let _ = ui.window().dispatch_event_with_result(WindowEvent::PointerScrolled {
                    position: slint::LogicalPosition::new(300.0, 300.0),
                    delta_x: 0.0,
                    delta_y: 60.0,
                });
                let _ = ui.window().dispatch_event_with_result(WindowEvent::KeyReleased { text: Key::Control.into() });
            });
            d.settle(300);
            let s1 = d.ctl.ux.lock().unwrap().view;
            println!("scale {:.3} -> {:.3} after ctrl+wheel up", s0.scale, s1.scale);
            d.shot("20-after-keys");
        }
        if scenario == "flows" || scenario == "all" {
            // Crop: drag the south-east handle of the result's crop rectangle inwards.
            let (w, h) = {
                let ux = d.ctl.ux.lock().unwrap();
                (ux.doc.as_ref().unwrap().width as f64, ux.doc.as_ref().unwrap().height as f64)
            };
            let (x1, y1) = {
                let ux = d.ctl.ux.lock().unwrap();
                openscissors::viewport::to_screen(&ux.view, w, h)
            };
            d.drag(1, (x1, y1), (x1 - 150.0, y1 - 90.0));
            let crop = d.ctl.ux.lock().unwrap().doc.as_ref().unwrap().params.crop;
            println!("crop after handle drag: {crop:?}");
            d.shot("30-cropped");
            d.ui(|c, _| c.set_crop(None));
            // Smart select with one click, then apply.
            d.ui(|c, _| c.set_tool(Tool::Select));
            d.ui(|c, _| c.smart_add(openscissors_core::model::Prompt::Point { x: 450.0, y: 300.0, keep: true }));
            let t = Instant::now();
            while d.ctl.ux.lock().unwrap().smart_busy || t.elapsed() < Duration::from_millis(500) {
                std::thread::sleep(Duration::from_millis(100));
                if t.elapsed() > Duration::from_secs(60) {
                    break;
                }
            }
            d.settle(500);
            println!(
                "smart select: prompts {} selection {}",
                d.ctl.ux.lock().unwrap().prompts.len(),
                d.ctl.st.doc.lock().unwrap().as_ref().unwrap().has_selection()
            );
            d.shot("31-smart");
            d.ui(|c, _| c.smart_apply());
            d.settle(500);
            // Hair tool stroke.
            d.ui(|c, _| c.set_tool(Tool::Hair));
            d.drag(0, (330.0, 280.0), (380.0, 300.0));
            d.shot("32-hair");
            // Export to a file.
            let target = d.out.join("export.png");
            let r = d.ctl.export_to(&target);
            println!("export: {:?} exists {}", r, target.exists());
            // Models: use the lightest model again, switch the memory option, delete nothing.
            d.ui(|c, _| c.set_half(true));
            d.ui(|c, _| c.set_half(false));
            println!("budget percent now {}", openscissors_core::sysmem::budget_percent());
        }
        if scenario == "script" {
            run_script(&d, args.get(4).expect("script path"));
        }
        d.ui(|_, _| {
            let _ = slint::quit_event_loop();
        });
    });
    slint::run_event_loop().expect("event loop");
}

/// A scenario written as JSON: a list of steps such as
/// `{"click":[x,y]}`, `{"rclick":[x,y]}`, `{"drag":[x0,y0,x1,y1]}`, `{"key":"z","ctrl":true}`,
/// `{"wheel":[x,y,dy],"ctrl":true}`, `{"shot":"name"}`, `{"wait":ms}`, `{"say":"text"}`.
fn run_script(d: &Driver, path: &str) {
    use slint::platform::{Key, PointerEventButton, WindowEvent};
    let steps: Vec<serde_json::Value> = serde_json::from_str(&std::fs::read_to_string(path).expect("script")).expect("json");
    for step in steps {
        let pt = |k: &str| step[k].as_array().map(|a| a.iter().map(|v| v.as_f64().unwrap_or(0.0) as f32).collect::<Vec<_>>());
        if let Some(p) = pt("click") {
            d.click(p[0], p[1]);
        } else if let Some(p) = pt("rclick") {
            d.ui(move |_, ui| {
                let pos = slint::LogicalPosition::new(p[0], p[1]);
                let w = ui.window();
                let _ = w.dispatch_event_with_result(WindowEvent::PointerMoved { position: pos });
                let _ = w.dispatch_event_with_result(WindowEvent::PointerPressed { position: pos, button: PointerEventButton::Right });
                let _ = w.dispatch_event_with_result(WindowEvent::PointerReleased { position: pos, button: PointerEventButton::Right });
            });
            d.settle(250);
        } else if let Some(p) = pt("drag") {
            let (side, from, to) =
                (if p[0] < 530.5 { 0 } else { 1 }, (p[0] as f64 - 60.0, p[1] as f64 - 60.0), (p[2] as f64 - 60.0, p[3] as f64 - 60.0));
            // Pane coordinates: the left pane starts at x = 48, y = 48.
            let off = if side == 0 { 48.0 } else { 48.0 + 1.0 + (1320.0 - 48.0 - 308.0 - 1.0) / 2.0 };
            let _ = (from, to);
            d.drag(side, (p[0] as f64 - off, p[1] as f64 - 48.0), (p[2] as f64 - off, p[3] as f64 - 48.0));
        } else if let Some(k) = step["key"].as_str() {
            let ctrl = step["ctrl"].as_bool().unwrap_or(false);
            let shift = step["shift"].as_bool().unwrap_or(false);
            let text: slint::SharedString = match k {
                "Escape" => Key::Escape.into(),
                "Backspace" => Key::Backspace.into(),
                "Return" => Key::Return.into(),
                "Left" => Key::LeftArrow.into(),
                "Right" => Key::RightArrow.into(),
                "Up" => Key::UpArrow.into(),
                "Down" => Key::DownArrow.into(),
                other => other.into(),
            };
            d.ui(move |_, ui| {
                let w = ui.window();
                if ctrl {
                    let _ = w.dispatch_event_with_result(WindowEvent::KeyPressed { text: Key::Control.into() });
                }
                if shift {
                    let _ = w.dispatch_event_with_result(WindowEvent::KeyPressed { text: Key::Shift.into() });
                }
                let _ = w.dispatch_event_with_result(WindowEvent::KeyPressed { text: text.clone() });
                let _ = w.dispatch_event_with_result(WindowEvent::KeyReleased { text });
                if shift {
                    let _ = w.dispatch_event_with_result(WindowEvent::KeyReleased { text: Key::Shift.into() });
                }
                if ctrl {
                    let _ = w.dispatch_event_with_result(WindowEvent::KeyReleased { text: Key::Control.into() });
                }
            });
            d.settle(250);
        } else if let Some(p) = pt("wheel") {
            let ctrl = step["ctrl"].as_bool().unwrap_or(false);
            d.ui(move |_, ui| {
                let w = ui.window();
                let pos = slint::LogicalPosition::new(p[0], p[1]);
                let _ = w.dispatch_event_with_result(WindowEvent::PointerMoved { position: pos });
                if ctrl {
                    let _ = w.dispatch_event_with_result(WindowEvent::KeyPressed { text: Key::Control.into() });
                }
                let _ = w.dispatch_event_with_result(WindowEvent::PointerScrolled { position: pos, delta_x: 0.0, delta_y: p[2] });
                if ctrl {
                    let _ = w.dispatch_event_with_result(WindowEvent::KeyReleased { text: Key::Control.into() });
                }
            });
            d.settle(250);
        } else if let Some(p) = pt("window") {
            d.ui(move |_, ui| {
                ui.window().set_size(slint::PhysicalSize::new(p[2] as u32, p[3] as u32));
                ui.window().set_position(slint::PhysicalPosition::new(p[0] as i32, p[1] as i32));
            });
            d.settle(3500);
        } else if let Some(a) = step["param"].as_array() {
            let (path, value) = (a[0].as_str().unwrap_or("").to_string(), a[1].clone());
            d.ui(move |c, _| c.set_param(&path, value));
            d.settle(300);
        } else if let Some(path) = step["open"].as_str() {
            d.ctl.open_path_sync(std::path::Path::new(path));
            d.settle(1500);
        } else if let Some(path) = step["export"].as_str() {
            println!("export {:?}", d.ctl.export_to(std::path::Path::new(path)));
            d.settle(300);
        } else if let Some(code) = step["lang"].as_str() {
            let idx = openscissors::i18n::LOCALES.iter().position(|(c, _)| *c == code).map(|i| i + 1).unwrap_or(1);
            d.ui(move |c, _| c.set_language(idx));
            d.settle(300);
        } else if let Some(name) = step["shot"].as_str() {
            d.shot(name);
        } else if let Some(ms) = step["wait"].as_u64() {
            d.settle(ms);
        } else if let Some(t) = step["say"].as_str() {
            println!("{t}");
        } else if let Some(t) = step["print"].as_str() {
            let v = d.ctl.ux.lock().unwrap();
            match t {
                "tool" => println!("tool {}", v.tool.as_str()),
                "marker" => println!("marker {}", v.marker),
                "view" => println!("view {:?}", v.view),
                "prompts" => println!("prompts {}", v.prompts.len()),
                "window" => {
                    drop(v);
                    let g = d.ui(|_, ui| (ui.window().position(), ui.window().size()));
                    println!("window at {},{} size {}x{}", g.0.x, g.0.y, g.1.width, g.1.height);
                }
                "dialog" => println!("dialog open {}", openscissors::dialogs::is_open()),
                "doc" => println!(
                    "doc {:?}",
                    v.doc.as_ref().map(|x| (
                        x.can_undo,
                        x.can_redo,
                        x.modified,
                        x.params.background.mode,
                        x.params.shadow.enabled,
                        x.params.mask.graphcut
                    ))
                ),
                _ => {}
            }
        }
    }
}
