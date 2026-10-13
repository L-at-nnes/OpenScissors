//! OpenScissors: the desktop application (Slint interface over `openscissors-core`).

use openscissors_core::sync::LockExt;

slint::include_modules!();

pub mod crash;
pub mod cropmath;
pub mod ctl;
pub mod dialogs;
pub mod i18n;
pub mod input;
pub mod models;
pub mod ops;
pub mod render;
pub mod state;
pub mod viewport;
pub mod window;
pub mod wire;

/// Build the window and run until it closes. An image or project path on the command line is opened.
pub fn run() -> Result<(), slint::PlatformError> {
    state::use_portable_temp();
    crash::install(state::Paths::resolve().data);
    let ui = MainWindow::new()?;
    let ctl = ctl::Ctl::new(&ui);
    wire::wire(&ui, &ctl);
    let saved = ctl.st.settings.guard().window;
    window::restore(&ui, saved);
    ui.show()?;
    window::settle(&ui, saved);
    if let Some(path) = std::env::args_os().nth(1) {
        let path = std::path::PathBuf::from(path);
        if path.is_file() {
            let c = ctl.clone();
            slint::Timer::single_shot(std::time::Duration::from_millis(150), move || c.open_path(path));
        }
    }
    ui.run()
}
