#![cfg_attr(all(not(debug_assertions), windows), windows_subsystem = "windows")]

fn main() {
    if let Err(e) = openscissors::run() {
        eprintln!("OpenScissors: {e}");
        std::process::exit(1);
    }
}
