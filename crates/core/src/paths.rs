//! Where OpenScissors keeps its files: one folder (models, settings, plug-ins, scratch files), the same for the window
//! and for the command line, so that both see the same models and the same saved defaults.

use std::ffi::OsString;
use std::path::{Path, PathBuf};

/// The folder of the application:
/// 1. `OPENSCISSORS_HOME` when it is set;
/// 2. `portable-dev/` at the top of the source tree in a debug build;
/// 3. the folder of the executable (the portable layout), when it can be written;
/// 4. otherwise a folder in the user's profile (a read-only drive, Program Files).
pub fn root() -> PathBuf {
    let exe_dir = std::env::current_exe().ok().and_then(|p| p.parent().map(Path::to_path_buf)).unwrap_or_else(|| PathBuf::from("."));
    let writable = is_writable(&exe_dir);
    pick_root(std::env::var_os("OPENSCISSORS_HOME"), cfg!(debug_assertions), exe_dir, writable)
}

/// The decision of [`root`], from what it looks at.
fn pick_root(home: Option<OsString>, debug: bool, exe_dir: PathBuf, exe_dir_writable: bool) -> PathBuf {
    match home {
        Some(h) if !h.is_empty() => PathBuf::from(h),
        _ if debug => Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..").join("portable-dev"),
        _ if exe_dir_writable => exe_dir,
        _ => fallback_dir(),
    }
}

fn is_writable(dir: &Path) -> bool {
    let probe = dir.join(format!(".osc-write-test-{}", std::process::id()));
    let ok = std::fs::write(&probe, b"").is_ok();
    let _ = std::fs::remove_file(probe);
    ok
}

/// The folder in the user's profile, used only when the folder of the executable cannot be written.
fn fallback_dir() -> PathBuf {
    let var = |name: &str| std::env::var_os(name).filter(|v| !v.is_empty()).map(PathBuf::from);
    let base = if cfg!(windows) {
        var("LOCALAPPDATA")
    } else if cfg!(target_os = "macos") {
        var("HOME").map(|h| h.join("Library").join("Application Support"))
    } else {
        var("XDG_DATA_HOME").or_else(|| var("HOME").map(|h| h.join(".local").join("share")))
    };
    base.unwrap_or_else(|| PathBuf::from(".")).join("OpenScissors")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_folder_is_chosen_in_order_and_never_left_unwritable() {
        let exe = PathBuf::from("exe-dir");
        assert_eq!(pick_root(Some("custom".into()), true, exe.clone(), false), PathBuf::from("custom"), "the variable wins");
        assert!(pick_root(None, true, exe.clone(), true).ends_with("portable-dev"), "debug builds use portable-dev");
        assert_eq!(pick_root(None, false, exe.clone(), true), exe, "the portable folder");
        assert_eq!(pick_root(Some(OsString::new()), false, exe.clone(), true), exe, "an empty variable counts as unset");
        let fallback = pick_root(None, false, exe.clone(), false);
        assert!(fallback.ends_with("OpenScissors") && fallback != exe, "a read-only folder: the profile, {fallback:?}");
    }

    #[test]
    fn a_folder_that_can_be_written_is_recognised() {
        let dir = std::env::temp_dir().join(format!("osc-paths-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        assert!(is_writable(&dir));
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 0, "the probe is removed");
        assert!(!is_writable(&dir.join("missing")));
        std::fs::remove_dir_all(dir).ok();
    }
}
