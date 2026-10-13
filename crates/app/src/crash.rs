//! A panic leaves a trace in `data/crash.log` (next to the settings, so it stays in the portable folder):
//! without it a crash on someone else's machine can only be described, not diagnosed.

use std::io::Write;
use std::panic::PanicHookInfo;
use std::path::{Path, PathBuf};

/// The log is started again when it passes this size.
const MAX_BYTES: u64 = 256 * 1024;

/// Keep the usual message on the console and also append the panic to `dir/crash.log`.
pub fn install(dir: PathBuf) {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let _ = append(&dir, &entry(info));
        previous(info);
    }));
}

fn entry(info: &PanicHookInfo<'_>) -> String {
    let thread = std::thread::current();
    let message = info.payload().downcast_ref::<&str>().map(|s| s.to_string()).or_else(|| info.payload().downcast_ref::<String>().cloned());
    let place = info.location().map(|l| format!("{}:{}:{}", l.file(), l.line(), l.column()));
    format_entry(
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0),
        thread.name().unwrap_or("unnamed"),
        place.as_deref(),
        message.as_deref(),
        &std::backtrace::Backtrace::force_capture().to_string(),
    )
}

fn format_entry(unix_secs: u64, thread: &str, place: Option<&str>, message: Option<&str>, backtrace: &str) -> String {
    format!(
        "[{unix_secs}] version {} thread '{thread}' panicked at {}: {}\n{backtrace}\n\n",
        env!("CARGO_PKG_VERSION"),
        place.unwrap_or("an unknown place"),
        message.unwrap_or("(no message)")
    )
}

fn append(dir: &Path, text: &str) -> std::io::Result<()> {
    let file = dir.join("crash.log");
    if std::fs::metadata(&file).is_ok_and(|m| m.len() > MAX_BYTES) {
        std::fs::remove_file(&file)?;
    }
    std::fs::OpenOptions::new().create(true).append(true).open(file)?.write_all(text.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_entry_says_when_where_and_why() {
        let e = format_entry(1_700_000_000, "render", Some("src/render.rs:10:5"), Some("index out of bounds"), "frame 0\nframe 1");
        assert!(e.starts_with("[1700000000] version "));
        assert!(e.contains("thread 'render' panicked at src/render.rs:10:5: index out of bounds\nframe 0\nframe 1"));
    }

    #[test]
    fn the_log_grows_by_appending_and_starts_again_when_too_big() {
        let dir = std::env::temp_dir().join(format!("osc-crash-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        append(&dir, "one\n").unwrap();
        append(&dir, "two\n").unwrap();
        assert_eq!(std::fs::read_to_string(dir.join("crash.log")).unwrap(), "one\ntwo\n");
        append(&dir, &"x".repeat(MAX_BYTES as usize + 1)).unwrap();
        append(&dir, "fresh\n").unwrap();
        assert_eq!(std::fs::read_to_string(dir.join("crash.log")).unwrap(), "fresh\n");
        std::fs::remove_dir_all(dir).ok();
    }
}
