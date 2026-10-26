//! Plug-ins: user-installed programs that post-process an exported result.
//!
//! A plug-in is a folder `plugins/<id>/` (next to the models folder) with a `plugin.json`:
//!
//! ```json
//! { "id": "optimize-png", "name": "Optimise PNG", "description": "Runs oxipng on the result",
//!   "command": "run.cmd", "args": ["{input}"], "extensions": ["png"] }
//! ```
//!
//! `command` is a file inside the plug-in folder (never a path outside it, never a shell string); `args` may use
//! `{input}` (the exported file) and `{dir}` (its folder). A plug-in is never run automatically: the user turns it
//! on, then clicks *Run* on the last saved file. It runs with the plug-in folder as working directory, is killed
//! after 60 seconds (with everything it started), and the last lines of its output are shown. Plug-ins are code the user installed: OpenScissors
//! grants them nothing, and says so when one is enabled.

use std::io::Read;
use std::path::{Component, Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::sync::LockExt;

pub const RUN_TIMEOUT: Duration = Duration::from_secs(60);
/// How long the output pipes may stay open after the plug-in itself has finished. A program it started and left
/// running keeps them open; past this delay that program is stopped instead of freezing the application.
const PIPE_GRACE: Duration = Duration::from_secs(2);
const OUTPUT_TAIL: usize = 2000;

fn default_args() -> Vec<String> {
    vec!["{input}".into()]
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Manifest {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub description: String,
    /// A file inside the plug-in folder.
    pub command: String,
    /// Arguments; `{input}` is the file to process, `{dir}` its folder.
    #[serde(default = "default_args")]
    pub args: Vec<String>,
    /// File extensions the plug-in accepts (lower case, no dot); empty = any.
    #[serde(default)]
    pub extensions: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct Plugin {
    pub manifest: Manifest,
    pub dir: PathBuf,
}

/// A folder found in the plug-ins directory: a usable plug-in or the reason it is not.
#[derive(Clone, Debug)]
pub struct Discovered {
    pub folder: String,
    pub plugin: std::result::Result<Plugin, String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct RunOutput {
    pub success: bool,
    /// Last part of what the program printed (standard output and error).
    pub output: String,
}

fn validate(m: &Manifest, dir: &Path) -> Result<()> {
    if m.id.is_empty() || !m.id.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-' || c == '_') {
        return Err(Error::msg("the plug-in id may only use a-z, 0-9, '-' and '_'"));
    }
    let cmd = Path::new(&m.command);
    if m.command.is_empty() || cmd.is_absolute() || cmd.components().any(|c| !matches!(c, Component::Normal(_))) {
        return Err(Error::msg("the command must be a file inside the plug-in folder"));
    }
    let full = dir.join(cmd);
    if !full.is_file() {
        return Err(Error::msg(format!("command file '{}' not found", m.command)));
    }
    let (canon_dir, canon_cmd) = (std::fs::canonicalize(dir)?, std::fs::canonicalize(&full)?);
    if !canon_cmd.starts_with(&canon_dir) {
        return Err(Error::msg("the command resolves outside the plug-in folder"));
    }
    for a in &m.args {
        let mut rest = a.as_str();
        while let Some(open) = rest.find('{') {
            let close = rest[open..].find('}').ok_or_else(|| Error::msg("unbalanced '{' in args"))? + open;
            let name = &rest[open + 1..close];
            if name != "input" && name != "dir" {
                return Err(Error::msg(format!("unknown placeholder '{{{name}}}' in args")));
            }
            rest = &rest[close + 1..];
        }
    }
    Ok(())
}

/// Read every `plugins/<folder>/plugin.json`.
pub fn discover(plugins_dir: &Path) -> Vec<Discovered> {
    let Ok(entries) = std::fs::read_dir(plugins_dir) else { return Vec::new() };
    let mut found: Vec<Discovered> = entries
        .flatten()
        .filter(|e| e.path().is_dir())
        .map(|e| {
            let dir = e.path();
            let folder = e.file_name().to_string_lossy().into_owned();
            let plugin = std::fs::read_to_string(dir.join("plugin.json"))
                .map_err(|_| "no plugin.json".to_string())
                .and_then(|t| serde_json::from_str::<Manifest>(&t).map_err(|e| format!("plugin.json: {e}")))
                .and_then(|m| validate(&m, &dir).map(|_| m).map_err(|e| e.to_string()))
                .map(|manifest| Plugin { manifest, dir });
            Discovered { folder, plugin }
        })
        .collect();
    found.sort_by(|a, b| a.folder.cmp(&b.folder));
    found
}

/// Replace `{input}` and `{dir}` in one pass: a file name that itself contains `{dir}` stays as it is.
fn expand(arg: &str, input: &Path) -> String {
    let dir = input.parent().map(|p| p.display().to_string()).unwrap_or_default();
    let file = input.display().to_string();
    let mut out = String::with_capacity(arg.len());
    let mut rest = arg;
    while let Some(i) = rest.find('{') {
        out.push_str(&rest[..i]);
        let tail = &rest[i..];
        if let Some(t) = tail.strip_prefix("{input}") {
            out.push_str(&file);
            rest = t;
        } else if let Some(t) = tail.strip_prefix("{dir}") {
            out.push_str(&dir);
            rest = t;
        } else {
            out.push('{');
            rest = &tail[1..];
        }
    }
    out.push_str(rest);
    out
}

/// Run `plugin` on `input` with the plug-in folder as working directory. Killed after `timeout`.
pub fn run(plugin: &Plugin, input: &Path, timeout: Duration) -> Result<RunOutput> {
    if !input.is_file() {
        return Err(Error::msg("the file to process does not exist"));
    }
    let ext = input.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
    let wanted = &plugin.manifest.extensions;
    if !wanted.is_empty() && !wanted.iter().any(|e| e.eq_ignore_ascii_case(&ext)) {
        return Err(Error::msg(format!("this plug-in handles: {}", wanted.join(", "))));
    }
    let mut cmd = Command::new(plugin.dir.join(&plugin.manifest.command));
    cmd.args(plugin.manifest.args.iter().map(|a| expand(a, input)))
        .current_dir(&plugin.dir)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0); // its own group, so that the whole tree can be stopped
    }
    let mut child = cmd.spawn()?;
    let tree = ProcessTree::attach(&child);
    // Drain both pipes on threads so a chatty program cannot block on a full pipe. They append to one shared buffer, so
    // what was printed so far is still there if a pipe never closes.
    let printed = Arc::new(Mutex::new(Vec::new()));
    let mut readers: Vec<JoinHandle<()>> = Vec::new();
    if let Some(s) = child.stdout.take() {
        readers.push(drain(s, printed.clone()));
    }
    if let Some(s) = child.stderr.take() {
        readers.push(drain(s, printed.clone()));
    }
    let start = Instant::now();
    let status = loop {
        if let Some(s) = child.try_wait()? {
            break s;
        }
        if start.elapsed() > timeout {
            tree.kill(&mut child);
            let _ = child.wait();
            return Err(Error::msg(format!("the plug-in did not finish within {} seconds and was stopped", timeout.as_secs())));
        }
        std::thread::sleep(Duration::from_millis(25));
    };
    if !wait_finished(&readers, PIPE_GRACE) {
        eprintln!("OpenScissors: plug-in '{}' left a program running after it finished; stopping it", plugin.manifest.id);
        tree.kill(&mut child);
        wait_finished(&readers, PIPE_GRACE);
    }
    let output = String::from_utf8_lossy(&printed.guard()).into_owned();
    let output = output.trim();
    let tail: String = output.chars().rev().take(OUTPUT_TAIL).collect::<Vec<_>>().into_iter().rev().collect();
    Ok(RunOutput { success: status.success(), output: tail })
}

/// Copy everything `r` produces into `into`, as it comes.
fn drain(mut r: impl Read + Send + 'static, into: Arc<Mutex<Vec<u8>>>) -> JoinHandle<()> {
    std::thread::spawn(move || {
        let mut chunk = [0u8; 4096];
        while let Ok(n) = r.read(&mut chunk) {
            if n == 0 {
                break;
            }
            into.guard().extend_from_slice(&chunk[..n]);
        }
    })
}

/// Wait up to `limit` for every reader to end; false if one is still going.
fn wait_finished(readers: &[JoinHandle<()>], limit: Duration) -> bool {
    let start = Instant::now();
    while !readers.iter().all(JoinHandle::is_finished) {
        if start.elapsed() >= limit {
            return false;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    true
}

/// A way to stop a program together with everything it started (a script usually runs other programs).
/// Windows: a job object that also kills its members when closed. Unix: the process group the plug-in leads.
struct ProcessTree {
    #[cfg(windows)]
    job: windows_sys::Win32::Foundation::HANDLE,
    #[cfg(unix)]
    group: u32,
}

#[cfg(windows)]
impl ProcessTree {
    fn attach(child: &Child) -> Self {
        use std::os::windows::io::AsRawHandle;
        use windows_sys::Win32::System::JobObjects::*;
        // SAFETY: plain Win32 calls on handles owned here; the limit structure is plain data, zero is a valid start.
        unsafe {
            let job = CreateJobObjectW(std::ptr::null(), std::ptr::null());
            if !job.is_null() {
                let mut limits: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
                limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
                let size = std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32;
                let set = SetInformationJobObject(job, JobObjectExtendedLimitInformation, std::ptr::addr_of!(limits).cast(), size);
                if set == 0 || AssignProcessToJobObject(job, child.as_raw_handle() as _) == 0 {
                    windows_sys::Win32::Foundation::CloseHandle(job);
                    return Self { job: std::ptr::null_mut() };
                }
            }
            Self { job }
        }
    }

    fn kill(&self, child: &mut Child) {
        if !self.job.is_null() {
            // SAFETY: `job` is a live job object handle.
            unsafe { windows_sys::Win32::System::JobObjects::TerminateJobObject(self.job, 1) };
        }
        let _ = child.kill();
    }
}

#[cfg(windows)]
impl Drop for ProcessTree {
    fn drop(&mut self) {
        if !self.job.is_null() {
            // SAFETY: closing the handle also ends whatever is still in the job.
            unsafe { windows_sys::Win32::Foundation::CloseHandle(self.job) };
        }
    }
}

#[cfg(unix)]
impl ProcessTree {
    fn attach(child: &Child) -> Self {
        Self { group: child.id() }
    }

    fn kill(&self, child: &mut Child) {
        let _ =
            Command::new("kill").args(["-KILL", "--", &format!("-{}", self.group)]).stdout(Stdio::null()).stderr(Stdio::null()).status();
        let _ = child.kill();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("osc-plug-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn script(body_win: &str, body_unix: &str) -> (&'static str, String) {
        if cfg!(windows) {
            ("run.cmd", format!("@echo off\r\n{body_win}\r\n"))
        } else {
            ("run.sh", format!("#!/bin/sh\n{body_unix}\n"))
        }
    }

    fn install(root: &Path, folder: &str, manifest: serde_json::Value, body_win: &str, body_unix: &str) -> PathBuf {
        let dir = root.join(folder);
        std::fs::create_dir_all(&dir).unwrap();
        let (name, text) = script(body_win, body_unix);
        std::fs::write(dir.join(name), text).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(dir.join(name), std::fs::Permissions::from_mode(0o755)).unwrap();
        }
        let mut m = manifest;
        m["command"] = serde_json::json!(name);
        std::fs::write(dir.join("plugin.json"), m.to_string()).unwrap();
        dir
    }

    #[test]
    fn manifests_are_validated() {
        let root = tmp("validate");
        let good = serde_json::json!({ "id": "ok-1", "name": "Ok" });
        install(&root, "good", good, "echo hi", "echo hi");
        let bad_id = install(&root, "bad-id", serde_json::json!({ "id": "Bad ID", "name": "x" }), "echo", "echo");
        let _ = bad_id;
        install(&root, "bad-arg", serde_json::json!({ "id": "a", "name": "x", "args": ["{input}", "{secret}"] }), "echo", "echo");
        let escape = install(&root, "escape", serde_json::json!({ "id": "e", "name": "x" }), "echo", "echo");
        let mut m: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(escape.join("plugin.json")).unwrap()).unwrap();
        m["command"] = serde_json::json!("../good/run.cmd");
        std::fs::write(escape.join("plugin.json"), m.to_string()).unwrap();
        let missing = root.join("missing");
        std::fs::create_dir_all(&missing).unwrap();
        std::fs::write(missing.join("plugin.json"), r#"{"id":"m","name":"m","command":"nope.exe"}"#).unwrap();
        std::fs::create_dir_all(root.join("empty")).unwrap();

        let found = discover(&root);
        let by = |f: &str| found.iter().find(|d| d.folder == f).unwrap();
        assert!(by("good").plugin.is_ok());
        assert!(by("bad-id").plugin.as_ref().unwrap_err().contains("id"));
        assert!(by("bad-arg").plugin.as_ref().unwrap_err().contains("placeholder"));
        assert!(by("escape").plugin.as_ref().unwrap_err().contains("inside"));
        assert!(by("missing").plugin.as_ref().unwrap_err().contains("not found"));
        assert!(by("empty").plugin.as_ref().unwrap_err().contains("plugin.json"));
        assert_eq!(by("good").plugin.as_ref().unwrap().manifest.args, vec!["{input}"], "default args");
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn a_plugin_runs_on_a_real_file_and_reports_its_output() {
        let root = tmp("run");
        let dir = install(
            &root,
            "p",
            serde_json::json!({ "id": "p", "name": "P", "extensions": ["png"] }),
            "echo processed> \"%~1.done\"\r\necho all good",
            "echo processed > \"$1.done\"; echo all good",
        );
        let file = root.join("result.png");
        std::fs::write(&file, b"x").unwrap();
        let plugin = discover(&root).into_iter().find(|d| d.folder == "p").unwrap().plugin.unwrap();
        assert_eq!(plugin.dir, dir);
        let out = run(&plugin, &file, Duration::from_secs(20)).unwrap();
        assert!(out.success && out.output.contains("all good"), "{out:?}");
        assert!(root.join("result.png.done").exists(), "the plug-in saw the file");

        let wrong = root.join("result.jpg");
        std::fs::write(&wrong, b"x").unwrap();
        assert!(run(&plugin, &wrong, Duration::from_secs(5)).unwrap_err().to_string().contains("png"));
        assert!(run(&plugin, &root.join("missing.png"), Duration::from_secs(5)).is_err());
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn failing_and_slow_plugins_are_handled() {
        let root = tmp("slow");
        install(&root, "fail", serde_json::json!({ "id": "fail", "name": "F" }), "echo boom 1>&2\r\nexit /b 3", "echo boom >&2; exit 3");
        install(&root, "slow", serde_json::json!({ "id": "slow", "name": "S" }), "ping -n 8 127.0.0.1 >nul", "sleep 7");
        let file = root.join("f.png");
        std::fs::write(&file, b"x").unwrap();
        let found = discover(&root);
        let get = |f: &str| found.iter().find(|d| d.folder == f).unwrap().plugin.clone().unwrap();
        let out = run(&get("fail"), &file, Duration::from_secs(20)).unwrap();
        assert!(!out.success && out.output.contains("boom"), "{out:?}");
        let started = Instant::now();
        let err = run(&get("slow"), &file, Duration::from_millis(500)).unwrap_err();
        assert!(err.to_string().contains("did not finish") && started.elapsed() < Duration::from_secs(5));
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn a_percent_sign_in_a_file_name_reaches_the_script_untouched() {
        let root = tmp("percent");
        install(&root, "p", serde_json::json!({ "id": "p", "name": "P" }), "echo [%~1]", "echo \"[$1]\"");
        // `%USERNAME%` would be replaced by cmd.exe when it reads the script arguments.
        let file = root.join("100%USERNAME%.png");
        if std::fs::write(&file, b"x").is_err() {
            return; // this system does not allow '%' in a file name
        }
        let plugin = discover(&root).into_iter().next().unwrap().plugin.unwrap();
        let out = run(&plugin, &file, Duration::from_secs(20)).unwrap();
        assert!(out.output.contains("100%USERNAME%.png"), "{out:?}");
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn a_program_left_running_by_a_plugin_does_not_freeze_the_wait() {
        let root = tmp("stray");
        // The script ends at once, but the program it started keeps the output pipes open for 30 seconds.
        install(
            &root,
            "stray",
            serde_json::json!({ "id": "stray", "name": "S" }),
            "start /b ping -n 30 127.0.0.1\r\necho finished",
            "sleep 30 &\necho finished",
        );
        let file = root.join("f.png");
        std::fs::write(&file, b"x").unwrap();
        let plugin = discover(&root).into_iter().next().unwrap().plugin.unwrap();
        let started = Instant::now();
        let out = run(&plugin, &file, Duration::from_secs(20)).unwrap();
        assert!(out.success && out.output.contains("finished"), "{out:?}");
        assert!(started.elapsed() < Duration::from_secs(10), "returned after {:?}", started.elapsed());
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn placeholders_are_replaced_in_one_pass() {
        let input = Path::new("/photos/{dir}/a{input}.png");
        assert_eq!(expand("{input}", input), "/photos/{dir}/a{input}.png");
        assert_eq!(expand("--in={input} --dir={dir} {other}", input), "--in=/photos/{dir}/a{input}.png --dir=/photos/{dir} {other}");
        assert_eq!(expand("plain {", input), "plain {");
    }
}
