//! Memory of the machine and of this process: to refuse work that cannot fit instead of crashing,
//! and to keep the application under its share of the RAM.

/// Default share of the physical memory the application may use (the web view is not counted).
pub const DEFAULT_BUDGET_PERCENT: u64 = 25;

static BUDGET_PERCENT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(DEFAULT_BUDGET_PERCENT);

/// Change the share of the physical memory the application may use (1 to 90).
pub fn set_budget_percent(percent: u64) {
    BUDGET_PERCENT.store(percent.clamp(1, 90), std::sync::atomic::Ordering::Relaxed);
}

/// The share of the physical memory the application may use now.
pub fn budget_percent() -> u64 {
    BUDGET_PERCENT.load(std::sync::atomic::Ordering::Relaxed)
}

/// Memory a new allocation could still get, in MB: the smaller of the free physical memory and
/// the room left in the commit limit (RAM plus page file), which is what makes allocations fail.
/// `None` when the platform does not tell.
pub fn available_memory_mb() -> Option<u64> {
    imp::available()
}

/// Installed physical memory in MB, `None` when unknown.
pub fn total_memory_mb() -> Option<u64> {
    imp::total()
}

/// Memory held by this process right now (working set) in MB, `None` when unknown.
pub fn process_memory_mb() -> Option<u64> {
    imp::process()
}

/// Processor time used by this process so far (all threads, user + kernel) in ms, `None` when unknown.
pub fn process_cpu_ms() -> Option<u64> {
    imp::cpu_ms()
}

/// What the application may use at most: [`budget_percent`] of the physical memory
/// (`OPENSCISSORS_BUDGET_MB` replaces it, to test a small machine). `None` (no limit) when the platform does not tell.
pub fn budget_mb() -> Option<u64> {
    if let Some(mb) = std::env::var("OPENSCISSORS_BUDGET_MB").ok().and_then(|v| v.parse().ok()) {
        return Some(mb);
    }
    total_memory_mb().map(|t| t * budget_percent() / 100)
}

/// Resident memory in MB from the text of `/proc/<pid>/status` (`VmRSS:  12345 kB`). The size is given in kB, so
/// the page size of the machine (4, 16 or 64 kB) does not matter.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn vm_rss_mb(status: &str) -> Option<u64> {
    let kb: u64 = status.lines().find_map(|l| l.strip_prefix("VmRSS:"))?.trim().trim_end_matches("kB").trim().parse().ok()?;
    Some(kb / 1024)
}

#[cfg(windows)]
mod imp {
    use windows_sys::Win32::Foundation::FILETIME;
    use windows_sys::Win32::System::ProcessStatus::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, GetProcessTimes};

    fn status() -> Option<MEMORYSTATUSEX> {
        let mut m: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
        m.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
        // SAFETY: `m` is a valid, correctly sized MEMORYSTATUSEX with `dwLength` set.
        (unsafe { GlobalMemoryStatusEx(&mut m) } != 0).then_some(m)
    }

    pub fn available() -> Option<u64> {
        status().map(|m| m.ullAvailPhys.min(m.ullAvailPageFile) / (1024 * 1024))
    }

    pub fn total() -> Option<u64> {
        status().map(|m| m.ullTotalPhys / (1024 * 1024))
    }

    pub fn process() -> Option<u64> {
        let mut c: PROCESS_MEMORY_COUNTERS = unsafe { std::mem::zeroed() };
        c.cb = std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
        // SAFETY: `c` is a valid, correctly sized structure and the pseudo handle needs no closing.
        (unsafe { GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) } != 0).then_some(c.WorkingSetSize as u64 / (1024 * 1024))
    }

    pub fn cpu_ms() -> Option<u64> {
        let zero = FILETIME { dwLowDateTime: 0, dwHighDateTime: 0 };
        let (mut created, mut exited, mut kernel, mut user) = (zero, zero, zero, zero);
        // SAFETY: all four pointers are valid FILETIMEs; the pseudo handle needs no closing.
        if unsafe { GetProcessTimes(GetCurrentProcess(), &mut created, &mut exited, &mut kernel, &mut user) } == 0 {
            return None;
        }
        let ticks = |f: FILETIME| ((f.dwHighDateTime as u64) << 32) | f.dwLowDateTime as u64; // 100 ns units
        Some((ticks(kernel) + ticks(user)) / 10_000)
    }
}

#[cfg(target_os = "linux")]
mod imp {
    fn meminfo(key: &str) -> Option<u64> {
        let text = std::fs::read_to_string("/proc/meminfo").ok()?;
        let kb: u64 = text.lines().find_map(|l| l.strip_prefix(key))?.trim().trim_end_matches("kB").trim().parse().ok()?;
        Some(kb / 1024)
    }

    pub fn available() -> Option<u64> {
        meminfo("MemAvailable:")
    }

    pub fn total() -> Option<u64> {
        meminfo("MemTotal:")
    }

    pub fn process() -> Option<u64> {
        super::vm_rss_mb(&std::fs::read_to_string("/proc/self/status").ok()?)
    }

    pub fn cpu_ms() -> Option<u64> {
        let text = std::fs::read_to_string("/proc/self/stat").ok()?;
        // Fields after the parenthesised name: utime and stime are the 12th and 13th. /proc counts them in USER_HZ,
        // which is 100 on every mainstream architecture whatever the kernel's own tick rate.
        let rest = text.rsplit_once(')')?.1;
        let mut f = rest.split_whitespace().skip(11);
        let (u, k): (u64, u64) = (f.next()?.parse().ok()?, f.next()?.parse().ok()?);
        Some((u + k) * 10)
    }
}

#[cfg(not(any(windows, target_os = "linux")))]
mod imp {
    pub fn available() -> Option<u64> {
        None
    }
    pub fn total() -> Option<u64> {
        None
    }
    pub fn process() -> Option<u64> {
        None
    }
    pub fn cpu_ms() -> Option<u64> {
        None
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn resident_memory_is_read_in_kilobytes_whatever_the_page_size() {
        let status = "Name:	openscissors
VmPeak:	 9999999 kB
VmRSS:	  524288 kB
Threads:	12
";
        assert_eq!(super::vm_rss_mb(status), Some(512));
        assert_eq!(
            super::vm_rss_mb(
                "Name:	x
"
            ),
            None
        );
        assert_eq!(super::vm_rss_mb("VmRSS: nonsense kB"), None);
    }

    #[test]
    fn reports_a_plausible_amount_on_supported_platforms() {
        if cfg!(any(windows, target_os = "linux")) {
            let free = super::available_memory_mb().expect("supported platform");
            let total = super::total_memory_mb().expect("supported platform");
            let mine = super::process_memory_mb().expect("supported platform");
            assert!(free > 16 && free < 16 * 1024 * 1024, "{free} MB");
            assert!(total >= free.min(total) && total > 256, "{total} MB");
            assert!(mine > 0 && mine < total, "{mine} MB");
            let t0 = super::process_cpu_ms().expect("supported platform");
            let spin = std::time::Instant::now();
            while spin.elapsed().as_millis() < 120 {}
            assert!(super::process_cpu_ms().unwrap() >= t0);
            if std::env::var_os("OPENSCISSORS_BUDGET_MB").is_none() {
                super::set_budget_percent(super::DEFAULT_BUDGET_PERCENT);
                assert_eq!(super::budget_mb(), Some(total * 25 / 100));
                super::set_budget_percent(50);
                assert_eq!(super::budget_mb(), Some(total / 2));
                super::set_budget_percent(super::DEFAULT_BUDGET_PERCENT);
            }
        }
    }
}
