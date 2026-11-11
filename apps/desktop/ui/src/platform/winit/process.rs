//! The process's age (for `FOLO_UI_TRACE`'s milliseconds since process creation).

use std::time::Duration;

/// How long ago the process was created, where the system says (Windows, Linux); on macOS the time
/// since the first call into this backend's platform, a few milliseconds later than process creation.
pub(crate) fn age() -> Option<Duration> {
    imp::age()
}

/// Marks the backend's start (the fallback clock where the process creation time is unknown).
pub(crate) fn mark_start() {
    let _ = START.get_or_init(std::time::Instant::now);
}

static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();

#[cfg(windows)]
mod imp {
    use std::time::Duration;
    use windows::Win32::Foundation::FILETIME;
    use windows::Win32::System::SystemInformation::GetSystemTimeAsFileTime;
    use windows::Win32::System::Threading::{GetCurrentProcess, GetProcessTimes};

    pub fn age() -> Option<Duration> {
        let (mut created, mut exited, mut kernel, mut user) = Default::default();
        // SAFETY: the pseudo handle of the current process is always valid; the out pointers are locals.
        unsafe { GetProcessTimes(GetCurrentProcess(), &mut created, &mut exited, &mut kernel, &mut user) }
            .ok()?;
        // SAFETY: no preconditions.
        let now = unsafe { GetSystemTimeAsFileTime() };
        let ticks = |t: FILETIME| (t.dwHighDateTime as u64) << 32 | t.dwLowDateTime as u64;
        Some(Duration::from_nanos(ticks(now).saturating_sub(ticks(created)) * 100))
    }
}

#[cfg(target_os = "linux")]
mod imp {
    use std::time::Duration;

    /// From `/proc`: the start time in clock ticks since boot (field 22 of `stat`, at the usual 100 Hz)
    /// against the uptime.
    pub fn age() -> Option<Duration> {
        let stat = std::fs::read_to_string("/proc/self/stat").ok()?;
        let after_name = &stat[stat.rfind(')')? + 2..];
        let start_ticks: f64 = after_name.split_whitespace().nth(19)?.parse().ok()?;
        let uptime: f64 =
            std::fs::read_to_string("/proc/uptime").ok()?.split_whitespace().next()?.parse().ok()?;
        Some(Duration::from_secs_f64((uptime - start_ticks / 100.0).max(0.0)))
    }
}

#[cfg(not(any(windows, target_os = "linux")))]
mod imp {
    use std::time::Duration;

    pub fn age() -> Option<Duration> {
        super::START.get().map(|start| start.elapsed())
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_process_is_young_but_not_unborn() {
        super::mark_start();
        let age = super::age().expect("an age");
        assert!(age < std::time::Duration::from_secs(3600));
    }
}
