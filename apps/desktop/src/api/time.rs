//! Minimal RFC 3339 (UTC, `Z`) parsing and formatting, plus human durations.

/// Days since 1970-01-01 for a civil date (Howard Hinnant's algorithm).
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let doy = (153 * (m + if m > 2 { -3 } else { 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

fn civil_from_days(z: i64) -> (i64, i64, i64) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// Parse `YYYY-MM-DDTHH:MM:SS[.fff](Z|±HH:MM)` to Unix seconds.
pub fn parse(s: &str) -> Option<i64> {
    let b = s.as_bytes();
    if b.len() < 19 {
        return None;
    }
    let n = |r: std::ops::Range<usize>| s.get(r)?.parse::<i64>().ok();
    let (y, mo, d) = (n(0..4)?, n(5..7)?, n(8..10)?);
    let (h, mi, se) = (n(11..13)?, n(14..16)?, n(17..19)?);
    let mut rest = &s[19..];
    if let Some(r) = rest.strip_prefix('.') {
        rest = r.trim_start_matches(|c: char| c.is_ascii_digit());
    }
    let offset = match rest {
        "Z" | "z" | "" => 0,
        _ => {
            let sign = if rest.starts_with('-') { -1 } else { 1 };
            let oh: i64 = rest.get(1..3)?.parse().ok()?;
            let om: i64 = rest.get(4..6)?.parse().ok()?;
            sign * (oh * 3600 + om * 60)
        }
    };
    Some(days_from_civil(y, mo, d) * 86_400 + h * 3600 + mi * 60 + se - offset)
}

pub fn format(t: i64) -> String {
    let (y, m, d) = civil_from_days(t.div_euclid(86_400));
    let s = t.rem_euclid(86_400);
    format!("{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z", s / 3600, s / 60 % 60, s % 60)
}

/// Clock time `HH:MM:SS` (UTC) for log lines.
pub fn clock(t: i64) -> String {
    let s = t.rem_euclid(86_400);
    format!("{:02}:{:02}:{:02}", s / 3600, s / 60 % 60, s % 60)
}

/// `clock` in the machine's local time, as Studio shows log times (the server's are UTC). Each
/// time is converted with the rules for its own date, so a July log reads in summer time in November.
pub fn local_clock(t: i64) -> String {
    clock(t + local_offset_at(t))
}

/// `YYYY-MM-DD HH:MM` in the machine's local time: the exact moment behind an "ago".
pub fn local_datetime(t: i64) -> String {
    let lt = t + local_offset_at(t);
    let (y, m, d) = civil_from_days(lt.div_euclid(86_400));
    let s = lt.rem_euclid(86_400);
    format!("{y:04}-{m:02}-{d:02} {:02}:{:02}", s / 3600, s / 60 % 60)
}

/// A deadline as a clock reading in local time: "14:32" when it falls on the same day as `now`,
/// else with its date (`local_datetime`).
pub fn local_deadline(t: i64, now: i64) -> String {
    let local = |x: i64| x + local_offset_at(x);
    if local(t).div_euclid(86_400) != local(now).div_euclid(86_400) {
        return local_datetime(t);
    }
    let s = local(t).rem_euclid(86_400);
    format!("{:02}:{:02}", s / 3600, s / 60 % 60)
}

/// Seconds east of UTC at `t` (Windows: the zone's rules for that date, daylight saving included;
/// UTC elsewhere).
#[cfg(windows)]
fn local_offset_at(t: i64) -> i64 {
    use windows_sys::Win32::Foundation::SYSTEMTIME;
    use windows_sys::Win32::System::Time::SystemTimeToTzSpecificLocalTime;
    let (y, mo, d) = civil_from_days(t.div_euclid(86_400));
    let s = t.rem_euclid(86_400);
    let utc = SYSTEMTIME {
        wYear: y as u16,
        wMonth: mo as u16,
        wDayOfWeek: 0,
        wDay: d as u16,
        wHour: (s / 3600) as u16,
        wMinute: (s / 60 % 60) as u16,
        wSecond: (s % 60) as u16,
        wMilliseconds: 0,
    };
    // SAFETY: both pointers are to live SYSTEMTIMEs; a null zone means the current one.
    let mut local: SYSTEMTIME = unsafe { std::mem::zeroed() };
    if unsafe { SystemTimeToTzSpecificLocalTime(std::ptr::null(), &utc, &mut local) } == 0 {
        return 0;
    }
    let days = days_from_civil(local.wYear as i64, local.wMonth as i64, local.wDay as i64);
    let lt = days * 86_400 + local.wHour as i64 * 3600 + local.wMinute as i64 * 60 + local.wSecond as i64;
    lt - t
}

#[cfg(not(windows))]
fn local_offset_at(_t: i64) -> i64 {
    0
}

/// "just now", "4 min ago", "3 h ago", "2 d ago", or a date for older times.
pub fn ago(then: i64, now: i64) -> String {
    let d = (now - then).max(0);
    match d {
        0..45 => "just now".into(),
        45..3600 => format!("{} min ago", (d + 30) / 60),
        3600..86_400 => format!("{} h ago", d / 3600),
        86_400..604_800 => format!("{} d ago", d / 86_400),
        _ => {
            let (y, m, dd) = civil_from_days(then.div_euclid(86_400));
            const MONTHS: [&str; 12] =
                ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
            format!("{} {dd}, {y}", MONTHS[(m - 1) as usize])
        }
    }
}

/// [`ago`] for a moment shown beside a running clock (a live run's "started …" next to its elapsed
/// time): seconds under a minute ("35 s ago", where "just now" sat beside "35s elapsed").
pub fn ago_live(then: i64, now: i64) -> String {
    match (now - then).max(0) {
        d @ 0..60 => format!("{d} s ago"),
        _ => ago(then, now),
    }
}

/// Compact duration: "42 s", "12 m 05 s", "3 h 20 m".
pub fn duration(secs: f64) -> String {
    let s = secs.max(0.0).round() as i64;
    // Units hug their numbers ("14m 55s"), like a clock reading, not prose.
    if s < 60 {
        format!("{s}s")
    } else if s < 3600 {
        format!("{}m {:02}s", s / 60, s % 60)
    } else {
        format!("{}h {:02}m", s / 3600, s / 60 % 60)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_live_runs_start_reads_like_its_elapsed_clock() {
        assert_eq!(ago_live(100, 135), "35 s ago");
        assert_eq!(ago(100, 135), "just now");
        assert_eq!(ago_live(100, 190), "2 min ago");
    }

    #[test]
    fn round_trip() {
        let t = parse("2026-11-08T06:00:00Z").unwrap();
        assert_eq!(format(t), "2026-11-08T06:00:00Z");
        assert_eq!(parse("1970-01-01T00:00:00Z"), Some(0));
        assert_eq!(parse("2026-11-08T07:00:00+01:00"), Some(t));
        assert_eq!(parse("2026-11-08T06:00:00.250Z"), Some(t));
        assert_eq!(parse("nope"), None);
    }

    #[test]
    fn humanised() {
        assert_eq!(ago(0, 10), "just now");
        assert_eq!(ago(0, 600), "10 min ago");
        assert_eq!(ago(0, 7200), "2 h ago");
        assert_eq!(duration(1840.0), "30m 40s");
        assert_eq!(duration(12_000.0), "3h 20m");
    }

    #[test]
    fn deadlines_name_the_date_only_on_another_day() {
        let now = parse("2026-11-08T07:30:00Z").unwrap();
        let today = local_deadline(now, now);
        assert!(today.len() == 5 && today.as_bytes()[2] == b':', "{today}");
        assert_eq!(local_deadline(now + 3 * 86_400, now), local_datetime(now + 3 * 86_400));
    }
}
