//! Small Win32 helpers: message parameter words, window longs, wide strings, the Windows version,
//! the system appearance, DWM dark mode, cursors, and the `FOLO_UI_*` developer switches.

use crate::{CursorStyle, Pixels, Point, WindowAppearance, px};
use std::sync::OnceLock;
use ::windows::Win32::Foundation::{HWND, LPARAM, WPARAM};
use ::windows::Win32::Graphics::Dwm::{DWMWA_USE_IMMERSIVE_DARK_MODE, DwmSetWindowAttribute};
use ::windows::Win32::System::Registry::{HKEY_CURRENT_USER, RRF_RT_REG_DWORD, RegGetValueW};
use ::windows::Win32::UI::WindowsAndMessaging::*;
use ::windows::core::{BOOL, w};

/// The low and high 16-bit words of a message parameter.
pub(crate) trait HiLoWord {
    fn hiword(&self) -> u16;
    fn loword(&self) -> u16;
    fn signed_hiword(&self) -> i16 {
        self.hiword() as i16
    }
    fn signed_loword(&self) -> i16 {
        self.loword() as i16
    }
}

impl HiLoWord for WPARAM {
    fn hiword(&self) -> u16 {
        ((self.0 >> 16) & 0xFFFF) as u16
    }
    fn loword(&self) -> u16 {
        (self.0 & 0xFFFF) as u16
    }
}

impl HiLoWord for LPARAM {
    fn hiword(&self) -> u16 {
        ((self.0 >> 16) & 0xFFFF) as u16
    }
    fn loword(&self) -> u16 {
        (self.0 & 0xFFFF) as u16
    }
}

/// `GetWindowLongPtrW`.
pub(crate) fn get_window_long(hwnd: HWND, index: WINDOW_LONG_PTR_INDEX) -> isize {
    // SAFETY: reading a window long has no memory-safety requirements; an invalid window gives 0.
    unsafe { GetWindowLongPtrW(hwnd, index) }
}

/// `SetWindowLongPtrW`; returns the previous value.
pub(crate) fn set_window_long(hwnd: HWND, index: WINDOW_LONG_PTR_INDEX, value: isize) -> isize {
    // SAFETY: as above; the callers own what the value means.
    unsafe { SetWindowLongPtrW(hwnd, index, value) }
}

/// A device-pixel position as window logical pixels (GPUI's `logical_point`).
#[inline]
pub(crate) fn logical_point(x: f32, y: f32, scale_factor: f32) -> Point<Pixels> {
    Point { x: px(x / scale_factor), y: px(y / scale_factor) }
}

/// Windows 10 or 11 (the custom title bar's top inset differs).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WindowsVersion {
    Win10,
    Win11,
}

impl WindowsVersion {
    /// The running version (`RtlGetVersion`, which does not lie without a manifest).
    pub(crate) fn current() -> Self {
        static VERSION: OnceLock<WindowsVersion> = OnceLock::new();
        *VERSION.get_or_init(|| {
            let mut info = ::windows::Win32::System::SystemInformation::OSVERSIONINFOW {
                dwOSVersionInfoSize: std::mem::size_of::<
                    ::windows::Win32::System::SystemInformation::OSVERSIONINFOW,
                >() as u32,
                ..Default::default()
            };
            // SAFETY: `info` is a valid, sized OSVERSIONINFOW.
            let status = unsafe { ::windows::Wdk::System::SystemServices::RtlGetVersion(&mut info) };
            if status.is_ok() && info.dwBuildNumber >= 22000 { Self::Win11 } else { Self::Win10 }
        })
    }
}

/// The apps' light / dark setting (what GPUI reads through `UISettings`' foreground colour).
pub(crate) fn system_appearance() -> WindowAppearance {
    let mut value: u32 = 1;
    let mut size = std::mem::size_of::<u32>() as u32;
    // SAFETY: the output buffer is a u32 and `size` says so.
    let status = unsafe {
        RegGetValueW(
            HKEY_CURRENT_USER,
            w!("Software\\Microsoft\\Windows\\CurrentVersion\\Themes\\Personalize"),
            w!("AppsUseLightTheme"),
            RRF_RT_REG_DWORD,
            None,
            Some(&mut value as *mut u32 as *mut _),
            Some(&mut size),
        )
    };
    if status.is_ok() && value == 0 { WindowAppearance::Dark } else { WindowAppearance::Light }
}

/// The dark frame DWM draws around the window in dark mode (GPUI's `configure_dwm_dark_mode`).
pub(crate) fn configure_dwm_dark_mode(hwnd: HWND, appearance: WindowAppearance) {
    let dark: BOOL = matches!(appearance, WindowAppearance::Dark | WindowAppearance::VibrantDark).into();
    // SAFETY: the attribute takes a BOOL of the given size.
    let _ = unsafe {
        DwmSetWindowAttribute(
            hwnd,
            DWMWA_USE_IMMERSIVE_DARK_MODE,
            &dark as *const BOOL as *const _,
            std::mem::size_of::<BOOL>() as u32,
        )
    };
}

/// A shared system cursor handle (never destroyed, so it may cross threads).
#[derive(Clone, Copy)]
struct SharedCursor(isize);

/// The system cursor for a style (`None`: hidden), GPUI's mapping.
pub(crate) fn load_cursor(style: CursorStyle) -> Option<HCURSOR> {
    static CURSORS: [OnceLock<SharedCursor>; 7] = [const { OnceLock::new() }; 7];
    let (slot, name) = match style {
        CursorStyle::IBeam | CursorStyle::IBeamCursorForVerticalLayout => (0, IDC_IBEAM),
        CursorStyle::Crosshair => (1, IDC_CROSS),
        CursorStyle::PointingHand | CursorStyle::DragLink => (2, IDC_HAND),
        CursorStyle::ResizeLeft
        | CursorStyle::ResizeRight
        | CursorStyle::ResizeLeftRight
        | CursorStyle::ResizeColumn => (3, IDC_SIZEWE),
        CursorStyle::ResizeUp | CursorStyle::ResizeDown | CursorStyle::ResizeUpDown | CursorStyle::ResizeRow => {
            (4, IDC_SIZENS)
        }
        CursorStyle::OperationNotAllowed => (5, IDC_NO),
        CursorStyle::None => return None,
        _ => (6, IDC_ARROW),
    };
    let cursor = CURSORS[slot].get_or_init(|| {
        // SAFETY: loading a shared system cursor; LR_SHARED handles are never freed.
        let handle = unsafe { LoadImageW(None, name, IMAGE_CURSOR, 0, 0, LR_DEFAULTSIZE | LR_SHARED) };
        SharedCursor(handle.map(|h| h.0 as isize).unwrap_or(0))
    });
    Some(HCURSOR(cursor.0 as _))
}

/// `FOLO_UI_OFFSCREEN=1`: windows open outside every display, are never activated and take no focus.
pub fn offscreen() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("FOLO_UI_OFFSCREEN").is_some_and(|v| v == "1"))
}

/// `FOLO_UI_TRACE=1`: print `first-present <ms since process creation>` to stderr once.
pub(crate) fn trace() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("FOLO_UI_TRACE").is_some_and(|v| v == "1"))
}

/// Milliseconds since this process was created (`GetProcessTimes`).
pub fn ms_since_process_start() -> f64 {
    use ::windows::Win32::Foundation::FILETIME;
    use ::windows::Win32::System::SystemInformation::GetSystemTimePreciseAsFileTime;
    use ::windows::Win32::System::Threading::{GetCurrentProcess, GetProcessTimes};
    let (mut created, mut exited, mut kernel, mut user) =
        (FILETIME::default(), FILETIME::default(), FILETIME::default(), FILETIME::default());
    // SAFETY: the pseudo handle of the current process and four valid out pointers.
    let ok = unsafe { GetProcessTimes(GetCurrentProcess(), &mut created, &mut exited, &mut kernel, &mut user) };
    if ok.is_err() {
        return f64::NAN;
    }
    // SAFETY: no arguments besides the return value.
    let now = unsafe { GetSystemTimePreciseAsFileTime() };
    let ticks = |t: FILETIME| (t.dwHighDateTime as u64) << 32 | t.dwLowDateTime as u64;
    ticks(now).saturating_sub(ticks(created)) as f64 / 10_000.0
}

/// Process CPU time (kernel + user) so far.
pub fn process_cpu_time() -> std::time::Duration {
    use ::windows::Win32::Foundation::FILETIME;
    use ::windows::Win32::System::Threading::{GetCurrentProcess, GetProcessTimes};
    let (mut created, mut exited, mut kernel, mut user) =
        (FILETIME::default(), FILETIME::default(), FILETIME::default(), FILETIME::default());
    // SAFETY: as in `ms_since_process_start`.
    let _ = unsafe { GetProcessTimes(GetCurrentProcess(), &mut created, &mut exited, &mut kernel, &mut user) };
    let ticks = |t: FILETIME| (t.dwHighDateTime as u64) << 32 | t.dwLowDateTime as u64;
    std::time::Duration::from_nanos((ticks(kernel) + ticks(user)) * 100)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn words_are_split_and_signed() {
        let l = LPARAM(((-5i16 as u16 as isize) << 16) | (-3i16 as u16 as isize));
        assert_eq!((l.signed_loword(), l.signed_hiword()), (-3, -5));
        let w = WPARAM(0x0078_0001);
        assert_eq!((w.loword(), w.hiword()), (1, 0x78));
    }

    #[test]
    fn logical_point_divides_by_the_scale() {
        let p = logical_point(150., 75., 1.5);
        assert_eq!((p.x, p.y), (px(100.), px(50.)));
    }

    #[test]
    fn cursors_load_and_none_hides() {
        assert!(load_cursor(CursorStyle::None).is_none());
        let arrow = load_cursor(CursorStyle::Arrow).unwrap();
        assert!(!arrow.is_invalid());
        assert_eq!(load_cursor(CursorStyle::ResizeColumn).unwrap(), load_cursor(CursorStyle::ResizeLeft).unwrap());
    }
}
