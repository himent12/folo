//! `WindowsDisplay`: a monitor, its bounds in its own logical pixels, and GPUI's display UUID (a v5 UUID
//! of the device name, every one of its 32 UTF-16 units as big-endian bytes) so `desktop.json` keeps
//! matching across the port.

use super::util::logical_point;
use crate::{Bounds, DevicePixels, DisplayId, Pixels, PlatformDisplay, point, size};
use anyhow::{Result, anyhow};
use smallvec::SmallVec;
use std::rc::Rc;
use uuid::Uuid;
use ::windows::Win32::Foundation::{LPARAM, POINT, RECT};
use ::windows::Win32::Graphics::Gdi::*;
use ::windows::Win32::UI::HiDpi::{GetDpiForMonitor, MDT_EFFECTIVE_DPI};
use ::windows::Win32::UI::WindowsAndMessaging::USER_DEFAULT_SCREEN_DPI;
use ::windows::core::BOOL;

/// A connected monitor.
#[derive(Debug, Clone, Copy)]
pub struct WindowsDisplay {
    pub(crate) handle: HMONITOR,
    display_id: DisplayId,
    pub(crate) scale_factor: f32,
    bounds: Bounds<Pixels>,
    physical_bounds: Bounds<DevicePixels>,
    uuid: Uuid,
}

// SAFETY: an HMONITOR is a plain identifier, valid on any thread.
unsafe impl Send for WindowsDisplay {}
// SAFETY: as above; the struct is immutable.
unsafe impl Sync for WindowsDisplay {}

impl WindowsDisplay {
    /// The display with this id (its index among the connected monitors).
    pub(crate) fn new(display_id: DisplayId) -> Option<Self> {
        let handle = *available_monitors().get(display_id.0 as usize)?;
        Self::with_handle_and_id(handle, display_id).ok()
    }

    /// The display a monitor handle names.
    pub(crate) fn with_handle(handle: HMONITOR) -> Result<Self> {
        let id = available_monitors()
            .iter()
            .position(|m| m.0 == handle.0)
            .ok_or_else(|| anyhow!("monitor {handle:?} is not connected"))?;
        Self::with_handle_and_id(handle, DisplayId(id as u32))
    }

    fn with_handle_and_id(handle: HMONITOR, display_id: DisplayId) -> Result<Self> {
        let info = monitor_info(handle)?;
        let r = info.monitorInfo.rcMonitor;
        let scale_factor = scale_factor_for_monitor(handle)?;
        let physical_size = size(DevicePixels(r.right - r.left), DevicePixels(r.bottom - r.top));
        Ok(Self {
            handle,
            display_id,
            scale_factor,
            bounds: Bounds {
                origin: logical_point(r.left as f32, r.top as f32, scale_factor),
                size: physical_size.to_pixels(scale_factor),
            },
            physical_bounds: Bounds { origin: point(DevicePixels(r.left), DevicePixels(r.top)), size: physical_size },
            uuid: generate_uuid(&info.szDevice),
        })
    }

    /// The primary monitor (the one holding the screen origin).
    pub(crate) fn primary() -> Option<Self> {
        // SAFETY: a plain query.
        let monitor = unsafe { MonitorFromPoint(POINT { x: 0, y: 0 }, MONITOR_DEFAULTTOPRIMARY) };
        if monitor.is_invalid() { None } else { Self::with_handle(monitor).ok() }
    }

    /// Every connected display.
    pub(crate) fn all() -> Vec<Rc<dyn PlatformDisplay>> {
        available_monitors()
            .into_iter()
            .enumerate()
            .filter_map(|(i, h)| Self::with_handle_and_id(h, DisplayId(i as u32)).ok())
            .map(|d| Rc::new(d) as Rc<dyn PlatformDisplay>)
            .collect()
    }

    /// Whether the centre of `bounds` (logical, at this display's scale) is on this display (GPUI's
    /// `check_given_bounds`).
    pub(crate) fn contains_center_of(&self, bounds: Bounds<Pixels>) -> bool {
        let c = bounds.center();
        let center = POINT { x: (c.x.0 * self.scale_factor) as i32, y: (c.y.0 * self.scale_factor) as i32 };
        // SAFETY: a plain query.
        let monitor = unsafe { MonitorFromPoint(center, MONITOR_DEFAULTTONULL) };
        !monitor.is_invalid() && Self::with_handle(monitor).is_ok_and(|d| d.uuid == self.uuid)
    }

    /// Whether the monitor is still connected.
    pub(crate) fn is_connected(handle: HMONITOR) -> bool {
        available_monitors().contains(&handle)
    }

    /// The monitor's rectangle in physical screen pixels.
    pub(crate) fn physical_bounds(&self) -> Bounds<DevicePixels> {
        self.physical_bounds
    }
}

impl PlatformDisplay for WindowsDisplay {
    fn id(&self) -> DisplayId {
        self.display_id
    }

    fn uuid(&self) -> Result<Uuid> {
        Ok(self.uuid)
    }

    fn bounds(&self) -> Bounds<Pixels> {
        self.bounds
    }
}

fn available_monitors() -> SmallVec<[HMONITOR; 4]> {
    let mut monitors: SmallVec<[HMONITOR; 4]> = SmallVec::new();
    // SAFETY: the callback only pushes into `monitors`, which outlives the call.
    let _ = unsafe {
        EnumDisplayMonitors(None, None, Some(monitor_enum_proc), LPARAM(&mut monitors as *mut _ as isize))
    };
    monitors
}

unsafe extern "system" fn monitor_enum_proc(monitor: HMONITOR, _: HDC, _: *mut RECT, data: LPARAM) -> BOOL {
    let monitors = data.0 as *mut SmallVec<[HMONITOR; 4]>;
    // SAFETY: `data` is the vector `available_monitors` passed, alive for the enumeration.
    unsafe { (*monitors).push(monitor) };
    BOOL(1)
}

fn monitor_info(monitor: HMONITOR) -> Result<MONITORINFOEXW> {
    let mut info = MONITORINFOEXW::default();
    info.monitorInfo.cbSize = std::mem::size_of::<MONITORINFOEXW>() as u32;
    // SAFETY: `info` is a sized MONITORINFOEXW, which GetMonitorInfoW accepts through the base pointer.
    let ok = unsafe { GetMonitorInfoW(monitor, &mut info as *mut MONITORINFOEXW as *mut MONITORINFO) };
    if ok.as_bool() { Ok(info) } else { Err(anyhow!(std::io::Error::last_os_error())) }
}

/// GPUI's display UUID: v5 (DNS namespace) of the whole `szDevice` array, each unit big-endian.
pub(crate) fn generate_uuid(device_name: &[u16]) -> Uuid {
    let name: Vec<u8> = device_name.iter().flat_map(|u| u.to_be_bytes()).collect();
    Uuid::new_v5(&Uuid::NAMESPACE_DNS, &name)
}

/// The monitor's effective DPI as a scale factor.
pub(crate) fn scale_factor_for_monitor(monitor: HMONITOR) -> Result<f32> {
    let (mut x, mut y) = (0, 0);
    // SAFETY: two valid out pointers.
    unsafe { GetDpiForMonitor(monitor, MDT_EFFECTIVE_DPI, &mut x, &mut y) }?;
    Ok(x as f32 / USER_DEFAULT_SCREEN_DPI as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uuid_is_gpuis_formula() {
        // "\\.\DISPLAY1" padded with zeros to 32 units, as szDevice is.
        let mut name = [0u16; 32];
        for (i, u) in r"\\.\DISPLAY1".encode_utf16().enumerate() {
            name[i] = u;
        }
        let bytes: Vec<u8> = name.iter().flat_map(|u| u.to_be_bytes()).collect();
        assert_eq!(bytes.len(), 64);
        assert_eq!(generate_uuid(&name), Uuid::new_v5(&Uuid::NAMESPACE_DNS, &bytes));
        assert_eq!(generate_uuid(&name).get_version_num(), 5);
    }

    #[test]
    fn displays_are_listed_with_their_ids() {
        let all = WindowsDisplay::all();
        for (i, d) in all.iter().enumerate() {
            assert_eq!(u32::from(d.id()), i as u32);
            assert!(d.bounds().size.width.0 > 0.);
        }
        if let Some(primary) = WindowsDisplay::primary() {
            assert!(all.iter().any(|d| d.uuid().unwrap() == primary.uuid));
        }
    }
}
