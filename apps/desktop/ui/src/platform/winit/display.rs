//! Displays: winit monitors as `PlatformDisplay`s, in logical pixels, with a stable UUID.

use crate::{Bounds, DisplayId, Pixels, PlatformDisplay, point, px, size};
use anyhow::Result;
use std::fmt;
use uuid::Uuid;
use winit::monitor::MonitorHandle;

/// A monitor, as the app sees it.
pub struct WinitDisplay {
    id: DisplayId,
    uuid: Uuid,
    bounds: Bounds<Pixels>,
}

impl WinitDisplay {
    /// The display for a monitor; `index` is its place in the system's list (the `DisplayId` for this
    /// run).
    pub(crate) fn new(index: usize, handle: MonitorHandle) -> Self {
        let scale = handle.scale_factor() as f32;
        let position = handle.position();
        let physical = handle.size();
        let bounds = Bounds::new(
            point(px(position.x as f32 / scale), px(position.y as f32 / scale)),
            size(px(physical.width as f32 / scale), px(physical.height as f32 / scale)),
        );
        let uuid = display_uuid(handle.name().as_deref(), position.x, position.y);
        Self { id: DisplayId::new(index as u32), uuid, bounds }
    }
}

/// A stable id: a v5 UUID of the monitor's name and position (winit gives no hardware id; GPUI's
/// Windows backend hashes the device name the same way).
pub(crate) fn display_uuid(name: Option<&str>, x: i32, y: i32) -> Uuid {
    let key = format!("{}@{x},{y}", name.unwrap_or("display"));
    Uuid::new_v5(&Uuid::NAMESPACE_DNS, key.as_bytes())
}

impl PlatformDisplay for WinitDisplay {
    fn id(&self) -> DisplayId {
        self.id
    }

    fn uuid(&self) -> Result<Uuid> {
        Ok(self.uuid)
    }

    fn bounds(&self) -> Bounds<Pixels> {
        self.bounds
    }
}

impl fmt::Debug for WinitDisplay {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WinitDisplay").field("id", &self.id).field("bounds", &self.bounds).finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uuids_are_stable_and_distinct() {
        assert_eq!(display_uuid(Some("DELL U2720Q"), 0, 0), display_uuid(Some("DELL U2720Q"), 0, 0));
        assert_ne!(display_uuid(Some("DELL U2720Q"), 0, 0), display_uuid(Some("DELL U2720Q"), 3840, 0));
        assert_ne!(display_uuid(Some("A"), 0, 0), display_uuid(None, 0, 0));
    }
}
