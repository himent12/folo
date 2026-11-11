//! File drops through OLE drag and drop: `FileDropEvent::{Entered, Pending, Submit, Exited}` with the
//! paths (`CF_HDROP`) and the position in window logical pixels, and the shell's drag image helper.

use super::util::logical_point;
use super::window::WindowInner;
use crate::{ExternalPaths, FileDropEvent, PlatformInput, Pixels, Point};
use smallvec::SmallVec;
use std::path::PathBuf;
use std::rc::{Rc, Weak};
use ::windows::Win32::Foundation::{POINT, POINTL, S_OK};
use ::windows::Win32::Graphics::Gdi::ScreenToClient;
use ::windows::Win32::System::Com::{DVASPECT_CONTENT, FORMATETC, IDataObject, TYMED_HGLOBAL};
use ::windows::Win32::System::Ole::*;
use ::windows::Win32::System::SystemServices::MODIFIERKEYS_FLAGS;
use ::windows::Win32::UI::Shell::{DragQueryFileW, HDROP};
use ::windows::core::{Ref, implement};

/// Registers the window as a drop target (OLE is initialized by the platform).
pub(crate) fn register(window: &Rc<WindowInner>) {
    let target: IDropTarget = DropTarget(Rc::downgrade(window)).into();
    // SAFETY: our window, on the thread that created it; OLE keeps its own reference.
    let _ = unsafe { RegisterDragDrop(window.hwnd, &target) };
}

/// Unregisters it (before the window is destroyed).
pub(crate) fn revoke(window: &WindowInner) {
    // SAFETY: as above; revoking twice only fails.
    let _ = unsafe { RevokeDragDrop(window.hwnd) };
}

#[implement(IDropTarget)]
struct DropTarget(Weak<WindowInner>);

impl DropTarget {
    fn send(&self, input: PlatformInput) {
        if let Some(window) = self.0.upgrade() {
            window.dispatch_input(input);
        }
    }

    fn position(&self, pt: &POINTL) -> Point<Pixels> {
        let Some(window) = self.0.upgrade() else { return Point::default() };
        let mut p = POINT { x: pt.x, y: pt.y };
        // SAFETY: a valid in/out pointer.
        let _ = unsafe { ScreenToClient(window.hwnd, &mut p) };
        logical_point(p.x as f32, p.y as f32, window.state.borrow().scale_factor)
    }

    fn helper(&self) -> Option<::windows::Win32::UI::Shell::IDropTargetHelper> {
        self.0.upgrade().and_then(|w| w.shared.drop_helper())
    }
}

/// The paths in a data object's `CF_HDROP`, if it has one.
fn dropped_paths(data: &IDataObject) -> Option<SmallVec<[PathBuf; 2]>> {
    let format = FORMATETC {
        cfFormat: CF_HDROP.0,
        ptd: std::ptr::null_mut(),
        dwAspect: DVASPECT_CONTENT.0,
        lindex: -1,
        tymed: TYMED_HGLOBAL.0 as u32,
    };
    // SAFETY: COM calls on the data object the shell gave us; the medium is released below.
    unsafe {
        if data.QueryGetData(&format) != S_OK {
            return None;
        }
        let mut medium = data.GetData(&format).ok()?;
        let mut paths = SmallVec::new();
        if !medium.u.hGlobal.is_invalid() {
            let hdrop = HDROP(medium.u.hGlobal.0);
            let count = DragQueryFileW(hdrop, u32::MAX, None);
            for i in 0..count {
                let len = DragQueryFileW(hdrop, i, None) as usize;
                let mut buffer = vec![0u16; len + 1];
                if DragQueryFileW(hdrop, i, Some(&mut buffer)) > 0 {
                    paths.push(PathBuf::from(String::from_utf16_lossy(&buffer[..len])));
                }
            }
        }
        ReleaseStgMedium(&mut medium);
        Some(paths)
    }
}

#[allow(non_snake_case)]
impl IDropTarget_Impl for DropTarget_Impl {
    fn DragEnter(
        &self,
        data: Ref<IDataObject>,
        _keys: MODIFIERKEYS_FLAGS,
        pt: &POINTL,
        effect: *mut DROPEFFECT,
    ) -> ::windows::core::Result<()> {
        let data = data.ok()?;
        let paths = dropped_paths(data);
        let accepted = if paths.is_some() { DROPEFFECT_COPY } else { DROPEFFECT_NONE };
        // SAFETY: OLE passes a valid effect pointer.
        unsafe { *effect = accepted };
        if let Some(paths) = paths {
            self.send(PlatformInput::FileDrop(FileDropEvent::Entered { position: self.position(pt), paths: ExternalPaths(paths) }));
        }
        if let (Some(helper), Some(window)) = (self.helper(), self.0.upgrade()) {
            // SAFETY: the helper draws the drag image over our window.
            let _ = unsafe { helper.DragEnter(window.hwnd, data, &POINT { x: pt.x, y: pt.y }, accepted) };
        }
        Ok(())
    }

    fn DragOver(&self, _keys: MODIFIERKEYS_FLAGS, pt: &POINTL, effect: *mut DROPEFFECT) -> ::windows::core::Result<()> {
        // SAFETY: OLE passes a valid effect pointer.
        unsafe { *effect = DROPEFFECT_COPY };
        if let Some(helper) = self.helper() {
            // SAFETY: as in DragEnter.
            let _ = unsafe { helper.DragOver(&POINT { x: pt.x, y: pt.y }, DROPEFFECT_COPY) };
        }
        self.send(PlatformInput::FileDrop(FileDropEvent::Pending { position: self.position(pt) }));
        Ok(())
    }

    fn DragLeave(&self) -> ::windows::core::Result<()> {
        if let Some(helper) = self.helper() {
            // SAFETY: as in DragEnter.
            let _ = unsafe { helper.DragLeave() };
        }
        self.send(PlatformInput::FileDrop(FileDropEvent::Exited));
        Ok(())
    }

    fn Drop(
        &self,
        data: Ref<IDataObject>,
        _keys: MODIFIERKEYS_FLAGS,
        pt: &POINTL,
        effect: *mut DROPEFFECT,
    ) -> ::windows::core::Result<()> {
        let data = data.ok()?;
        // SAFETY: OLE passes a valid effect pointer.
        unsafe { *effect = DROPEFFECT_COPY };
        if let Some(helper) = self.helper() {
            // SAFETY: as in DragEnter.
            let _ = unsafe { helper.Drop(data, &POINT { x: pt.x, y: pt.y }, DROPEFFECT_COPY) };
        }
        self.send(PlatformInput::FileDrop(FileDropEvent::Submit { position: self.position(pt) }));
        Ok(())
    }
}
