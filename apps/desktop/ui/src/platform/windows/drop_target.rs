// Adapted from GPUI (Apache-2.0): crates/gpui/src/platform/windows/window.rs (WindowsDragDropHandler)
//! File drops through OLE drag and drop: `FileDropEvent::{Entered, Pending, Submit, Exited}` with the
//! paths (`CF_HDROP`) and the position in window logical pixels, and the shell's drag image helper.

use super::util::logical_point;
use super::window::WindowInner;
use crate::{ExternalPaths, FileDropEvent, Pixels, PlatformInput, Point};
use ::windows::Win32::Foundation::{POINT, POINTL, S_OK};
use ::windows::Win32::Graphics::Gdi::ScreenToClient;
use ::windows::Win32::System::Com::{DVASPECT_CONTENT, FORMATETC, IDataObject, TYMED_HGLOBAL};
use ::windows::Win32::System::Ole::*;
use ::windows::Win32::System::SystemServices::MODIFIERKEYS_FLAGS;
use ::windows::Win32::UI::Shell::{DragQueryFileW, HDROP};
use ::windows::core::{Ref, implement};
use smallvec::SmallVec;
use std::path::PathBuf;
use std::rc::{Rc, Weak};

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

/// Answers through OLE's `pdwEffect` out pointer (never null from OLE; a null one is ignored).
fn set_effect(effect: *mut DROPEFFECT, value: DROPEFFECT) {
    if !effect.is_null() {
        // SAFETY: OLE passes a valid effect pointer for the duration of the call; not null (checked).
        unsafe { *effect = value };
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
        set_effect(effect, accepted);
        if let Some(paths) = paths {
            self.send(PlatformInput::FileDrop(FileDropEvent::Entered {
                position: self.position(pt),
                paths: ExternalPaths(paths),
            }));
        }
        if let (Some(helper), Some(window)) = (self.helper(), self.0.upgrade()) {
            // SAFETY: the helper draws the drag image over our window.
            let _ = unsafe { helper.DragEnter(window.hwnd, data, &POINT { x: pt.x, y: pt.y }, accepted) };
        }
        Ok(())
    }

    fn DragOver(
        &self,
        _keys: MODIFIERKEYS_FLAGS,
        pt: &POINTL,
        effect: *mut DROPEFFECT,
    ) -> ::windows::core::Result<()> {
        set_effect(effect, DROPEFFECT_COPY);
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
        set_effect(effect, DROPEFFECT_COPY);
        if let Some(helper) = self.helper() {
            // SAFETY: as in DragEnter.
            let _ = unsafe { helper.Drop(data, &POINT { x: pt.x, y: pt.y }, DROPEFFECT_COPY) };
        }
        self.send(PlatformInput::FileDrop(FileDropEvent::Submit { position: self.position(pt) }));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ::windows::Win32::Foundation::{HGLOBAL, TRUE};
    use ::windows::Win32::System::Com::{
        COINIT_APARTMENTTHREADED, CoInitializeEx, CoUninitialize, STGMEDIUM, STGMEDIUM_0,
    };
    use ::windows::Win32::System::Memory::{GMEM_MOVEABLE, GlobalAlloc, GlobalLock, GlobalUnlock};
    use ::windows::Win32::UI::Shell::{DROPFILES, SHCreateDataObject};
    use std::mem::ManuallyDrop;

    /// A shell data object holding `paths` as `CF_HDROP` (what Explorer hands a drop target).
    fn data_object(paths: &[&str]) -> IDataObject {
        let mut wide: Vec<u16> = Vec::new();
        for p in paths {
            wide.extend(p.encode_utf16());
            wide.push(0);
        }
        wide.push(0);
        let header = std::mem::size_of::<DROPFILES>();
        // SAFETY: a movable block big enough for the header and the strings, filled while locked; the
        // data object takes it over (fRelease).
        unsafe {
            let block = GlobalAlloc(GMEM_MOVEABLE, header + wide.len() * 2).unwrap();
            let at = GlobalLock(block) as *mut u8;
            at.cast::<DROPFILES>().write(DROPFILES {
                pFiles: header as u32,
                fWide: TRUE,
                ..Default::default()
            });
            std::ptr::copy_nonoverlapping(wide.as_ptr(), at.add(header).cast::<u16>(), wide.len());
            let _ = GlobalUnlock(block);
            let data: IDataObject = SHCreateDataObject(None, None, None).unwrap();
            let format = FORMATETC {
                cfFormat: CF_HDROP.0,
                ptd: std::ptr::null_mut(),
                dwAspect: DVASPECT_CONTENT.0,
                lindex: -1,
                tymed: TYMED_HGLOBAL.0 as u32,
            };
            let medium = STGMEDIUM {
                tymed: TYMED_HGLOBAL.0 as u32,
                u: STGMEDIUM_0 { hGlobal: HGLOBAL(block.0) },
                pUnkForRelease: ManuallyDrop::new(None),
            };
            data.SetData(&format, &medium, true).unwrap();
            data
        }
    }

    #[test]
    fn a_drop_of_files_gives_their_paths() {
        // SAFETY: COM for this test thread, released at the end.
        unsafe { CoInitializeEx(None, COINIT_APARTMENTTHREADED).ok().unwrap() };
        let paths = [r"C:\images\a.png", r"D:\data\ünïcødé bild.jpg"];
        let got = dropped_paths(&data_object(&paths)).expect("CF_HDROP is there");
        assert_eq!(got.iter().map(|p| p.to_string_lossy().into_owned()).collect::<Vec<_>>(), paths);
        // SAFETY: an empty shell data object.
        let empty: IDataObject = unsafe { SHCreateDataObject(None, None, None) }.unwrap();
        assert!(dropped_paths(&empty).is_none(), "no CF_HDROP, no paths");
        // SAFETY: pairs the CoInitializeEx above.
        unsafe { CoUninitialize() };
    }
}
