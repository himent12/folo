// Adapted from GPUI (Apache-2.0): crates/gpui/src/platform/windows/platform.rs (open_target, file dialogs, reveal)
//! The shell: `open_url` (`ShellExecuteW`), `reveal_path` (Explorer with the file selected) and the
//! file dialogs (`IFileOpenDialog` / `IFileSaveDialog`), as GPUI does them.

use crate::PathPromptOptions;
use ::windows::Win32::Foundation::{ERROR_FILE_NOT_FOUND, HWND};
use ::windows::Win32::System::Com::{CLSCTX_ALL, CoCreateInstance, CoTaskMemFree};
use ::windows::Win32::UI::Shell::Common::{COMDLG_FILTERSPEC, ITEMIDLIST};
use ::windows::Win32::UI::Shell::*;
use ::windows::Win32::UI::WindowsAndMessaging::SW_SHOWDEFAULT;
use ::windows::core::{HSTRING, w};
use anyhow::{Context as _, Result, anyhow};
use std::ffi::OsStr;
use std::path::{Path, PathBuf};

/// Opens a URL or file with its default handler.
pub(crate) fn open_target(target: impl AsRef<OsStr>) -> Result<()> {
    // SAFETY: all strings outlive the call.
    let ret = unsafe {
        ShellExecuteW(None, w!("open"), &HSTRING::from(target.as_ref()), None, None, SW_SHOWDEFAULT)
    };
    if ret.0 as isize <= 32 {
        Err(anyhow!("unable to open target: {}", std::io::Error::last_os_error()))
    } else {
        Ok(())
    }
}

/// Opens Explorer on the file's folder with the file selected (its folder alone if that fails as
/// GPUI's fallback does).
pub(crate) fn reveal_in_explorer(target: &Path) -> Result<()> {
    let dir = target.parent().context("no parent folder")?;
    // SAFETY: COM calls on the shell's desktop folder; every PIDL it returns is freed below.
    unsafe {
        let desktop = SHGetDesktopFolder()?;
        let parse = |path: &Path| -> Result<*mut ITEMIDLIST> {
            let mut item = std::ptr::null_mut();
            desktop.ParseDisplayName(
                HWND::default(),
                None,
                &HSTRING::from(path),
                None,
                &mut item,
                std::ptr::null_mut(),
            )?;
            Ok(item)
        };
        let dir_item = parse(dir)?;
        let file_item = match parse(target) {
            Ok(item) => item,
            Err(e) => {
                CoTaskMemFree(Some(dir_item as _));
                return Err(e);
            }
        };
        let highlight = [file_item as *const ITEMIDLIST];
        let result = SHOpenFolderAndSelectItems(dir_item, Some(&highlight), 0);
        CoTaskMemFree(Some(dir_item as _));
        CoTaskMemFree(Some(file_item as _));
        match result {
            Ok(()) => Ok(()),
            Err(e)
                if e.code().0 == ERROR_FILE_NOT_FOUND.to_hresult().0
                    || e.code().0 == ERROR_FILE_NOT_FOUND.0 as i32 =>
            {
                open_target(dir)
            }
            Err(e) => Err(anyhow!("cannot reveal {}: {e}", target.display())),
        }
    }
}

/// The open dialog, modal over `owner`. `Ok(None)`: cancelled.
pub(crate) fn file_open_dialog(
    options: &PathPromptOptions,
    owner: Option<HWND>,
) -> Result<Option<Vec<PathBuf>>> {
    // SAFETY: COM calls on a dialog object this function owns; strings outlive the calls.
    unsafe {
        let dialog: IFileOpenDialog = CoCreateInstance(&FileOpenDialog, None, CLSCTX_ALL)?;
        let mut flags = FOS_FILEMUSTEXIST;
        if options.multiple {
            flags |= FOS_ALLOWMULTISELECT;
        }
        if options.directories {
            flags |= FOS_PICKFOLDERS;
        }
        dialog.SetOptions(flags)?;
        if let Some(prompt) = &options.prompt {
            let prompt: &str = prompt;
            dialog.SetOkButtonLabel(&HSTRING::from(prompt))?;
        }
        if dialog.Show(owner).is_err() {
            return Ok(None);
        }
        let results = dialog.GetResults()?;
        let count = results.GetCount()?;
        if count == 0 {
            return Ok(None);
        }
        let mut paths = Vec::with_capacity(count as usize);
        for i in 0..count {
            let item = results.GetItemAt(i)?;
            let name = item.GetDisplayName(SIGDN_FILESYSPATH)?;
            let path = name.to_string();
            CoTaskMemFree(Some(name.0 as _));
            paths.push(PathBuf::from(path?));
        }
        Ok(Some(paths))
    }
}

/// The save dialog in `directory` with `suggested_name`, modal over `owner`. `Ok(None)`: cancelled.
pub(crate) fn file_save_dialog(
    directory: &Path,
    suggested_name: Option<&str>,
    owner: Option<HWND>,
) -> Result<Option<PathBuf>> {
    // SAFETY: as in `file_open_dialog`.
    unsafe {
        let dialog: IFileSaveDialog = CoCreateInstance(&FileSaveDialog, None, CLSCTX_ALL)?;
        if !directory.as_os_str().is_empty()
            && let Ok(full) = directory.canonicalize()
        {
            // `canonicalize` gives a `\\?\` path, which the shell does not parse.
            let full = full.to_string_lossy().trim_start_matches(r"\\?\").to_string();
            if let Ok(folder) = SHCreateItemFromParsingName::<_, _, IShellItem>(&HSTRING::from(full), None) {
                let _ = dialog.SetFolder(&folder);
            }
        }
        if let Some(name) = suggested_name {
            let _ = dialog.SetFileName(&HSTRING::from(name));
        }
        dialog.SetFileTypes(&[COMDLG_FILTERSPEC { pszName: w!("All files"), pszSpec: w!("*.*") }])?;
        if dialog.Show(owner).is_err() {
            return Ok(None);
        }
        let item = dialog.GetResult()?;
        let name = item.GetDisplayName(SIGDN_FILESYSPATH)?;
        let path = name.to_string();
        CoTaskMemFree(Some(name.0 as _));
        Ok(Some(PathBuf::from(path?)))
    }
}
