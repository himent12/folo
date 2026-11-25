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
use anyhow::{Result, anyhow};
use std::ffi::{OsStr, OsString};
use std::os::windows::ffi::{OsStrExt, OsStringExt};
use std::path::{Path, PathBuf};

/// Strips Windows' `\\?\` extended-length prefix (converting `\\?\UNC\server\share` to
/// `\\server\share`) losslessly in UTF-16 so shell parsing APIs that reject verbatim paths accept it.
fn normalize_shell_path(path: &Path) -> PathBuf {
    const VERBATIM: [u16; 4] = [b'\\' as u16, b'\\' as u16, b'?' as u16, b'\\' as u16];
    let wide: Vec<u16> = path.as_os_str().encode_wide().collect();
    let Some(after_prefix) = wide.strip_prefix(&VERBATIM) else {
        return path.to_path_buf();
    };
    let eq_ascii = |u: u16, upper: u8| u == upper as u16 || u == upper.to_ascii_lowercase() as u16;
    if after_prefix.len() >= 4
        && eq_ascii(after_prefix[0], b'U')
        && eq_ascii(after_prefix[1], b'N')
        && eq_ascii(after_prefix[2], b'C')
        && after_prefix[3] == b'\\' as u16
    {
        let mut out = Vec::with_capacity(2 + after_prefix.len() - 4);
        out.extend_from_slice(&[b'\\' as u16, b'\\' as u16]);
        out.extend_from_slice(&after_prefix[4..]);
        PathBuf::from(OsString::from_wide(&out))
    } else {
        PathBuf::from(OsString::from_wide(after_prefix))
    }
}

/// How Explorer should reveal a target path.
#[derive(Debug, PartialEq, Eq)]
enum RevealPlan {
    /// Open `dir` with `target` selected (falling back to opening `dir` if `target` cannot be parsed).
    SelectInParent { dir: PathBuf, target: PathBuf },
    /// The path is a filesystem root (such as `C:\` or `\\server\share`): open it directly.
    OpenDirectly(PathBuf),
}

/// Plans how `reveal_in_explorer` should open `target`, resolving relative paths against `cwd`.
fn plan_reveal_with_cwd(target: &Path, cwd: Option<&Path>) -> Result<RevealPlan> {
    let normalized = normalize_shell_path(target);
    if normalized.as_os_str().is_empty() {
        return Err(anyhow!("cannot reveal an empty path"));
    }
    let resolved = if normalized.is_relative() {
        match cwd {
            Some(base) => normalize_shell_path(&base.join(&normalized)),
            None => normalized,
        }
    } else {
        normalized
    };
    match resolved.parent() {
        Some(dir) if !dir.as_os_str().is_empty() => {
            Ok(RevealPlan::SelectInParent { dir: dir.to_path_buf(), target: resolved })
        }
        None => Ok(RevealPlan::OpenDirectly(resolved)),
        Some(_) => Err(anyhow!("no parent folder for {}", resolved.display())),
    }
}

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
/// GPUI's fallback does), or opens the folder directly when `target` is a filesystem root.
pub(crate) fn reveal_in_explorer(target: &Path) -> Result<()> {
    let cwd = std::env::current_dir().ok();
    let (dir, target) = match plan_reveal_with_cwd(target, cwd.as_deref())? {
        RevealPlan::OpenDirectly(path) => return open_target(path),
        RevealPlan::SelectInParent { dir, target } => (dir, target),
    };
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
        let dir_item = parse(&dir)?;
        let file_item = match parse(&target) {
            Ok(item) => item,
            Err(_) => {
                CoTaskMemFree(Some(dir_item as _));
                return open_target(&dir);
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
                open_target(&dir)
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
            let full = normalize_shell_path(&full);
            if let Ok(folder) =
                SHCreateItemFromParsingName::<_, _, IShellItem>(&HSTRING::from(full.as_path()), None)
            {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_shell_path_strips_verbatim_and_unc_prefixes() {
        assert_eq!(
            normalize_shell_path(Path::new(r"\\?\D:\folo\models\best.folo")),
            PathBuf::from(r"D:\folo\models\best.folo")
        );
        assert_eq!(
            normalize_shell_path(Path::new(r"\\?\UNC\server\share\runs\best.folo")),
            PathBuf::from(r"\\server\share\runs\best.folo")
        );
        assert_eq!(
            normalize_shell_path(Path::new(r"\\?\unc\server\share\dir")),
            PathBuf::from(r"\\server\share\dir")
        );
        assert_eq!(
            normalize_shell_path(Path::new(r"C:\plain\path.txt")),
            PathBuf::from(r"C:\plain\path.txt")
        );
    }

    #[test]
    fn plan_reveal_handles_files_roots_relative_and_empty_paths() {
        assert_eq!(
            plan_reveal_with_cwd(Path::new(r"\\?\C:\folo\models\best.folo"), None).unwrap(),
            RevealPlan::SelectInParent {
                dir: PathBuf::from(r"C:\folo\models"),
                target: PathBuf::from(r"C:\folo\models\best.folo"),
            }
        );
        assert_eq!(
            plan_reveal_with_cwd(Path::new(r"\\?\C:\"), None).unwrap(),
            RevealPlan::OpenDirectly(PathBuf::from(r"C:\"))
        );
        assert_eq!(
            plan_reveal_with_cwd(Path::new(r"\\?\UNC\server\share"), None).unwrap(),
            RevealPlan::OpenDirectly(PathBuf::from(r"\\server\share"))
        );
        assert_eq!(
            plan_reveal_with_cwd(Path::new("best.folo"), Some(Path::new(r"D:\work"))).unwrap(),
            RevealPlan::SelectInParent {
                dir: PathBuf::from(r"D:\work"),
                target: PathBuf::from(r"D:\work\best.folo"),
            }
        );
        assert!(plan_reveal_with_cwd(Path::new("best.folo"), None).is_err());
        assert!(plan_reveal_with_cwd(Path::new(""), Some(Path::new(r"D:\work"))).is_err());
    }
}
