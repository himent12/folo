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
use std::path::{Component, Path, PathBuf};

const VERBATIM_PREFIX: [u16; 4] = [b'\\' as u16, b'\\' as u16, b'?' as u16, b'\\' as u16];

fn is_verbatim_prefix(path: &Path) -> bool {
    let wide: Vec<u16> = path.as_os_str().encode_wide().collect();
    wide.starts_with(&VERBATIM_PREFIX)
}

/// Whether a single UTF-16 path segment names a reserved DOS device (`CON`, `PRN`, `AUX`, `NUL`,
/// `COM1`..=`COM9`, `LPT1`..=`LPT9`), either bare or before the first `.` (e.g. `NUL.txt`).
fn is_dos_device_name(seg: &[u16]) -> bool {
    let stem = seg.split(|&u| u == b'.' as u16).next().unwrap_or(seg);
    let stem = match stem.len() {
        3 | 4 => stem,
        _ => return false,
    };
    let mut buf = [0u8; 4];
    for (dst, &u) in buf.iter_mut().zip(stem) {
        if u > 0x7F {
            return false;
        }
        *dst = (u as u8).to_ascii_uppercase();
    }
    match &buf[..stem.len()] {
        b"CON" | b"PRN" | b"AUX" | b"NUL" => true,
        [b'C', b'O', b'M', d] | [b'L', b'P', b'T', d] => (b'1'..=b'9').contains(d),
        _ => false,
    }
}

/// Whether `units` contains any component whose meaning would change under Win32 DOS normalization
/// (forward slashes, `.`/`..`, trailing dots/spaces, or reserved DOS device names).
fn has_dos_unsafe_component(units: &[u16]) -> bool {
    const DOT: u16 = b'.' as u16;
    const SPACE: u16 = b' ' as u16;
    const SLASH: u16 = b'/' as u16;
    const BACKSLASH: u16 = b'\\' as u16;
    if units.contains(&SLASH) {
        return true;
    }
    units.split(|&u| u == BACKSLASH).any(|seg| {
        !seg.is_empty() && (matches!(seg.last().copied(), Some(DOT | SPACE)) || is_dos_device_name(seg))
    })
}

/// Strips Windows' `\\?\` extended-length prefix only when the path is a DOS drive (`\\?\C:\...`)
/// or UNC (`\\?\UNC\server\share...`) path whose meaning survives DOS normalization. Non-drive
/// verbatim paths (such as `\\?\Volume{GUID}\...`) and paths with DOS-unsafe components (trailing
/// dots/spaces, `.`/`..`, `/`) are kept verbatim rather than silently redirected.
fn normalize_shell_path(path: &Path) -> PathBuf {
    let wide: Vec<u16> = path.as_os_str().encode_wide().collect();
    let Some(after_prefix) = wide.strip_prefix(&VERBATIM_PREFIX) else {
        return path.to_path_buf();
    };
    let eq_ascii = |u: u16, upper: u8| u == upper as u16 || u == upper.to_ascii_lowercase() as u16;
    if after_prefix.len() > 4
        && eq_ascii(after_prefix[0], b'U')
        && eq_ascii(after_prefix[1], b'N')
        && eq_ascii(after_prefix[2], b'C')
        && after_prefix[3] == b'\\' as u16
    {
        let rest = &after_prefix[4..];
        if !has_dos_unsafe_component(rest) {
            let mut out = Vec::with_capacity(2 + rest.len());
            out.extend_from_slice(&[b'\\' as u16, b'\\' as u16]);
            out.extend_from_slice(rest);
            return PathBuf::from(OsString::from_wide(&out));
        }
    } else if after_prefix.len() >= 3
        && matches!(after_prefix[0], 0x41..=0x5A | 0x61..=0x7A)
        && after_prefix[1] == b':' as u16
        && after_prefix[2] == b'\\' as u16
        && !has_dos_unsafe_component(&after_prefix[3..])
    {
        return PathBuf::from(OsString::from_wide(after_prefix));
    }
    path.to_path_buf()
}

/// Replaces `/` with `\` losslessly in UTF-16 so shell display-name parsing accepts mixed-slash paths.
fn to_backslashes(path: &Path) -> PathBuf {
    let wide: Vec<u16> =
        path.as_os_str().encode_wide().map(|u| if u == b'/' as u16 { b'\\' as u16 } else { u }).collect();
    PathBuf::from(OsString::from_wide(&wide))
}

/// Lexically collapses `.` and `..` components without touching the filesystem.
fn collapse_dots(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    let mut depth = 0usize;
    for comp in path.components() {
        match comp {
            Component::Prefix(p) => out.push(p.as_os_str()),
            Component::RootDir => out.push(comp.as_os_str()),
            Component::CurDir => {}
            Component::ParentDir => {
                if depth > 0 {
                    out.pop();
                    depth -= 1;
                }
            }
            Component::Normal(part) => {
                out.push(part);
                depth += 1;
            }
        }
    }
    out
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
    if target.as_os_str().is_empty() {
        return Err(anyhow!("cannot reveal an empty path"));
    }
    let normalized = normalize_shell_path(target);
    if is_verbatim_prefix(&normalized) {
        return Err(anyhow!("cannot reveal unsupported verbatim path {}", target.display()));
    }
    let slashed = to_backslashes(&normalized);
    let mut has_prefix = false;
    let mut has_root = false;
    for comp in slashed.components() {
        match comp {
            Component::Prefix(_) => has_prefix = true,
            Component::RootDir => has_root = true,
            _ => break,
        }
    }
    if has_prefix && !has_root {
        return Err(anyhow!("cannot reveal drive-relative path {}", target.display()));
    }
    let resolved = if !has_prefix {
        let base =
            cwd.ok_or_else(|| anyhow!("no working directory for relative path {}", target.display()))?;
        let base = normalize_shell_path(base);
        if is_verbatim_prefix(&base) {
            return Err(anyhow!("cannot reveal relative path against verbatim cwd {}", base.display()));
        }
        to_backslashes(&base).join(&slashed)
    } else {
        slashed
    };
    let collapsed = collapse_dots(&resolved);
    match collapsed.parent() {
        Some(dir) if !dir.as_os_str().is_empty() => {
            Ok(RevealPlan::SelectInParent { dir: dir.to_path_buf(), target: collapsed })
        }
        None => Ok(RevealPlan::OpenDirectly(collapsed)),
        Some(_) => Err(anyhow!("no parent folder for {}", collapsed.display())),
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

/// Opens an existing directory in Explorer without risking executing a file.
fn open_folder(dir: &Path) -> Result<()> {
    if !dir.is_dir() {
        return Err(anyhow!("not a directory: {}", dir.display()));
    }
    // SAFETY: all strings outlive the call.
    let ret = unsafe { ShellExecuteW(None, w!("explore"), &HSTRING::from(dir), None, None, SW_SHOWDEFAULT) };
    if ret.0 as isize <= 32 {
        Err(anyhow!(
            "unable to open folder {} (ShellExecuteW code {}): {}",
            dir.display(),
            ret.0 as isize,
            std::io::Error::last_os_error()
        ))
    } else {
        Ok(())
    }
}

/// Whether `path` has a Windows drive prefix without a root slash (e.g. `C:foo`), whose target
/// depends on per-drive ambient state rather than an explicit root or the process working directory.
fn is_drive_relative(path: &Path) -> bool {
    let mut has_prefix = false;
    let mut has_root = false;
    for comp in path.components() {
        match comp {
            Component::Prefix(_) => has_prefix = true,
            Component::RootDir => has_root = true,
            _ => break,
        }
    }
    has_prefix && !has_root
}

/// Production preparation path shared by `reveal_in_explorer` and tests: rejects empty,
/// unsupported verbatim, and drive-relative paths before resolving relative paths against the
/// process working directory.
fn prepare_reveal(target: &Path) -> Result<RevealPlan> {
    if target.as_os_str().is_empty() {
        return Err(anyhow!("cannot reveal an empty path"));
    }
    let stripped = normalize_shell_path(target);
    if is_verbatim_prefix(&stripped) {
        return Err(anyhow!("cannot reveal unsupported verbatim path {}", target.display()));
    }
    let slashed = to_backslashes(&stripped);
    if is_drive_relative(&slashed) {
        return Err(anyhow!("cannot reveal drive-relative path {}", target.display()));
    }
    let abs = std::path::absolute(&slashed)?;
    plan_reveal_with_cwd(&abs, None)
}

/// Opens Explorer on the file's folder with the file selected (its folder alone if that fails as
/// GPUI's fallback does), or opens the folder directly when `target` is a filesystem root.
pub(crate) fn reveal_in_explorer(target: &Path) -> Result<()> {
    let (dir, target) = match prepare_reveal(target)? {
        RevealPlan::OpenDirectly(path) => return open_folder(&path),
        RevealPlan::SelectInParent { dir, target } => (dir, target),
    };
    if !dir.is_dir() {
        return Err(anyhow!("parent is not a directory: {}", dir.display()));
    }
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
                return open_folder(&dir);
            }
        };
        // `SHOpenFolderAndSelectItems` expects child PIDLs relative to `dir_item`; `ILFindLastID`
        // points to the final item ID within `file_item` (owned by `file_item`, freed with it).
        let child_item = ILFindLastID(file_item as *const ITEMIDLIST);
        let highlight = [child_item as *const ITEMIDLIST];
        let result = SHOpenFolderAndSelectItems(dir_item, Some(&highlight), 0);
        CoTaskMemFree(Some(dir_item as _));
        CoTaskMemFree(Some(file_item as _));
        match result {
            Ok(()) => Ok(()),
            Err(e)
                if e.code().0 == ERROR_FILE_NOT_FOUND.to_hresult().0
                    || e.code().0 == ERROR_FILE_NOT_FOUND.0 as i32 =>
            {
                open_folder(&dir)
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
        // Non-drive verbatim paths and paths with DOS-unsafe components must NOT be stripped.
        assert_eq!(
            normalize_shell_path(Path::new(r"\\?\Volume{1234}\dir\file")),
            PathBuf::from(r"\\?\Volume{1234}\dir\file")
        );
        assert_eq!(normalize_shell_path(Path::new(r"\\?\C:\dir\file.")), PathBuf::from(r"\\?\C:\dir\file."));
        assert_eq!(
            normalize_shell_path(Path::new(r"\\?\C:\dir\trailing ")),
            PathBuf::from(r"\\?\C:\dir\trailing ")
        );
        // Reserved DOS device names (and their extension forms) keep the verbatim prefix.
        assert_eq!(
            normalize_shell_path(Path::new(r"\\?\C:\runs\NUL.txt")),
            PathBuf::from(r"\\?\C:\runs\NUL.txt")
        );
        assert_eq!(
            normalize_shell_path(Path::new(r"\\?\C:\runs\com1.log")),
            PathBuf::from(r"\\?\C:\runs\com1.log")
        );
        assert_eq!(normalize_shell_path(Path::new(r"\\?\C:\runs\CON")), PathBuf::from(r"\\?\C:\runs\CON"));
        // Non-device names sharing a prefix still normalize normally.
        assert_eq!(
            normalize_shell_path(Path::new(r"\\?\C:\runs\null.txt")),
            PathBuf::from(r"C:\runs\null.txt")
        );
    }

    #[test]
    fn plan_reveal_handles_files_roots_relative_slashes_dots_and_unsupported_paths() {
        assert_eq!(
            plan_reveal_with_cwd(Path::new(r"\\?\C:\folo\models\best.folo"), None).unwrap(),
            RevealPlan::SelectInParent {
                dir: PathBuf::from(r"C:\folo\models"),
                target: PathBuf::from(r"C:\folo\models\best.folo"),
            }
        );
        assert_eq!(
            plan_reveal_with_cwd(Path::new("D:/folo/models/m.onnx"), None).unwrap(),
            RevealPlan::SelectInParent {
                dir: PathBuf::from(r"D:\folo\models"),
                target: PathBuf::from(r"D:\folo\models\m.onnx"),
            }
        );
        assert_eq!(
            plan_reveal_with_cwd(Path::new(r"D:\work\sub\..\best.folo"), None).unwrap(),
            RevealPlan::SelectInParent {
                dir: PathBuf::from(r"D:\work"),
                target: PathBuf::from(r"D:\work\best.folo"),
            }
        );
        assert_eq!(
            plan_reveal_with_cwd(Path::new(r"D:\work\.."), None).unwrap(),
            RevealPlan::OpenDirectly(PathBuf::from(r"D:\"))
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
            plan_reveal_with_cwd(Path::new("runs/exp1/best.folo"), Some(Path::new(r"D:\work"))).unwrap(),
            RevealPlan::SelectInParent {
                dir: PathBuf::from(r"D:\work\runs\exp1"),
                target: PathBuf::from(r"D:\work\runs\exp1\best.folo"),
            }
        );
        assert_eq!(
            plan_reveal_with_cwd(Path::new(r"\models\best.folo"), Some(Path::new(r"D:\work"))).unwrap(),
            RevealPlan::SelectInParent {
                dir: PathBuf::from(r"D:\models"),
                target: PathBuf::from(r"D:\models\best.folo"),
            }
        );
        // Non-drive verbatim, trailing-dot verbatim, drive-relative, missing cwd, and empty paths fail cleanly.
        assert!(
            plan_reveal_with_cwd(Path::new(r"\\?\Volume{1234}\dir\file"), Some(Path::new(r"D:\work")))
                .is_err()
        );
        assert!(plan_reveal_with_cwd(Path::new(r"\\?\C:\dir\file."), Some(Path::new(r"D:\work"))).is_err());
        assert!(plan_reveal_with_cwd(Path::new(r"C:best.folo"), Some(Path::new(r"D:\work"))).is_err());
        assert!(plan_reveal_with_cwd(Path::new("best.folo"), None).is_err());
        assert!(plan_reveal_with_cwd(Path::new(""), Some(Path::new(r"D:\work"))).is_err());
        // Production `prepare_reveal` rejects drive-relative and DOS-device verbatim paths too.
        assert!(prepare_reveal(Path::new(r"C:best.folo")).is_err());
        assert!(prepare_reveal(Path::new(r"\\?\C:\runs\NUL.txt")).is_err());
        assert!(prepare_reveal(Path::new(r"\\?\Volume{1234}\dir\file")).is_err());
    }
}
