//! The system services the winit backend offers: clipboard text (arboard), file dialogs (rfd: the xdg
//! portal on Linux, no GTK), opening URLs and revealing files (`open` / `xdg-open`).

use crate::{ClipboardItem, PathPromptOptions, oneshot};
use anyhow::Result;
use std::path::{Path, PathBuf};
use std::process::Command;

/// The clipboard, opened on first use and kept (on X11 the owner must stay alive to serve pastes). On
/// Linux arboard uses Wayland's data-control protocol when the compositor offers it, else X11.
#[derive(Default)]
pub(crate) struct Clipboard {
    clipboard: Option<arboard::Clipboard>,
    /// Whether a failure to open it was reported (once; later uses retry quietly).
    reported: bool,
}

impl Clipboard {
    fn get(&mut self) -> Option<&mut arboard::Clipboard> {
        if self.clipboard.is_none() {
            match arboard::Clipboard::new() {
                Ok(clipboard) => self.clipboard = Some(clipboard),
                Err(error) => {
                    if !std::mem::replace(&mut self.reported, true) {
                        super::log_error(&format!("no clipboard: {error}"));
                    }
                }
            }
        }
        self.clipboard.as_mut()
    }

    pub fn write(&mut self, item: ClipboardItem) {
        if let (Some(clipboard), Some(text)) = (self.get(), item.text()) {
            let _ = clipboard.set_text(text);
        }
    }

    pub fn read(&mut self) -> Option<ClipboardItem> {
        self.get()?.get_text().ok().map(ClipboardItem::new_string)
    }
}

/// Shows the open dialog `options` describe; the receiver gets the paths (`None`: cancelled).
pub(crate) fn prompt_for_paths(
    options: PathPromptOptions,
) -> oneshot::Receiver<Result<Option<Vec<PathBuf>>>> {
    let (tx, rx) = oneshot::channel();
    on_thread(move || {
        let mut dialog = rfd::FileDialog::new();
        if let Some(prompt) = &options.prompt {
            dialog = dialog.set_title(prompt.as_ref());
        }
        let folders = options.directories && !options.files;
        let paths = match (folders, options.multiple) {
            (true, true) => dialog.pick_folders(),
            (true, false) => dialog.pick_folder().map(|p| vec![p]),
            (false, true) => dialog.pick_files(),
            (false, false) => dialog.pick_file().map(|p| vec![p]),
        };
        let _ = tx.send(Ok(paths));
    });
    rx
}

/// Shows the save dialog in `directory`; the receiver gets the path (`None`: cancelled).
pub(crate) fn prompt_for_new_path(
    directory: &Path,
    suggested_name: Option<&str>,
) -> oneshot::Receiver<Result<Option<PathBuf>>> {
    let (tx, rx) = oneshot::channel();
    let directory = directory.to_path_buf();
    let suggested_name = suggested_name.map(str::to_owned);
    on_thread(move || {
        let mut dialog = rfd::FileDialog::new().set_directory(&directory);
        if let Some(name) = &suggested_name {
            dialog = dialog.set_file_name(name);
        }
        let _ = tx.send(Ok(dialog.save_file()));
    });
    rx
}

/// Runs a dialog on its own thread, so the event loop keeps running while it is up. On macOS rfd hands
/// the panel to the main thread itself (GCD's main queue, which the running `NSApplication` serves), so
/// AppKit's main-thread rule holds without the dialog running inside a winit callback.
fn on_thread(job: impl FnOnce() + Send + 'static) {
    let _ = std::thread::Builder::new().name("folo-ui-dialog".into()).spawn(job);
}

/// The command that opens a URL in the default browser.
pub(crate) fn open_url_command(url: &str) -> Command {
    let mut command;
    if cfg!(target_os = "macos") {
        command = Command::new("open");
        command.arg(url);
    } else if cfg!(windows) {
        command = Command::new("rundll32");
        command.args(["url.dll,FileProtocolHandler", url]);
    } else {
        command = Command::new("xdg-open");
        command.arg(url);
    }
    command
}

/// The command that shows `path` in the file manager: selected in Finder / Explorer; on Linux its
/// parent directory opened (there is no portable "select").
pub(crate) fn reveal_path_command(path: &Path) -> Command {
    let mut command;
    if cfg!(target_os = "macos") {
        command = Command::new("open");
        command.arg("-R").arg(path);
    } else if cfg!(windows) {
        command = Command::new("explorer");
        let mut select = std::ffi::OsString::from("/select,");
        select.push(path);
        command.arg(select);
    } else {
        command = Command::new("xdg-open");
        command.arg(if path.is_dir() { path } else { path.parent().unwrap_or(path) });
    }
    command
}

/// Starts `command` and reaps it on a thread, so no zombie is left.
pub(crate) fn spawn_detached(mut command: Command) {
    if let Ok(mut child) = command.spawn() {
        let _ = std::thread::Builder::new().name("folo-ui-open".into()).spawn(move || child.wait());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn commands_name_the_system_tools() {
        let url = open_url_command("https://example.com/a?b=1&c=2");
        let args: Vec<_> = url.get_args().map(|a| a.to_string_lossy().into_owned()).collect();
        assert_eq!(
            args.last().map(String::as_str),
            Some("https://example.com/a?b=1&c=2"),
            "one argument, unparsed"
        );
        let reveal = reveal_path_command(Path::new("/tmp/run/weights.folo"));
        let program = reveal.get_program().to_string_lossy().into_owned();
        assert!(["open", "explorer", "xdg-open"].contains(&program.as_str()));
        if program == "xdg-open" {
            assert_eq!(reveal.get_args().next().unwrap(), "/tmp/run");
        }
    }
}
