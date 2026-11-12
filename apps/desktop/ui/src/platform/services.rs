//! The services that open windows of their own (file dialogs, the browser, the file manager) behind one
//! off-screen gate, which every OS backend (Win32 and winit) puts in front of its own: with
//! `FOLO_UI_OFFSCREEN=1` none of them is launched. The dialogs answer with an error (`REFUSED`), and
//! `open_url` and `reveal_path` say on stderr what they did not open. A dialog of a window that is never
//! active has no owner, so Windows centred it on the primary display; Explorer, the browser and the
//! portals open windows of other processes, which no backend can keep off the displays.

use crate::{PathPromptOptions, oneshot};
use anyhow::Result;
use std::path::{Path, PathBuf};

/// Why a file dialog, `open_url` or `reveal_path` did nothing off screen.
pub(crate) const REFUSED: &str =
    "FOLO_UI_OFFSCREEN=1: file dialogs, Explorer and the browser stay closed (they would open on a display)";

/// How a backend launches each service (tests put a recorder in its place).
pub(crate) trait Launcher {
    /// Opens `url` (not empty) in the default browser.
    fn open_url(&self, url: &str);
    /// Shows `path` (not empty) in the file manager.
    fn reveal_path(&self, path: &Path);
    /// Shows the open dialog.
    fn prompt_for_paths(&self, options: PathPromptOptions)
    -> oneshot::Receiver<Result<Option<Vec<PathBuf>>>>;
    /// Shows the save dialog.
    fn prompt_for_new_path(
        &self,
        directory: &Path,
        suggested_name: Option<&str>,
    ) -> oneshot::Receiver<Result<Option<PathBuf>>>;
}

/// A backend's services behind the off-screen gate; its `Platform` methods call these.
pub(crate) struct Services {
    /// `FOLO_UI_OFFSCREEN=1` as the backend read it: nothing is launched.
    offscreen: bool,
    launcher: Box<dyn Launcher>,
}

impl Services {
    /// `launcher` behind the gate, closed when `offscreen`.
    pub(crate) fn new(offscreen: bool, launcher: Box<dyn Launcher>) -> Self {
        Services { offscreen, launcher }
    }

    /// Whether the gate is closed.
    #[cfg(test)]
    pub(crate) fn offscreen(&self) -> bool {
        self.offscreen
    }

    pub(crate) fn open_url(&self, url: &str) {
        if url.is_empty() {
            return;
        }
        if self.offscreen {
            eprintln!("folo-ui: not opening {url}: {REFUSED}");
            return;
        }
        self.launcher.open_url(url);
    }

    pub(crate) fn reveal_path(&self, path: &Path) {
        if path.as_os_str().is_empty() {
            return;
        }
        if self.offscreen {
            eprintln!("folo-ui: not showing {} in the file manager: {REFUSED}", path.display());
            return;
        }
        self.launcher.reveal_path(path);
    }

    pub(crate) fn prompt_for_paths(
        &self,
        options: PathPromptOptions,
    ) -> oneshot::Receiver<Result<Option<Vec<PathBuf>>>> {
        if self.offscreen {
            return refused();
        }
        self.launcher.prompt_for_paths(options)
    }

    pub(crate) fn prompt_for_new_path(
        &self,
        directory: &Path,
        suggested_name: Option<&str>,
    ) -> oneshot::Receiver<Result<Option<PathBuf>>> {
        if self.offscreen {
            return refused();
        }
        self.launcher.prompt_for_new_path(directory, suggested_name)
    }
}

/// A dialog's answer off screen: `REFUSED`, at once.
fn refused<T>() -> oneshot::Receiver<Result<T>> {
    let (tx, rx) = oneshot::channel();
    let _ = tx.send(Err(anyhow::anyhow!(REFUSED)));
    rx
}

/// A launcher that launches nothing and notes each call, for the backends' tests.
#[cfg(test)]
#[derive(Clone, Default)]
pub(crate) struct Recorder(pub(crate) std::rc::Rc<std::cell::RefCell<Vec<String>>>);

#[cfg(test)]
impl Launcher for Recorder {
    fn open_url(&self, url: &str) {
        self.0.borrow_mut().push(format!("url {url}"));
    }

    fn reveal_path(&self, path: &Path) {
        self.0.borrow_mut().push(format!("reveal {}", path.display()));
    }

    fn prompt_for_paths(
        &self,
        _options: PathPromptOptions,
    ) -> oneshot::Receiver<Result<Option<Vec<PathBuf>>>> {
        self.0.borrow_mut().push("open dialog".into());
        let (tx, rx) = oneshot::channel();
        let _ = tx.send(Ok(None));
        rx
    }

    fn prompt_for_new_path(
        &self,
        _directory: &Path,
        _suggested_name: Option<&str>,
    ) -> oneshot::Receiver<Result<Option<PathBuf>>> {
        self.0.borrow_mut().push("save dialog".into());
        let (tx, rx) = oneshot::channel();
        let _ = tx.send(Ok(None));
        rx
    }
}

/// Calls each service once through `services`; what the recorder behind it saw, and whether each dialog
/// answered with `REFUSED`.
#[cfg(test)]
pub(crate) fn call_each(
    open_url: impl Fn(&str),
    reveal_path: impl Fn(&Path),
    prompt_for_paths: impl Fn(PathPromptOptions) -> oneshot::Receiver<Result<Option<Vec<PathBuf>>>>,
    prompt_for_new_path: impl Fn(&Path, Option<&str>) -> oneshot::Receiver<Result<Option<PathBuf>>>,
) -> (bool, bool) {
    let refused = |e: &anyhow::Error| e.to_string() == REFUSED;
    open_url("https://example.com/docs");
    reveal_path(Path::new("D:/folo/models/m.onnx"));
    let options = PathPromptOptions { files: true, directories: false, multiple: false, prompt: None };
    let mut paths = prompt_for_paths(options);
    let mut new_path = prompt_for_new_path(Path::new("D:/folo"), Some("x.txt"));
    let a = matches!(paths.try_recv(), Ok(Some(Err(e))) if refused(&e));
    let b = matches!(new_path.try_recv(), Ok(Some(Err(e))) if refused(&e));
    (a, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn each(services: &Services) -> (bool, bool) {
        call_each(
            |u| services.open_url(u),
            |p| services.reveal_path(p),
            |o| services.prompt_for_paths(o),
            |d, n| services.prompt_for_new_path(d, n),
        )
    }

    #[test]
    fn off_screen_nothing_is_launched_and_the_dialogs_say_why() {
        let recorder = Recorder::default();
        let services = Services::new(true, Box::new(recorder.clone()));
        assert_eq!(each(&services), (true, true));
        assert!(recorder.0.borrow().is_empty(), "launched: {:?}", recorder.0.borrow());
    }

    #[test]
    fn on_screen_each_service_reaches_the_backend() {
        let recorder = Recorder::default();
        let services = Services::new(false, Box::new(recorder.clone()));
        assert_eq!(each(&services), (false, false));
        assert_eq!(
            *recorder.0.borrow(),
            ["url https://example.com/docs", "reveal D:/folo/models/m.onnx", "open dialog", "save dialog"]
        );
        // Nothing to open: nothing is launched.
        services.open_url("");
        services.reveal_path(Path::new(""));
        assert_eq!(recorder.0.borrow().len(), 4);
    }
}
