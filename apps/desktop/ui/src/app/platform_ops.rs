//! App methods that forward to the platform: clipboard, displays, file prompts, URLs, quit, activate.
//! Owner tag: `"runtime"`.
#![allow(dead_code)] // skeleton

use std::{
    path::{Path, PathBuf},
    rc::Rc,
};

use crate::{App, ClipboardItem, PathPromptOptions, PlatformDisplay};

impl App {
    /// Writes to the system clipboard.
    pub fn write_to_clipboard(&self, item: ClipboardItem) {
        self.platform.write_to_clipboard(item)
    }

    /// Reads the system clipboard.
    pub fn read_from_clipboard(&self) -> Option<ClipboardItem> {
        self.platform.read_from_clipboard()
    }

    /// Every display.
    pub fn displays(&self) -> Vec<Rc<dyn PlatformDisplay>> {
        self.platform.displays()
    }

    /// The primary display.
    pub fn primary_display(&self) -> Option<Rc<dyn PlatformDisplay>> {
        self.platform.primary_display()
    }

    /// Opens the system's file / folder picker. The receiver resolves to the chosen paths (`None` when
    /// cancelled); awaiting it gives `Err` if the platform dropped the request.
    pub fn prompt_for_paths(
        &self,
        options: PathPromptOptions,
    ) -> crate::oneshot::Receiver<anyhow::Result<Option<Vec<PathBuf>>>> {
        self.platform.prompt_for_paths(options)
    }

    /// Opens a URL in the default browser.
    pub fn open_url(&self, url: &str) {
        self.platform.open_url(url)
    }

    /// Shows a file in the system's file manager.
    pub fn reveal_path(&self, path: &Path) {
        self.platform.reveal_path(path)
    }

    /// Brings the app to the front.
    pub fn activate(&self, ignoring_other_apps: bool) {
        self.platform.activate(ignoring_other_apps)
    }

    /// Quits: runs the quit observers (waiting briefly for their futures), then stops the platform loop.
    pub fn quit(&mut self) {
        todo!("runtime")
    }
}

impl crate::Bounds<crate::Pixels> {
    /// Bounds of `size` centred on display `display_id` (or the primary display), as GPUI's.
    pub fn centered(
        display_id: Option<crate::DisplayId>,
        size: crate::Size<crate::Pixels>,
        cx: &App,
    ) -> Self {
        let display = display_id
            .and_then(|id| cx.displays().into_iter().find(|d| d.id() == id))
            .or_else(|| cx.primary_display());
        display.map_or(crate::Bounds { origin: crate::Point::default(), size }, |display| {
            crate::Bounds::centered_at(display.bounds().center(), size)
        })
    }
}
