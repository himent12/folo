//! App methods that forward to the platform: clipboard, displays, file prompts, URLs, quit, activate.
//! Owner tag: `"runtime"`.

use std::{
    path::{Path, PathBuf},
    rc::Rc,
    time::Duration,
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
    ///
    /// GPUI's `quit` asks the platform, whose quit callback runs `shutdown`; our platforms have no quit
    /// callback, so this is `shutdown` then `platform.quit()`: the `on_app_quit` callbacks run once, every
    /// window is dropped, and their futures get `SHUTDOWN_TIMEOUT` (100 ms, as GPUI) on this thread.
    pub fn quit(&mut self) {
        if self.quitting {
            return;
        }
        self.quitting = true;
        let futures: Vec<_> =
            self.quit_observers.remove(&()).into_iter().map(|mut on_quit| on_quit(self)).collect();
        // A leased window (the update that called `quit`) is dropped when that update ends.
        let windows: Vec<_> = self.windows.drain().collect();
        self.window_order.clear();
        drop(windows);

        let all = async move {
            for future in futures {
                future.await;
            }
        };
        if self.background_executor.block_with_timeout(SHUTDOWN_TIMEOUT, all).is_none() {
            eprintln!("folo-ui: timed out waiting for on_app_quit");
        }
        self.platform.quit();
    }
}

/// How long `quit` waits for the `on_app_quit` futures (GPUI's value).
const SHUTDOWN_TIMEOUT: Duration = Duration::from_millis(100);

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
