// Adapted from GPUI (Apache-2.0): crates/gpui/src/platform/windows/platform.rs
//! `WindowsPlatform`: the message loop on the thread that created it, a message-only window for
//! main-thread wake-ups, and the `Platform` methods (displays, windows, clipboard, dialogs, shell).

use super::dispatcher::{MainQueue, WindowsDispatcher};
use super::display::WindowsDisplay;
use super::pacer::Pacer;
use super::util::*;
use super::window::{WindowsWindow, module_handle};
use super::{WM_FOLO_CALL, WM_FOLO_MAIN_TASKS, WM_FOLO_QUIT, clipboard, offscreen, shell};
use crate::platform::services::{Launcher, Services};
use crate::*;
use ::windows::Win32::Foundation::*;
use ::windows::Win32::System::Com::{CLSCTX_INPROC_SERVER, CoCreateInstance};
use ::windows::Win32::System::Ole::{OleInitialize, OleUninitialize};
use ::windows::Win32::UI::HiDpi::{
    DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2, SetProcessDpiAwarenessContext,
};
use ::windows::Win32::UI::Shell::{CLSID_DragDropHelper, IDropTargetHelper};
use ::windows::Win32::UI::WindowsAndMessaging::*;
use ::windows::core::{PCWSTR, w};
use anyhow::{Context as _, Result};
use std::cell::{Cell, OnceCell, RefCell};
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::Arc;

/// What the platform's windows share (main thread only).
pub(crate) struct Shared {
    message_window: Cell<HWND>,
    pub(crate) dispatcher: Arc<WindowsDispatcher>,
    pub(crate) pacer: Arc<Pacer>,
    /// The cursor `set_cursor_style` asked for (`None`: hidden).
    pub(crate) cursor: Cell<Option<HCURSOR>>,
    windows: RefCell<Vec<isize>>,
    pub(crate) icon: HICON,
    drop_helper: OnceCell<Option<IDropTargetHelper>>,
    /// `first-present` has been printed (or tracing is off).
    pub(crate) traced: Cell<bool>,
    /// Test-only client-space cursor override (device pixels) so tests can exercise pointer-dependent
    /// cancellation paths without moving a window onto a display or touching the real cursor.
    #[cfg(test)]
    pub(crate) test_client_cursor: Cell<Option<POINT>>,
}

impl Shared {
    /// The message-only window main-thread calls are posted to.
    pub(crate) fn message_window(&self) -> HWND {
        self.message_window.get()
    }

    /// The shell's drag image helper, made on the first drag.
    pub(crate) fn drop_helper(&self) -> Option<IDropTargetHelper> {
        self.drop_helper
            .get_or_init(|| {
                // SAFETY: creating an in-process COM object on an OLE-initialized thread.
                unsafe { CoCreateInstance(&CLSID_DragDropHelper, None, CLSCTX_INPROC_SERVER) }.ok()
            })
            .clone()
    }

    /// A window was destroyed; the app quits when its last window is gone (GPUI's rule).
    pub(crate) fn window_closed(&self, hwnd: HWND) {
        let mut windows = self.windows.borrow_mut();
        windows.retain(|&h| h != hwnd.0 as isize);
        if windows.is_empty() {
            // SAFETY: posting WM_QUIT to this thread's queue.
            unsafe { PostQuitMessage(0) };
        }
    }

    /// Our window that is active now, if any (the owner of a dialog).
    fn active_window(&self) -> Option<HWND> {
        // SAFETY: a plain query.
        let active = unsafe { ::windows::Win32::UI::Input::KeyboardAndMouse::GetActiveWindow() };
        self.windows.borrow().contains(&(active.0 as isize)).then_some(active)
    }
}

/// The Win32 platform. Create it on the thread that will run the message loop.
pub struct WindowsPlatform {
    shared: Rc<Shared>,
    text_system: Arc<dyn PlatformTextSystem>,
    quit_callbacks: RefCell<Vec<Box<dyn FnMut()>>>,
    /// The file dialogs, the browser and Explorer, closed off screen.
    services: Services,
    /// Off screen: keeps every other window of the process off the displays.
    _parker: Option<offscreen::Parker>,
}

impl WindowsPlatform {
    /// Sets per-monitor-v2 DPI awareness and OLE up on this thread, and makes the message window.
    /// `text_system` is what `text_system()` returns (the DirectWrite backend in the app).
    pub fn new(text_system: Arc<dyn PlatformTextSystem>) -> Result<Self> {
        // SAFETY: process and thread set-up calls with no pointers; DPI awareness fails harmlessly
        // when a manifest already set it.
        unsafe {
            let _ = SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
            OleInitialize(None).context("unable to initialize OLE")?;
        }
        let dispatcher = Arc::new(WindowsDispatcher::new());
        let message_window = create_message_window(dispatcher.main.clone())?;
        dispatcher.set_main_window(message_window);
        let shared = Rc::new(Shared {
            message_window: Cell::new(message_window),
            dispatcher,
            pacer: Arc::new(Pacer::default()),
            cursor: Cell::new(load_cursor(CursorStyle::Arrow)),
            windows: RefCell::new(Vec::new()),
            icon: load_icon(),
            drop_helper: OnceCell::new(),
            traced: Cell::new(false),
            #[cfg(test)]
            test_client_cursor: Cell::new(None),
        });
        let _parker = if offscreen() { offscreen::Parker::install() } else { None };
        let services = Services::new(offscreen(), Box::new(Win32Launcher(shared.clone())));
        Ok(Self { shared, text_system, quit_callbacks: RefCell::new(Vec::new()), services, _parker })
    }

    /// Called once the message loop has ended (after `quit`, or when the last window closed).
    pub fn on_quit(&self, callback: Box<dyn FnMut()>) {
        self.quit_callbacks.borrow_mut().push(callback);
    }

    /// The dispatcher, as its concrete type.
    pub fn windows_dispatcher(&self) -> Arc<WindowsDispatcher> {
        self.shared.dispatcher.clone()
    }

    /// Opens a window and returns it as its concrete type (tests and tools reach the handle).
    pub fn open_windows_window(&self, params: WindowParams) -> Result<WindowsWindow> {
        let window = WindowsWindow::new(params, self.shared.clone())?;
        self.shared.windows.borrow_mut().push(window.hwnd().0 as isize);
        Ok(window)
    }

    /// Opens an off-screen window directly without mutating process environment variables.
    #[cfg(test)]
    pub(crate) fn open_windows_window_offscreen(&self, params: WindowParams) -> Result<WindowsWindow> {
        let window = WindowsWindow::new_with_offscreen(params, self.shared.clone(), true)?;
        self.shared.windows.borrow_mut().push(window.hwnd().0 as isize);
        Ok(window)
    }

    /// Puts `launcher` behind the gate in place of the shell's, closed when `offscreen` (tests launch
    /// nothing real); the gate's state as the platform built it is returned.
    #[cfg(test)]
    pub(crate) fn launch_with(&mut self, offscreen: bool, launcher: Box<dyn Launcher>) -> bool {
        std::mem::replace(&mut self.services, Services::new(offscreen, launcher)).offscreen()
    }

    /// The windows opened so far (tests reach a core window's HWND through it).
    #[cfg(test)]
    pub(crate) fn window_handles(&self) -> Vec<HWND> {
        self.shared.windows.borrow().iter().map(|&h| HWND(h as _)).collect()
    }

    /// Overrides the client-space cursor position (device pixels) read by `cancel_presses` and
    /// `mouse_position`, so tests stay deterministic and off screen without touching the real cursor.
    #[cfg(test)]
    pub(crate) fn set_test_client_cursor(&self, point: Option<POINT>) {
        self.shared.test_client_cursor.set(point);
    }
}

impl Drop for WindowsPlatform {
    fn drop(&mut self) {
        self.shared.dispatcher.main.clear();
        // Also when the loop never ran (`run` stops it too; a second call does nothing).
        self.shared.pacer.shut_down();
        // SAFETY: destroying our message window and undoing OleInitialize, on the same thread.
        unsafe {
            let _ = DestroyWindow(self.shared.message_window());
            OleUninitialize();
        }
    }
}

impl Platform for WindowsPlatform {
    fn dispatcher(&self) -> Arc<dyn PlatformDispatcher> {
        self.shared.dispatcher.clone()
    }

    fn text_system(&self) -> Arc<dyn PlatformTextSystem> {
        self.text_system.clone()
    }

    fn run(&self, on_finish_launching: Box<dyn 'static + FnOnce()>) {
        on_finish_launching();
        let mut msg = MSG::default();
        // SAFETY: the standard loop; GetMessageW blocks while there is nothing to do. Keys are not
        // translated here: the window translates only those the app did not handle.
        unsafe {
            while GetMessageW(&mut msg, None, 0, 0).0 > 0 {
                DispatchMessageW(&msg);
            }
        }
        for callback in self.quit_callbacks.borrow_mut().iter_mut() {
            callback();
        }
        // Nothing runs main-thread work any more: drop what is queued and let the dispatcher's threads go,
        // and stop the frame pacer's thread.
        self.shared.dispatcher.shut_down();
        self.shared.pacer.shut_down();
    }

    fn quit(&self) {
        // SAFETY: posting to our own message window; it calls PostQuitMessage from the loop.
        let _ =
            unsafe { PostMessageW(Some(self.shared.message_window()), WM_FOLO_QUIT, WPARAM(0), LPARAM(0)) };
    }

    fn activate(&self, _ignoring_other_apps: bool) {}

    fn displays(&self) -> Vec<Rc<dyn PlatformDisplay>> {
        WindowsDisplay::all()
    }

    fn primary_display(&self) -> Option<Rc<dyn PlatformDisplay>> {
        WindowsDisplay::primary().map(|d| Rc::new(d) as Rc<dyn PlatformDisplay>)
    }

    fn open_window(&self, options: WindowParams) -> Result<Box<dyn PlatformWindow>> {
        Ok(Box::new(self.open_windows_window(options)?))
    }

    fn window_appearance(&self) -> WindowAppearance {
        system_appearance()
    }

    fn open_url(&self, url: &str) {
        self.services.open_url(url);
    }

    fn prompt_for_paths(
        &self,
        options: PathPromptOptions,
    ) -> oneshot::Receiver<Result<Option<Vec<PathBuf>>>> {
        self.services.prompt_for_paths(options)
    }

    fn prompt_for_new_path(
        &self,
        directory: &Path,
        suggested_name: Option<&str>,
    ) -> oneshot::Receiver<Result<Option<PathBuf>>> {
        self.services.prompt_for_new_path(directory, suggested_name)
    }

    fn reveal_path(&self, path: &Path) {
        self.services.reveal_path(path);
    }

    fn set_cursor_style(&self, style: CursorStyle) {
        let cursor = load_cursor(style);
        let previous = self.shared.cursor.replace(cursor);
        // Showing or hiding takes effect now; another shape at the next WM_SETCURSOR (GPUI's rule).
        if previous.is_some() != cursor.is_some() {
            // SAFETY: setting a shared system cursor (or none).
            unsafe { SetCursor(cursor) };
        }
    }

    fn write_to_clipboard(&self, item: ClipboardItem) {
        clipboard::write_to_clipboard(self.shared.message_window(), item);
    }

    fn read_from_clipboard(&self) -> Option<ClipboardItem> {
        clipboard::read_from_clipboard(self.shared.message_window())
    }
}

/// The shell's launches and the common file dialogs (behind `Services`' off-screen gate).
struct Win32Launcher(Rc<Shared>);

impl Launcher for Win32Launcher {
    fn open_url(&self, url: &str) {
        let url = url.to_string();
        std::thread::spawn(move || {
            let _ = shell::open_target(&url);
        });
    }

    fn prompt_for_paths(
        &self,
        options: PathPromptOptions,
    ) -> oneshot::Receiver<Result<Option<Vec<PathBuf>>>> {
        let (tx, rx) = oneshot::channel();
        let owner = self.0.active_window();
        super::post_call(
            self.0.message_window(),
            Box::new(move || {
                let _ = tx.send(shell::file_open_dialog(&options, owner));
            }),
        );
        rx
    }

    fn prompt_for_new_path(
        &self,
        directory: &Path,
        suggested_name: Option<&str>,
    ) -> oneshot::Receiver<Result<Option<PathBuf>>> {
        let (tx, rx) = oneshot::channel();
        let owner = self.0.active_window();
        let (directory, name) = (directory.to_path_buf(), suggested_name.map(str::to_string));
        super::post_call(
            self.0.message_window(),
            Box::new(move || {
                let _ = tx.send(shell::file_save_dialog(&directory, name.as_deref(), owner));
            }),
        );
        rx
    }

    fn reveal_path(&self, path: &Path) {
        let path = path.to_path_buf();
        std::thread::spawn(move || {
            // The shell's folder and selection calls need COM on this thread.
            // SAFETY: COM set-up and tear-down around the call, on this thread.
            unsafe {
                let ok = ::windows::Win32::System::Com::CoInitializeEx(
                    None,
                    ::windows::Win32::System::Com::COINIT_APARTMENTTHREADED,
                )
                .is_ok();
                let _ = shell::reveal_in_explorer(&path);
                if ok {
                    ::windows::Win32::System::Com::CoUninitialize();
                }
            }
        });
    }
}

/// The executable's icon resource 1, as GPUI loads it (none if the executable has no icon).
fn load_icon() -> HICON {
    // SAFETY: loading a shared resource from our own module.
    let handle = unsafe {
        LoadImageW(Some(module_handle().into()), PCWSTR(1 as _), IMAGE_ICON, 0, 0, LR_DEFAULTSIZE | LR_SHARED)
    };
    handle.map(|h| HICON(h.0)).unwrap_or_default()
}

const MESSAGE_CLASS_NAME: PCWSTR = w!("FOLO::PlatformWindow");

fn create_message_window(queue: Arc<MainQueue>) -> Result<HWND> {
    let class = WNDCLASSW {
        lpfnWndProc: Some(message_window_procedure),
        lpszClassName: MESSAGE_CLASS_NAME,
        ..Default::default()
    };
    // SAFETY: registering (again is harmless) and creating a message-only window; the queue is
    // boxed into the window's user data and freed at WM_NCDESTROY.
    unsafe {
        RegisterClassW(&class);
        let hwnd = CreateWindowExW(
            WINDOW_EX_STYLE(0),
            MESSAGE_CLASS_NAME,
            None,
            WINDOW_STYLE(0),
            0,
            0,
            0,
            0,
            Some(HWND_MESSAGE),
            None,
            None,
            None,
        )?;
        set_window_long(hwnd, GWLP_USERDATA, Box::into_raw(Box::new(queue)) as isize);
        Ok(hwnd)
    }
}

unsafe extern "system" fn message_window_procedure(
    hwnd: HWND,
    msg: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    match msg {
        WM_FOLO_MAIN_TASKS => {
            let queue = get_window_long(hwnd, GWLP_USERDATA) as *const Arc<MainQueue>;
            if !queue.is_null() {
                // SAFETY: the boxed queue set at creation, alive until WM_NCDESTROY; cloned so a task
                // that destroys the window cannot free it under us.
                let queue = unsafe { (*queue).clone() };
                queue.run_pending();
            }
            LRESULT(0)
        }
        WM_FOLO_CALL => {
            let call = lparam.0 as *mut Box<dyn FnOnce()>;
            if !call.is_null() {
                // SAFETY: `post_call` leaked exactly this box for this message (not null: checked).
                let call = unsafe { Box::from_raw(call) };
                call();
            }
            LRESULT(0)
        }
        WM_FOLO_QUIT => {
            // SAFETY: posting WM_QUIT to this thread's queue.
            unsafe { PostQuitMessage(0) };
            LRESULT(0)
        }
        WM_NCDESTROY => {
            let queue = set_window_long(hwnd, GWLP_USERDATA, 0) as *mut Arc<MainQueue>;
            if !queue.is_null() {
                // SAFETY: the box leaked at creation; nothing reads it after this.
                drop(unsafe { Box::from_raw(queue) });
            }
            LRESULT(0)
        }
        // SAFETY: default processing.
        _ => unsafe { DefWindowProcW(hwnd, msg, wparam, lparam) },
    }
}
