//! The modal file dialog (only on a private desktop, `--dialog`): while `IFileOpenDialog` runs its
//! own message loop, main-thread tasks and frames must keep running; cancelling it gives `Ok(None)`.

use crate::common::*;
use ::windows::Win32::Foundation::*;
use ::windows::Win32::System::Threading::GetCurrentThreadId;
use ::windows::Win32::UI::WindowsAndMessaging::*;
use ::windows::core::BOOL;
use folo_ui::windows::{WindowsPlatform, WindowsWindow};
use folo_ui::*;
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

/// The visible dialog box (`#32770`) of thread `thread`, if one is up.
fn find_dialog(thread: u32) -> Option<isize> {
    unsafe extern "system" fn each(hwnd: HWND, found: LPARAM) -> BOOL {
        let mut class = [0u16; 16];
        // SAFETY: a valid buffer; `found` is the Option the caller passed.
        unsafe {
            let n = GetClassNameW(hwnd, &mut class) as usize;
            if IsWindowVisible(hwnd).as_bool() && String::from_utf16_lossy(&class[..n]) == "#32770" {
                *(found.0 as *mut Option<isize>) = Some(hwnd.0 as isize);
                return BOOL(0);
            }
        }
        BOOL(1)
    }
    let mut found: Option<isize> = None;
    // SAFETY: the callback only writes `found`, which outlives the call.
    let _ = unsafe { EnumThreadWindows(thread, Some(each), LPARAM(&mut found as *mut _ as isize)) };
    found
}

pub fn dialog(
    checks: &mut Checks,
    platform: &Rc<WindowsPlatform>,
    window: &Rc<RefCell<WindowsWindow>>,
    log: &Rc<RefCell<Log>>,
    frame: &Rc<RefCell<Frame>>,
) {
    // SAFETY: a plain query.
    let main_thread = unsafe { GetCurrentThreadId() };
    let open = Arc::new(AtomicBool::new(false));
    let ran_while_open = Arc::new(AtomicUsize::new(0));
    let frames_before = Rc::new(RefCell::new(0usize));
    let d = platform.dispatcher();
    // A helper thread: waits for the dialog, queues main-thread tasks while it is up, then cancels it.
    let helper = {
        let (open, ran) = (open.clone(), ran_while_open.clone());
        std::thread::spawn(move || {
            let mut dialog = None;
            for _ in 0..250 {
                dialog = find_dialog(main_thread);
                if dialog.is_some() {
                    break;
                }
                std::thread::sleep(Duration::from_millis(20));
            }
            let Some(dialog) = dialog else { return false };
            open.store(true, Ordering::SeqCst);
            for _ in 0..10 {
                let (ran, open, dd) = (ran.clone(), open.clone(), d.clone());
                let (runnable, task) = async_task::spawn(
                    async move {
                        if open.load(Ordering::SeqCst) {
                            ran.fetch_add(1, Ordering::SeqCst);
                        }
                    },
                    move |r| dd.dispatch_on_main_thread(r),
                );
                task.detach();
                runnable.schedule();
                std::thread::sleep(Duration::from_millis(30));
            }
            open.store(false, Ordering::SeqCst);
            // SAFETY: posting Cancel to the dialog.
            unsafe {
                PostMessageW(Some(HWND(dialog as _)), WM_COMMAND, WPARAM(IDCANCEL.0 as usize), LPARAM(0))
                    .is_ok()
            }
        })
    };
    // An animation runs meanwhile: its frames come through the dialog's loop.
    *frames_before.borrow_mut() = log.borrow().frames.len();
    frame.borrow_mut().animate = 100_000;
    window.borrow().request_frame();
    let mut rx = platform.prompt_for_paths(PathPromptOptions {
        files: true,
        directories: false,
        multiple: true,
        prompt: None,
    });
    let mut result = None;
    let done = pump_until(Duration::from_secs(10), || {
        if let Ok(Some(r)) = rx.try_recv() {
            result = Some(r);
        }
        result.is_some()
    });
    frame.borrow_mut().animate = 0;
    let cancelled = helper.join().unwrap_or(false);
    let frames = log.borrow().frames.len() - *frames_before.borrow();
    let ran = ran_while_open.load(Ordering::SeqCst);
    checks.check(
        "file dialog opens and cancels",
        done && cancelled && matches!(result, Some(Ok(None))),
        format!("{result:?}"),
    );
    checks.check(
        "main-thread tasks and frames run while it is up",
        ran >= 8 && frames > 30,
        format!("{ran} task(s), {frames} frame(s)"),
    );
}
