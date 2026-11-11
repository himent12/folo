//! Smoke check of the winit backend on a real window, off screen and never focused
//! (`FOLO_UI_OFFSCREEN=1` is forced): open a window, present a test pattern, run background /
//! main-thread / timer jobs, and on Windows drive the window with posted messages (never real input).
//! Prints PASS / FAIL per step; exits 0 only if all pass.
//!
//! `cargo run -p folo-ui --example winit_smoke --features winit,portable-text`

#[cfg(not(any(not(windows), feature = "winit")))]
fn main() {
    eprintln!("build with --features winit");
    std::process::exit(2);
}

#[cfg(any(not(windows), feature = "winit"))]
fn main() {
    smoke::main();
}

#[cfg(any(not(windows), feature = "winit"))]
mod smoke {
    use folo_ui::text::portable::CosmicTextSystem;
    use folo_ui::winit::WinitPlatform;
    use folo_ui::{
        Bounds, DevicePixels, DispatchEventResult, Framebuffer, Pixels, Platform, PlatformDispatcher,
        PlatformInput, PlatformInputHandler, PlatformWindow, Point, RequestFrameOptions, Size,
        TitlebarOptions, UTF16Selection, WindowControlArea, WindowKind, WindowParams, oneshot, point, px,
        size,
    };
    use std::cell::{Cell, RefCell};
    use std::future::Future;
    use std::ops::Range;
    use std::rc::Rc;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::{Duration, Instant};

    pub fn main() {
        // SAFETY: nothing else runs yet (single thread, before any other thread is spawned).
        unsafe { std::env::set_var("FOLO_UI_OFFSCREEN", "1") };
        // A watchdog: a hung loop is a failure, not a hang.
        std::thread::spawn(|| {
            std::thread::sleep(Duration::from_secs(30));
            println!("FAIL watchdog: the smoke run did not finish in 30 s");
            std::process::exit(1);
        });

        let platform =
            Rc::new(WinitPlatform::new(Arc::new(CosmicTextSystem::new())).expect("winit platform"));
        let report = Rc::new(Report::default());
        let dispatcher = platform.dispatcher();
        {
            let (platform2, report2, dispatcher2) = (platform.clone(), report.clone(), dispatcher.clone());
            platform.run(Box::new(move || {
                spawn_main(&dispatcher2.clone(), async move {
                    script(&platform2, &dispatcher2, &report2).await;
                    platform2.quit();
                });
            }));
        }
        let failed = report.print();
        std::process::exit(if failed == 0 { 0 } else { 1 });
    }

    #[derive(Default)]
    struct Report(RefCell<Vec<(bool, String)>>);

    impl Report {
        fn check(&self, ok: bool, what: impl Into<String>) {
            self.0.borrow_mut().push((ok, what.into()));
        }
        fn print(&self) -> usize {
            let results = self.0.borrow();
            for (ok, what) in results.iter() {
                println!("{} {what}", if *ok { "PASS" } else { "FAIL" });
            }
            results.iter().filter(|(ok, _)| !ok).count()
        }
    }

    /// What the window's callbacks saw.
    #[derive(Default)]
    struct Seen {
        frames: Cell<usize>,
        presented_frames: Cell<usize>,
        last_options: Cell<Option<RequestFrameOptions>>,
        resizes: RefCell<Vec<(Size<Pixels>, f32)>>,
        inputs: RefCell<Vec<PlatformInput>>,
        texts: RefCell<Vec<String>>,
        should_close: Cell<usize>,
        hit_area: Cell<Option<WindowControlArea>>,
    }

    fn spawn_main(dispatcher: &Arc<dyn PlatformDispatcher>, future: impl Future<Output = ()> + 'static) {
        let d = dispatcher.clone();
        let (runnable, task) = async_task::spawn_local(future, move |r| d.dispatch_on_main_thread(r));
        runnable.schedule();
        task.detach();
    }

    fn sleep(dispatcher: &Arc<dyn PlatformDispatcher>, delay: Duration) -> oneshot::Receiver<()> {
        let (tx, rx) = oneshot::channel();
        let d = dispatcher.clone();
        let (runnable, task) = async_task::spawn(
            async move {
                let _ = tx.send(());
            },
            move |r| d.dispatch_after(delay, r),
        );
        runnable.schedule();
        task.detach();
        rx
    }

    /// The test pattern: a gradient with a white border, at `width` x `height`.
    fn pattern(width: u32, height: u32) -> Framebuffer {
        let mut fb = Framebuffer::new(width, height);
        for y in 0..height {
            for x in 0..width {
                let edge = x == 0 || y == 0 || x + 1 == width || y + 1 == height;
                let pixel = if edge {
                    0x00ff_ffff
                } else {
                    folo_ui::pack_rgb((x % 256) as u8, (y % 256) as u8, 0x80)
                };
                fb.set_pixel(x, y, pixel);
            }
        }
        fb
    }

    struct Recorder(Rc<Seen>);

    impl PlatformInputHandler for Recorder {
        fn selected_text_range(&mut self, _ignore_disabled_input: bool) -> Option<UTF16Selection> {
            Some(UTF16Selection { range: 0..0, reversed: false })
        }
        fn marked_text_range(&mut self) -> Option<Range<usize>> {
            None
        }
        fn text_for_range(
            &mut self,
            _range: Range<usize>,
            _adjusted: &mut Option<Range<usize>>,
        ) -> Option<String> {
            None
        }
        fn replace_text_in_range(&mut self, _range: Option<Range<usize>>, text: &str) {
            self.0.texts.borrow_mut().push(text.to_owned());
        }
        fn replace_and_mark_text_in_range(
            &mut self,
            _r: Option<Range<usize>>,
            text: &str,
            _s: Option<Range<usize>>,
        ) {
            self.0.texts.borrow_mut().push(format!("marked:{text}"));
        }
        fn unmark_text(&mut self) {}
        fn bounds_for_range(&mut self, _range: Range<usize>) -> Option<Bounds<Pixels>> {
            None
        }
        fn character_index_for_point(&mut self, _point: Point<Pixels>) -> Option<usize> {
            None
        }
    }

    async fn script(platform: &Rc<WinitPlatform>, dispatcher: &Arc<dyn PlatformDispatcher>, report: &Report) {
        let title = format!("folo-ui winit smoke {}", std::process::id());
        let params = WindowParams {
            bounds: Bounds::new(point(px(100.), px(100.)), size(px(960.), px(620.))),
            titlebar: Some(TitlebarOptions {
                title: Some(title.clone().into()),
                appears_transparent: true,
                traffic_light_position: None,
            }),
            kind: WindowKind::Normal,
            is_movable: true,
            is_resizable: true,
            is_minimizable: true,
            focus: true,
            show: true,
            display_id: None,
            window_min_size: Some(size(px(400.), px(300.))),
            app_id: Some("folo-ui-smoke".into()),
        };
        let opened = Instant::now();
        let window = match platform.open_window(params) {
            Ok(window) => Rc::new(RefCell::new(window)),
            Err(error) => return report.check(false, format!("open window: {error:#}")),
        };
        let seen = Rc::new(Seen::default());
        install_callbacks(&window, &seen);
        window.borrow_mut().set_input_handler(Box::new(Recorder(seen.clone())));

        // Off screen: outside every display, and not active.
        let bounds = window.borrow().bounds();
        let displays = platform.displays();
        let on_a_display = displays.iter().any(|d| d.bounds().intersects(&bounds));
        report.check(
            !displays.is_empty() && !on_a_display,
            format!("window opens outside all {} displays at {bounds:?}", displays.len()),
        );
        report.check(!window.borrow().is_active(), "window is not active");
        let content = window.borrow().content_size();
        report.check(content == size(px(960.), px(620.)), format!("content size {content:?} is 960x620"));
        let primary = platform.primary_display().map(|d| d.uuid().is_ok());
        report.check(primary == Some(true), "primary display has a uuid");

        // First frame: the system's first redraw reaches on_request_frame and presents.
        wait_for(dispatcher, Duration::from_secs(3), || seen.presented_frames.get() >= 1).await;
        report.check(
            seen.presented_frames.get() >= 1,
            format!("first frame presented {:.0} ms after open", opened.elapsed().as_secs_f64() * 1e3),
        );

        // A requested frame arrives once, then the window is idle (no frames without a request).
        let before = seen.frames.get();
        window.borrow().request_frame();
        window.borrow().request_frame();
        wait_for(dispatcher, Duration::from_secs(2), || seen.frames.get() > before).await;
        sleep(dispatcher, Duration::from_millis(150)).await.ok();
        let requested = seen.frames.get() - before;
        report.check(requested == 1, format!("two coalesced request_frame calls gave {requested} frame(s)"));
        let before = seen.frames.get();
        sleep(dispatcher, Duration::from_millis(300)).await.ok();
        report.check(seen.frames.get() == before, "no frames while nothing is requested");

        // Resize: the resize callback and a synchronous frame that must present.
        seen.resizes.borrow_mut().clear();
        let frames_before = seen.frames.get();
        window.borrow_mut().resize(size(px(800.), px(500.)));
        let target = size(px(800.), px(500.));
        wait_for(dispatcher, Duration::from_secs(2), || {
            seen.resizes.borrow().iter().any(|(s, _)| *s == target)
        })
        .await;
        let resized = seen.resizes.borrow().clone();
        report.check(resized.iter().any(|(s, _)| *s == target), format!("resize callback got {resized:?}"));
        report.check(seen.frames.get() > frames_before, "the resize drew a frame");
        let options = seen.last_options.get();
        report.check(
            options.is_some_and(|o| o.require_presentation),
            format!("resize drew a frame requiring presentation ({options:?})"),
        );

        // Partial present after a full one.
        let scale = window.borrow().scale_factor();
        let (w, h) = ((800. * scale).round() as u32, (500. * scale).round() as u32);
        let mut frame = pattern(w, h);
        window.borrow_mut().present(
            &frame,
            &[Bounds::new(
                point(DevicePixels(0), DevicePixels(0)),
                size(DevicePixels(w as i32), DevicePixels(h as i32)),
            )],
        );
        for x in 10..74 {
            for y in 10..74 {
                frame.set_pixel(x, y, 0x00ff_0000);
            }
        }
        let damage = [Bounds::new(
            point(DevicePixels(10), DevicePixels(10)),
            size(DevicePixels(64), DevicePixels(64)),
        )];
        let started = Instant::now();
        window.borrow_mut().present(&frame, &damage);
        report.check(
            true,
            format!("partial present of 64x64 took {:.2} ms", started.elapsed().as_secs_f64() * 1e3),
        );

        dispatcher_jobs(dispatcher, report).await;

        #[cfg(windows)]
        posted_input::run(&title, &window, &seen, dispatcher, report).await;
        #[cfg(not(windows))]
        report.check(true, "posted input: Windows only (the translation functions are unit-tested)");

        #[cfg(windows)]
        {
            let cpu_before = posted_input::process_cpu();
            sleep(dispatcher, Duration::from_secs(2)).await.ok();
            let cpu = posted_input::process_cpu() - cpu_before;
            report.check(
                cpu < Duration::from_millis(40),
                format!("idle: {:.1} ms of CPU over 2 s", cpu.as_secs_f64() * 1e3),
            );
        }

        drop(window);
    }

    fn install_callbacks(window: &Rc<RefCell<Box<dyn PlatformWindow>>>, seen: &Rc<Seen>) {
        let w = window.borrow();
        let (s, weak) = (seen.clone(), Rc::downgrade(window));
        w.on_request_frame(Box::new(move |options| {
            s.frames.set(s.frames.get() + 1);
            s.last_options.set(Some(options));
            let Some(window) = weak.upgrade() else { return };
            let Ok(mut window) = window.try_borrow_mut() else { return };
            let scale = window.scale_factor();
            let content = window.content_size();
            let (w, h) = (
                (f32::from(content.width) * scale).round() as u32,
                (f32::from(content.height) * scale).round() as u32,
            );
            window.present(&pattern(w, h), &[]);
            s.presented_frames.set(s.presented_frames.get() + 1);
        }));
        let s = seen.clone();
        w.on_resize(Box::new(move |size, scale| s.resizes.borrow_mut().push((size, scale))));
        let s = seen.clone();
        w.on_input(Box::new(move |input| {
            s.inputs.borrow_mut().push(input);
            DispatchEventResult { propagate: true, default_prevented: false }
        }));
        let s = seen.clone();
        w.on_should_close(Box::new(move || {
            s.should_close.set(s.should_close.get() + 1);
            false
        }));
        let s = seen.clone();
        w.on_hit_test_window_control(Box::new(move || s.hit_area.get()));
    }

    async fn wait_for(dispatcher: &Arc<dyn PlatformDispatcher>, limit: Duration, done: impl Fn() -> bool) {
        let deadline = Instant::now() + limit;
        while !done() && Instant::now() < deadline {
            sleep(dispatcher, Duration::from_millis(10)).await.ok();
        }
    }

    async fn dispatcher_jobs(dispatcher: &Arc<dyn PlatformDispatcher>, report: &Report) {
        // 1000 background jobs.
        let done = Arc::new(AtomicUsize::new(0));
        for _ in 0..1000 {
            let (d, done) = (dispatcher.clone(), done.clone());
            let (runnable, task) = async_task::spawn(
                async move {
                    done.fetch_add(1, Ordering::SeqCst);
                },
                move |r| d.dispatch(r),
            );
            runnable.schedule();
            task.detach();
        }
        wait_for(dispatcher, Duration::from_secs(5), || done.load(Ordering::SeqCst) == 1000).await;
        report.check(done.load(Ordering::SeqCst) == 1000, "1000 background jobs ran");

        // Main-thread jobs run in order, on the main thread.
        let order = Rc::new(RefCell::new(Vec::new()));
        for i in 0..20 {
            let (order, d) = (order.clone(), dispatcher.clone());
            spawn_main(dispatcher, async move {
                assert!(d.is_main_thread());
                order.borrow_mut().push(i);
            });
        }
        wait_for(dispatcher, Duration::from_secs(2), || order.borrow().len() == 20).await;
        report.check(
            *order.borrow() == (0..20).collect::<Vec<_>>(),
            "20 main-thread jobs ran in order on the main thread",
        );

        // Timers fire in deadline order, not before their time.
        let fired = Arc::new(parking_lot::Mutex::new(Vec::new()));
        let start = Instant::now();
        for (i, ms) in [(0u32, 90u64), (1, 30), (2, 60)] {
            let (d, fired) = (dispatcher.clone(), fired.clone());
            let (runnable, task) =
                async_task::spawn(async move { fired.lock().push((i, start.elapsed())) }, move |r| {
                    d.dispatch_after(Duration::from_millis(ms), r)
                });
            runnable.schedule();
            task.detach();
        }
        wait_for(dispatcher, Duration::from_secs(2), || fired.lock().len() == 3).await;
        let fired = fired.lock().clone();
        let order: Vec<u32> = fired.iter().map(|f| f.0).collect();
        let on_time = fired.iter().all(|(i, at)| *at >= Duration::from_millis([90, 30, 60][*i as usize]));
        report.check(order == [1, 2, 0] && on_time, format!("timers fired in order {order:?}, none early"));
    }

    #[cfg(windows)]
    mod posted_input {
        use super::*;
        use folo_ui::{FileDropEvent, KeyDownEvent, MouseButton, ScrollDelta};
        use windows::Win32::Foundation::{HWND, LPARAM, WPARAM};
        use windows::Win32::System::Threading::{GetCurrentProcess, GetProcessTimes};
        use windows::Win32::UI::WindowsAndMessaging::{
            FindWindowW, PostMessageW, WM_CHAR, WM_KEYDOWN, WM_KEYUP, WM_LBUTTONDOWN, WM_LBUTTONUP,
            WM_MOUSEMOVE, WM_MOUSEWHEEL,
        };
        use windows::core::HSTRING;

        /// The process's user + kernel CPU time.
        pub fn process_cpu() -> Duration {
            let (mut c, mut e, mut k, mut u) = Default::default();
            // SAFETY: the pseudo handle of this process is valid; the out pointers are locals.
            unsafe { GetProcessTimes(GetCurrentProcess(), &mut c, &mut e, &mut k, &mut u) }
                .expect("GetProcessTimes");
            let ticks = |t: windows::Win32::Foundation::FILETIME| {
                (t.dwHighDateTime as u64) << 32 | t.dwLowDateTime as u64
            };
            Duration::from_nanos((ticks(k) + ticks(u)) * 100)
        }

        fn post(hwnd: HWND, message: u32, wparam: usize, lparam: isize) {
            // SAFETY: posting to a window of this thread; the message carries plain values.
            let _ = unsafe { PostMessageW(Some(hwnd), message, WPARAM(wparam), LPARAM(lparam)) };
        }

        fn xy(x: i32, y: i32) -> isize {
            ((y as u32 as isize) << 16) | (x as u16 as isize)
        }

        fn key(hwnd: HWND, vk: usize, scan: isize, extended: bool, ch: Option<char>) {
            let lparam = 1 | (scan << 16) | if extended { 1 << 24 } else { 0 };
            post(hwnd, WM_KEYDOWN, vk, lparam);
            if let Some(ch) = ch {
                post(hwnd, WM_CHAR, ch as usize, lparam);
            }
            post(hwnd, WM_KEYUP, vk, lparam | (1 << 30) | (1 << 31));
        }

        pub async fn run(
            title: &str,
            window: &Rc<RefCell<Box<dyn PlatformWindow>>>,
            seen: &Rc<Seen>,
            dispatcher: &Arc<dyn PlatformDispatcher>,
            report: &Report,
        ) {
            // SAFETY: a plain lookup by title (this process's window).
            let Ok(hwnd) = (unsafe { FindWindowW(None, &HSTRING::from(title)) }) else {
                return report.check(false, "posted input: window not found by title");
            };
            let scale = window.borrow().scale_factor();
            seen.inputs.borrow_mut().clear();

            // Mouse: a move, a wheel notch, two clicks over a "Close" caption button.
            post(hwnd, WM_MOUSEMOVE, 0, xy(120, 60));
            post(hwnd, WM_MOUSEWHEEL, (120usize) << 16, xy(0, 0));
            seen.hit_area.set(Some(WindowControlArea::Close));
            for _ in 0..2 {
                post(hwnd, WM_LBUTTONDOWN, 1, xy(120, 60));
                post(hwnd, WM_LBUTTONUP, 0, xy(120, 60));
            }
            wait_for(dispatcher, Duration::from_secs(2), || seen.should_close.get() >= 2).await;
            seen.hit_area.set(None);
            let inputs = seen.inputs.borrow().clone();
            let want = point(px(120. / scale), px(60. / scale));
            let moved = inputs.iter().any(|i| matches!(i, PlatformInput::MouseMove(m) if (m.position - want).x.abs() < px(0.01) && (m.position - want).y.abs() < px(0.01)));
            report.check(moved, format!("posted WM_MOUSEMOVE gave a MouseMove at {want:?} (scale {scale})"));
            let wheel = inputs.iter().find_map(|i| match i {
                PlatformInput::ScrollWheel(w) => Some(w.delta),
                _ => None,
            });
            report.check(
                matches!(wheel, Some(ScrollDelta::Lines(p)) if p.y == 3.0),
                format!("one wheel notch is 3 lines ({wheel:?})"),
            );
            let counts: Vec<usize> = inputs
                .iter()
                .filter_map(|i| match i {
                    PlatformInput::MouseDown(d) if d.button == MouseButton::Left => Some(d.click_count),
                    _ => None,
                })
                .collect();
            report.check(counts == [1, 2], format!("left presses have click counts {counts:?}"));
            report.check(
                seen.should_close.get() == 2,
                format!("Close caption area asked should_close {} times on release", seen.should_close.get()),
            );
            let ups = inputs.iter().filter(|i| matches!(i, PlatformInput::MouseUp(_))).count();
            report.check(ups == 2, "the presses reached the app first (and their releases)");
            let drops =
                inputs.iter().any(|i| matches!(i, PlatformInput::FileDrop(FileDropEvent::Pending { .. })));
            report.check(!drops, "no stray file-drop events");

            // Keys: "a" (typed into the input handler), Enter, F11, Left.
            seen.inputs.borrow_mut().clear();
            seen.texts.borrow_mut().clear();
            key(hwnd, 0x41, 0x1E, false, Some('a'));
            key(hwnd, 0x0D, 0x1C, false, Some('\r'));
            key(hwnd, 0x7A, 0x57, false, None);
            key(hwnd, 0x25, 0x4B, true, None);
            wait_for(dispatcher, Duration::from_secs(2), || {
                seen.inputs.borrow().iter().filter(|i| matches!(i, PlatformInput::KeyDown(_))).count() >= 4
            })
            .await;
            let downs: Vec<(String, Option<String>)> = seen
                .inputs
                .borrow()
                .iter()
                .filter_map(|i| match i {
                    PlatformInput::KeyDown(KeyDownEvent { keystroke, .. }) => {
                        Some((keystroke.unparse(), keystroke.key_char.clone()))
                    }
                    _ => None,
                })
                .collect();
            let want: Vec<(String, Option<String>)> = vec![
                ("a".into(), Some("a".into())),
                ("enter".into(), None),
                ("f11".into(), None),
                ("left".into(), None),
            ];
            report.check(downs == want, format!("posted keys gave {downs:?}"));
            let texts = seen.texts.borrow().clone();
            report.check(texts == ["a"], format!("only \"a\" was typed into the input handler ({texts:?})"));
        }
    }
}
