//! A real-window check of the Win32 backend that the user never sees: `FOLO_UI_OFFSCREEN=1` is forced
//! on, so the 960x620 window opens outside every display, never activated, with no taskbar button.
//! It presents a test pattern and drives the window with posted messages (never real input), checking
//! what the callbacks receive. Prints PASS / FAIL per step and exits 0 only if all pass.
//!
//!   cargo run -p folo-ui --release --example win_smoke            # all checks and the timings
//!   cargo run -p folo-ui --release --example win_smoke -- --quick # skip the 5 s idle-CPU check

#[cfg(not(windows))]
fn main() {
    eprintln!("win_smoke runs on Windows only");
}

#[cfg(windows)]
fn main() {
    smoke::main();
}

#[cfg(windows)]
mod smoke {
    use folo_ui::windows::{WindowsPlatform, WindowsWindow, ms_since_process_start, process_cpu_time};
    use folo_ui::*;
    use std::cell::{Cell, RefCell};
    use std::ops::Range;
    use std::rc::Rc;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::{Duration, Instant};
    use ::windows::Win32::Foundation::*;
    use ::windows::Win32::Graphics::Gdi::*;
    use ::windows::Win32::Storage::Xps::{PRINT_WINDOW_FLAGS, PrintWindow};
    use ::windows::Win32::UI::Input::KeyboardAndMouse::*;
    use ::windows::Win32::UI::WindowsAndMessaging::*;

    /// A text system that is never asked for anything (the example draws no text).
    struct NoText;

    impl PlatformTextSystem for NoText {
        fn add_fonts(&self, _: Vec<std::borrow::Cow<'static, [u8]>>) -> anyhow::Result<()> {
            Ok(())
        }
        fn all_font_names(&self) -> Vec<String> {
            Vec::new()
        }
        fn font_id(&self, _: &Font) -> anyhow::Result<FontId> {
            anyhow::bail!("no fonts")
        }
        fn font_metrics(&self, _: FontId) -> FontMetrics {
            unreachable!("no fonts")
        }
        fn typographic_bounds(&self, _: FontId, _: GlyphId) -> anyhow::Result<Bounds<f32>> {
            anyhow::bail!("no fonts")
        }
        fn advance(&self, _: FontId, _: GlyphId) -> anyhow::Result<Size<f32>> {
            anyhow::bail!("no fonts")
        }
        fn glyph_for_char(&self, _: FontId, _: char) -> Option<GlyphId> {
            None
        }
        fn glyph_raster_bounds(&self, _: &RenderGlyphParams) -> anyhow::Result<Bounds<DevicePixels>> {
            anyhow::bail!("no fonts")
        }
        fn rasterize_glyph(&self, _: &RenderGlyphParams, _: Bounds<DevicePixels>) -> anyhow::Result<(Size<DevicePixels>, Vec<u8>)> {
            anyhow::bail!("no fonts")
        }
        fn layout_line(&self, _: &str, _: Pixels, _: &[FontRun]) -> LineLayout {
            LineLayout::default()
        }
        fn glyph_blend(&self) -> GlyphBlend {
            GlyphBlend::IDENTITY
        }
    }

    /// What the callbacks saw.
    #[derive(Default)]
    struct Log {
        inputs: Vec<PlatformInput>,
        text: Vec<String>,
        resizes: Vec<(Size<Pixels>, f32)>,
        frames: Vec<(Instant, RequestFrameOptions)>,
        /// A frame request arrived while a resize was being reported (the synchronous draw).
        frame_inside_resize: usize,
        in_resize: bool,
        hover: Vec<bool>,
    }

    struct Checks {
        fails: usize,
    }

    impl Checks {
        fn check(&mut self, name: &str, ok: bool, detail: impl std::fmt::Display) {
            if ok {
                println!("PASS  {name}  {detail}");
            } else {
                self.fails += 1;
                println!("FAIL  {name}  {detail}");
            }
        }
    }

    /// The fake input handler: records the text calls.
    struct Handler(Rc<RefCell<Log>>);

    impl PlatformInputHandler for Handler {
        fn selected_text_range(&mut self, _: bool) -> Option<UTF16Selection> {
            Some(UTF16Selection { range: 0..0, reversed: false })
        }
        fn marked_text_range(&mut self) -> Option<Range<usize>> {
            None
        }
        fn text_for_range(&mut self, _: Range<usize>, _: &mut Option<Range<usize>>) -> Option<String> {
            None
        }
        fn replace_text_in_range(&mut self, _: Option<Range<usize>>, text: &str) {
            self.0.borrow_mut().text.push(format!("replace:{text}"));
        }
        fn replace_and_mark_text_in_range(&mut self, _: Option<Range<usize>>, text: &str, _: Option<Range<usize>>) {
            self.0.borrow_mut().text.push(format!("mark:{text}"));
        }
        fn unmark_text(&mut self) {}
        fn bounds_for_range(&mut self, _: Range<usize>) -> Option<Bounds<Pixels>> {
            Some(Bounds::new(point(px(10.), px(10.)), size(px(1.), px(16.))))
        }
        fn character_index_for_point(&mut self, _: Point<Pixels>) -> Option<usize> {
            None
        }
    }

    /// Dispatches whatever is queued, then waits (in the message wait, not a sleep) until `done` or
    /// the timeout. Returns whether `done` became true.
    fn pump_until(timeout: Duration, mut done: impl FnMut() -> bool) -> bool {
        let deadline = Instant::now() + timeout;
        loop {
            pump();
            if done() {
                return true;
            }
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return false;
            }
            // SAFETY: waiting on this thread's message queue only.
            unsafe { MsgWaitForMultipleObjects(None, false, left.as_millis().min(5) as u32, QS_ALLINPUT) };
        }
    }

    fn pump() {
        let mut msg = MSG::default();
        // SAFETY: the standard peek / dispatch on this thread; WM_QUIT goes back for `run`.
        unsafe {
            while PeekMessageW(&mut msg, None, 0, 0, PM_REMOVE).as_bool() {
                if msg.message == WM_QUIT {
                    PostQuitMessage(msg.wParam.0 as i32);
                    return;
                }
                DispatchMessageW(&msg);
            }
        }
    }

    fn post(hwnd: HWND, msg: u32, w: usize, l: isize) {
        // SAFETY: posting to our own window.
        unsafe { PostMessageW(Some(hwnd), msg, WPARAM(w), LPARAM(l)).unwrap() };
    }

    fn send(hwnd: HWND, msg: u32, w: usize, l: isize) -> isize {
        // SAFETY: our window, on its own thread: a direct call of the window procedure.
        unsafe { SendMessageW(hwnd, msg, Some(WPARAM(w)), Some(LPARAM(l))).0 }
    }

    fn pt(x: i32, y: i32) -> isize {
        ((y as i16 as u16 as isize) << 16) | (x as i16 as u16 as isize)
    }

    /// Sets this thread's key state (what GetKeyState reads): no real input is injected.
    fn set_keys(down: &[VIRTUAL_KEY]) {
        let mut state = [0u8; 256];
        for k in down {
            state[k.0 as usize] = 0x80;
        }
        // SAFETY: a 256-byte key state for this thread.
        unsafe { SetKeyboardState(&state).unwrap() };
    }

    fn key_lparam(vk: VIRTUAL_KEY, held: bool, up: bool) -> isize {
        // SAFETY: a plain layout query.
        let scan = unsafe { MapVirtualKeyW(vk.0 as u32, MAPVK_VK_TO_VSC) } as isize;
        let mut l = 1 | (scan << 16);
        if held || up {
            l |= 1 << 30;
        }
        if up {
            l |= 1 << 31;
        }
        l
    }

    fn rect(hwnd: HWND) -> RECT {
        let mut r = RECT::default();
        // SAFETY: a valid out pointer.
        unsafe { GetWindowRect(hwnd, &mut r).unwrap() };
        r
    }

    fn off_every_monitor(r: RECT) -> bool {
        // SAFETY: a plain query.
        unsafe { MonitorFromRect(&r, MONITOR_DEFAULTTONULL) }.is_invalid()
    }

    fn foreground() -> HWND {
        // SAFETY: a plain query.
        unsafe { GetForegroundWindow() }
    }

    /// A pattern that differs per pixel and per frame: colour bands, a grid and a frame number.
    fn pattern(fb: &mut Framebuffer, seed: u32) {
        for y in 0..fb.height {
            for x in 0..fb.width {
                let v = if (x / 64 + y / 64) % 2 == 0 { 0x0020_3040 } else { 0x00c0_8020 };
                fb.set_pixel(x, y, v ^ ((x.wrapping_mul(7) ^ y.wrapping_mul(13) ^ seed) & 0x00ff_ffff));
            }
        }
    }

    /// The client area as the compositor has it (`PrintWindow` with PW_CLIENTONLY | PW_RENDERFULLCONTENT).
    fn print_client(hwnd: HWND, width: i32, height: i32) -> Vec<u32> {
        // The compositor picks GDI's work up at its next pass: give it two.
        pump_until(Duration::from_millis(30), || false);
        // SAFETY: a memory DC and a top-down DIB section of the client size, read and freed here.
        unsafe {
            let screen = GetDC(None);
            let dc = CreateCompatibleDC(Some(screen));
            let info = BITMAPINFO {
                bmiHeader: BITMAPINFOHEADER {
                    biSize: std::mem::size_of::<BITMAPINFOHEADER>() as u32,
                    biWidth: width,
                    biHeight: -height,
                    biPlanes: 1,
                    biBitCount: 32,
                    biCompression: BI_RGB.0,
                    ..Default::default()
                },
                ..Default::default()
            };
            let mut bits = std::ptr::null_mut();
            let bitmap = CreateDIBSection(Some(dc), &info, DIB_RGB_COLORS, &mut bits, None, 0).unwrap();
            let old = SelectObject(dc, bitmap.into());
            let ok = PrintWindow(hwnd, dc, PRINT_WINDOW_FLAGS(3)).as_bool();
            let _ = GdiFlush();
            let pixels = if ok {
                std::slice::from_raw_parts(bits as *const u32, (width * height) as usize).iter().map(|p| p & 0x00ff_ffff).collect()
            } else {
                Vec::new()
            };
            SelectObject(dc, old);
            let _ = DeleteObject(bitmap.into());
            let _ = DeleteDC(dc);
            ReleaseDC(None, screen);
            pixels
        }
    }

    /// Where a capture differs from the frame (empty if it does not).
    fn mismatch(shot: &[u32], frame: &Framebuffer) -> String {
        if shot.len() != frame.pixels.len() {
            return format!(", capture has {} px, frame {}x{}", shot.len(), frame.width, frame.height);
        }
        let bad: Vec<usize> = (0..shot.len()).filter(|&i| shot[i] != frame.pixels[i]).collect();
        match bad.first() {
            None => String::new(),
            Some(&i) => format!(
                ", {} differ, first at ({}, {}): {:06x} vs {:06x}",
                bad.len(),
                i % frame.width as usize,
                i / frame.width as usize,
                shot[i],
                frame.pixels[i]
            ),
        }
    }

    fn client_size(hwnd: HWND) -> (i32, i32) {
        let mut r = RECT::default();
        // SAFETY: a valid out pointer.
        unsafe { GetClientRect(hwnd, &mut r).unwrap() };
        (r.right, r.bottom)
    }

    fn client_origin(hwnd: HWND) -> POINT {
        let mut p = POINT::default();
        // SAFETY: a valid in/out pointer.
        let _ = unsafe { ClientToScreen(hwnd, &mut p) };
        p
    }

    fn take_inputs(log: &Rc<RefCell<Log>>) -> Vec<PlatformInput> {
        std::mem::take(&mut log.borrow_mut().inputs)
    }

    fn take_text(log: &Rc<RefCell<Log>>) -> Vec<String> {
        std::mem::take(&mut log.borrow_mut().text)
    }

    /// The window's shared parts the frame callback needs.
    struct Frame {
        buffer: Framebuffer,
        seed: u32,
        /// Keep asking for frames (an animation) while this is above zero.
        animate: usize,
        /// Present only this rectangle next time (else everything).
        damage: Option<Bounds<DevicePixels>>,
    }

    pub fn main() {
        // SAFETY: set before any other thread exists; the backend reads it once.
        unsafe { std::env::set_var("FOLO_UI_OFFSCREEN", "1") };
        let quick = std::env::args().any(|a| a == "--quick");
        let started_ms = ms_since_process_start();
        let platform = Rc::new(WindowsPlatform::new(Arc::new(NoText)).expect("platform"));
        let checks = Rc::new(RefCell::new(Checks { fails: 0 }));
        let quit_ran = Rc::new(Cell::new(false));
        platform.on_quit(Box::new({
            let quit_ran = quit_ran.clone();
            move || quit_ran.set(true)
        }));
        let p = platform.clone();
        let c = checks.clone();
        platform.run(Box::new(move || run_checks(&p, &mut c.borrow_mut(), quick, started_ms)));
        let mut checks = checks.borrow_mut();
        checks.check("run returned after the last window closed; on_quit ran", quit_ran.get(), "");
        let fails = checks.fails;
        drop(checks);
        drop(platform);
        println!("{}", if fails == 0 { "all checks passed".to_string() } else { format!("{fails} check(s) failed") });
        std::process::exit(if fails == 0 { 0 } else { 1 });
    }

    fn run_checks(platform: &Rc<WindowsPlatform>, checks: &mut Checks, quick: bool, started_ms: f64) {
        let fg0 = foreground();
        let log = Rc::new(RefCell::new(Log::default()));
        let primary = platform.primary_display().expect("display");
        let bounds = Bounds::new(primary.bounds().origin + point(px(100.), px(100.)), size(px(960.), px(620.)));
        let params = WindowParams {
            bounds,
            titlebar: Some(TitlebarOptions { title: Some("win_smoke".into()), appears_transparent: true, traffic_light_position: None }),
            kind: WindowKind::Normal,
            is_movable: true,
            is_resizable: true,
            is_minimizable: true,
            focus: false,
            show: true,
            display_id: Some(primary.id()),
            window_min_size: Some(size(px(320.), px(200.))),
            app_id: None,
        };
        let window: Rc<RefCell<WindowsWindow>> = Rc::new(RefCell::new(platform.open_windows_window(params).expect("window")));
        let hwnd = window.borrow().hwnd();
        let scale = window.borrow().scale_factor();
        let frame = Rc::new(RefCell::new(Frame { buffer: Framebuffer::default(), seed: 1, animate: 0, damage: None }));
        let first_present_ms = Rc::new(Cell::new(f64::NAN));
        // The core's part, faked: draw a pattern of the client size and present it.
        {
            let w = window.borrow();
            let (log2, frame2, window2, first) = (log.clone(), frame.clone(), Rc::downgrade(&window), first_present_ms.clone());
            w.on_request_frame(Box::new(move |options| {
                let mut l = log2.borrow_mut();
                l.frames.push((Instant::now(), options));
                if l.in_resize {
                    l.frame_inside_resize += 1;
                }
                drop(l);
                let Some(window) = window2.upgrade() else { return };
                let mut f = frame2.borrow_mut();
                let (cw, ch) = client_size(hwnd);
                if f.buffer.width != cw as u32 || f.buffer.height != ch as u32 {
                    f.buffer.resize(cw as u32, ch as u32);
                    let seed = f.seed;
                    pattern(&mut f.buffer, seed);
                }
                let full = [Bounds::new(point(DevicePixels(0), DevicePixels(0)), size(DevicePixels(cw), DevicePixels(ch)))];
                let damage = f.damage.take().map(|d| [d]).unwrap_or(full);
                window.borrow_mut().present(&f.buffer, &damage);
                if first.get().is_nan() {
                    first.set(ms_since_process_start());
                }
                if f.animate > 0 {
                    f.animate -= 1;
                    window.borrow().request_frame();
                }
            }));
            let l = log.clone();
            w.on_input(Box::new(move |input| {
                // Ctrl+K is "handled by the app"; everything else propagates.
                let handled = matches!(&input, PlatformInput::KeyDown(k) if k.keystroke.key == "k" && k.keystroke.modifiers.control);
                l.borrow_mut().inputs.push(input);
                DispatchEventResult { propagate: !handled, default_prevented: false }
            }));
            let l = log.clone();
            w.on_resize(Box::new(move |s, f| {
                l.borrow_mut().resizes.push((s, f));
                l.borrow_mut().in_resize = true;
            }));
            let l = log.clone();
            w.on_hover_status_change(Box::new(move |h| l.borrow_mut().hover.push(h)));
            // The app's caption areas: the top 4 px resize, then a 40 px title bar with three buttons.
            let l = log.clone();
            w.on_hit_test_window_control(Box::new(move || {
                let l = l.borrow();
                let p = l.inputs.iter().rev().find_map(|i| match i {
                    PlatformInput::MouseMove(m) => Some(m.position),
                    _ => None,
                })?;
                let (x, y) = (p.x / px(1.), p.y / px(1.));
                match (x, y) {
                    (_, y) if !(4. ..40.).contains(&y) => None,
                    (x, _) if (800. ..860.).contains(&x) => Some(WindowControlArea::Min),
                    (x, _) if (860. ..920.).contains(&x) => Some(WindowControlArea::Max),
                    (x, _) if x >= 920. => Some(WindowControlArea::Close),
                    _ => Some(WindowControlArea::Drag),
                }
            }));
            w.on_should_close(Box::new(|| true));
        }
        window.borrow_mut().set_input_handler(Box::new(Handler(log.clone())));

        // ---- First frame.
        let presented = pump_until(Duration::from_secs(5), || !first_present_ms.get().is_nan());
        let since_open = first_present_ms.get() - started_ms;
        checks.check("first frame presented", presented, format!("{:.1} ms after process start ({since_open:.1} ms after main)", first_present_ms.get()));
        let r = rect(hwnd);
        checks.check("window is outside every monitor", off_every_monitor(r), format!("({}, {}) - ({}, {})", r.left, r.top, r.right, r.bottom));
        let (cw, ch) = client_size(hwnd);
        let want = (960. * scale).round() as i32;
        checks.check("client area is 960x620 logical", cw == want && ch == (620. * scale).round() as i32, format!("{cw}x{ch} device px at scale {scale}"));
        checks.check("not activated, foreground unchanged", foreground() == fg0 && !window.borrow().is_active(), "");
        checks.check("window_bounds is the requested place", window.borrow().window_bounds() == WindowBounds::Windowed(window.borrow().window_bounds().get_bounds()) && (window.borrow().window_bounds().get_bounds().size.width / px(1.) - 960.).abs() < 0.01, format!("{:?}", window.borrow().window_bounds()));

        // ---- PrintWindow equals what was presented, after a full and a partial present.
        let shot = print_client(hwnd, cw, ch);
        let same = shot == frame.borrow().buffer.pixels;
        let diff = mismatch(&shot, &frame.borrow().buffer);
        checks.check("PrintWindow(3) of the client equals the presented frame", same, format!("{} px{diff}", shot.len()));
        {
            let mut f = frame.borrow_mut();
            f.seed = 99;
            let mut next = f.buffer.clone();
            pattern(&mut next, 99);
            // Only the 64x64 square at (128, 64) changes and is presented.
            for y in 64..128 {
                for x in 128..192 {
                    let v = next.pixel(x, y);
                    f.buffer.set_pixel(x, y, v);
                }
            }
            f.damage = Some(Bounds::new(point(DevicePixels(128), DevicePixels(64)), size(DevicePixels(64), DevicePixels(64))));
        }
        window.borrow().request_frame();
        let n = log.borrow().frames.len();
        pump_until(Duration::from_secs(2), || log.borrow().frames.len() > n);
        let shot = print_client(hwnd, cw, ch);
        checks.check("a 64x64 damage present shows up, the rest is kept", shot == frame.borrow().buffer.pixels, "");

        hit_tests(checks, hwnd, scale, &log);
        mouse(checks, hwnd, scale, &log);
        keys(checks, hwnd, &log);
        resize(checks, hwnd, scale, &log, &window);
        modes(checks, hwnd, &window, fg0);
        pacing(checks, &window, &log, &frame);
        dispatcher(checks, platform);
        if !quick {
            idle(checks, &log);
        }
        timings(checks, hwnd, &window, &frame);
        checks.check("still not activated, foreground unchanged", foreground() == fg0, format!("{:?} (was {fg0:?}, window {hwnd:?})", foreground()));

        // ---- Close: WM_CLOSE asks should_close, destroys the window, and the loop ends.
        post(hwnd, WM_CLOSE, 0, 0);
    }

    fn hit_tests(checks: &mut Checks, hwnd: HWND, scale: f32, log: &Rc<RefCell<Log>>) {
        let o = client_origin(hwnd);
        let r = rect(hwnd);
        let at = |x: f32, y: f32| (o.x + (x * scale) as i32, o.y + (y * scale) as i32);
        let cases: [(&str, (i32, i32), u32); 9] = [
            ("title bar (Drag)", at(400., 20.), HTCAPTION),
            ("Minimize", at(830., 20.), HTMINBUTTON),
            ("Maximize", at(890., 20.), HTMAXBUTTON),
            ("Close", at(940., 20.), HTCLOSE),
            ("the page", at(400., 300.), HTCLIENT),
            ("the top edge strip", (o.x + (400. * scale) as i32, o.y + 1), HTTOP),
            ("the left edge", (r.left + 3, (r.top + r.bottom) / 2), HTLEFT),
            ("the right edge", (r.right - 3, (r.top + r.bottom) / 2), HTRIGHT),
            ("the bottom edge", ((r.left + r.right) / 2, r.bottom - 3), HTBOTTOM),
        ];
        for (name, (x, y), want) in cases {
            post(hwnd, WM_NCMOUSEMOVE, HTCLIENT as usize, pt(x, y));
            pump();
            let got = send(hwnd, WM_NCHITTEST, 0, pt(x, y)) as u32;
            checks.check(&format!("WM_NCHITTEST over {name}"), got == want, format!("got {got}, want {want}"));
        }
        take_inputs(log);
    }

    fn mouse(checks: &mut Checks, hwnd: HWND, scale: f32, log: &Rc<RefCell<Log>>) {
        let lp = |x: f32, y: f32| pt((x * scale) as i32, (y * scale) as i32);
        let near = |p: Point<Pixels>, x: f32, y: f32| (p.x / px(1.) - x).abs() < 1. && (p.y / px(1.) - y).abs() < 1.;
        set_keys(&[]);
        post(hwnd, WM_MOUSEMOVE, 0, lp(150., 90.));
        pump();
        let got = take_inputs(log);
        let ok = matches!(got.as_slice(), [PlatformInput::MouseMove(m)] if near(m.position, 150., 90.) && m.pressed_button.is_none());
        checks.check("mouse move in logical px", ok, format!("{got:?}"));
        checks.check("hover reported", log.borrow().hover.first() == Some(&true), format!("{:?}", log.borrow().hover));

        // Two quick left clicks: counts 1 then 2, with capture while pressed.
        post(hwnd, WM_LBUTTONDOWN, 1, lp(200., 100.));
        pump();
        // SAFETY: a plain query.
        let captured = unsafe { GetCapture() } == hwnd;
        post(hwnd, WM_MOUSEMOVE, 1, lp(201., 100.));
        post(hwnd, WM_LBUTTONUP, 0, lp(201., 100.));
        post(hwnd, WM_LBUTTONDOWN, 1, lp(201., 101.));
        post(hwnd, WM_LBUTTONUP, 0, lp(201., 101.));
        pump();
        // SAFETY: a plain query.
        let released = unsafe { GetCapture() }.is_invalid();
        let got = take_inputs(log);
        let counts: Vec<(String, usize)> = got
            .iter()
            .filter_map(|i| match i {
                PlatformInput::MouseDown(d) => Some((format!("down {:?}", d.button), d.click_count)),
                PlatformInput::MouseUp(u) => Some((format!("up {:?}", u.button), u.click_count)),
                _ => None,
            })
            .collect();
        let want = vec![("down Left".into(), 1), ("up Left".into(), 1), ("down Left".into(), 2), ("up Left".into(), 2)];
        checks.check("left double click counts 1, 2", counts == want, format!("{counts:?}"));
        let drag = got.iter().any(|i| matches!(i, PlatformInput::MouseMove(m) if m.pressed_button == Some(MouseButton::Left) && near(m.position, 201., 100.)));
        checks.check("move while pressed reports the button", drag, "");
        checks.check("capture while pressed, released after", captured && released, "");

        // Right, middle and the navigation buttons; Shift held.
        set_keys(&[VK_SHIFT, VK_LSHIFT]);
        post(hwnd, WM_RBUTTONDOWN, 2, lp(10., 20.));
        post(hwnd, WM_RBUTTONUP, 0, lp(10., 20.));
        post(hwnd, WM_MBUTTONDOWN, 0x10, lp(11., 20.));
        post(hwnd, WM_MBUTTONUP, 0, lp(11., 20.));
        post(hwnd, WM_XBUTTONDOWN, 1 << 16, lp(12., 20.));
        post(hwnd, WM_XBUTTONUP, 1 << 16, lp(12., 20.));
        post(hwnd, WM_XBUTTONDOWN, 2 << 16, lp(13., 20.));
        post(hwnd, WM_XBUTTONUP, 2 << 16, lp(13., 20.));
        pump();
        let got = take_inputs(log);
        let downs: Vec<(MouseButton, bool, usize)> = got
            .iter()
            .filter_map(|i| match i {
                PlatformInput::MouseDown(d) => Some((d.button, d.modifiers.shift, d.click_count)),
                _ => None,
            })
            .collect();
        let want = vec![
            (MouseButton::Right, true, 1),
            (MouseButton::Middle, true, 1),
            (MouseButton::Navigate(NavigationDirection::Back), true, 1),
            (MouseButton::Navigate(NavigationDirection::Forward), true, 1),
        ];
        checks.check("right, middle, back, forward with shift", downs == want, format!("{downs:?}"));

        // Wheel: lines, by the system setting; shift makes it horizontal; WM_MOUSEHWHEEL is horizontal.
        let mut lines = 3u32;
        let mut chars = 3u32;
        // SAFETY: u32 parameters.
        unsafe {
            let _ = SystemParametersInfoW(SPI_GETWHEELSCROLLLINES, 0, Some(&mut lines as *mut u32 as *mut _), SYSTEM_PARAMETERS_INFO_UPDATE_FLAGS(0));
            let _ = SystemParametersInfoW(SPI_GETWHEELSCROLLCHARS, 0, Some(&mut chars as *mut u32 as *mut _), SYSTEM_PARAMETERS_INFO_UPDATE_FLAGS(0));
        }
        let o = client_origin(hwnd);
        let screen = pt(o.x + (300. * scale) as i32, o.y + (200. * scale) as i32);
        set_keys(&[]);
        post(hwnd, WM_MOUSEWHEEL, (120usize) << 16, screen);
        pump();
        set_keys(&[VK_SHIFT, VK_LSHIFT]);
        post(hwnd, WM_MOUSEWHEEL, ((-120i16 as u16) as usize) << 16, screen);
        pump();
        set_keys(&[]);
        post(hwnd, WM_MOUSEHWHEEL, (120usize) << 16, screen);
        pump();
        let got = take_inputs(log);
        let deltas: Vec<(f32, f32, bool)> = got
            .iter()
            .filter_map(|i| match i {
                PlatformInput::ScrollWheel(w) if near(w.position, 300., 200.) => match w.delta {
                    ScrollDelta::Lines(d) => Some((d.x, d.y, w.modifiers.shift)),
                    ScrollDelta::Pixels(_) => None,
                },
                _ => None,
            })
            .collect();
        let want = vec![(0., lines as f32, false), (-(chars as f32), 0., true), (-(chars as f32), 0., false)];
        checks.check("wheel lines (vertical, shift horizontal, horizontal)", deltas == want, format!("{deltas:?}, want {want:?}"));
    }

    fn keys(checks: &mut Checks, hwnd: HWND, log: &Rc<RefCell<Log>>) {
        take_inputs(log);
        take_text(log);
        // (name, keys held, virtual key, key, shift, ctrl, key_char, text inserted)
        let cases: [(&str, &[VIRTUAL_KEY], VIRTUAL_KEY, &str, bool, bool, Option<&str>, &[&str]); 10] = [
            ("a", &[], VK_A, "a", false, false, Some("a"), &["replace:a"]),
            ("A", &[VK_SHIFT, VK_LSHIFT], VK_A, "a", true, false, Some("A"), &["replace:A"]),
            ("Enter", &[], VK_RETURN, "enter", false, false, None, &[]),
            ("Ctrl+K (handled: no text)", &[VK_CONTROL, VK_LCONTROL], VK_K, "k", false, true, None, &[]),
            ("Shift+Tab", &[VK_SHIFT, VK_LSHIFT], VK_TAB, "tab", true, false, None, &[]),
            ("F11", &[], VK_F11, "f11", false, false, None, &[]),
            ("Left", &[], VK_LEFT, "left", false, false, None, &[]),
            ("Up", &[], VK_UP, "up", false, false, None, &[]),
            ("Right", &[], VK_RIGHT, "right", false, false, None, &[]),
            ("Down", &[], VK_DOWN, "down", false, false, None, &[]),
        ];
        for (name, held, vk, key, shift, ctrl, key_char, text) in cases {
            set_keys(held);
            post(hwnd, WM_KEYDOWN, vk.0 as usize, key_lparam(vk, false, false));
            pump();
            post(hwnd, WM_KEYUP, vk.0 as usize, key_lparam(vk, false, true));
            pump();
            let got = take_inputs(log);
            let inserted = take_text(log);
            let down = got.iter().find_map(|i| match i {
                PlatformInput::KeyDown(k) => Some(k.clone()),
                _ => None,
            });
            let up = got.iter().any(|i| matches!(i, PlatformInput::KeyUp(k) if k.keystroke.key == key));
            let ok = down.as_ref().is_some_and(|d| {
                d.keystroke.key == key
                    && d.keystroke.modifiers.shift == shift
                    && d.keystroke.modifiers.control == ctrl
                    && d.keystroke.key_char.as_deref() == key_char
                    && !d.is_held
            }) && up
                && inserted == text;
            checks.check(&format!("key {name}"), ok, format!("{:?}, text {inserted:?}", down.map(|d| d.keystroke)));
        }
        // Auto-repeat: the previous-state bit makes is_held.
        set_keys(&[]);
        post(hwnd, WM_KEYDOWN, VK_LEFT.0 as usize, key_lparam(VK_LEFT, true, false));
        pump();
        let got = take_inputs(log);
        checks.check("key repeat is_held", matches!(got.as_slice(), [PlatformInput::KeyDown(k)] if k.is_held), format!("{got:?}"));
        // A modifier press reports the change once.
        set_keys(&[VK_SHIFT, VK_LSHIFT]);
        post(hwnd, WM_KEYDOWN, VK_SHIFT.0 as usize, key_lparam(VK_SHIFT, false, false));
        post(hwnd, WM_KEYDOWN, VK_SHIFT.0 as usize, key_lparam(VK_SHIFT, true, false));
        pump();
        set_keys(&[]);
        post(hwnd, WM_KEYUP, VK_SHIFT.0 as usize, key_lparam(VK_SHIFT, false, true));
        pump();
        let got = take_inputs(log);
        let changes: Vec<bool> = got
            .iter()
            .filter_map(|i| match i {
                PlatformInput::ModifiersChanged(m) => Some(m.modifiers.shift),
                _ => None,
            })
            .collect();
        checks.check("modifiers changed: shift on (once), off", changes == [true, false], format!("{changes:?}"));
        // A surrogate pair in two WM_CHARs is one character.
        post(hwnd, WM_CHAR, 0xD83D, 1);
        post(hwnd, WM_CHAR, 0xDE00, 1);
        pump();
        let text = take_text(log);
        checks.check("WM_CHAR surrogate pair", text == ["replace:\u{1F600}"], format!("{text:?}"));
        set_keys(&[]);
    }

    fn resize(checks: &mut Checks, hwnd: HWND, scale: f32, log: &Rc<RefCell<Log>>, window: &Rc<RefCell<WindowsWindow>>) {
        let r = rect(hwnd);
        log.borrow_mut().resizes.clear();
        log.borrow_mut().frame_inside_resize = 0;
        let grow = (100. * scale) as i32;
        // SAFETY: resizing our own window without activating it.
        unsafe {
            SetWindowPos(hwnd, None, 0, 0, r.right - r.left + grow, r.bottom - r.top + grow, SWP_NOMOVE | SWP_NOZORDER | SWP_NOACTIVATE).unwrap();
        }
        // No pumping: both must have happened inside SetWindowPos.
        let l = log.borrow();
        let ok = l.resizes.last().is_some_and(|(s, f)| (s.width / px(1.) - 1060.).abs() < 1. && (s.height / px(1.) - 720.).abs() < 1. && *f == scale);
        checks.check("resize reports the new logical size and scale", ok, format!("{:?}", l.resizes));
        let expose = l.frames.last().is_some_and(|(_, o)| o.require_presentation);
        checks.check("resize draws synchronously (require_presentation)", l.frame_inside_resize >= 1 && expose, format!("{} frame(s) inside", l.frame_inside_resize));
        drop(l);
        log.borrow_mut().in_resize = false;
        let (cw, ch) = client_size(hwnd);
        checks.check("content_size follows", window.borrow().content_size() == size(px(cw as f32 / scale), px(ch as f32 / scale)), "");
        // SAFETY: back to the original size.
        unsafe { SetWindowPos(hwnd, None, 0, 0, r.right - r.left, r.bottom - r.top, SWP_NOMOVE | SWP_NOZORDER | SWP_NOACTIVATE).unwrap() };
        log.borrow_mut().in_resize = false;
        pump();
    }

    fn modes(checks: &mut Checks, hwnd: HWND, window: &Rc<RefCell<WindowsWindow>>, fg0: HWND) {
        let restored = window.borrow().window_bounds();
        let b = restored.get_bounds();
        window.borrow().zoom();
        let maxed = pump_until(Duration::from_secs(3), || window.borrow().is_maximized());
        let wb = window.borrow().window_bounds();
        checks.check("zoom maximizes", maxed, "");
        checks.check("maximized window_bounds keeps the restored bounds", wb == WindowBounds::Maximized(b), format!("{wb:?}"));
        let (r, fg) = (rect(hwnd), foreground());
        checks.check("maximized stays off screen, foreground unchanged", off_every_monitor(r) && fg == fg0, format!("({}, {}) - ({}, {}), foreground {fg:?} (was {fg0:?}, window {hwnd:?})", r.left, r.top, r.right, r.bottom));
        window.borrow().zoom();
        let back = pump_until(Duration::from_secs(3), || !window.borrow().is_maximized());
        pump_until(Duration::from_millis(200), || false);
        let wb = window.borrow().window_bounds();
        checks.check("zoom again restores, window_bounds windowed as before", back && wb == restored, format!("{wb:?}"));

        window.borrow().toggle_fullscreen();
        let full = pump_until(Duration::from_secs(3), || window.borrow().is_fullscreen());
        pump_until(Duration::from_millis(200), || false);
        let display = window.borrow().display().unwrap().bounds();
        let wb = window.borrow().window_bounds();
        let content = window.borrow().content_size();
        checks.check("full screen covers the display's size", full && (content.width / px(1.) - display.size.width / px(1.)).abs() < 1., format!("{content:?} vs {:?}", display.size));
        checks.check("full screen window_bounds keeps the restored bounds", wb == WindowBounds::Fullscreen(b), format!("{wb:?}"));
        let (r, fg) = (rect(hwnd), foreground());
        checks.check("full screen stays off screen, foreground unchanged", off_every_monitor(r) && fg == fg0, format!("({}, {}) - ({}, {}), foreground {fg:?} (was {fg0:?}, window {hwnd:?})", r.left, r.top, r.right, r.bottom));
        window.borrow().toggle_fullscreen();
        let left = pump_until(Duration::from_secs(3), || !window.borrow().is_fullscreen());
        pump_until(Duration::from_millis(200), || false);
        let wb = window.borrow().window_bounds();
        checks.check("leaving full screen restores style and place", left && wb == restored && window.borrow().content_size() == b.size, format!("{wb:?}"));
    }

    fn pacing(checks: &mut Checks, window: &Rc<RefCell<WindowsWindow>>, log: &Rc<RefCell<Log>>, frame: &Rc<RefCell<Frame>>) {
        // Coalescing: many requests before the next refresh are one frame.
        pump_until(Duration::from_millis(100), || false);
        let n = log.borrow().frames.len();
        for _ in 0..50 {
            window.borrow().request_frame();
        }
        pump_until(Duration::from_millis(150), || false);
        let got = log.borrow().frames.len() - n;
        checks.check("50 requests before a refresh draw once", got == 1, format!("{got} frame(s)"));

        // An animation: each frame asks for the next; ticks follow the compositor.
        let n = log.borrow().frames.len();
        frame.borrow_mut().animate = 240;
        window.borrow().request_frame();
        pump_until(Duration::from_secs(5), || frame.borrow().animate == 0);
        pump_until(Duration::from_millis(50), || false);
        let l = log.borrow();
        let times: Vec<Instant> = l.frames[n..].iter().map(|(t, _)| *t).collect();
        let intervals: Vec<f64> = times.windows(2).map(|w| (w[1] - w[0]).as_secs_f64() * 1000.).collect();
        let mean = intervals.iter().sum::<f64>() / intervals.len().max(1) as f64;
        let mut sorted = intervals.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = sorted.get(sorted.len() / 2).copied().unwrap_or(0.);
        let min = sorted.first().copied().unwrap_or(0.);
        let period = refresh_ms();
        checks.check(
            "animation ticks at the display refresh",
            (median - period).abs() < period * 0.25 && min > period * 0.4,
            format!("{} frames, median {median:.2} ms, mean {mean:.2} ms, min {min:.2} ms; refresh {period:.2} ms ({:.1} Hz)", times.len(), 1000. / period),
        );
    }

    fn refresh_ms() -> f64 {
        use ::windows::Win32::Graphics::Dwm::{DWM_TIMING_INFO, DwmGetCompositionTimingInfo};
        let mut info = DWM_TIMING_INFO { cbSize: std::mem::size_of::<DWM_TIMING_INFO>() as u32, ..Default::default() };
        // SAFETY: a sized DWM_TIMING_INFO.
        let _ = unsafe { DwmGetCompositionTimingInfo(HWND::default(), &mut info) };
        let (n, d) = (info.rateRefresh.uiNumerator as f64, info.rateRefresh.uiDenominator as f64);
        if n > 0. { 1000. * d / n } else { 16.67 }
    }

    fn dispatcher(checks: &mut Checks, platform: &Rc<WindowsPlatform>) {
        let d = platform.dispatcher();
        // 1000 background jobs.
        let done = Arc::new(AtomicUsize::new(0));
        let start = Instant::now();
        for _ in 0..1000 {
            let (done, dd) = (done.clone(), d.clone());
            let (runnable, task) = async_task::spawn(async move { done.fetch_add(1, Ordering::SeqCst); }, move |r| dd.dispatch(r));
            task.detach();
            runnable.schedule();
        }
        let ok = pump_until(Duration::from_secs(5), || done.load(Ordering::SeqCst) == 1000);
        checks.check("1000 background jobs run", ok, format!("{:.1} ms, {} pool thread(s)", start.elapsed().as_secs_f64() * 1000., platform.windows_dispatcher().pool_threads()));

        // Main-thread jobs, queued from the main thread and from a background thread, run in order.
        let order = Rc::new(RefCell::new(Vec::new()));
        for i in 0..100 {
            let (order, dd) = (order.clone(), d.clone());
            let (runnable, task) = async_task::spawn_local(async move { order.borrow_mut().push(i) }, move |r| dd.dispatch_on_main_thread(r));
            task.detach();
            runnable.schedule();
        }
        let ok = pump_until(Duration::from_secs(2), || order.borrow().len() == 100);
        checks.check("main-thread jobs run in order on the main thread", ok && *order.borrow() == (0..100).collect::<Vec<_>>() && d.is_main_thread(), "");
        let from_bg = Arc::new(parking_lot::Mutex::new(Vec::new()));
        let dd = d.clone();
        let fb = from_bg.clone();
        std::thread::spawn(move || {
            for i in 0..100 {
                let (fb, d2) = (fb.clone(), dd.clone());
                let main = dd.clone();
                let (runnable, task) = async_task::spawn(async move { fb.lock().push((i, main.is_main_thread())) }, move |r| d2.dispatch_on_main_thread(r));
                task.detach();
                runnable.schedule();
            }
        })
        .join()
        .unwrap();
        let ok = pump_until(Duration::from_secs(2), || from_bg.lock().len() == 100);
        let v = from_bg.lock().clone();
        checks.check("main-thread jobs from a background thread run in order there", ok && v.iter().enumerate().all(|(i, &(j, main))| i == j && main), "");

        // Timers: one thread, deadline order.
        let fired = Arc::new(parking_lot::Mutex::new(Vec::new()));
        let start = Instant::now();
        for (i, ms) in [(0, 45u64), (1, 15), (2, 30), (3, 60)] {
            let (fired, dd) = (fired.clone(), d.clone());
            let (runnable, task) = async_task::spawn(
                async move { fired.lock().push((i, start.elapsed())) },
                move |r| dd.dispatch_after(Duration::from_millis(ms), r),
            );
            task.detach();
            runnable.schedule();
        }
        let ok = pump_until(Duration::from_secs(2), || fired.lock().len() == 4);
        let v = fired.lock().clone();
        let order: Vec<i32> = v.iter().map(|(i, _)| *i).collect();
        let late = v.iter().zip([15u64, 30, 45, 60]).all(|((_, t), ms)| *t >= Duration::from_millis(ms));
        checks.check("timers fire in deadline order, not early", ok && order == [1, 2, 0, 3] && late, format!("{v:?}"));
    }

    fn idle(checks: &mut Checks, log: &Rc<RefCell<Log>>) {
        pump_until(Duration::from_millis(300), || false);
        let frames = log.borrow().frames.len();
        let cpu0 = process_cpu_time();
        let t0 = Instant::now();
        let mut messages = 0usize;
        // Block in the message wait for 5 s, dispatching whatever arrives (nothing should).
        while t0.elapsed() < Duration::from_secs(5) {
            let left = Duration::from_secs(5).saturating_sub(t0.elapsed());
            // SAFETY: waiting on this thread's queue.
            unsafe { MsgWaitForMultipleObjects(None, false, left.as_millis() as u32, QS_ALLINPUT) };
            let mut msg = MSG::default();
            // SAFETY: as in `pump`.
            unsafe {
                while PeekMessageW(&mut msg, None, 0, 0, PM_REMOVE).as_bool() {
                    messages += 1;
                    DispatchMessageW(&msg);
                }
            }
        }
        let cpu = process_cpu_time() - cpu0;
        let frames = log.borrow().frames.len() - frames;
        checks.check(
            "idle window is idle (5 s, no input)",
            cpu < Duration::from_millis(20) && frames == 0,
            format!("process CPU {:.2} ms over {:.1} s (GetProcessTimes), {messages} message(s), {frames} frame(s)", cpu.as_secs_f64() * 1000., t0.elapsed().as_secs_f64()),
        );
    }

    fn timings(checks: &mut Checks, hwnd: HWND, window: &Rc<RefCell<WindowsWindow>>, frame: &Rc<RefCell<Frame>>) {
        let r0 = rect(hwnd);
        let (bw, bh) = {
            let (cw, ch) = client_size(hwnd);
            (r0.right - r0.left - cw, r0.bottom - r0.top - ch)
        };
        let mut lines = Vec::new();
        for (w, h) in [(1920, 1200), (3840, 2160)] {
            // SAFETY: resizing our own window without activating it.
            unsafe { SetWindowPos(hwnd, None, 0, 0, w + bw, h + bh, SWP_NOMOVE | SWP_NOZORDER | SWP_NOACTIVATE).unwrap() };
            pump();
            let (cw, ch) = client_size(hwnd);
            let mut fb = Framebuffer::new(cw as u32, ch as u32);
            pattern(&mut fb, 7);
            let full = [Bounds::new(point(DevicePixels(0), DevicePixels(0)), size(DevicePixels(cw), DevicePixels(ch)))];
            let small = [Bounds::new(point(DevicePixels(200), DevicePixels(200)), size(DevicePixels(64), DevicePixels(64)))];
            let time = |damage: &[Bounds<DevicePixels>], n: usize| {
                let mut v: Vec<f64> = (0..n)
                    .map(|_| {
                        let t = Instant::now();
                        window.borrow_mut().present(&fb, damage);
                        // SAFETY: no arguments.
                        let _ = unsafe { GdiFlush() };
                        t.elapsed().as_secs_f64() * 1000.
                    })
                    .collect();
                v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                v[n / 2]
            };
            time(&full, 3);
            let full_ms = time(&full, 30);
            let small_ms = time(&small, 200);
            let same = print_client(hwnd, cw, ch) == fb.pixels;
            checks.check(&format!("present {cw}x{ch} matches PrintWindow"), same && cw == w && ch == h, format!("client {cw}x{ch}"));
            lines.push(format!("present {cw}x{ch} full: {full_ms:.3} ms, 64x64: {small_ms:.4} ms (median)"));
        }
        for l in lines {
            println!("      {l}");
        }
        // SAFETY: back to the original size; the frame callback repaints.
        unsafe { SetWindowPos(hwnd, None, 0, 0, r0.right - r0.left, r0.bottom - r0.top, SWP_NOMOVE | SWP_NOZORDER | SWP_NOACTIVATE).unwrap() };
        frame.borrow_mut().buffer = Framebuffer::default();
        pump();
    }
}
