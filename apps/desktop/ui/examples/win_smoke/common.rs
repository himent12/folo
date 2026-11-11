//! The pieces every check shares: the fake core's records, the fake input handler, the message
//! pump, posting helpers and captures.

use ::windows::Win32::Foundation::*;
use ::windows::Win32::Graphics::Gdi::*;
use ::windows::Win32::Storage::Xps::{PRINT_WINDOW_FLAGS, PrintWindow};
use ::windows::Win32::UI::Input::KeyboardAndMouse::*;
use ::windows::Win32::UI::WindowsAndMessaging::*;
use folo_ui::*;
use std::cell::RefCell;
use std::ops::Range;
use std::rc::Rc;
use std::time::{Duration, Instant};

/// A text system that is never asked for anything (the example draws no text).
pub struct NoText;

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
    fn rasterize_glyph(
        &self,
        _: &RenderGlyphParams,
        _: Bounds<DevicePixels>,
    ) -> anyhow::Result<(Size<DevicePixels>, Vec<u8>)> {
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
pub struct Log {
    pub inputs: Vec<PlatformInput>,
    pub text: Vec<String>,
    pub resizes: Vec<(Size<Pixels>, f32)>,
    pub frames: Vec<(Instant, RequestFrameOptions)>,
    /// A frame request arrived while a resize was being reported (the synchronous draw).
    pub frame_inside_resize: usize,
    pub in_resize: bool,
    pub hover: Vec<bool>,
}

pub struct Checks {
    pub fails: usize,
}

impl Checks {
    pub fn check(&mut self, name: &str, ok: bool, detail: impl std::fmt::Display) {
        if ok {
            println!("PASS  {name}  {detail}");
        } else {
            self.fails += 1;
            println!("FAIL  {name}  {detail}");
        }
    }
}

/// The fake input handler: records the text calls.
pub struct Handler(pub Rc<RefCell<Log>>);

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
    fn replace_and_mark_text_in_range(
        &mut self,
        _: Option<Range<usize>>,
        text: &str,
        _: Option<Range<usize>>,
    ) {
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
pub fn pump_until(timeout: Duration, mut done: impl FnMut() -> bool) -> bool {
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

pub fn pump() {
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

pub fn post(hwnd: HWND, msg: u32, w: usize, l: isize) {
    // SAFETY: posting to our own window.
    unsafe { PostMessageW(Some(hwnd), msg, WPARAM(w), LPARAM(l)).unwrap() };
}

pub fn send(hwnd: HWND, msg: u32, w: usize, l: isize) -> isize {
    // SAFETY: our window, on its own thread: a direct call of the window procedure.
    unsafe { SendMessageW(hwnd, msg, Some(WPARAM(w)), Some(LPARAM(l))).0 }
}

pub fn pt(x: i32, y: i32) -> isize {
    ((y as i16 as u16 as isize) << 16) | (x as i16 as u16 as isize)
}

/// Sets this thread's key state (what GetKeyState reads): no real input is injected.
pub fn set_keys(down: &[VIRTUAL_KEY]) {
    let mut state = [0u8; 256];
    for k in down {
        state[k.0 as usize] = 0x80;
    }
    // SAFETY: a 256-byte key state for this thread.
    unsafe { SetKeyboardState(&state).unwrap() };
}

pub fn key_lparam(vk: VIRTUAL_KEY, held: bool, up: bool) -> isize {
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

pub fn rect(hwnd: HWND) -> RECT {
    let mut r = RECT::default();
    // SAFETY: a valid out pointer.
    unsafe { GetWindowRect(hwnd, &mut r).unwrap() };
    r
}

pub fn off_every_monitor(r: RECT) -> bool {
    // SAFETY: a plain query.
    unsafe { MonitorFromRect(&r, MONITOR_DEFAULTTONULL) }.is_invalid()
}

pub fn foreground() -> HWND {
    // SAFETY: a plain query.
    unsafe { GetForegroundWindow() }
}

/// A pattern that differs per pixel and per frame: colour bands, a grid and a frame number.
pub fn pattern(fb: &mut Framebuffer, seed: u32) {
    for y in 0..fb.height {
        for x in 0..fb.width {
            let v = if (x / 64 + y / 64) % 2 == 0 { 0x0020_3040 } else { 0x00c0_8020 };
            fb.set_pixel(x, y, v ^ ((x.wrapping_mul(7) ^ y.wrapping_mul(13) ^ seed) & 0x00ff_ffff));
        }
    }
}

/// The client area as the compositor has it (`PrintWindow` with PW_CLIENTONLY | PW_RENDERFULLCONTENT).
pub fn print_client(hwnd: HWND, width: i32, height: i32) -> Vec<u32> {
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
            std::slice::from_raw_parts(bits as *const u32, (width * height) as usize)
                .iter()
                .map(|p| p & 0x00ff_ffff)
                .collect()
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
pub fn mismatch(shot: &[u32], frame: &Framebuffer) -> String {
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

pub fn client_size(hwnd: HWND) -> (i32, i32) {
    let mut r = RECT::default();
    // SAFETY: a valid out pointer.
    unsafe { GetClientRect(hwnd, &mut r).unwrap() };
    (r.right, r.bottom)
}

pub fn client_origin(hwnd: HWND) -> POINT {
    let mut p = POINT::default();
    // SAFETY: a valid in/out pointer.
    let _ = unsafe { ClientToScreen(hwnd, &mut p) };
    p
}

pub fn take_inputs(log: &Rc<RefCell<Log>>) -> Vec<PlatformInput> {
    std::mem::take(&mut log.borrow_mut().inputs)
}

pub fn take_text(log: &Rc<RefCell<Log>>) -> Vec<String> {
    std::mem::take(&mut log.borrow_mut().text)
}

/// The window's shared parts the frame callback needs.
pub struct Frame {
    pub buffer: Framebuffer,
    pub seed: u32,
    /// Keep asking for frames (an animation) while this is above zero.
    pub animate: usize,
    /// Present only this rectangle next time (else everything).
    pub damage: Option<Bounds<DevicePixels>>,
}
