//! The Win32 platform's run loop without any window on screen (its message window is message-only):
//! main-thread tasks from another thread run in order on the loop's thread, a timer fires on the pool,
//! `quit` ends the loop and `on_quit` runs.
#![cfg(windows)]

use folo_ui::windows::WindowsPlatform;
use folo_ui::*;
use parking_lot::Mutex;
use std::borrow::Cow;
use std::cell::Cell;
use std::rc::Rc;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// A text system nothing asks for anything.
struct NoText;

impl PlatformTextSystem for NoText {
    fn add_fonts(&self, _: Vec<Cow<'static, [u8]>>) -> anyhow::Result<()> {
        Ok(())
    }
    fn all_font_names(&self) -> Vec<String> {
        Vec::new()
    }
    fn font_id(&self, _: &Font) -> anyhow::Result<FontId> {
        anyhow::bail!("no fonts")
    }
    fn font_metrics(&self, _: FontId) -> FontMetrics {
        panic!("no fonts")
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

#[test]
fn the_loop_runs_tasks_in_order_and_quits() {
    let platform = Rc::new(WindowsPlatform::new(Arc::new(NoText)).unwrap());
    let quit_ran = Rc::new(Cell::new(false));
    platform.on_quit(Box::new({
        let quit_ran = quit_ran.clone();
        move || quit_ran.set(true)
    }));
    let seen: Arc<Mutex<Vec<(String, bool)>>> = Arc::default();
    let start = Instant::now();
    let p = platform.clone();
    let s = seen.clone();
    platform.run(Box::new(move || {
        let d = p.dispatcher();
        assert!(d.is_main_thread());
        // Quits once the timer has fired: a main-thread task waiting on a channel.
        let (tx, rx) = oneshot::channel::<()>();
        let quitter = p.clone();
        let dd = d.clone();
        let (runnable, task) = async_task::spawn_local(
            async move {
                let _ = rx.await;
                quitter.quit();
            },
            move |r| dd.dispatch_on_main_thread(r),
        );
        task.detach();
        runnable.schedule();
        std::thread::spawn(move || {
            for i in 0..100 {
                let (s, main) = (s.clone(), d.clone());
                let dd = d.clone();
                let (runnable, task) = async_task::spawn(
                    async move { s.lock().push((format!("main {i}"), main.is_main_thread())) },
                    move |r| dd.dispatch_on_main_thread(r),
                );
                task.detach();
                runnable.schedule();
            }
            let dd = d.clone();
            let (s, main) = (s.clone(), d.clone());
            let (runnable, task) = async_task::spawn(
                async move {
                    s.lock().push(("timer".into(), main.is_main_thread()));
                    let _ = tx.send(());
                },
                move |r| dd.dispatch_after(Duration::from_millis(30), r),
            );
            task.detach();
            runnable.schedule();
        });
    }));
    let seen = seen.lock().clone();
    assert!(quit_ran.get(), "on_quit ran");
    assert_eq!(seen.len(), 101);
    for (i, (name, on_main)) in seen[..100].iter().enumerate() {
        assert_eq!((name.as_str(), *on_main), (format!("main {i}").as_str(), true));
    }
    assert_eq!(seen[100], ("timer".to_string(), false), "the timer runs on the pool");
    assert!(start.elapsed() >= Duration::from_millis(30) && start.elapsed() < Duration::from_secs(2));
}
