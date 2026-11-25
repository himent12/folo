//! The tile renderer on a synthetic busy frame that resembles FOLO Desktop at scale 1.5: a title bar
//! and sidebar, panels with 1 px borders and rounded corners, rows of text, a grid of thumbnails,
//! dialog shadows and line charts. Run: `cargo bench -p folo-ui --bench raster` (release).
//! `-- 1080` or `-- 4k` runs one size.

use folo_ui::raster::Renderer;
use folo_ui::*;
use std::sync::Arc;
use std::time::{Duration, Instant};

const SCALE: f32 = 1.5;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn i(&mut self, lo: i32, hi: i32) -> i32 {
        lo + (self.next() % (hi - lo) as u64) as i32
    }

    fn f(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
}

fn sp(v: f32) -> ScaledPixels {
    ScaledPixels::from(v)
}

fn rect(x: f32, y: f32, w: f32, h: f32) -> Bounds<ScaledPixels> {
    Bounds::new(point(sp(x), sp(y)), size(sp(w), sp(h)))
}

/// Glyph-like masks: a few strokes (vertical, horizontal, diagonal) with anti-aliased edges.
fn glyph_masks(rng: &mut Rng) -> Vec<Arc<AlphaMask>> {
    (0..120)
        .map(|_| {
            let (w, h) = (rng.i(5, 13), rng.i(11, 19));
            let mut d = vec![0f32; (w * h) as usize];
            for _ in 0..rng.i(1, 4) {
                let (kind, at, width) = (rng.i(0, 3), rng.f(), 1.2 + rng.f() * 1.3);
                for y in 0..h {
                    for x in 0..w {
                        let (fx, fy) = (x as f32 + 0.5, y as f32 + 0.5);
                        let dist = match kind {
                            0 => (fx - (1. + at * (w as f32 - 2.))).abs(),
                            1 => (fy - (2. + at * (h as f32 - 4.))).abs(),
                            _ => (fx - (1. + at * 2.) - (fy * (w as f32 - 4.) / h as f32)).abs(),
                        };
                        let c = (width / 2. + 0.5 - dist).clamp(0., 1.);
                        let v = &mut d[(y * w + x) as usize];
                        *v = v.max(c);
                    }
                }
            }
            let d: Vec<u8> = d.iter().map(|v| (v * 255. + 0.5) as u8).collect();
            Arc::new(AlphaMask::new(size(DevicePixels(w), DevicePixels(h)), d))
        })
        .collect()
}

fn icon_masks(rng: &mut Rng) -> Vec<Arc<AlphaMask>> {
    (0..8)
        .map(|_| {
            let s = 48;
            let d: Vec<u8> =
                (0..s * s).map(|_| if rng.i(0, 3) == 0 { 0 } else { rng.i(0, 256) as u8 }).collect();
            Arc::new(AlphaMask::new(size(DevicePixels(s), DevicePixels(s)), d))
        })
        .collect()
}

fn images(rng: &mut Rng, n: usize) -> Vec<Arc<RenderImage>> {
    (0..n)
        .map(|_| {
            let mut buf = ::image::RgbaImage::new(320, 240);
            let base = [rng.i(0, 256) as u8, rng.i(0, 256) as u8, rng.i(0, 256) as u8];
            for (x, y, p) in buf.enumerate_pixels_mut() {
                *p = ::image::Rgba([base[0] ^ (x as u8), base[1] ^ (y as u8), base[2], 255]);
            }
            Arc::new(RenderImage::new(vec![::image::Frame::new(buf)]))
        })
        .collect()
}

fn quad(
    b: Bounds<ScaledPixels>,
    clip: Bounds<ScaledPixels>,
    bg: Hsla,
    border: Option<Hsla>,
    radius: f32,
) -> Quad {
    Quad {
        bounds: b,
        clip,
        background: bg.into(),
        border_color: border.unwrap_or_default(),
        corner_radii: Corners::all(sp(radius)),
        border_widths: Edges::all(sp(if border.is_some() { SCALE } else { 0. })),
        border_style: BorderStyle::Solid,
    }
}

struct Assets {
    glyphs: Vec<Arc<AlphaMask>>,
    icons: Vec<Arc<AlphaMask>>,
    images: Vec<Arc<RenderImage>>,
    paths: Vec<Vec<(f32, f32)>>,
}

/// Pushes a run of `n` glyphs from `x` on baseline `y`.
#[allow(clippy::too_many_arguments)]
fn glyphs(
    s: &mut Scene,
    a: &Assets,
    rng: &mut Rng,
    x: f32,
    y: f32,
    n: usize,
    clip: Bounds<ScaledPixels>,
    color: Hsla,
) {
    let mut gx = x.round();
    for _ in 0..n {
        let m = a.glyphs[rng.i(0, a.glyphs.len() as i32) as usize].clone();
        let (mw, mh) = (m.size.width.0 as f32, m.size.height.0 as f32);
        s.push_mono_sprite(MonoSprite {
            bounds: rect(gx, (y - mh).round(), mw, mh),
            clip,
            color,
            mask: m,
            kind: SpriteKind::Glyph,
        });
        gx += mw - 1.;
    }
}

/// The busy frame, `w` x `h` device px: title bar, sidebar, a grid of dataset cards (thumbnail,
/// detection boxes, labels, caption) taking about two thirds of the frame, and an inspector column
/// of elevated chart panels. `scroll` moves the card grid up by that many device px; `hover` changes
/// one 200x40 button's colour.
fn busy_scene(a: &Assets, w: f32, h: f32, scroll: f32, hover: bool) -> Scene {
    let mut s = Scene::new();
    let mut rng = Rng(1234);
    let frame = rect(0., 0., w, h);
    let text =
        [hsla(0., 0., 0.92, 1.), hsla(0., 0., 0.62, 1.), hsla(0.58, 0.8, 0.65, 1.), hsla(0., 0., 0.45, 1.)];
    let panel = hsla(0.62, 0.1, 0.12, 1.);
    let border = hsla(0., 0., 1., 0.08);
    let (title_h, side_w) = (40. * SCALE, 240. * SCALE);
    let insp_w = (200. * w / 1920.).round();
    s.push_quad(quad(frame, frame, hsla(0.62, 0.12, 0.09, 1.), None, 0.));
    // Title bar: tabs and buttons.
    s.push_quad(quad(rect(0., 0., w, title_h), frame, hsla(0.62, 0.12, 0.07, 1.), None, 0.));
    for k in 0..(w / 64.) as i32 {
        let x = 300. + k as f32 * 54.;
        if x + 48. < w {
            s.push_quad(quad(rect(x, 12., 46., 36.), frame, hsla(0.62, 0.1, 0.13, 1.), None, 6.));
            glyphs(&mut s, a, &mut rng, x + 8., 36., 2, frame, text[1]);
        }
    }
    glyphs(&mut s, a, &mut rng, 20., 38., 14, frame, text[0]);
    // Sidebar: sections of rows (row background, icon, label, count badge).
    let side = rect(0., title_h, side_w, h - title_h);
    s.push_quad(quad(side, frame, hsla(0.62, 0.12, 0.08, 1.), None, 0.));
    s.push_quad(quad(rect(side_w - SCALE, title_h, SCALE, h - title_h), frame, border, None, 0.));
    let mut y = title_h + 12.;
    let mut k = 0;
    while y + 40. < h {
        if k % 6 == 0 {
            s.push_quad(quad(
                rect(8., y, side_w - 16., 6. * 42. + 30.),
                side,
                panel,
                Some(border),
                8. * SCALE,
            ));
            glyphs(&mut s, a, &mut rng, 20., y + 22., 12, side, text[3]);
            y += 30.;
        }
        s.push_quad(quad(rect(14., y, side_w - 28., 36.), side, hsla(0.62, 0.1, 0.15, 0.5), None, 6.));
        s.push_mono_sprite(MonoSprite {
            bounds: rect(22., y + 6., 24., 24.),
            clip: side,
            color: text[1],
            mask: a.icons[k % 8].clone(),
            kind: SpriteKind::Svg,
        });
        glyphs(&mut s, a, &mut rng, 56., y + 25., 22, side, text[1]);
        s.push_quad(quad(rect(side_w - 64., y + 8., 36., 20.), side, hsla(0.58, 0.4, 0.3, 1.), None, 10.));
        glyphs(&mut s, a, &mut rng, side_w - 58., y + 24., 3, side, text[0]);
        y += 42.;
        k += 1;
    }
    // The card grid (scrolls).
    let grid = rect(side_w, title_h, w - side_w - insp_w, h - title_h);
    let (cw, ch, gap) = (248., 228., 8.);
    let cols = ((w - side_w - insp_w - gap) / (cw + gap)) as i32;
    let first_row = (scroll / (ch + gap)) as i32;
    let rows = ((h - title_h) / (ch + gap)).ceil() as i32 + 1;
    for row in first_row..first_row + rows {
        for col in 0..cols {
            let mut rng = Rng((row * 1000 + col) as u64);
            let x = side_w + gap + col as f32 * (cw + gap);
            let y = title_h + gap + row as f32 * (ch + gap) - scroll;
            let card = rect(x, y, cw, ch);
            s.push_quad(quad(card, grid, panel, Some(border), 8. * SCALE));
            s.push_image(Image {
                bounds: rect(x + 4., y + 4., 240., 180.),
                clip: grid,
                corner_radii: Corners::all(sp(6.)),
                image: a.images[(row * cols + col) as usize % a.images.len()].clone(),
                frame: 0,
                grayscale: false,
                opacity: 1.,
            });
            // Detection boxes with label tags.
            for _ in 0..12 {
                let (bx, by) = (x + 4. + rng.f() * 180., y + 18. + rng.f() * 120.);
                let (bw, bh) = (20. + rng.f() * 50., 20. + rng.f() * 40.);
                let color = hsla(rng.f(), 0.8, 0.55, 1.);
                s.push_quad(quad(rect(bx, by, bw, bh), grid, transparent_black(), Some(color), 2.));
                s.push_quad(quad(rect(bx, by - 14., 40., 14.), grid, color, None, 0.));
                glyphs(&mut s, a, &mut rng, bx + 2., by - 1., 6, grid, hsla(0., 0., 0.05, 1.));
            }
            // Caption, chips, separators.
            s.push_quad(quad(rect(x + 1., y + 188., cw - 2., SCALE), grid, border, None, 0.));
            glyphs(&mut s, a, &mut rng, x + 8., y + 206., 25, grid, text[0]);
            for c in 0..3 {
                let cx = x + 8. + c as f32 * 60.;
                s.push_quad(quad(rect(cx, y + 210., 54., 14.), grid, hsla(0.58, 0.5, 0.3, 0.6), None, 7.));
                glyphs(&mut s, a, &mut rng, cx + 4., y + 223., 5, grid, text[2]);
            }
            s.push_quad(quad(rect(x + 1., y + ch - 3., cw - 2., SCALE), grid, border, None, 0.));
        }
    }
    // The inspector: buttons on top, then elevated chart panels with shadows.
    let ix = w - insp_w;
    let insp = rect(ix, title_h, insp_w, h - title_h);
    s.push_quad(quad(insp, frame, hsla(0.62, 0.12, 0.08, 1.), None, 0.));
    for b in 0..3 {
        let bcol = if b == 0 && hover { hsla(0.58, 0.6, 0.5, 1.) } else { hsla(0.58, 0.5, 0.4, 1.) };
        let by = title_h + 12. + b as f32 * 48.;
        s.push_quad(quad(rect(ix + 16., by, 200., 40.), insp, bcol, Some(border), 6. * SCALE));
        glyphs(&mut s, a, &mut rng, ix + 30., by + 27., 12, insp, text[0]);
    }
    let charts = ((w * h) / (1920. * 1200.) * 8.).round() as usize;
    let chart_cols = (insp_w / 200.).floor().max(1.) as usize;
    let chart_w = insp_w / chart_cols as f32 - 16.;
    let chart_h = (h - title_h - 170.) / (charts as f32 / chart_cols as f32).ceil() - 12.;
    for c in 0..charts {
        let cx = ix + 8. + (c % chart_cols) as f32 * (chart_w + 16.);
        let cy = title_h + 160. + (c / chart_cols) as f32 * (chart_h + 12.);
        if c % 4 < 3 {
            s.push_shadow(Shadow {
                bounds: rect(cx, cy + 6., chart_w, chart_h),
                clip: insp,
                corner_radii: Corners::all(sp(12.)),
                color: hsla(0., 0., 0., 0.4),
                blur_radius: sp(18.),
            });
        }
        s.push_quad(quad(rect(cx, cy, chart_w, chart_h), insp, panel, Some(border), 12.));
        glyphs(&mut s, a, &mut rng, cx + 12., cy + 22., 20, insp, text[0]);
        for g in 0..12 {
            let gy = cy + 30. + g as f32 * (chart_h - 40.) / 12.;
            s.push_quad(quad(rect(cx + 8., gy, chart_w - 16., SCALE), insp, border, None, 0.));
        }
        for bar in 0..60 {
            let bh = 4. + rng.f() * 24.;
            let bx = cx + 8. + bar as f32 * (chart_w - 16.) / 60.;
            s.push_quad(quad(
                rect(bx, cy + chart_h - 8. - bh, 2., bh),
                insp,
                hsla(0.55, 0.6, 0.45, 0.8),
                None,
                0.,
            ));
        }
        for l in 0..10 {
            glyphs(
                &mut s,
                a,
                &mut rng,
                cx + 10. + l as f32 * (chart_w - 20.) / 10.,
                cy + chart_h - 10.,
                4,
                insp,
                text[3],
            );
        }
        for l in 0..3 {
            glyphs(&mut s, a, &mut rng, cx + 12., cy + 44. + l as f32 * 20., 20, insp, text[1]);
        }
        // As the app builds a chart line: `PathBuilder` in logical px (lyon's triangles), scaled.
        let mut b = PathBuilder::stroke(px(1.5));
        for (i, &(x, y)) in a.paths[c % a.paths.len()].iter().enumerate() {
            let x = (cx + 8. + x * (chart_w - 16.)) / SCALE;
            let y = (cy + 30. + y * (chart_h - 70.)) / SCALE;
            if i == 0 {
                b.move_to(point(px(x), px(y)));
            } else {
                b.line_to(point(px(x), px(y)));
            }
        }
        let built = b.build().expect("a chart line tessellates").scale(SCALE);
        s.push_path(Path {
            bounds: built.bounds,
            clip: insp,
            path: Arc::new(built.to_path_data()),
            paint: hsla(0.55, 0.8, 0.6, 1.).into(),
        });
    }
    s
}

const WARMUP: usize = 3;

fn percentile(sorted: &[Duration], pct: usize) -> Duration {
    if sorted.is_empty() {
        return Duration::ZERO;
    }
    let pct = pct.min(100);
    let rank = (sorted.len() as u128 * pct as u128).div_ceil(100) as usize;
    sorted[rank.saturating_sub(1).min(sorted.len() - 1)]
}

fn p50_p95(mut v: Vec<Duration>) -> (Duration, Duration) {
    v.sort();
    (percentile(&v, 50), percentile(&v, 95))
}

fn median(v: Vec<Duration>) -> Duration {
    p50_p95(v).0
}

fn ms(d: Duration) -> String {
    format!("{:7.3} ms", d.as_secs_f64() * 1000.)
}

fn count(s: &Scene) -> (usize, usize, usize, usize, usize) {
    let mut c = (0, 0, 0, 0, 0);
    for p in &s.primitives {
        match p {
            Primitive::Quad(_) => c.0 += 1,
            Primitive::MonoSprite(_) => c.1 += 1,
            Primitive::Image(_) => c.2 += 1,
            Primitive::Shadow(_) => c.3 += 1,
            Primitive::Path(_) => c.4 += 1,
            _ => {}
        }
    }
    c
}

fn run(name: &str, w: u32, h: u32, a: &Assets) {
    let (wf, hf) = (w as f32, h as f32);
    let base = busy_scene(a, wf, hf, 0., false);
    let hover = busy_scene(a, wf, hf, 0., true);
    let (q, g, i, sh, p) = count(&base);
    println!(
        "\n{name}: {w}x{h} at scale {SCALE}: {} primitives ({q} quads, {g} sprites, {i} images, {sh} shadows, {p} paths)",
        base.len()
    );
    let blend = GlyphBlend::new([0.149_355_4, -0.226_006_7, 0.026_559_83, -0.034_591_47], 1.0);
    let iters = 40;
    let mut fb = Framebuffer::new(w, h);

    // Cold: empty caches (shadow masks, image copies, path masks, tables made on the way).
    let mut pool = Renderer::new();
    let t = Instant::now();
    pool.render(&base, blend, &mut fb);
    let cold = t.elapsed();
    let st = pool.stats().clone();
    println!(
        "  cold first frame (pool):     {}  (prepare {}, raster {})",
        ms(cold),
        ms(st.prepare),
        ms(st.raster)
    );

    let mut one = Renderer::with_threads(1);
    let mut fb1 = Framebuffer::new(w, h);
    one.render(&base, blend, &mut fb1);
    let full = |r: &mut Renderer, fb: &mut Framebuffer| {
        for _ in 0..WARMUP {
            r.invalidate();
            r.render(&base, blend, fb);
        }
        let mut v = Vec::with_capacity(iters);
        let mut last = None;
        for _ in 0..iters {
            r.invalidate();
            let t = Instant::now();
            r.render(&base, blend, fb);
            v.push(t.elapsed());
            last = Some(r.stats().clone());
        }
        (p50_p95(v), last.unwrap())
    };
    let ((d50, d95), st) = full(&mut one, &mut fb1);
    println!(
        "  full repaint, 1 thread:      p50 {} p95 {}  (bin+hash {}, prepare {}, raster {})",
        ms(d50),
        ms(d95),
        ms(st.bin_hash),
        ms(st.prepare),
        ms(st.raster)
    );
    let ((d50, d95), st) = full(&mut pool, &mut fb);
    println!(
        "  full repaint, pool ({}):      p50 {} p95 {}  (bin+hash {}, prepare {}, raster {}; {} tiles)",
        st.threads_used,
        ms(d50),
        ms(d95),
        ms(st.bin_hash),
        ms(st.prepare),
        ms(st.raster),
        st.dirty_tiles
    );
    assert_eq!(fb.pixels, fb1.pixels, "pool and single thread differ");

    // Unchanged.
    for _ in 0..WARMUP {
        pool.render(&base, blend, &mut fb);
    }
    let mut v = Vec::with_capacity(iters * 5);
    let (mut bh, mut rs, mut bi) =
        (Vec::with_capacity(iters * 5), Vec::with_capacity(iters * 5), Vec::with_capacity(iters * 5));
    for _ in 0..iters * 5 {
        let t = Instant::now();
        let d = pool.render(&base, blend, &mut fb);
        v.push(t.elapsed());
        bh.push(pool.stats().bin_hash);
        rs.push(pool.stats().resolve);
        bi.push(pool.stats().bin);
        assert!(d.is_empty());
    }
    println!(
        "  bin + hash:                  {}  (resolve {}, bin {})",
        ms(median(bh)),
        ms(median(rs)),
        ms(median(bi))
    );
    let (u50, u95) = p50_p95(v);
    println!("  unchanged frame:             p50 {} p95 {}", ms(u50), ms(u95));

    // One 200x40 region changes (a button's hover colour).
    for k in 0..WARMUP {
        let s = if k.is_multiple_of(2) { &hover } else { &base };
        pool.render(s, blend, &mut fb);
    }
    pool.render(&base, blend, &mut fb);
    let mut v = Vec::with_capacity(iters * 5);
    let mut tiles = 0;
    for k in 0..iters * 5 {
        let s = if k.is_multiple_of(2) { &hover } else { &base };
        let t = Instant::now();
        pool.render(s, blend, &mut fb);
        v.push(t.elapsed());
        tiles = pool.stats().dirty_tiles;
    }
    let (s50, s95) = p50_p95(v);
    println!("  small change (200x40):       p50 {} p95 {}  ({tiles} tiles)", ms(s50), ms(s95));

    // Scroll: the left two thirds of the content move 11 px per frame (new geometry every frame).
    let scenes: Vec<Scene> = (1..=iters).map(|k| busy_scene(a, wf, hf, k as f32 * 11., false)).collect();
    for s in scenes.iter().take(WARMUP) {
        pool.render(s, blend, &mut fb);
    }
    pool.render(&base, blend, &mut fb);
    let mut v = Vec::with_capacity(scenes.len());
    let mut last = None;
    for s in &scenes {
        let t = Instant::now();
        pool.render(s, blend, &mut fb);
        v.push(t.elapsed());
        last = Some(pool.stats().clone());
    }
    let st = last.unwrap();
    let (sc50, sc95) = p50_p95(v);
    println!(
        "  scroll ({:.0}% of tiles):       p50 {} p95 {}  (bin+hash {}, prepare {}, raster {}; {} of {} tiles)",
        st.dirty_tiles as f64 * 100. / st.tiles as f64,
        ms(sc50),
        ms(sc95),
        ms(st.bin_hash),
        ms(st.prepare),
        ms(st.raster),
        st.dirty_tiles,
        st.tiles
    );
    println!("  renderer caches:             {:.1} MB", pool.cache_bytes() as f64 / (1 << 20) as f64);
}

/// Single-thread full repaints of the 4K frame with only some primitive kinds, to see what costs.
fn breakdown(a: &Assets) {
    let (w, h) = (3840, 2160);
    let base = busy_scene(a, w as f32, h as f32, 0., false);
    let blend = GlyphBlend::new([0.149_355_4, -0.226_006_7, 0.026_559_83, -0.034_591_47], 1.0);
    fn plain(q: &Quad) -> bool {
        q.corner_radii == Corners::default() && q.border_widths == Edges::default()
    }
    type Keep = fn(&Primitive) -> bool;
    let kinds: [(&str, Keep); 9] = [
        ("all", |_| true),
        ("quads", |p| matches!(p, Primitive::Quad(_))),
        ("plain", |p| matches!(p, Primitive::Quad(q) if plain(q))),
        ("rounded", |p| matches!(p, Primitive::Quad(q) if !plain(q))),
        ("glyphs", |p| matches!(p, Primitive::MonoSprite(s) if s.kind == SpriteKind::Glyph)),
        ("svgs", |p| matches!(p, Primitive::MonoSprite(s) if s.kind == SpriteKind::Svg)),
        ("images", |p| matches!(p, Primitive::Image(_))),
        ("shadows", |p| matches!(p, Primitive::Shadow(_))),
        ("paths", |p| matches!(p, Primitive::Path(_))),
    ];
    println!(
        "
4K breakdown, full repaint on 1 thread (p50/p95 of 15, after {WARMUP} warm-up):"
    );
    for (name, keep) in kinds {
        let mut s = Scene::new();
        s.primitives = base.primitives.iter().filter(|p| keep(p)).cloned().collect();
        let mut r = Renderer::with_threads(1);
        let mut fb = Framebuffer::new(w, h);
        r.render(&s, blend, &mut fb);
        for _ in 0..WARMUP {
            r.invalidate();
            r.render(&s, blend, &mut fb);
        }
        let mut v = Vec::with_capacity(15);
        for _ in 0..15 {
            r.invalidate();
            let t = Instant::now();
            r.render(&s, blend, &mut fb);
            v.push(t.elapsed());
        }
        let st = r.stats();
        let (b50, b95) = p50_p95(v);
        println!(
            "  {name:8} p50 {} p95 {}  (raster {}, prepare {})",
            ms(b50),
            ms(b95),
            ms(st.raster),
            ms(st.prepare)
        );
    }
}

fn main() {
    let arg = std::env::args().nth(1).unwrap_or_default();
    if arg == "breakdown" {
        let mut rng = Rng(7);
        let paths = (0..32).map(|_| (0..500).map(|i| (i as f32 / 499., rng.f())).collect()).collect();
        let a = Assets {
            glyphs: glyph_masks(&mut rng),
            icons: icon_masks(&mut rng),
            images: images(&mut rng, 96),
            paths,
        };
        breakdown(&a);
        return;
    }
    let mut rng = Rng(7);
    let paths = (0..32)
        .map(|_| {
            let mut y = 0.5f32;
            (0..500)
                .map(|i| {
                    y = (y + (rng.f() - 0.5) * 0.08).clamp(0., 1.);
                    (i as f32 / 499., y)
                })
                .collect()
        })
        .collect();
    let a = Assets {
        glyphs: glyph_masks(&mut rng),
        icons: icon_masks(&mut rng),
        images: images(&mut rng, 96),
        paths,
    };
    if arg != "4k" && arg != "breakdown" {
        run("1080p", 1920, 1200, &a);
    }
    if arg != "1080" {
        run("4K", 3840, 2160, &a);
    }
}
