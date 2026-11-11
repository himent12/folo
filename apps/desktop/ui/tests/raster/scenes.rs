//! Seeded random scenes covering every primitive kind, clips, overlap and off-frame bounds.

use folo_ui::*;
use std::sync::Arc;

/// splitmix64.
pub struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    pub fn f(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (self.next() >> 40) as f32 / (1u64 << 24) as f32 * (hi - lo)
    }

    pub fn i(&mut self, lo: i32, hi: i32) -> i32 {
        lo + (self.next() % (hi - lo) as u64) as i32
    }

    pub fn chance(&mut self, n: u64) -> bool {
        self.next().is_multiple_of(n)
    }

    pub fn pick<T: Copy>(&mut self, v: &[T]) -> T {
        v[(self.next() % v.len() as u64) as usize]
    }
}

pub fn sp(v: f32) -> ScaledPixels {
    ScaledPixels::from(v)
}

pub fn rect(x: f32, y: f32, w: f32, h: f32) -> Bounds<ScaledPixels> {
    Bounds::new(point(sp(x), sp(y)), size(sp(w), sp(h)))
}

pub fn dsize(w: i32, h: i32) -> Size<DevicePixels> {
    size(DevicePixels(w), DevicePixels(h))
}

/// A straight-alpha BGRA image of random pixels (some translucent when `alpha`).
pub fn image(rng: &mut Rng, w: u32, h: u32, alpha: bool) -> Arc<RenderImage> {
    let mut buf = ::image::RgbaImage::new(w, h);
    for p in buf.pixels_mut() {
        let a = if alpha && rng.chance(3) { rng.i(0, 256) as u8 } else { 255 };
        *p = ::image::Rgba([rng.i(0, 256) as u8, rng.i(0, 256) as u8, rng.i(0, 256) as u8, a]);
    }
    Arc::new(RenderImage::new(vec![::image::Frame::new(buf)]))
}

/// The masks and images scenes draw (shared, so ids repeat across frames).
pub struct Assets {
    pub glyphs: Vec<Arc<AlphaMask>>,
    pub icons: Vec<Arc<AlphaMask>>,
    pub emoji: Vec<Arc<ColorMask>>,
    pub images: Vec<Arc<RenderImage>>,
}

impl Assets {
    pub fn new(rng: &mut Rng) -> Self {
        let alpha = |rng: &mut Rng, w: i32, h: i32| {
            let data: Vec<u8> =
                (0..w * h).map(|_| if rng.chance(3) { 0 } else { rng.i(0, 256) as u8 }).collect();
            Arc::new(AlphaMask::new(dsize(w, h), data))
        };
        let glyphs = (0..12)
            .map(|_| {
                let (w, h) = (rng.i(2, 16), rng.i(3, 20));
                alpha(rng, w, h)
            })
            .collect();
        let icons = (0..4)
            .map(|_| {
                let s = rng.i(8, 40);
                alpha(rng, s, s)
            })
            .collect();
        let emoji = (0..3)
            .map(|_| {
                let (w, h) = (rng.i(6, 20), rng.i(6, 20));
                let mut data = Vec::new();
                for _ in 0..w * h {
                    let a = rng.pick(&[0u8, 255, 128, 40]);
                    let c = |rng: &mut Rng| (rng.i(0, 256) as u32 * a as u32 / 255) as u8;
                    data.extend_from_slice(&[c(rng), c(rng), c(rng), a]);
                }
                Arc::new(ColorMask::new(dsize(w, h), data))
            })
            .collect();
        let images = vec![image(rng, 24, 18, false), image(rng, 40, 30, true), image(rng, 7, 9, true)];
        Assets { glyphs, icons, emoji, images }
    }
}

fn color(rng: &mut Rng) -> Hsla {
    hsla(rng.f(0., 1.), rng.f(0., 1.), rng.f(0., 1.), rng.pick(&[1., 1., 0.6, 0.25, 0.05]))
}

fn background(rng: &mut Rng) -> Background {
    match rng.i(0, 10) {
        0 => linear_gradient(
            rng.f(0., 360.),
            linear_color_stop(color(rng), rng.f(0., 0.4)),
            linear_color_stop(color(rng), rng.f(0.6, 1.)),
        ),
        1 => linear_gradient(
            rng.f(0., 360.),
            linear_color_stop(color(rng), 0.),
            linear_color_stop(color(rng), 1.),
        )
        .color_space(ColorSpace::Oklab),
        2 => pattern_slash(color(rng), rng.f(1., 4.), rng.f(1., 6.)),
        _ => color(rng).into(),
    }
}

/// A random clip: mostly the whole frame, sometimes a fractional sub-rectangle.
fn clip(rng: &mut Rng, w: f32, h: f32) -> Bounds<ScaledPixels> {
    if rng.chance(3) {
        let x = rng.f(-20., w * 0.6);
        let y = rng.f(-20., h * 0.6);
        rect(x, y, rng.f(10., w), rng.f(10., h))
    } else {
        rect(0., 0., w, h)
    }
}

/// One random primitive inside (and around) a `w` x `h` frame.
pub fn primitive(rng: &mut Rng, a: &Assets, w: f32, h: f32) -> Primitive {
    let x = rng.f(-40., w + 10.);
    let y = rng.f(-40., h + 10.);
    let frac = |rng: &mut Rng, v: f32| if rng.chance(2) { v.round() } else { v };
    let c = clip(rng, w, h);
    match rng.i(0, 16) {
        0..=5 => {
            let r = |rng: &mut Rng| if rng.chance(2) { 0. } else { rng.f(0.5, 14.) };
            let bw = |rng: &mut Rng| if rng.chance(2) { 0. } else { rng.pick(&[1., 1.5, 2., 3.]) };
            Primitive::Quad(Quad {
                bounds: rect(frac(rng, x), frac(rng, y), rng.f(1., 120.), rng.f(1., 90.)),
                clip: c,
                background: background(rng),
                border_color: color(rng),
                corner_radii: Corners {
                    top_left: sp(r(rng)),
                    top_right: sp(r(rng)),
                    bottom_right: sp(r(rng)),
                    bottom_left: sp(r(rng)),
                },
                border_widths: Edges {
                    top: sp(bw(rng)),
                    right: sp(bw(rng)),
                    bottom: sp(bw(rng)),
                    left: sp(bw(rng)),
                },
                border_style: if rng.chance(6) { BorderStyle::Dashed } else { BorderStyle::Solid },
            })
        }
        6 => Primitive::Shadow(Shadow {
            bounds: rect(x, y, rng.f(5., 100.), rng.f(5., 80.)),
            clip: c,
            corner_radii: Corners::all(sp(rng.f(0., 10.))),
            color: hsla(0., 0., 0., rng.f(0.1, 0.6)),
            blur_radius: sp(rng.f(0.5, 10.)),
        }),
        7..=10 => {
            let m = a.glyphs[rng.i(0, a.glyphs.len() as i32) as usize].clone();
            Primitive::MonoSprite(MonoSprite {
                bounds: rect(x.round(), y.round(), m.size.width.0 as f32, m.size.height.0 as f32),
                clip: c,
                color: color(rng),
                mask: m,
                kind: SpriteKind::Glyph,
            })
        }
        11 => {
            let m = a.icons[rng.i(0, a.icons.len() as i32) as usize].clone();
            let s = (m.size.width.0 as f32 / 2.).ceil();
            Primitive::MonoSprite(MonoSprite {
                bounds: rect(x.round(), y.round(), s, s),
                clip: c,
                color: color(rng),
                mask: m,
                kind: SpriteKind::Svg,
            })
        }
        12 => {
            let m = a.emoji[rng.i(0, a.emoji.len() as i32) as usize].clone();
            Primitive::ColorSprite(ColorSprite {
                bounds: rect(x.round(), y.round(), m.size.width.0 as f32, m.size.height.0 as f32),
                clip: c,
                mask: m,
                opacity: rng.pick(&[1., 0.5]),
            })
        }
        13 => {
            let img = a.images[rng.i(0, a.images.len() as i32) as usize].clone();
            let s = img.size(0);
            let (iw, ih) = if rng.chance(2) {
                (s.width.0 as f32, s.height.0 as f32)
            } else {
                (rng.f(4., 80.).round(), rng.f(4., 60.).round())
            };
            Primitive::Image(Image {
                bounds: rect(x.floor(), y.floor(), iw, ih),
                clip: c,
                corner_radii: Corners::all(sp(rng.pick(&[0., 0., 4., 9.]))),
                image: img,
                frame: 0,
                grayscale: rng.chance(4),
                opacity: rng.pick(&[1., 1., 0.7]),
            })
        }
        14 => {
            let n = rng.i(3, 12);
            let points: Vec<_> =
                (0..n).map(|_| point(sp(x + rng.f(0., 80.)), sp(y + rng.f(0., 60.)))).collect();
            let kind = if rng.chance(2) {
                PathKind::Fill(if rng.chance(2) { FillRule::NonZero } else { FillRule::EvenOdd })
            } else {
                PathKind::Stroke(StrokeStyle { width: sp(rng.f(0.5, 4.)), ..Default::default() })
            };
            let closed = matches!(kind, PathKind::Fill(_));
            Primitive::Path(Path {
                bounds: rect(x - 4., y - 4., 88., 68.),
                clip: c,
                path: Arc::new(PathData::new(vec![Contour { points, closed }], kind)),
                paint: if rng.chance(3) {
                    linear_gradient(
                        180.,
                        linear_color_stop(color(rng), 0.),
                        linear_color_stop(color(rng), 1.),
                    )
                } else {
                    color(rng).into()
                },
            })
        }
        _ => {
            let wavy = rng.chance(2);
            let t = rng.pick(&[1., 1.5, 2.]);
            Primitive::Underline(Underline {
                bounds: rect(x, y, rng.f(5., 100.), if wavy { t * 3. } else { t }),
                clip: c,
                color: color(rng),
                thickness: sp(t),
                wavy,
            })
        }
    }
}

/// A scene of `n` random primitives.
pub fn scene(rng: &mut Rng, a: &Assets, n: usize, w: f32, h: f32) -> Scene {
    let mut s = Scene::new();
    // A full-frame backdrop most of the time, so occlusion is exercised.
    if rng.chance(2) {
        s.push_quad(Quad {
            bounds: rect(0., 0., w, h),
            clip: rect(0., 0., w, h),
            background: hsla(0.6, 0.1, 0.1, 1.).into(),
            border_color: transparent_black(),
            corner_radii: Corners::default(),
            border_widths: Edges::default(),
            border_style: BorderStyle::Solid,
        });
    }
    for _ in 0..n {
        s.push(primitive(rng, a, w, h));
    }
    s
}

/// `s` with a few primitives changed: moved, replaced, removed or inserted.
pub fn mutate(rng: &mut Rng, a: &Assets, s: &Scene, w: f32, h: f32) -> Scene {
    let mut out = s.clone();
    for _ in 0..rng.i(1, 5) {
        let n = out.primitives.len();
        match rng.i(0, 4) {
            0 if n > 0 => {
                let i = rng.i(0, n as i32) as usize;
                out.primitives[i] = primitive(rng, a, w, h);
            }
            1 if n > 0 => {
                let i = rng.i(0, n as i32) as usize;
                out.primitives.remove(i);
            }
            2 if n > 0 => {
                // Move a quad by a whole or a fractional step.
                let i = rng.i(0, n as i32) as usize;
                if let Primitive::Quad(q) = &mut out.primitives[i] {
                    q.bounds.origin.x =
                        sp(f64::from(q.bounds.origin.x.round()) as f32 + rng.pick(&[1., 0.5, -3.]));
                }
            }
            _ => {
                let i = rng.i(0, n as i32 + 1) as usize;
                out.primitives.insert(i, primitive(rng, a, w, h));
            }
        }
    }
    out
}
