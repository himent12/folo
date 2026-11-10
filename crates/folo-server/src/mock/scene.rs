//! Procedural mock images: a scene is generated from `(seed, index)`, its labels are exact, and it can be rendered
//! at any size.

use folo_api::BBox;

use crate::raster::{Canvas, Rgb, hsv, shade};
use crate::util::{Rng, mix64};

/// Visual theme of a mock dataset.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Theme {
    /// Circles / squares / triangles / stars on textured gradients.
    Shapes,
    /// Vehicles, people and bicycles on a road.
    Traffic,
    /// Bolts and hex nuts on a conveyor belt.
    Fasteners,
}

#[derive(Clone, Copy, Debug)]
enum ShapeKind {
    Circle,
    Square,
    Triangle,
    Star,
}

#[derive(Clone, Debug)]
enum Draw {
    Ellipse {
        color: Rgb,
    },
    Rect {
        color: Rgb,
    },
    Poly {
        pts: Vec<(f32, f32)>,
        color: Rgb,
    },
    Vehicle {
        color: Rgb,
        wheels: u32,
    },
    Person {
        color: Rgb,
    },
    Bicycle {
        color: Rgb,
    },
    Bolt {
        color: Rgb,
    },
    /// A hex nut: its outline, and a hole in the middle.
    Nut {
        pts: Vec<(f32, f32)>,
        color: Rgb,
    },
}

#[derive(Clone, Debug)]
struct Obj {
    class: u32,
    // Unclipped geometry box (pixels).
    geo: [f32; 4],
    draw: Draw,
}

/// A generated image: size, background parameters and objects.
#[derive(Clone, Debug)]
pub struct Scene {
    /// Width in pixels.
    pub w: u32,
    /// Height in pixels.
    pub h: u32,
    theme: Theme,
    seed: u64,
    bg: [Rgb; 4],
    objs: Vec<Obj>,
}

fn kind_for(class: u32, name: &str) -> ShapeKind {
    match name.to_ascii_lowercase().as_str() {
        "circle" => ShapeKind::Circle,
        "square" | "rectangle" => ShapeKind::Square,
        "triangle" => ShapeKind::Triangle,
        "star" => ShapeKind::Star,
        _ => [ShapeKind::Circle, ShapeKind::Square, ShapeKind::Triangle, ShapeKind::Star][class as usize % 4],
    }
}

fn soft_color(r: &mut Rng) -> Rgb {
    hsv(r.range(0.0, 360.0), r.range(0.05, 0.35), r.range(0.25, 0.85))
}

fn star_points(cx: f32, cy: f32, r: f32, rot: f32) -> Vec<(f32, f32)> {
    (0..10)
        .map(|i| {
            let a = rot + i as f32 * std::f32::consts::PI / 5.0;
            let rr = if i % 2 == 0 { r } else { r * 0.45 };
            (cx + rr * a.sin(), cy - rr * a.cos())
        })
        .collect()
}

fn poly_box(pts: &[(f32, f32)]) -> [f32; 4] {
    pts.iter().fold([f32::MAX, f32::MAX, f32::MIN, f32::MIN], |b, &(x, y)| {
        [b[0].min(x), b[1].min(y), b[2].max(x), b[3].max(y)]
    })
}

impl Scene {
    /// Generates image `index` of a dataset.
    pub fn generate(theme: Theme, classes: &[String], w: u32, h: u32, seed: u64, index: u64) -> Scene {
        let s = mix64(seed ^ mix64(index.wrapping_add(1)));
        let mut r = Rng::new(s);
        let bg = [soft_color(&mut r), soft_color(&mut r), soft_color(&mut r), soft_color(&mut r)];
        let mut scene = Scene { w, h, theme, seed: s, bg, objs: Vec::new() };
        let (wf, hf) = (w as f32, h as f32);
        let nc = classes.len().max(1) as u64;
        match theme {
            Theme::Shapes => {
                let n = if r.chance(0.012) { 0 } else { 1 + r.below(7) + r.below(2) };
                let side = wf.min(hf);
                for _ in 0..n {
                    let class = r.below(nc) as u32;
                    let kind = kind_for(class, classes.get(class as usize).map_or("", |c| c.as_str()));
                    let size = side * r.range(0.06, 0.3);
                    let (cx, cy) =
                        (r.range(size * 0.5, wf - size * 0.5), r.range(size * 0.5, hf - size * 0.5));
                    let color = hsv(r.range(0.0, 360.0), r.range(0.55, 0.95), r.range(0.6, 1.0));
                    let (geo, draw) = match kind {
                        ShapeKind::Circle => {
                            let rad = size * 0.5;
                            ([cx - rad, cy - rad, cx + rad, cy + rad], Draw::Ellipse { color })
                        }
                        ShapeKind::Square => {
                            let (hw, hh) = (size * 0.5 * r.range(0.8, 1.15), size * 0.5 * r.range(0.8, 1.15));
                            ([cx - hw, cy - hh, cx + hw, cy + hh], Draw::Rect { color })
                        }
                        ShapeKind::Triangle => {
                            let rad = size * 0.55;
                            let rot = r.range(-0.4, 0.4);
                            let pts: Vec<_> = (0..3)
                                .map(|i| {
                                    let a = rot + i as f32 * std::f32::consts::TAU / 3.0;
                                    (cx + rad * a.sin(), cy - rad * a.cos())
                                })
                                .collect();
                            (poly_box(&pts), Draw::Poly { pts, color })
                        }
                        ShapeKind::Star => {
                            let pts = star_points(cx, cy, size * 0.55, r.range(0.0, 1.2));
                            (poly_box(&pts), Draw::Poly { pts, color })
                        }
                    };
                    scene.objs.push(Obj { class, geo, draw });
                }
            }
            Theme::Traffic => {
                let horizon = hf * 0.38;
                let n = 3 + r.below(9) + r.below(3);
                let mut objs = Vec::new();
                for _ in 0..n {
                    let pick = r.f32();
                    // car, truck, bus, person, bicycle
                    let class: u32 = match pick {
                        p if p < 0.55 => 0,
                        p if p < 0.67 => 1,
                        p if p < 0.73 => 2,
                        p if p < 0.93 => 3,
                        _ => 4,
                    }
                    .min(nc as u32 - 1);
                    let foot = horizon + 12.0 + (hf * 1.02 - horizon - 12.0) * r.f32().powf(0.8);
                    let p = ((foot - horizon) / (hf - horizon)).clamp(0.0, 1.1);
                    let base = 14.0 + 190.0 * p;
                    let (bw, bh) = match class {
                        0 => (base * 1.5, base * 0.72),
                        1 => (base * 1.9, base * 1.25),
                        2 => (base * 2.7, base * 1.15),
                        3 => (base * 0.3, base * 0.9),
                        _ => (base * 0.75, base * 0.5),
                    };
                    let cx = r.range(-bw * 0.2, wf + bw * 0.2);
                    let geo = [cx - bw * 0.5, foot - bh, cx + bw * 0.5, foot];
                    let draw = match class {
                        0..=2 => Draw::Vehicle {
                            color: hsv(r.range(0.0, 360.0), r.range(0.0, 0.8), r.range(0.3, 0.95)),
                            wheels: if class == 2 { 3 } else { 2 },
                        },
                        3 => Draw::Person {
                            color: hsv(r.range(0.0, 360.0), r.range(0.3, 0.8), r.range(0.3, 0.8)),
                        },
                        _ => Draw::Bicycle { color: hsv(r.range(0.0, 360.0), 0.7, 0.8) },
                    };
                    objs.push((foot, Obj { class, geo, draw }));
                }
                objs.sort_by(|a, b| a.0.total_cmp(&b.0));
                scene.objs = objs.into_iter().map(|(_, o)| o).collect();
                scene.bg = [[118, 162, 214], [214, 228, 240], [92, 94, 98], [58, 60, 64]];
            }
            Theme::Fasteners => {
                let n = if r.chance(0.03) { 0 } else { 1 + r.below(5) + r.below(2) };
                let metals = [[148, 156, 168], [182, 164, 118], [116, 124, 136]];
                scene.bg = [[44, 48, 54], [62, 66, 74], [36, 40, 46], [52, 56, 64]];
                for k in 0..n {
                    let class = if k == 0 || nc <= 1 { 0 } else { r.below(nc) as u32 };
                    let color = shade(metals[r.below(3) as usize], r.range(0.85, 1.15));
                    if class == 0 {
                        let len = r.range(120.0, 260.0);
                        let thick = len * r.range(0.24, 0.34);
                        let cx = r.range(len * 0.6, wf - len * 0.6);
                        let cy = r.range(thick * 0.8, hf - thick * 0.8);
                        let geo = [cx - len * 0.5, cy - thick * 0.5, cx + len * 0.5, cy + thick * 0.5];
                        scene.objs.push(Obj { class: 0, geo, draw: Draw::Bolt { color } });
                    } else {
                        let rad = r.range(36.0, 78.0);
                        let cx = r.range(rad * 1.2, wf - rad * 1.2);
                        let cy = r.range(rad * 1.2, hf - rad * 1.2);
                        // The box is the hexagon's own (flat top and bottom: 0.87 of its width tall).
                        let pts: Vec<_> = (0..6)
                            .map(|i| {
                                let a = i as f32 * std::f32::consts::TAU / 6.0;
                                (cx + rad * a.cos(), cy + rad * a.sin())
                            })
                            .collect();
                        scene.objs.push(Obj { class, geo: poly_box(&pts), draw: Draw::Nut { pts, color } });
                    }
                }
            }
        }
        scene
    }

    /// Ground-truth boxes, clipped to the image; boxes that end up thinner than 2 px are dropped.
    pub fn labels(&self) -> Vec<BBox> {
        let (wf, hf) = (self.w as f32, self.h as f32);
        self.objs
            .iter()
            .filter_map(|o| {
                let b = BBox {
                    class: o.class,
                    x1: o.geo[0].clamp(0.0, wf).round(),
                    y1: o.geo[1].clamp(0.0, hf).round(),
                    x2: o.geo[2].clamp(0.0, wf).round(),
                    y2: o.geo[3].clamp(0.0, hf).round(),
                };
                (b.x2 - b.x1 >= 2.0 && b.y2 - b.y1 >= 2.0).then_some(b)
            })
            .collect()
    }

    /// Renders at full size, or with the long side scaled down to `long_side`.
    pub fn render(&self, long_side: Option<u32>) -> Canvas {
        let long = self.w.max(self.h) as f32;
        let scale = long_side.map_or(1.0, |l| (l as f32 / long).min(1.0));
        let cw = ((self.w as f32 * scale).round() as u32).max(1);
        let ch = ((self.h as f32 * scale).round() as u32).max(1);
        let mut c = Canvas::new(cw, ch, scale);
        let (wf, hf) = (self.w as f32, self.h as f32);
        match self.theme {
            Theme::Shapes => {
                c.gradient(0.0, hf, self.bg[0], self.bg[1], 22.0, wf.min(hf) / 14.0, self.seed);
            }
            Theme::Traffic => {
                let horizon = hf * 0.38;
                c.gradient(0.0, horizon, self.bg[0], self.bg[1], 4.0, 90.0, self.seed);
                c.gradient(horizon, hf, self.bg[2], self.bg[3], 10.0, 6.0, self.seed ^ 1);
                let vp = (wf * 0.5, horizon);
                for k in [-1.2f32, -0.4, 0.4, 1.2] {
                    c.line(vp, (wf * 0.5 + k * wf * 0.55, hf), 2.0 + 10.0 * k.abs(), [210, 206, 190]);
                }
            }
            Theme::Fasteners => {
                c.gradient(0.0, hf, self.bg[0], self.bg[1], 14.0, 28.0, self.seed);
                for slot in 1..6 {
                    let y = hf * (slot as f32 / 6.0);
                    c.line((0.0, y), (wf, y), 3.0, self.bg[2]);
                }
            }
        }
        let shadow_off = wf.min(hf) * 0.008;
        for o in &self.objs {
            let [x1, y1, x2, y2] = o.geo;
            let (bw, bh) = (x2 - x1, y2 - y1);
            let cx = (x1 + x2) * 0.5;
            match &o.draw {
                Draw::Ellipse { color } => {
                    c.ellipse(
                        cx + shadow_off,
                        (y1 + y2) * 0.5 + shadow_off,
                        bw * 0.5,
                        bh * 0.5,
                        [20, 20, 24],
                    );
                    c.ellipse(cx, (y1 + y2) * 0.5, bw * 0.5, bh * 0.5, *color);
                }
                Draw::Rect { color } => {
                    c.rect(x1 + shadow_off, y1 + shadow_off, x2 + shadow_off, y2 + shadow_off, [20, 20, 24]);
                    c.rect(x1, y1, x2, y2, *color);
                }
                Draw::Poly { pts, color } => {
                    let sh: Vec<_> = pts.iter().map(|&(x, y)| (x + shadow_off, y + shadow_off)).collect();
                    c.polygon(&sh, [20, 20, 24]);
                    c.polygon(pts, *color);
                }
                Draw::Vehicle { color, wheels } => {
                    let wr = bh * 0.2;
                    c.rect(x1, y1 + bh * 0.3, x2, y2 - wr * 0.6, *color);
                    c.rect(x1 + bw * 0.12, y1, x2 - bw * 0.12, y1 + bh * 0.34, shade(*color, 0.9));
                    c.rect(x1 + bw * 0.18, y1 + bh * 0.06, x2 - bw * 0.18, y1 + bh * 0.3, [40, 58, 80]);
                    for i in 0..*wheels {
                        let t = (i as f32 + 0.5) / *wheels as f32;
                        c.ellipse(x1 + bw * (0.1 + 0.8 * t), y2 - wr, wr, wr, [18, 18, 18]);
                    }
                }
                Draw::Person { color } => {
                    let hr = bw * 0.42;
                    c.ellipse(cx, y1 + hr, hr, hr, [224, 186, 150]);
                    c.rect(x1 + bw * 0.1, y1 + hr * 2.0, x2 - bw * 0.1, y1 + bh * 0.6, *color);
                    c.rect(x1 + bw * 0.15, y1 + bh * 0.6, cx - bw * 0.05, y2, [40, 44, 60]);
                    c.rect(cx + bw * 0.05, y1 + bh * 0.6, x2 - bw * 0.15, y2, [40, 44, 60]);
                }
                Draw::Bicycle { color } => {
                    let wr = bh * 0.32;
                    c.ellipse(x1 + wr, y2 - wr, wr, wr, [24, 24, 24]);
                    c.ellipse(x2 - wr, y2 - wr, wr, wr, [24, 24, 24]);
                    let lw = (bh * 0.07).max(1.0);
                    c.line((x1 + wr, y2 - wr), (cx, y1 + bh * 0.25), lw, *color);
                    c.line((cx, y1 + bh * 0.25), (x2 - wr, y2 - wr), lw, *color);
                    c.line((x1 + bw * 0.3, y1), (x2 - bw * 0.3, y1), lw, [30, 30, 30]);
                }
                Draw::Bolt { color } => {
                    let head_w = bw * 0.24;
                    let shaft_h = bh * 0.48;
                    let cy = (y1 + y2) * 0.5;
                    c.rect(
                        x1 + shadow_off,
                        y1 + shadow_off,
                        x1 + head_w + shadow_off,
                        y2 + shadow_off,
                        [18, 20, 24],
                    );
                    c.rect(
                        x1 + head_w,
                        cy - shaft_h * 0.5 + shadow_off,
                        x2 + shadow_off,
                        cy + shaft_h * 0.5 + shadow_off,
                        [18, 20, 24],
                    );
                    c.rect(x1, y1, x1 + head_w, y2, shade(*color, 1.08));
                    c.rect(x1 + head_w, cy - shaft_h * 0.5, x2, cy + shaft_h * 0.5, *color);
                }
                Draw::Nut { pts, color } => {
                    let cy = (y1 + y2) * 0.5;
                    let hole = bw * 0.5 * 0.42;
                    let sh: Vec<_> = pts.iter().map(|&(x, y)| (x + shadow_off, y + shadow_off)).collect();
                    c.polygon(&sh, [18, 20, 24]);
                    c.polygon(pts, *color);
                    c.ellipse(cx, cy, hole, hole, [34, 38, 44]);
                }
            }
        }
        c
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn deterministic_and_inside_image() {
        for (theme, classes, w, h) in [
            (Theme::Shapes, names(&["circle", "square", "triangle"]), 640, 640),
            (Theme::Traffic, names(&["car", "truck", "bus", "person", "bicycle"]), 1280, 720),
            (Theme::Fasteners, names(&["bolt", "nut"]), 1920, 1080),
        ] {
            for i in 0..50 {
                let a = Scene::generate(theme, &classes, w, h, 9, i).labels();
                let b = Scene::generate(theme, &classes, w, h, 9, i).labels();
                assert_eq!(a, b);
                for bx in &a {
                    assert!((bx.class as usize) < classes.len());
                    assert!(bx.x1 >= 0.0 && bx.x2 <= w as f32 && bx.x1 < bx.x2);
                    assert!(bx.y1 >= 0.0 && bx.y2 <= h as f32 && bx.y1 < bx.y2);
                }
            }
        }
    }

    #[test]
    fn fastener_boxes_are_what_is_drawn() {
        let classes = names(&["bolt", "nut"]);
        let (mut bolts, mut nuts) = (0, 0);
        for i in 0..40 {
            for o in Scene::generate(Theme::Fasteners, &classes, 1920, 1080, 5, i).objs {
                match &o.draw {
                    // The head spans the box's full height from its left edge; the shaft ends at its right edge.
                    Draw::Bolt { .. } => bolts += 1,
                    // Every side of the box touches the hexagon.
                    Draw::Nut { pts, .. } => {
                        nuts += 1;
                        let b = poly_box(pts);
                        for k in 0..4 {
                            assert!((b[k] - o.geo[k]).abs() < 1e-3, "{:?} vs {:?}", b, o.geo);
                        }
                        let (w, h) = (o.geo[2] - o.geo[0], o.geo[3] - o.geo[1]);
                        assert!((h / w - 3f32.sqrt() / 2.0).abs() < 1e-3, "a regular hexagon: {w} x {h}");
                    }
                    _ => panic!("only bolts and nuts"),
                }
            }
        }
        assert!(bolts > 20 && nuts > 20, "{bolts} bolts, {nuts} nuts");
    }

    #[test]
    fn renders_at_thumbnail_size() {
        let s = Scene::generate(Theme::Traffic, &names(&["car"]), 1280, 720, 1, 0);
        let c = s.render(Some(256));
        assert_eq!((c.w, c.h), (256, 144));
        let full = s.render(None);
        assert_eq!((full.w, full.h), (1280, 720));
    }
}
