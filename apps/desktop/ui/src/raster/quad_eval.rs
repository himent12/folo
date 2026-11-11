// Adapted from GPUI (Apache-2.0): crates/gpui/src/platform/windows/shaders.hlsl (quad_fragment, quad_sdf,
// pick_corner_radius, over, dash_alpha, corner_dash_velocity, quarter_ellipse_sdf)
//! GPUI's quad fragment shader, pixel for pixel. The tile renderer calls it only for pixels near
//! edges and corners; interiors and straight bands are spans (`quad.rs`).

use super::blend::{rgba, sat};
use super::gradient::Paint;
use crate::{BorderStyle, Quad};

const M_PI_F: f32 = 3.141_592_7;
/// The SDF threshold for including a pixel (half a pixel).
const AA: f32 = 0.5;

/// A quad's shader inputs in plain floats.
#[derive(Clone, Debug)]
pub(crate) struct QuadShader {
    pub origin: [f32; 2],
    pub size: [f32; 2],
    /// top-left, top-right, bottom-right, bottom-left.
    pub radii: [f32; 4],
    /// top, right, bottom, left.
    pub widths: [f32; 4],
    pub border: [f32; 4],
    pub paint: Paint,
    pub dashed: bool,
}

/// `pick_corner_radius`.
#[inline]
pub(crate) fn pick_corner_radius(c2p: [f32; 2], radii: &[f32; 4]) -> f32 {
    if c2p[0] < 0. {
        if c2p[1] < 0. { radii[0] } else { radii[3] }
    } else if c2p[1] < 0. {
        radii[1]
    } else {
        radii[2]
    }
}

/// `quad_sdf_impl`.
#[inline]
pub(crate) fn quad_sdf_impl(cctp: [f32; 2], r: f32) -> f32 {
    if r == 0. {
        cctp[0].max(cctp[1])
    } else {
        let (mx, my) = (cctp[0].max(0.), cctp[1].max(0.));
        (mx * mx + my * my).sqrt() + cctp[0].max(cctp[1]).min(0.) - r
    }
}

/// `quad_sdf` of the pixel centre `p` against `origin`, `size`, `radii`.
#[inline]
pub(crate) fn quad_sdf(p: [f32; 2], origin: [f32; 2], size: [f32; 2], radii: &[f32; 4]) -> f32 {
    let half = [size[0] / 2., size[1] / 2.];
    let c2p = [p[0] - (origin[0] + half[0]), p[1] - (origin[1] + half[1])];
    let r = pick_corner_radius(c2p, radii);
    let ctp = [c2p[0].abs() - half[0], c2p[1].abs() - half[1]];
    quad_sdf_impl([ctp[0] + r, ctp[1] + r], r)
}

/// `over` (straight colours).
fn over(below: [f32; 4], above: [f32; 4]) -> [f32; 4] {
    let alpha = above[3] + below[3] * (1.0 - above[3]);
    if alpha <= 0. {
        return [0.; 4];
    }
    let k = below[3] * (1.0 - above[3]);
    [
        (above[0] * above[3] + below[0] * k) / alpha,
        (above[1] * above[3] + below[1] * k) / alpha,
        (above[2] * above[3] + below[2] * k) / alpha,
        alpha,
    ]
}

fn corner_dash_velocity(dv1: f32, dv2: f32) -> f32 {
    if dv1 == 0.0 {
        dv2
    } else if dv2 == 0.0 {
        dv1
    } else {
        dv1.min(dv2)
    }
}

fn dash_alpha(t: f32, period: f32, length: f32, dash_velocity: f32, aa: f32) -> f32 {
    let half_period = period / 2.0;
    let half_length = length / 2.0;
    let centered = (t + half_period - half_length) % period - half_period;
    let signed_distance = centered.abs() - half_length;
    sat(aa - signed_distance / dash_velocity)
}

fn quarter_ellipse_sdf(pt: [f32; 2], radii: [f32; 2]) -> f32 {
    let v = [pt[0] / radii[0], pt[1] / radii[1]];
    let unit = (v[0] * v[0] + v[1] * v[1]).sqrt() - 1.0;
    unit * (radii[0] + radii[1]) * -0.5
}

impl QuadShader {
    pub fn new(q: &Quad) -> Self {
        let origin = [q.bounds.origin.x.0, q.bounds.origin.y.0];
        let size = [q.bounds.size.width.0, q.bounds.size.height.0];
        let r = &q.corner_radii;
        let w = &q.border_widths;
        QuadShader {
            origin,
            size,
            radii: [r.top_left.0, r.top_right.0, r.bottom_right.0, r.bottom_left.0],
            widths: [w.top.0, w.right.0, w.bottom.0, w.left.0],
            border: rgba(q.border_color),
            paint: Paint::new(&q.background, origin, size),
            dashed: q.border_style == BorderStyle::Dashed,
        }
    }

    pub fn unrounded(&self) -> bool {
        self.radii.iter().all(|&r| r == 0.)
    }

    pub fn unbordered(&self) -> bool {
        self.widths.iter().all(|&w| w == 0.)
    }

    /// The shader's output at the centre of pixel (`x`, `y`): straight rgb and alpha, with `bg` the
    /// background colour there.
    pub fn eval_with(&self, x: i32, y: i32, bg: [f32; 4]) -> [f32; 4] {
        let unrounded = self.unrounded();
        if self.unbordered() && unrounded {
            return bg;
        }
        let pos = [x as f32 + 0.5, y as f32 + 0.5];
        let size = self.size;
        let half = [size[0] / 2., size[1] / 2.];
        let the_point = [pos[0] - self.origin[0], pos[1] - self.origin[1]];
        let c2p = [the_point[0] - half[0], the_point[1] - half[1]];
        let r = pick_corner_radius(c2p, &self.radii);
        let [w_t, w_r, w_b, w_l] = self.widths;
        let border = [if c2p[0] < 0. { w_l } else { w_r }, if c2p[1] < 0. { w_t } else { w_b }];
        let reduced = [if border[0] == 0. { -AA } else { border[0] }, if border[1] == 0. { -AA } else { border[1] }];
        let ctp = [c2p[0].abs() - half[0], c2p[1].abs() - half[1]];
        let cctp = [ctp[0] + r, ctp[1] + r];
        let near_rounded = cctp[0] >= 0. && cctp[1] >= 0.;
        let sbic = [ctp[0] + reduced[0], ctp[1] + reduced[1]];
        let beyond_inner = sbic[0] > 0. || sbic[1] > 0.;
        let within_inner = sbic[0] < -AA && sbic[1] < -AA;
        if within_inner && !near_rounded {
            return bg;
        }
        let outer = quad_sdf_impl(cctp, r);
        let inner = if cctp[0] <= 0. || cctp[1] <= 0. {
            -sbic[0].max(sbic[1])
        } else if beyond_inner {
            -1.0
        } else if reduced[0] == reduced[1] {
            -(outer + reduced[0])
        } else {
            let er = [(r - reduced[0]).max(0.), (r - reduced[1]).max(0.)];
            quarter_ellipse_sdf(cctp, er)
        };
        let border_sdf = inner.max(outer);
        let mut color = bg;
        if border_sdf < AA {
            let mut bc = self.border;
            if self.dashed {
                bc[3] *= self.dash(the_point, c2p, cctp, near_rounded, r, unrounded);
            }
            let blended = over(bg, bc);
            let f = sat(AA - inner);
            color = std::array::from_fn(|i| bg[i] + (blended[i] - bg[i]) * f);
        }
        color[3] *= sat(AA - outer);
        color
    }

    /// The shader's output at pixel (`x`, `y`) with its own background.
    pub fn eval(&self, x: i32, y: i32) -> [f32; 4] {
        let bg = self.paint.color_at(x as f32 + 0.5, y as f32 + 0.5);
        self.eval_with(x, y, bg)
    }

    /// The dash pattern's alpha (border_style == 1).
    fn dash(&self, the_point: [f32; 2], c2p: [f32; 2], cctp: [f32; 2], near: bool, r: f32, unrounded: bool) -> f32 {
        let size = self.size;
        let (dash_length_per_width, dash_gap_per_width) = (2.0f32, 1.0f32);
        let dash_period_per_width = dash_length_per_width + dash_gap_per_width;
        let dv_numerator = 1.0 / dash_period_per_width;
        let [w_t, w_r, w_b, w_l] = self.widths;
        let (t, mut max_t, dash_velocity);
        if unrounded {
            let is_horizontal = cctp[0] < cctp[1];
            let dashed_border = [w_b.max(w_t), w_r.max(w_l)];
            let bw = if is_horizontal { dashed_border[0] } else { dashed_border[1] };
            dash_velocity = dv_numerator / bw;
            t = (if is_horizontal { the_point[0] } else { the_point[1] }) * dash_velocity;
            max_t = (if is_horizontal { size[0] } else { size[1] }) * dash_velocity;
        } else {
            let [r_tl, r_tr, r_br, r_bl] = self.radii;
            let dv = |w: f32| if w <= 0. { 0. } else { dv_numerator / w };
            let (dv_t, dv_r, dv_b, dv_l) = (dv(w_t), dv(w_r), dv(w_b), dv(w_l));
            let s_t = (size[0] - r_tl - r_tr) * dv_t;
            let s_r = (size[1] - r_tr - r_br) * dv_r;
            let s_b = (size[0] - r_br - r_bl) * dv_b;
            let s_l = (size[1] - r_bl - r_tl) * dv_l;
            let cdv_tr = corner_dash_velocity(dv_t, dv_r);
            let cdv_br = corner_dash_velocity(dv_b, dv_r);
            let cdv_bl = corner_dash_velocity(dv_b, dv_l);
            let cdv_tl = corner_dash_velocity(dv_t, dv_l);
            let c_tr = r_tr * (M_PI_F / 2.0) * cdv_tr;
            let c_br = r_br * (M_PI_F / 2.0) * cdv_br;
            let c_bl = r_bl * (M_PI_F / 2.0) * cdv_bl;
            let c_tl = r_tl * (M_PI_F / 2.0) * cdv_tl;
            let upto_tr = s_t;
            let upto_r = upto_tr + c_tr;
            let upto_br = upto_r + s_r;
            let upto_b = upto_br + c_br;
            let upto_bl = upto_b + s_b;
            let upto_l = upto_bl + c_bl;
            let upto_tl = upto_l + s_l;
            max_t = upto_tl + c_tl;
            if near {
                let corner_t = cctp[1].atan2(cctp[0]) * r;
                if c2p[0] >= 0. {
                    if c2p[1] < 0. {
                        dash_velocity = cdv_tr;
                        t = upto_r - corner_t * dash_velocity;
                    } else {
                        dash_velocity = cdv_br;
                        t = upto_br + corner_t * dash_velocity;
                    }
                } else if c2p[1] >= 0. {
                    dash_velocity = cdv_bl;
                    t = upto_l - corner_t * dash_velocity;
                } else {
                    dash_velocity = cdv_tl;
                    t = upto_tl + corner_t * dash_velocity;
                }
            } else {
                let is_horizontal = cctp[0] < cctp[1];
                if is_horizontal {
                    if c2p[1] < 0. {
                        dash_velocity = dv_t;
                        t = (the_point[0] - r_tl) * dash_velocity;
                    } else {
                        dash_velocity = dv_b;
                        t = upto_bl - (the_point[0] - r_bl) * dash_velocity;
                    }
                } else if c2p[0] < 0. {
                    dash_velocity = dv_l;
                    t = upto_tl - (the_point[1] - r_tl) * dash_velocity;
                } else {
                    dash_velocity = dv_r;
                    t = upto_r + (the_point[1] - r_tr) * dash_velocity;
                }
            }
        }
        let dash_length = dash_length_per_width / dash_period_per_width;
        if unrounded {
            max_t -= dash_length;
        }
        if max_t >= 1.0 {
            let dash_count = max_t.floor();
            let dash_period = max_t / dash_count;
            dash_alpha(t, dash_period, dash_length, dash_velocity, AA)
        } else if unrounded {
            let dash_gap = max_t - dash_length;
            if dash_gap > 0.0 { dash_alpha(t, dash_length + dash_gap, dash_length, dash_velocity, AA) } else { 1.0 }
        } else {
            1.0
        }
    }
}
