//! Decoupled 8-state constant-velocity Kalman box filter for ByteTrack (`[cx, cy, a, h, vx, vy, va, vh]`).
//!
//! Because ByteTrack's transition matrix `F`, observation matrix `H`, initial covariance `P0`, process noise `Q`,
//! and measurement noise `R` are all block-diagonal across the four box coordinates `(cx, cy, a, h)` (with
//! `a = w / h`), four independent 2×2 Kalman filters are algebraically identical to the full 8×8 matrix update
//! with O(1) flops, zero heap allocation, and guaranteed positive-definite variance.

use crate::boxes::Xyxy;

/// ByteTrack Kalman position std weight (`1 / 20`).
const STD_WEIGHT_POS: f64 = 1.0 / 20.0;
/// ByteTrack Kalman velocity std weight (`1 / 160`).
const STD_WEIGHT_VEL: f64 = 1.0 / 160.0;

/// 1D constant-velocity Kalman state `[position, velocity]` with symmetric 2×2 covariance `[[p00, p01], [p01, p11]]`.
#[derive(Clone, Copy, Debug)]
struct Kalman1D {
    pos: f64,
    vel: f64,
    p00: f64,
    p01: f64,
    p11: f64,
}

impl Kalman1D {
    #[inline]
    fn new(pos: f64, std_pos: f64, std_vel: f64) -> Self {
        Kalman1D { pos, vel: 0.0, p00: std_pos * std_pos, p01: 0.0, p11: std_vel * std_vel }
    }

    #[inline]
    fn predict(&mut self, q_std_pos: f64, q_std_vel: f64) {
        self.pos += self.vel;
        self.p00 = self.p00 + 2.0 * self.p01 + self.p11 + q_std_pos * q_std_pos;
        self.p01 += self.p11;
        self.p11 += q_std_vel * q_std_vel;
    }

    #[inline]
    fn update(&mut self, z: f64, r_std: f64) {
        let r = r_std * r_std;
        let s = self.p00 + r;
        if s <= 0.0 || !s.is_finite() {
            return;
        }
        let y = z - self.pos;
        let k0 = self.p00 / s;
        let k1 = self.p01 / s;
        let p01_prior = self.p01;
        self.pos += k0 * y;
        self.vel += k1 * y;
        let one_minus_k0 = (1.0 - k0).max(0.0);
        self.p00 = (one_minus_k0 * self.p00).max(1e-12);
        self.p01 = one_minus_k0 * p01_prior;
        self.p11 = (self.p11 - k1 * p01_prior).max(1e-12);
    }
}

/// Decoupled 8-state constant-velocity Kalman box tracker over `(cx, cy, aspect_ratio, height)`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct KalmanBox {
    cx: Kalman1D,
    cy: Kalman1D,
    a: Kalman1D,
    h: Kalman1D,
}

impl KalmanBox {
    pub(crate) fn from_bbox(b: Xyxy) -> Self {
        let w = (b.x2 as f64 - b.x1 as f64).max(1e-3);
        let h = (b.y2 as f64 - b.y1 as f64).max(1e-3);
        let cx = b.x1 as f64 + 0.5 * w;
        let cy = b.y1 as f64 + 0.5 * h;
        let a = w / h;
        let sp = 2.0 * STD_WEIGHT_POS * h;
        let sv = 10.0 * STD_WEIGHT_VEL * h;
        KalmanBox {
            cx: Kalman1D::new(cx, sp, sv),
            cy: Kalman1D::new(cy, sp, sv),
            a: Kalman1D::new(a, 1e-2, 1e-5),
            h: Kalman1D::new(h, sp, sv),
        }
    }

    pub(crate) fn predict(&mut self, coasting: bool) {
        if coasting || self.h.pos + self.h.vel <= 1e-3 {
            // ByteTrack guard: dampen height velocity while coasting or if predicted height would collapse.
            self.h.vel = 0.0;
        }
        let h_ref = self.h.pos.max(1.0);
        let qp = STD_WEIGHT_POS * h_ref;
        let qv = STD_WEIGHT_VEL * h_ref;
        self.cx.predict(qp, qv);
        self.cy.predict(qp, qv);
        self.a.predict(1e-2, 1e-5);
        self.h.predict(qp, qv);
        self.h.pos = self.h.pos.max(1e-3);
        self.a.pos = self.a.pos.max(1e-3);
    }

    pub(crate) fn update(&mut self, b: Xyxy) {
        let w = (b.x2 as f64 - b.x1 as f64).max(1e-3);
        let h = (b.y2 as f64 - b.y1 as f64).max(1e-3);
        let cx = b.x1 as f64 + 0.5 * w;
        let cy = b.y1 as f64 + 0.5 * h;
        let a = w / h;
        let r = STD_WEIGHT_POS * self.h.pos.max(1.0);
        self.cx.update(cx, r);
        self.cy.update(cy, r);
        self.a.update(a, 1e-1);
        self.h.update(h, r);
        self.h.pos = self.h.pos.max(1e-3);
        self.a.pos = self.a.pos.max(1e-3);
    }

    pub(crate) fn bbox(&self) -> Xyxy {
        let h = self.h.pos.max(1e-3);
        let w = (self.a.pos.max(1e-3) * h).max(1e-3);
        let x1 = (self.cx.pos - 0.5 * w) as f32;
        let y1 = (self.cy.pos - 0.5 * h) as f32;
        let x2 = (self.cx.pos + 0.5 * w) as f32;
        let y2 = (self.cy.pos + 0.5 * h) as f32;
        Xyxy::new(x1, y1, x2, y2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Direct full 8×8 matrix Kalman reference (`P' = F P F^T + Q`, `K = P H^T (H P H^T + R)^{-1}`,
    /// `P' = (I - K H) P`) to pin exact algebraic equivalence of `KalmanBox`.
    struct RefKalman8 {
        x: [f64; 8],
        p: [[f64; 8]; 8],
    }

    impl RefKalman8 {
        fn from_bbox(b: Xyxy) -> Self {
            let w = (b.x2 as f64 - b.x1 as f64).max(1e-3);
            let h = (b.y2 as f64 - b.y1 as f64).max(1e-3);
            let x = [b.x1 as f64 + 0.5 * w, b.y1 as f64 + 0.5 * h, w / h, h, 0.0, 0.0, 0.0, 0.0];
            let sp = 2.0 * STD_WEIGHT_POS * h;
            let sv = 10.0 * STD_WEIGHT_VEL * h;
            let stds = [sp, sp, 1e-2, sp, sv, sv, 1e-5, sv];
            let mut p = [[0.0; 8]; 8];
            for i in 0..8 {
                p[i][i] = stds[i] * stds[i];
            }
            RefKalman8 { x, p }
        }

        fn predict(&mut self, coasting: bool) {
            if coasting || self.x[3] + self.x[7] <= 1e-3 {
                self.x[7] = 0.0;
            }
            let h_ref = self.x[3].max(1.0);
            let qp = STD_WEIGHT_POS * h_ref;
            let qv = STD_WEIGHT_VEL * h_ref;
            let q_stds = [qp, qp, 1e-2, qp, qv, qv, 1e-5, qv];
            let mut f = [[0.0; 8]; 8];
            for (i, row) in f.iter_mut().enumerate() {
                row[i] = 1.0;
                if i < 4 {
                    row[i + 4] = 1.0;
                }
            }
            // x' = F x
            let mut nx = [0.0; 8];
            for i in 0..8 {
                for j in 0..8 {
                    nx[i] += f[i][j] * self.x[j];
                }
            }
            // P' = F P F^T + Q
            let mut fp = [[0.0; 8]; 8];
            for i in 0..8 {
                for j in 0..8 {
                    for k in 0..8 {
                        fp[i][j] += f[i][k] * self.p[k][j];
                    }
                }
            }
            let mut np = [[0.0; 8]; 8];
            for i in 0..8 {
                for j in 0..8 {
                    for k in 0..8 {
                        np[i][j] += fp[i][k] * f[j][k];
                    }
                }
                np[i][i] += q_stds[i] * q_stds[i];
            }
            self.x = nx;
            self.p = np;
        }

        fn update(&mut self, b: Xyxy) {
            let w = (b.x2 as f64 - b.x1 as f64).max(1e-3);
            let h = (b.y2 as f64 - b.y1 as f64).max(1e-3);
            let z = [b.x1 as f64 + 0.5 * w, b.y1 as f64 + 0.5 * h, w / h, h];
            let r = STD_WEIGHT_POS * self.x[3].max(1.0);
            let r_stds = [r, r, 1e-1, r];
            // S = H P H^T + R is diagonal 4x4 because P is block-diagonal.
            let mut k = [[0.0; 4]; 8];
            for j in 0..4 {
                let s_jj = self.p[j][j] + r_stds[j] * r_stds[j];
                for i in 0..8 {
                    k[i][j] = self.p[i][j] / s_jj;
                }
            }
            let mut innov = [0.0; 4];
            for j in 0..4 {
                innov[j] = z[j] - self.x[j];
            }
            for i in 0..8 {
                for j in 0..4 {
                    self.x[i] += k[i][j] * innov[j];
                }
            }
            // P' = (I - K H) P
            let mut np = [[0.0; 8]; 8];
            for i in 0..8 {
                for j in 0..8 {
                    let mut sum = self.p[i][j];
                    for m in 0..4 {
                        sum -= k[i][m] * self.p[m][j];
                    }
                    np[i][j] = sum;
                }
            }
            self.p = np;
        }
    }

    #[test]
    fn decoupled_kalman_matches_full_8x8_matrix_reference_across_steps() {
        let obs = [
            Xyxy::new(70.0, 60.0, 130.0, 140.0),  // cx=100, 60x80
            Xyxy::new(80.0, 61.0, 140.0, 141.0),  // cx=110
            Xyxy::new(90.0, 62.0, 150.0, 142.0),  // cx=120
            Xyxy::new(100.0, 63.0, 160.0, 143.0), // cx=130
            Xyxy::new(110.0, 64.0, 170.0, 144.0), // cx=140
            Xyxy::new(120.0, 65.0, 180.0, 145.0), // cx=150
            Xyxy::new(130.0, 66.0, 190.0, 146.0), // cx=160
        ];
        let mut fast = KalmanBox::from_bbox(obs[0]);
        let mut ref8 = RefKalman8::from_bbox(obs[0]);

        for &z in &obs[1..] {
            fast.predict(false);
            ref8.predict(false);
            fast.update(z);
            ref8.update(z);

            let states = [(fast.cx, 0usize), (fast.cy, 1), (fast.a, 2), (fast.h, 3)];
            for (k1d, idx) in states {
                assert!((k1d.pos - ref8.x[idx]).abs() < 1e-10, "pos[{idx}] mismatch");
                assert!((k1d.vel - ref8.x[idx + 4]).abs() < 1e-10, "vel[{idx}] mismatch");
                assert!((k1d.p00 - ref8.p[idx][idx]).abs() < 1e-10, "p00[{idx}] mismatch");
                assert!((k1d.p01 - ref8.p[idx][idx + 4]).abs() < 1e-10, "p01[{idx}] mismatch");
                assert!((k1d.p11 - ref8.p[idx + 4][idx + 4]).abs() < 1e-10, "p11[{idx}] mismatch");
            }
        }

        // Next prediction step after x=100..=160: exact Kalman center is ~168.02 (not 169.65).
        fast.predict(false);
        ref8.predict(false);
        assert!((fast.cx.pos - ref8.x[0]).abs() < 1e-10);
        assert!((fast.cx.pos - 168.02).abs() < 0.05, "unexpected predicted cx: {}", fast.cx.pos);

        // Coasting step + re-acquisition update also matches exact 8x8 matrix math.
        fast.predict(true);
        ref8.predict(true);
        fast.update(Xyxy::new(145.0, 68.0, 205.0, 148.0));
        ref8.update(Xyxy::new(145.0, 68.0, 205.0, 148.0));
        assert!((fast.cx.pos - ref8.x[0]).abs() < 1e-10);
        assert!((fast.cx.vel - ref8.x[4]).abs() < 1e-10);
        assert!((fast.cx.p11 - ref8.p[4][4]).abs() < 1e-10);
    }
}
