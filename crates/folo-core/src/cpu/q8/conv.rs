//! The int8 conv: packed s8 weights ([`QConv`]) and the driver ([`conv`]) that runs the micro-kernel over
//! output tiles with the epilogue.
//!
//! B is read in place, as the `f32` path does: a 1×1 stride-1 conv reads the input's C4 planes themselves, every
//! other conv a copy of its input padded with the zero point and split into stride phases (`T[a][b][i][j] =
//! x_pad[s·i + a][s·j + b]`), so tap `(ky, kx)` of group `g` is a contiguous run starting at row `ky / s`, column
//! `kx / s` of phase `(ky mod s, kx mod s)`, and output pixel `(oy, ox)` is column `oy · wq + ox` of every run.
//! The `wq − ow` extra columns per output row are computed and dropped.

use rayon::prelude::*;

use super::kernel::{Epi, MR, NR, Out, Tile, kernel};
use super::{C4, Grid, SLACK, recycle, words};

/// A conv's weights, quantised and packed for the int8 kernel: symmetric s8 per output channel after each
/// input channel's activation scale is folded into its column, plus the zero-point correction and the `f32`
/// bias of the epilogue.
#[derive(Debug, Clone)]
pub struct QConv {
    /// Output channels (a multiple of 4).
    pub o: usize,
    /// Input channels (a multiple of 4).
    pub c: usize,
    /// Kernel size `k × k`.
    pub k: usize,
    /// Stride.
    pub stride: usize,
    /// Zero padding on each side.
    pub pad: usize,
    /// SiLU in the epilogue (else identity).
    pub silu: bool,
    /// K steps: `c / 4 · k · k`.
    ks: usize,
    /// `[o / 4][ks][MR]` words of four s8 weights (input channels `4g..4g+4` of tap `(ky, kx)`, step
    /// `(g, ky, kx)`).
    a: Vec<i32>,
    /// `Σ_k q[o, k] · zero(k)` per output channel.
    corr: Vec<i32>,
    /// Per output channel: the weight scale (input scales folded in).
    scale: Vec<f32>,
    bias: Vec<f32>,
}

/// Symmetric per-output-channel s8 weights of `w` (`[o, K]` row-major): `w[r, k] ≈ scale[r] · q[r, k]`, `q` in
/// `-127..=127` (`round(w / scale)`, ties away from zero). Returns `(q, scale)`.
pub fn quantize_weights(w: &[f32], o: usize) -> (Vec<i8>, Vec<f32>) {
    let k = w.len() / o.max(1);
    let mut q = vec![0i8; w.len()];
    let mut scales = vec![0f32; o];
    for r in 0..o {
        let row = &w[r * k..(r + 1) * k];
        let m = row.iter().fold(0f32, |a, &v| a.max(v.abs()));
        let s = if m > 0.0 { m / 127.0 } else { 1.0 };
        scales[r] = s;
        for (d, &v) in q[r * k..(r + 1) * k].iter_mut().zip(row) {
            *d = (v / s).round().clamp(-127.0, 127.0) as i8;
        }
    }
    (q, scales)
}

impl QConv {
    /// Quantise and pack the `[o, c, k, k]` weights `w` with `bias` for inputs whose channel `i` is on grid
    /// `grids[i]`. `o` and `c` must be multiples of 4.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        w: &[f32],
        bias: &[f32],
        (o, c, k): (usize, usize, usize),
        stride: usize,
        pad: usize,
        silu: bool,
        grids: &[Grid],
    ) -> QConv {
        assert!(o % 4 == 0 && c % 4 == 0, "QConv: channels {o} → {c} must be multiples of 4");
        assert_eq!(w.len(), o * c * k * k, "QConv: weight length");
        assert_eq!(grids.len(), c, "QConv: one grid per input channel");
        assert_eq!(bias.len(), o, "QConv: bias length");
        let taps = k * k;
        let folded: Vec<f32> = w.iter().enumerate().map(|(i, &v)| v * grids[(i / taps) % c].scale).collect();
        let (q, scale) = quantize_weights(&folded, o);
        let ks = c / 4 * taps;
        let mut a = vec![0i32; o * ks];
        let mut corr = vec![0i32; o];
        for oc in 0..o {
            let row = &q[oc * c * taps..(oc + 1) * c * taps];
            corr[oc] = row.iter().enumerate().map(|(i, &v)| v as i32 * grids[i / taps].zero as i32).sum();
            for g in 0..c / 4 {
                for t in 0..taps {
                    let bytes = [0, 1, 2, 3].map(|j| row[(4 * g + j) * taps + t] as u8);
                    a[(oc / 4 * ks + g * taps + t) * MR + oc % 4] = i32::from_le_bytes(bytes);
                }
            }
        }
        QConv { o, c, k, stride, pad, silu, ks, a, corr, scale, bias: bias.to_vec() }
    }

    /// Output size of an `h × w` input.
    pub fn out_hw(&self, h: usize, w: usize) -> (usize, usize) {
        ((h + 2 * self.pad - self.k) / self.stride + 1, (w + 2 * self.pad - self.k) / self.stride + 1)
    }

    fn pointwise(&self) -> bool {
        self.k == 1 && self.stride == 1 && self.pad == 0
    }
}

/// Where a conv writes.
pub enum QOut<'a> {
    /// u8 codes on `grid`, C4, `[n][o / 4][oh · ow]` words (plus [`SLACK`] after them).
    U8 {
        /// The output words.
        data: &'a mut [u32],
        /// The output grid.
        grid: Grid,
    },
    /// `f32`, `[n][o][oh · ow]`.
    F32 {
        /// The output values.
        data: &'a mut [f32],
    },
}

/// The B rows of one image: row pointers into the input planes or a padded copy, the column count and the
/// padded row width.
struct Rows {
    buf: Vec<u32>,
    rows: Vec<*const u32>,
    n: usize,
    wq: usize,
}

// SAFETY: the row pointers point into `_buf` or into input slices that outlive the conv; they are only read.
unsafe impl Sync for Rows {}
// SAFETY: as above.
unsafe impl Send for Rows {}

/// An output pointer shared by tasks that write disjoint elements.
struct OutPtr<T>(*mut T);
// SAFETY: tasks write disjoint elements (each owns its tiles).
unsafe impl<T> Sync for OutPtr<T> {}
// SAFETY: as above.
unsafe impl<T> Send for OutPtr<T> {}
impl<T> OutPtr<T> {
    fn at(&self, i: usize) -> *mut T {
        // SAFETY: callers stay inside the buffer (checked against its length before the tasks start).
        unsafe { self.0.add(i) }
    }
}

fn image_rows(xs: &[C4<'_>], img: usize, h: usize, w: usize, cv: &QConv) -> Rows {
    let hw = h * w;
    if cv.pointwise() {
        let mut rows = Vec::with_capacity(cv.ks);
        for x in xs {
            let base = x.offset + img * x.img_stride;
            rows.extend((0..x.c / 4).map(|g| x.data[base + g * hw..].as_ptr()));
        }
        return Rows { buf: Vec::new(), rows, n: hw, wq: w };
    }
    let (s, p, k) = (cv.stride, cv.pad, cv.k);
    let (oh, _) = cv.out_hw(h, w);
    let (hq, wq) = ((h + 2 * p).div_ceil(s), (w + 2 * p).div_ceil(s));
    let plane = hq * wq;
    // The phases some tap reads, each with its slot.
    let mut slot = vec![usize::MAX; s * s];
    let mut phases = Vec::new();
    for ky in 0..k {
        for kx in 0..k {
            let ph = (ky % s) * s + kx % s;
            if slot[ph] == usize::MAX {
                slot[ph] = phases.len();
                phases.push(ph);
            }
        }
    }
    let per_g = phases.len() * plane;
    let groups = cv.c / 4;
    let mut buf = words(groups * per_g + wq * k + NR + SLACK);
    let mut srcs = Vec::with_capacity(groups);
    for x in xs {
        let base = x.offset + img * x.img_stride;
        srcs.extend((0..x.c / 4).map(|g| (&x.data[base + g * hw..base + (g + 1) * hw], x.zero)));
    }
    let fill = |(g, dst): (usize, &mut [u32])| {
        let (src, z) = srcs[g];
        let zw = u32::from_le_bytes([z; 4]);
        for (si, &ph) in phases.iter().enumerate() {
            let (a, b) = (ph / s, ph % s);
            let d = &mut dst[si * plane..(si + 1) * plane];
            for (i, drow) in d.chunks_exact_mut(wq).enumerate() {
                let y = (s * i + a) as isize - p as isize;
                if y < 0 || y >= h as isize {
                    drow.fill(zw);
                    continue;
                }
                let srow = &src[y as usize * w..(y as usize + 1) * w];
                for (j, v) in drow.iter_mut().enumerate() {
                    let x = (s * j + b) as isize - p as isize;
                    *v = if x < 0 || x >= w as isize { zw } else { srow[x as usize] };
                }
            }
        }
    };
    let (body, _) = buf.split_at_mut(groups * per_g);
    if body.len() >= 1 << 15 {
        body.par_chunks_mut(per_g).enumerate().for_each(fill);
    } else {
        body.chunks_mut(per_g).enumerate().for_each(fill);
    }
    let base = buf.as_ptr();
    let mut rows = Vec::with_capacity(cv.ks);
    for g in 0..groups {
        for ky in 0..k {
            for kx in 0..k {
                let off = g * per_g + slot[(ky % s) * s + kx % s] * plane + (ky / s) * wq + kx / s;
                // SAFETY: `off` is inside `buf`, which `Rows` keeps alive (moving the Vec does not move its heap block).
                rows.push(unsafe { base.add(off) });
            }
        }
    }
    Rows { buf, rows, n: oh * wq, wq }
}

/// Smallest amount of work (MACs) worth a task of its own: below `threads` of these the pool is not filled.
pub(super) const TASK_MACS: usize = 1 << 20;

/// Tests set this to split even their small convs into many tasks (0: [`TASK_MACS`]).
#[cfg(test)]
pub(super) static TEST_TASK_MACS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn task_macs() -> usize {
    #[cfg(test)]
    {
        let t = TEST_TASK_MACS.load(std::sync::atomic::Ordering::Relaxed);
        if t > 0 {
            return t;
        }
    }
    TASK_MACS
}

/// Input bytes of one image up to which tasks split the output channels first (see [`plan`]).
const SPLIT_CHANNELS_MAX: usize = 384 << 10;

/// Task blocks `(chunks, channel groups)` per task for `n` images of `chunks` pixel chunks and `og` channel groups,
/// `unit` MACs each (one chunk of one group), the image's input taking `input` bytes: about three tasks per thread
/// so work stealing evens out fast and slow cores, but no task under [`TASK_MACS`].
///
/// A small input (up to [`SPLIT_CHANNELS_MAX`], which stays in a core's L2) is split by channel groups first:
/// every task reads the whole input but writes whole output planes, where pixel blocks of a few chunks would share
/// cache lines with their neighbours on every plane (at 20², 1×1 128 → 256 on 28 threads: 0.107 → 0.072 ms). A
/// big one is split by pixel chunks first, so each task reads its own part of it once.
pub(super) fn plan(
    n: usize,
    chunks: usize,
    og: usize,
    unit: usize,
    input: usize,
    threads: usize,
) -> (usize, usize) {
    let macs = n * chunks * og * unit;
    let want = if threads > 1 { (3 * threads).min(macs / task_macs()).max(1) } else { 1 };
    let per_img = want.div_ceil(n);
    if input <= SPLIT_CHANNELS_MAX {
        let og_blk = og.div_ceil(per_img.min(og));
        let obs = og.div_ceil(og_blk);
        (chunks.div_ceil(per_img.div_ceil(obs).min(chunks)), og_blk)
    } else {
        let ch_blk = chunks.div_ceil(per_img.min(chunks));
        let cbs = chunks.div_ceil(ch_blk);
        (ch_blk, og.div_ceil(per_img.div_ceil(cbs).min(og)))
    }
}

/// `out = epilogue(conv(cat(xs)))` for `n` images of `h × w` (docs: [`super`]). `xs` are the input parts, in
/// channel order; `res` is a residual on the output's shape (C4, `[n][o / 4][oh · ow]`) and grid, added after
/// the activation. Runs on the current rayon pool; the result does not depend on its size.
pub fn conv(
    xs: &[C4<'_>],
    n: usize,
    h: usize,
    w: usize,
    cv: &QConv,
    res: Option<(C4<'_>, Grid)>,
    out: QOut<'_>,
) {
    assert_eq!(xs.iter().map(|x| x.c).sum::<usize>(), cv.c, "q8::conv: input channels");
    for x in xs {
        x.check(n, h * w);
    }
    let (oh, ow) = cv.out_hw(h, w);
    let ohw = oh * ow;
    let og = cv.o / 4;
    if let Some((r, _)) = &res {
        assert_eq!(r.c, cv.o, "q8::conv: residual channels");
        r.check(n, ohw);
    }
    let (qptr, fptr, out_kind) = match out {
        QOut::U8 { data, grid } => {
            assert!(data.len() >= n * og * ohw, "q8::conv: output too short");
            (
                OutPtr(data.as_mut_ptr()),
                OutPtr(std::ptr::null_mut::<f32>()),
                Out::U8 { inv: grid.inv(), zero: grid.zero as i32 },
            )
        }
        QOut::F32 { data } => {
            assert!(data.len() >= n * cv.o * ohw, "q8::conv: output too short");
            (OutPtr(std::ptr::null_mut::<u32>()), OutPtr(data.as_mut_ptr()), Out::F32)
        }
    };
    let rows: Vec<Rows> = (0..n).map(|i| image_rows(xs, i, h, w, cv)).collect();
    let chunks = rows[0].n.div_ceil(NR);
    let (ch_blk, og_blk) =
        plan(n, chunks, og, MR * NR * cv.ks * 4, cv.c * rows[0].n, rayon::current_num_threads());
    let (cbs, obs) = (chunks.div_ceil(ch_blk), og.div_ceil(og_blk));
    let kern = kernel();
    let res = res.map(|(r, g)| (r, (g.scale, g.zero as i32)));
    (0..n * cbs * obs).into_par_iter().for_each(|t| {
        let (img, rest) = (t / (cbs * obs), t % (cbs * obs));
        let (cb, ob) = (rest / obs, rest % obs);
        let r = &rows[img];
        let mut acc: Tile = [[0; NR]; MR];
        let mut resw = [0u32; NR];
        let mut qw = [0u32; NR];
        let mut fv = [[0f32; NR]; MR];
        for chunk in cb * ch_blk..((cb + 1) * ch_blk).min(chunks) {
            let col = chunk * NR;
            let cols = NR.min(r.n - col);
            // Output pixel of each column (`usize::MAX`: dropped). Columns are `oy · wq + ox`; when `wq == ow` and
            // the chunk is whole, they are the pixels `col..col + NR` themselves (`r.n = oh · wq`), so only
            // `pix[0]` is read.
            let contiguous = r.wq == ow && cols == NR;
            let mut pix = [usize::MAX; NR];
            if contiguous {
                pix[0] = col;
            } else {
                let (mut oy, mut ox) = (col / r.wq, col % r.wq);
                for px in pix.iter_mut().take(cols) {
                    if oy < oh && ox < ow {
                        *px = oy * ow + ox;
                    }
                    ox += 1;
                    if ox == r.wq {
                        (oy, ox) = (oy + 1, 0);
                    }
                }
            }
            for g in ob * og_blk..((ob + 1) * og_blk).min(og) {
                // SAFETY: the panel holds `ks · MR` words; every row has `col + NR` readable words (planes
                // followed by SLACK, or the padded copy's slack).
                unsafe { kern(cv.ks, cv.a[g * cv.ks * MR..].as_ptr(), r.rows.as_ptr(), col, &mut acc) };
                let e = Epi {
                    corr: cv.corr[4 * g..4 * g + 4].try_into().unwrap(),
                    scale: cv.scale[4 * g..4 * g + 4].try_into().unwrap(),
                    bias: cv.bias[4 * g..4 * g + 4].try_into().unwrap(),
                    silu: cv.silu,
                    res: res.as_ref().map(|(_, q)| *q),
                    out: out_kind,
                };
                if let Some((rv, _)) = &res {
                    let base = rv.offset + img * rv.img_stride + g * ohw;
                    if contiguous {
                        resw.copy_from_slice(&rv.data[base + pix[0]..base + pix[0] + NR]);
                    } else {
                        for (d, &px) in resw.iter_mut().zip(&pix) {
                            *d = if px == usize::MAX { 0 } else { rv.data[base + px] };
                        }
                    }
                }
                if cols == NR {
                    e.apply_full(&acc, &resw, &mut qw, &mut fv);
                } else {
                    e.apply(&acc, cols, &resw, &mut qw, &mut fv);
                }
                match out_kind {
                    Out::U8 { .. } if contiguous => {
                        let at = (img * og + g) * ohw + pix[0];
                        // SAFETY: the NR words at `at` are this task's own output words, inside `data`.
                        unsafe { std::ptr::copy_nonoverlapping(qw.as_ptr(), qptr.at(at), NR) };
                    }
                    Out::U8 { .. } => {
                        let base = (img * og + g) * ohw;
                        for (j, &px) in pix.iter().enumerate().take(cols) {
                            if px != usize::MAX {
                                // SAFETY: (img, g, px) is this task's own output word, inside `data`.
                                unsafe { *qptr.at(base + px) = qw[j] };
                            }
                        }
                    }
                    Out::F32 if contiguous => {
                        for (rr, row) in fv.iter().enumerate() {
                            let at = (img * cv.o + 4 * g + rr) * ohw + pix[0];
                            // SAFETY: the NR values at `at` are this task's own outputs, inside `data`.
                            unsafe { std::ptr::copy_nonoverlapping(row.as_ptr(), fptr.at(at), NR) };
                        }
                    }
                    Out::F32 => {
                        for (rr, row) in fv.iter().enumerate() {
                            let base = (img * cv.o + 4 * g + rr) * ohw;
                            for (j, &px) in pix.iter().enumerate().take(cols) {
                                if px != usize::MAX {
                                    // SAFETY: as above, for the f32 output.
                                    unsafe { *fptr.at(base + px) = row[j] };
                                }
                            }
                        }
                    }
                }
            }
        }
    });
    for r in rows {
        recycle(r.buf);
    }
}
