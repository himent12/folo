//! The int8 conv: packed s8 weights ([`QConv`]) and the driver ([`conv`] / [`conv_per_channel`]) that runs the
//! micro-kernel over output tiles with the epilogue.
//!
//! B is read in place, as the `f32` path does: a 1×1 stride-1 conv reads the input's C4 planes themselves, every
//! other conv a copy of its input padded with the per-channel zero points and split into stride phases (`T[a][b][i][j] =
//! x_pad[s·i + a][s·j + b]`), so tap `(ky, kx)` of group `g` is a contiguous run starting at row `ky / s`, column
//! `kx / s` of phase `(ky mod s, kx mod s)`, and output pixel `(oy, ox)` is column `oy · wq + ox` of every run.
//! The `wq − ow` extra columns per output row are computed and dropped.

use rayon::prelude::*;

use super::kernel::{Epi, MR, NR, Out, Tile, TileF32, accumulate_part_f32, kernel};
use super::{C4, Grid, SLACK, recycle, words};

/// One independently-quantised K-slice of a [`QConv`] (1 part for ordinary convs; `P` parts when concatenated
/// branches have disparate activation scales and are decoupled to preserve full 7-bit weight precision).
#[derive(Debug, Clone)]
struct PartWeights {
    /// K steps of this part: `c_p / 4 · k · k`.
    ks: usize,
    /// Offset in `Rows::rows` where this part's K-rows start.
    row_offset: usize,
    /// `[o / 4][ks][MR]` words of four s8 weights.
    a: Vec<i32>,
    /// `Σ_{k ∈ part} q[o, k] · zero(k)` per output channel.
    corr: Vec<i32>,
    /// Per output channel weight scale for this part (input scales folded in).
    scale: Vec<f32>,
}

/// A conv's weights, quantised and packed for the int8 kernel: symmetric s8 per output channel after each
/// input channel's activation scale is folded into its column, plus the zero-point correction and the `f32`
/// bias of the epilogue. Its geometry is fixed by [`QConv::new`] / [`QConv::new_parts`].
#[derive(Debug, Clone)]
pub struct QConv {
    /// Output channels (a multiple of 4).
    pub(super) o: usize,
    /// Input channels (a multiple of 4).
    pub(super) c: usize,
    /// Kernel size `k × k`.
    pub(super) k: usize,
    /// Stride.
    pub(super) stride: usize,
    /// Zero padding on each side.
    pub(super) pad: usize,
    /// SiLU in the epilogue (else identity).
    pub(super) silu: bool,
    /// The zero point of each input channel, which `corr` and the padding assume.
    zeros: Vec<u8>,
    /// Total K steps: `c / 4 · k · k`.
    ks: usize,
    /// One part normally, or `P` decoupled parts for multi-branch concats with mismatched activation scales.
    parts: Vec<PartWeights>,
    bias: Vec<f32>,
}

/// Longest reduction (`c · k²`) of a [`QConv`]: `|Σ q · x|` and `|Σ q · zero|` are each at most
/// `127 · 255 · K`, so their difference stays inside `i32` for `K ≤ 2¹⁵`.
pub const MAX_REDUCTION: usize = 1 << 15;
/// Largest stride of a [`QConv`] (its padded copy has `stride²` phases).
pub const MAX_STRIDE: usize = 64;
/// Largest padding of a [`QConv`].
pub const MAX_PAD: usize = 1 << 12;

/// Output size of a `k × k` conv with `stride` and `pad` over an `h × w` input, or `None` when the kernel does not
/// fit the padded input, the stride is 0 or the sizes overflow.
pub fn out_size(h: usize, w: usize, k: usize, stride: usize, pad: usize) -> Option<(usize, usize)> {
    let side = |v: usize| {
        let p = pad.checked_mul(2).and_then(|p| v.checked_add(p))?;
        (k >= 1 && stride >= 1 && p >= k).then(|| (p - k) / stride + 1)
    };
    Some((side(h)?, side(w)?))
}

/// [`out_size`] when [`conv`] also takes the buffer sizes: `n` images of `c` channels in, `o` out, and no buffer
/// it reads or allocates (input, padded copy, `f32` output) is over `isize::MAX` bytes, the most one allocation
/// can hold (on 32-bit targets too). `None` otherwise.
#[allow(clippy::too_many_arguments)]
pub fn conv_size(
    n: usize,
    c: usize,
    o: usize,
    h: usize,
    w: usize,
    k: usize,
    stride: usize,
    pad: usize,
) -> Option<(usize, usize)> {
    let (oh, ow) = out_size(h, w, k, stride, pad)?;
    // Elements of 4 bytes (u8 codes in C4 words, or f32), at most isize::MAX bytes.
    let fits = |v: Option<usize>| v.and_then(|v| v.checked_mul(4)).is_some_and(|b| b <= isize::MAX as usize);
    let side = |v: usize| v.checked_add(pad.checked_mul(2)?)?.checked_add(stride);
    let copy = side(h).zip(side(w)).and_then(|(a, b)| a.checked_mul(b)).and_then(|v| v.checked_mul(c));
    let input = n.checked_mul(c).zip(h.checked_mul(w)).and_then(|(a, b)| a.checked_mul(b));
    let output = n.checked_mul(o).zip(oh.checked_mul(ow)).and_then(|(a, b)| a.checked_mul(b));
    (fits(copy) && fits(input) && fits(output)).then_some((oh, ow))
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

/// Pack one K-slice covering input channels `ch0..ch0 + pc` into [`PartWeights`].
fn pack_part(w: &[f32], (o, c, taps): (usize, usize, usize), ch0: usize, part_grids: &[Grid]) -> PartWeights {
    let pc = part_grids.len();
    assert!(pc.is_multiple_of(4), "QConv part channels must be a multiple of 4, got {pc}");
    let mut folded = Vec::with_capacity(o * pc * taps);
    for oc in 0..o {
        let base = (oc * c + ch0) * taps;
        for (ic, g) in part_grids.iter().enumerate() {
            let s = g.scale;
            let row = &w[base + ic * taps..base + (ic + 1) * taps];
            folded.extend(row.iter().map(|&v| v * s));
        }
    }
    let (q, scale) = quantize_weights(&folded, o);
    let ks = pc / 4 * taps;
    let mut a = vec![0i32; o * ks];
    let mut corr = vec![0i32; o];
    for oc in 0..o {
        let row = &q[oc * pc * taps..(oc + 1) * pc * taps];
        corr[oc] = row.iter().enumerate().map(|(i, &v)| v as i32 * part_grids[i / taps].zero as i32).sum();
        for g in 0..pc / 4 {
            for t in 0..taps {
                let bytes = [0, 1, 2, 3].map(|j| row[(4 * g + j) * taps + t] as u8);
                a[(oc / 4 * ks + g * taps + t) * MR + oc % 4] = i32::from_le_bytes(bytes);
            }
        }
    }
    PartWeights { ks, row_offset: ch0 / 4 * taps, a, corr, scale }
}

/// Whether concatenated parts have disparate enough activation scales (`max / min > 1.5`) to decouple weight
/// quantisation across parts.
pub fn should_decouple_parts(part_grids: &[&[Grid]]) -> bool {
    if part_grids.len() <= 1 {
        return false;
    }
    let mut lo = f32::INFINITY;
    let mut hi = 0.0f32;
    for pg in part_grids {
        let m = pg.iter().map(|g| g.scale).fold(0.0f32, f32::max);
        if m > 0.0 {
            lo = lo.min(m);
            hi = hi.max(m);
        }
    }
    lo.is_finite() && hi > 1.5 * lo
}

impl QConv {
    /// Quantise and pack the `[o, c, k, k]` weights `w` with `bias` for inputs whose channel `i` is on grid
    /// `grids[i]`. `o` and `c` must be multiples of 4, and `c · k²` at most [`MAX_REDUCTION`] (the `i32` sums).
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
        Self::new_parts(w, bias, (o, c, k), stride, pad, silu, &[grids], false)
    }

    /// Quantise and pack `[o, c, k, k]` weights over channel-concatenated input parts (`part_grids`). When
    /// `decouple` is true (and `part_grids.len() > 1`), each part is quantised independently so a small-scale
    /// branch is never crushed by a large-scale neighbour; at runtime, each tile runs `P` sequential VNNI passes
    /// accumulating into a 384-byte L1 `TileF32` buffer (zero YMM spills, zero extra VNNI steps).
    #[allow(clippy::too_many_arguments)]
    pub fn new_parts(
        w: &[f32],
        bias: &[f32],
        (o, c, k): (usize, usize, usize),
        stride: usize,
        pad: usize,
        silu: bool,
        part_grids: &[&[Grid]],
        decouple: bool,
    ) -> QConv {
        assert!(
            QConv::supports((o, c, k), stride, pad),
            "QConv: {o} → {c} channels (multiples of 4), kernel {k}, stride {stride} (1..={MAX_STRIDE}), padding \
             {pad} (at most {MAX_PAD}), c · k² at most {MAX_REDUCTION}"
        );
        let red = c * k * k;
        assert_eq!(Some(w.len()), red.checked_mul(o), "QConv: weight length");
        assert_eq!(bias.len(), o, "QConv: bias length");
        let total_c: usize = part_grids.iter().map(|pg| pg.len()).sum();
        assert_eq!(total_c, c, "QConv: one grid per input channel");
        let taps = k * k;
        let ks = c / 4 * taps;
        let all_grids: Vec<Grid> = part_grids.iter().flat_map(|pg| pg.iter().copied()).collect();
        let zeros: Vec<u8> = all_grids.iter().map(|g| g.zero).collect();
        let parts = if decouple && part_grids.len() > 1 {
            let mut ps = Vec::with_capacity(part_grids.len());
            let mut ch0 = 0;
            for pg in part_grids {
                ps.push(pack_part(w, (o, c, taps), ch0, pg));
                ch0 += pg.len();
            }
            ps
        } else {
            vec![pack_part(w, (o, c, taps), 0, &all_grids)]
        };
        QConv { o, c, k, stride, pad, silu, zeros, ks, parts, bias: bias.to_vec() }
    }

    /// Number of independently-quantised K-parts (`1` normally, `> 1` when concat scales were decoupled).
    pub fn num_parts(&self) -> usize {
        self.parts.len()
    }

    /// Whether [`QConv::new`] takes this geometry: `o` and `c` multiples of 4, `k ≥ 1`, `c · k²` at most
    /// [`MAX_REDUCTION`], stride in `1..=`[`MAX_STRIDE`] and padding at most [`MAX_PAD`].
    pub fn supports((o, c, k): (usize, usize, usize), stride: usize, pad: usize) -> bool {
        o.is_multiple_of(4)
            && c.is_multiple_of(4)
            && k >= 1
            && (1..=MAX_STRIDE).contains(&stride)
            && pad <= MAX_PAD
            && k.checked_mul(k).and_then(|t| t.checked_mul(c)).is_some_and(|r| r <= MAX_REDUCTION)
    }

    /// Output channels.
    pub fn out_channels(&self) -> usize {
        self.o
    }

    /// Output size of an `h × w` input. Panics when the kernel does not fit the padded input or the sizes
    /// overflow.
    pub fn out_hw(&self, h: usize, w: usize) -> (usize, usize) {
        out_size(h, w, self.k, self.stride, self.pad)
            .unwrap_or_else(|| panic!("QConv: {h}×{w} input under a {0}×{0} kernel", self.k))
    }

    fn pointwise(&self) -> bool {
        self.k == 1 && self.stride == 1 && self.pad == 0
    }
}

/// Where a conv writes.
pub enum QOut<'a> {
    /// u8 codes on a single `grid`, C4, `[n][o / 4][oh · ow]` words (plus [`SLACK`] after them).
    U8 {
        /// The output words.
        data: &'a mut [u32],
        /// The output grid.
        grid: Grid,
    },
    /// u8 codes on per-channel `grids` (length `o`), C4, `[n][o / 4][oh · ow]` words (plus [`SLACK`]).
    U8PerChannel {
        /// The output words.
        data: &'a mut [u32],
        /// One output grid per channel (`grids.len() == o`).
        grids: &'a [Grid],
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

// SAFETY: the row pointers point into `buf` or into input slices that outlive the conv; they are only read.
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
        srcs.extend((0..x.c / 4).map(|g| &x.data[base + g * hw..base + (g + 1) * hw]));
    }
    let fill = |(g, dst): (usize, &mut [u32])| {
        let src = srcs[g];
        let zw = u32::from_le_bytes([
            cv.zeros[4 * g],
            cv.zeros[4 * g + 1],
            cv.zeros[4 * g + 2],
            cv.zeros[4 * g + 3],
        ]);
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
                // SAFETY: `off` is inside `buf`, which `Rows` keeps alive.
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

/// Task blocks `(chunks, channel groups)` per task for `n` images of `chunks` pixel chunks and `og` channel groups.
pub(super) fn plan(
    n: usize,
    chunks: usize,
    og: usize,
    unit: usize,
    input: usize,
    (threads, min_macs): (usize, usize),
) -> (usize, usize) {
    let macs = n.saturating_mul(chunks).saturating_mul(og).saturating_mul(unit);
    let want = if threads > 1 { (3 * threads).min(macs / min_macs.max(1)).max(1) } else { 1 };
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
    let mut ch = 0;
    for x in xs {
        x.check(n, h.checked_mul(w).expect("q8::conv: h · w overflows"));
        assert!(
            cv.zeros[ch..ch + x.c].iter().all(|&z| z == x.zero),
            "q8::conv: input zero point differs from the weights'"
        );
        ch += x.c;
    }
    let res_grids: Option<Vec<Grid>> = res.as_ref().map(|(_, g)| vec![*g; cv.o]);
    let res_pc = res.as_ref().zip(res_grids.as_deref()).map(|((r, _), gs)| (*r, gs));
    conv_inner(xs, n, h, w, cv, res_pc, out);
}

/// Per-channel variant of [`conv`]: each input part carries its own per-channel zero points (`in_zeros`, checked
/// against the weights' packed zero points), `res` (if any) carries per-channel grids (`[o]`), and `out` may use
/// [`QOut::U8PerChannel`].
#[allow(clippy::too_many_arguments)]
pub fn conv_per_channel(
    xs: &[C4<'_>],
    in_zeros: &[&[u8]],
    n: usize,
    h: usize,
    w: usize,
    cv: &QConv,
    res: Option<(C4<'_>, &[Grid])>,
    out: QOut<'_>,
) {
    assert_eq!(xs.len(), in_zeros.len(), "q8::conv_per_channel: zero-point slice count");
    assert_eq!(xs.iter().map(|x| x.c).sum::<usize>(), cv.c, "q8::conv: input channels");
    let mut ch = 0;
    for (x, zs) in xs.iter().zip(in_zeros) {
        x.check(n, h.checked_mul(w).expect("q8::conv: h · w overflows"));
        assert_eq!(zs.len(), x.c, "q8::conv_per_channel: zero-point length");
        assert_eq!(&cv.zeros[ch..ch + x.c], *zs, "q8::conv: input zero point differs from the weights'");
        ch += x.c;
    }
    conv_inner(xs, n, h, w, cv, res, out);
}

fn conv_inner(
    xs: &[C4<'_>],
    n: usize,
    h: usize,
    w: usize,
    cv: &QConv,
    res: Option<(C4<'_>, &[Grid])>,
    out: QOut<'_>,
) {
    let (oh, ow) = cv.out_hw(h, w);
    let sizes = conv_size(n, cv.c, cv.o, h, w, cv.k, cv.stride, cv.pad);
    assert!(sizes.is_some(), "q8::conv: {n} × {h}×{w} overflows");
    let ohw = oh * ow;
    let words_n = |c: usize| n * c * ohw;
    let og = cv.o / 4;
    if let Some((r, gs)) = &res {
        assert_eq!(r.c, cv.o, "q8::conv: residual channels");
        assert_eq!(gs.len(), cv.o, "q8::conv: residual grids length");
        r.check(n, ohw);
    }
    type GroupGrid = ([f32; MR], [i32; MR]);
    let (qptr, fptr, is_u8, group_out): (OutPtr<u32>, OutPtr<f32>, bool, Vec<GroupGrid>) = match out {
        QOut::U8 { data, grid } => {
            assert!(data.len() >= words_n(og), "q8::conv: output too short");
            let pair = ([grid.inv(); MR], [grid.zero as i32; MR]);
            (OutPtr(data.as_mut_ptr()), OutPtr(std::ptr::null_mut()), true, vec![pair; og])
        }
        QOut::U8PerChannel { data, grids } => {
            assert!(data.len() >= words_n(og), "q8::conv: output too short");
            assert_eq!(grids.len(), cv.o, "q8::conv: output grids length");
            let pairs = (0..og)
                .map(|g| {
                    (
                        [
                            grids[4 * g].inv(),
                            grids[4 * g + 1].inv(),
                            grids[4 * g + 2].inv(),
                            grids[4 * g + 3].inv(),
                        ],
                        [
                            grids[4 * g].zero as i32,
                            grids[4 * g + 1].zero as i32,
                            grids[4 * g + 2].zero as i32,
                            grids[4 * g + 3].zero as i32,
                        ],
                    )
                })
                .collect();
            (OutPtr(data.as_mut_ptr()), OutPtr(std::ptr::null_mut()), true, pairs)
        }
        QOut::F32 { data } => {
            assert!(data.len() >= words_n(cv.o), "q8::conv: output too short");
            (OutPtr(std::ptr::null_mut()), OutPtr(data.as_mut_ptr()), false, Vec::new())
        }
    };
    let group_res: Vec<([f32; MR], [i32; MR])> = match &res {
        Some((_, gs)) => (0..og)
            .map(|g| {
                (
                    [gs[4 * g].scale, gs[4 * g + 1].scale, gs[4 * g + 2].scale, gs[4 * g + 3].scale],
                    [
                        gs[4 * g].zero as i32,
                        gs[4 * g + 1].zero as i32,
                        gs[4 * g + 2].zero as i32,
                        gs[4 * g + 3].zero as i32,
                    ],
                )
            })
            .collect(),
        None => Vec::new(),
    };
    let res_view = res.map(|(r, _)| r);
    let rows: Vec<Rows> = (0..n).map(|i| image_rows(xs, i, h, w, cv)).collect();
    let chunks = rows[0].n.div_ceil(NR);
    let (ch_blk, og_blk) = plan(
        n,
        chunks,
        og,
        MR * NR * cv.ks * 4,
        cv.c * rows[0].n,
        (rayon::current_num_threads(), task_macs()),
    );
    let (cbs, obs) = (chunks.div_ceil(ch_blk), og.div_ceil(og_blk));
    let kern = kernel();
    let single_part = cv.parts.len() == 1;
    let zero_corr = [0i32; MR];
    let unit_scale = [1.0f32; MR];
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
                let out_kind = if is_u8 {
                    let (inv, zero) = &group_out[g];
                    Out::U8 { inv, zero }
                } else {
                    Out::F32
                };
                let res_opt = res_view.as_ref().map(|_| (&group_res[g].0, &group_res[g].1));
                if let Some(rv) = &res_view {
                    let base = rv.offset + img * rv.img_stride + g * ohw;
                    if contiguous {
                        resw.copy_from_slice(&rv.data[base + pix[0]..base + pix[0] + NR]);
                    } else {
                        for (d, &px) in resw.iter_mut().zip(&pix) {
                            *d = if px == usize::MAX { 0 } else { rv.data[base + px] };
                        }
                    }
                }
                if single_part {
                    let p0 = &cv.parts[0];
                    // SAFETY: the panel holds `ks · MR` words; every row has `col + NR` readable words.
                    unsafe { kern(p0.ks, p0.a[g * p0.ks * MR..].as_ptr(), r.rows.as_ptr(), col, &mut acc) };
                    let e = Epi {
                        corr: p0.corr[4 * g..4 * g + 4].try_into().unwrap(),
                        scale: p0.scale[4 * g..4 * g + 4].try_into().unwrap(),
                        bias: cv.bias[4 * g..4 * g + 4].try_into().unwrap(),
                        silu: cv.silu,
                        res: res_opt,
                        out: out_kind,
                    };
                    if cols == NR {
                        e.apply_full(&acc, &resw, &mut qw, &mut fv);
                    } else {
                        e.apply(&acc, cols, &resw, &mut qw, &mut fv);
                    }
                } else {
                    // Register-safe sequential per-part VNNI accumulation into L1-resident `tile_f32` (384 B).
                    let mut tile_f32: TileF32 = [[0.0; NR]; MR];
                    for part in &cv.parts {
                        // SAFETY: each part's panel holds `part.ks · MR` words and `rows[row_offset..]` has
                        // `part.ks` valid row pointers with `col + NR` readable words.
                        unsafe {
                            kern(
                                part.ks,
                                part.a[g * part.ks * MR..].as_ptr(),
                                r.rows[part.row_offset..].as_ptr(),
                                col,
                                &mut acc,
                            );
                        }
                        accumulate_part_f32(
                            &acc,
                            part.corr[4 * g..4 * g + 4].try_into().unwrap(),
                            part.scale[4 * g..4 * g + 4].try_into().unwrap(),
                            &mut tile_f32,
                        );
                    }
                    let e = Epi {
                        corr: &zero_corr,
                        scale: &unit_scale,
                        bias: cv.bias[4 * g..4 * g + 4].try_into().unwrap(),
                        silu: cv.silu,
                        res: res_opt,
                        out: out_kind,
                    };
                    if cols == NR {
                        e.apply_f32_full(&tile_f32, &resw, &mut qw, &mut fv);
                    } else {
                        e.apply_f32(&tile_f32, cols, &resw, &mut qw, &mut fv);
                    }
                }
                if is_u8 {
                    if contiguous {
                        let at = (img * og + g) * ohw + pix[0];
                        // SAFETY: the NR words at `at` are this task's own output words, inside `data`.
                        unsafe { std::ptr::copy_nonoverlapping(qw.as_ptr(), qptr.at(at), NR) };
                    } else {
                        let base = (img * og + g) * ohw;
                        for (j, &px) in pix.iter().enumerate().take(cols) {
                            if px != usize::MAX {
                                // SAFETY: (img, g, px) is this task's own output word, inside `data`.
                                unsafe { *qptr.at(base + px) = qw[j] };
                            }
                        }
                    }
                } else if contiguous {
                    for (rr, row) in fv.iter().enumerate() {
                        let at = (img * cv.o + 4 * g + rr) * ohw + pix[0];
                        // SAFETY: the NR values at `at` are this task's own outputs, inside `data`.
                        unsafe { std::ptr::copy_nonoverlapping(row.as_ptr(), fptr.at(at), NR) };
                    }
                } else {
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
    });
    for r in rows {
        recycle(r.buf);
    }
}
