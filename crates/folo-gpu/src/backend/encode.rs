//! Kernel calls several ops share: the elementwise kernel, the BatchNorm and conv problem descriptions, convs in
//! pieces of the batch, and matmul on strided operands.

use folo_core::backend::{Conv2dParams, UnaryOp, View};
use folo_core::{DType as CoreDType, Error, Result, Shape, Storage};

use super::convert::{Core, bn_act, f32_only, kdt, nchw, split3, strided_of};
use super::{BACKEND_NAME, GpuBackend};
use crate::error::GpuError;
use crate::kernels::conv::{Conv2dDesc, ConvInput};
use crate::kernels::elementwise::UnaryOp as GpuUnary;
use crate::kernels::ew::{self, EwInput, EwOp};
use crate::kernels::gemm::{self, BatchStrides, GemmDesc};
use crate::kernels::norm::BnDesc;
use crate::kernels::pooling::Window2d;
use crate::kernels::strided::{self, Elem, Indices, Strided};
use crate::profile;

/// `(transposed, leading dimension)` of a `rows × cols` matrix with strides `(rs, cs)` as the GEMM can
/// address it, if it can.
fn mat_operand(rows: usize, cols: usize, rs: usize, cs: usize) -> Option<(bool, usize)> {
    let (rs1, cs1) = (if rows == 1 { cols } else { rs }, if cols == 1 { 1 } else { cs });
    if cs1 == 1 && rs1 >= cols {
        return Some((false, rs1));
    }
    let (rs2, cs2) = (if rows == 1 { 1 } else { rs }, if cols == 1 { rows } else { cs });
    if rs2 == 1 && cs2 >= rows {
        return Some((true, cs2));
    }
    None
}

/// Where a conv can read `v` in place: an NCHW view whose images each hold their `C·H·W` elements contiguously
/// (a dense tensor, or a channel slice of one, as `narrow` along dim 1 gives), images apart by at least that.
pub(crate) fn conv_input(v: &View<'_>) -> Option<ConvInput> {
    let l = v.layout;
    let (d, s) = (l.dims(), l.strides());
    if d.len() != 4 || !l.fits(v.storage.len()) {
        return None;
    }
    let (n, c, h, w) = (d[0], d[1], d[2], d[3]);
    let image = c * h * w;
    let inner = (w <= 1 || s[3] == 1) && (h <= 1 || s[2] == w) && (c <= 1 || s[1] == h * w);
    let n_stride = if n <= 1 { image } else { s[0] };
    (inner && n_stride >= image).then_some(ConvInput { offset: l.offset(), n_stride })
}

impl GpuBackend {
    /// The BatchNorm problem of `x` (f32 or f16 data; the parameters and statistics are f32).
    pub(super) fn bn_desc(&self, x: &View<'_>, act: Option<UnaryOp>, op: &'static str) -> Result<BnDesc> {
        let act = bn_act(act).ok_or(Error::NotSupported { op, backend: BACKEND_NAME })?;
        let (n, c, h, w) = nchw(x.shape());
        Ok(BnDesc { n, c, hw: h * w, act, dtype: kdt(x.storage.dtype()) })
    }

    pub(super) fn ew(&self, op: EwOp, out_shape: &Shape, inputs: &[View<'_>]) -> Result<Storage> {
        let n = out_shape.numel();
        let _p = profile::scope(|| format!("ew {op:?} {:?}", out_shape.dims()));
        // The values' dtype (a `where` condition is input 0): the output's too.
        let value = if matches!(op, EwOp::Where { .. }) { &inputs[1] } else { &inputs[0] };
        let dtype = value.storage.dtype();
        let out = self.alloc(dtype, n)?;
        let views: Vec<Strided> = inputs.iter().map(|v| strided_of(v.layout)).collect();
        let bufs = inputs.iter().map(|v| self.buf(v)).collect::<Result<Vec<_>>>()?;
        let ins: Vec<EwInput<'_>> =
            bufs.iter().zip(&views).map(|(b, v)| EwInput { buf: b, view: v }).collect();
        ew::elementwise(&self.ctx, op, out_shape.dims(), &ins, out.raw(), kdt(dtype)).core()?;
        Ok(self.wrap(dtype, n, out))
    }

    /// The conv problem of `x` and `w` (stored as `dtype`).
    pub(super) fn conv_desc(&self, x: &Shape, w: &Shape, p: &Conv2dParams, dtype: CoreDType) -> Conv2dDesc {
        let (n, c, h, wi) = nchw(x);
        let (o, _, kh, kw) = nchw(w);
        Conv2dDesc {
            n,
            c_in: c,
            h,
            w: wi,
            c_out: o,
            groups: p.groups,
            win: Window2d {
                kernel: (kh, kw),
                stride: (p.stride[0], p.stride[1]),
                padding: (p.padding[0], p.padding[1]),
                dilation: (p.dilation[0], p.dilation[1]),
            },
            act: GpuUnary::Identity,
            dtype: kdt(dtype),
        }
    }

    /// Runs the conv kernel call `f(d, x, out)` in pieces of the batch when `N · groups` exceeds one dispatch's
    /// 65535 workgroup layers (the conv kernels put `(image, group)` on the grid's z axis): each piece's images
    /// are copied to a temporary and its result copied back. `x_per` / `out_per`: elements per image.
    pub(super) fn conv_by_batch(
        &self,
        d: &Conv2dDesc,
        (x, x_per): (&wgpu::Buffer, usize),
        (out, out_per): (&wgpu::Buffer, usize),
        f: impl Fn(&Conv2dDesc, &wgpu::Buffer, &wgpu::Buffer) -> std::result::Result<(), GpuError>,
    ) -> Result<()> {
        // An even piece keeps f16 offsets 4-byte aligned, so the copies stay buffer copies; with 32768 groups or
        // more the pieces are single images, which the strided copies also move from and to odd f16 offsets.
        let piece = match 65535 / d.groups.max(1) {
            p @ 0..=1 => p,
            p => p & !1,
        };
        if d.n * d.groups <= 65535 || piece == 0 {
            return f(d, x, out).core();
        }
        let el = if d.dtype == crate::DType::F16 { Elem::F16 } else { Elem::F32 };
        let eb = d.dtype.size() as u64;
        for n0 in (0..d.n).step_by(piece) {
            let pd = Conv2dDesc { n: piece.min(d.n - n0), ..d.clone() };
            let xb = ((pd.n * x_per) as u64 * eb).next_multiple_of(4);
            let ob = ((pd.n * out_per) as u64 * eb).next_multiple_of(4);
            let (tx, to) = (self.ctx.alloc(xb).core()?, self.ctx.alloc(ob).core()?);
            let (xv, ov) = (Strided::dense(&[pd.n * x_per]), Strided::dense(&[pd.n * out_per]));
            strided::gather_strided(&self.ctx, el, x, &Strided { offset: n0 * x_per, ..xv }, tx.raw())
                .core()?;
            f(&pd, tx.raw(), to.raw()).core()?;
            let dst = Strided { offset: n0 * out_per, ..ov.clone() };
            strided::scatter_strided(&self.ctx, el, to.raw(), &ov, out, &dst).core()?;
        }
        Ok(())
    }

    /// Encodes `a @ b` for strided views, or returns `None` when an operand's layout is not addressable
    /// (the caller densifies and retries).
    pub(super) fn try_matmul(&self, a: &View<'_>, b: &View<'_>, out: &wgpu::Buffer) -> Result<Option<()>> {
        let (la, lb) = (a.layout, b.layout);
        let r = la.rank();
        let (m, k, n) = (la.dims()[r - 2], la.dims()[r - 1], lb.dims()[r - 1]);
        let Some((ta, lda)) = mat_operand(m, k, la.strides()[r - 2], la.strides()[r - 1]) else {
            return Ok(None);
        };
        let Some((tb, ldb)) = mat_operand(k, n, lb.strides()[r - 2], lb.strides()[r - 1]) else {
            return Ok(None);
        };
        let bdims = &la.dims()[..r - 2];
        let mut sc = vec![m * n; bdims.len()];
        for i in (0..bdims.len().saturating_sub(1)).rev() {
            sc[i] = sc[i + 1] * bdims[i + 1];
        }
        let (cd, cs) = strided::coalesce(bdims, &[&la.strides()[..r - 2], &lb.strides()[..r - 2], &sc]);
        if cd.len() > 2 {
            return Ok(None);
        }
        let get = |v: &Vec<usize>, i: usize| v.get(i).copied().unwrap_or(0);
        let (outer, inner) = match cd.len() {
            0 => (1, 1),
            1 => (cd[0], 1),
            _ => (cd[0], cd[1]),
        };
        let mut g = GemmDesc::new(m, n, k, ta, tb, outer);
        g.dtype = kdt(a.storage.dtype());
        g.lda = lda;
        g.ldb = ldb;
        g.ldc = n;
        g.batch = (outer, inner);
        g.a = BatchStrides { offset: la.offset(), outer: get(&cs[0], 0), inner: get(&cs[0], 1) };
        g.b = BatchStrides { offset: lb.offset(), outer: get(&cs[1], 0), inner: get(&cs[1], 1) };
        g.c = BatchStrides { offset: 0, outer: get(&cs[2], 0), inner: get(&cs[2], 1) };
        // A shared B and a batch of A rows that form one strided matrix: one taller GEMM.
        if cd.len() == 1 && g.b.outer == 0 && !ta && g.a.outer == m * lda {
            g.m = outer * m;
            g.batch = (1, 1);
            g.a.outer = 0;
            g.c.outer = 0;
        }
        // One dispatch takes at most 65535 batch matrices (workgroup layers): larger batches go in pieces.
        let (outer, inner) = g.batch;
        let (step_i, step_o) = (inner.min(65535), (65535 / inner).max(1));
        let (ba, bb) = (self.buf(a)?, self.buf(b)?);
        for o in (0..outer).step_by(step_o) {
            for i in (0..inner).step_by(step_i) {
                let mut p = g.clone();
                p.batch = (step_o.min(outer - o), step_i.min(inner - i));
                for (s, base) in [(&mut p.a, &g.a), (&mut p.b, &g.b), (&mut p.c, &g.c)] {
                    s.offset = base.offset + o * base.outer + i * base.inner;
                }
                gemm::gemm(&self.ctx, &p, ba, bb, None, out).core()?;
            }
        }
        Ok(Some(()))
    }

    /// Direct zero-cleared `index_select` backward (skips the extra `full` + `copy_strided` of the default).
    pub(super) fn run_index_select_backward(
        &self,
        grad: View<'_>,
        dim: usize,
        idx: View<'_>,
        input_shape: &Shape,
    ) -> Result<Storage> {
        let op = "index_select_backward";
        f32_only(&grad, op)?;
        self.check_fits(&grad, op)?;
        self.check_fits(&idx, op)?;
        if dim >= input_shape.rank() {
            return Err(Error::DimOutOfRange { op, dim: dim as isize, rank: input_shape.rank() });
        }
        let (outer, d, inner) = split3(input_shape.dims(), dim);
        let len = input_shape.numel();
        let out = self.alloc(CoreDType::F32, len)?;
        if len > 0 {
            self.ctx.clear(out.raw(), Elem::F32.buffer_bytes(len).core()?);
            let (id, i64) = self.indices(&idx, op)?;
            let gd = self.dense(&grad)?;
            let ix = Indices { buf: id.raw(), i64, len: idx.layout.numel() };
            strided::index_add(&self.ctx, out.raw(), (outer, d, inner), ix, gd.raw()).core()?;
        }
        Ok(self.wrap(CoreDType::F32, len, out))
    }

    /// Direct zero-cleared `gather` backward (skips the extra `full` + `copy_strided` of the default).
    pub(super) fn run_gather_backward(
        &self,
        grad: View<'_>,
        dim: usize,
        idx: View<'_>,
        input_shape: &Shape,
    ) -> Result<Storage> {
        let op = "gather_backward";
        f32_only(&grad, op)?;
        self.check_fits(&grad, op)?;
        self.check_fits(&idx, op)?;
        if dim >= input_shape.rank() {
            return Err(Error::DimOutOfRange { op, dim: dim as isize, rank: input_shape.rank() });
        }
        let (outer, d, inner) = split3(input_shape.dims(), dim);
        let di = idx.shape().dims()[dim];
        let len = input_shape.numel();
        let out = self.alloc(CoreDType::F32, len)?;
        if len > 0 {
            self.ctx.clear(out.raw(), Elem::F32.buffer_bytes(len).core()?);
            let (id, i64) = self.indices(&idx, op)?;
            let gd = self.dense(&grad)?;
            let ix = Indices { buf: id.raw(), i64, len: idx.layout.numel() };
            strided::scatter_add(&self.ctx, out.raw(), (outer, d, inner), ix, di, gd.raw()).core()?;
        }
        Ok(self.wrap(CoreDType::F32, len, out))
    }
}
