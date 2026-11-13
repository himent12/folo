//! [`GpuBackend`]'s candidate-first detection hooks (folo-core's `conv2d_act_sites`, `detect_class_sites` and
//! `detect_decode_sites`, docs/MODEL.md §9.5) over [`crate::kernels::conv::conv2d_sites`] and
//! [`crate::kernels::detect_sites`].

use folo_core::backend::{ClassSites, Conv2dParams, SiteLevel, UnaryOp, View};
use folo_core::{DType as CoreDType, Error, Result, Storage};

use super::convert::{Core, conv_act, kdt, nchw};
use super::encode::conv_input;
use super::{BACKEND_NAME, GpuBackend};
use crate::error::GpuError;
use crate::kernels::conv::{self, ConvInput};
use crate::kernels::detect_sites::{self, DecodeLevel, HEADER_WORDS, RECORD_WORDS, SiteLists};

impl GpuBackend {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn sites_conv(
        &self,
        x: View<'_>,
        w: View<'_>,
        bias: Option<View<'_>>,
        p: &Conv2dParams,
        act: Option<UnaryOp>,
        sites: View<'_>,
        to_map: bool,
    ) -> Result<Storage> {
        let unsupported = || Error::NotSupported { op: "conv2d_act_sites", backend: BACKEND_NAME };
        let dtype = x.storage.dtype();
        let mut d = self.conv_desc(x.shape(), w.shape(), p, dtype);
        d.act = conv_act(act).ok_or_else(unsupported)?;
        if d.groups != 1 || d.n > 65535 {
            return Err(unsupported());
        }
        let (oh, ow) = d.output().core()?;
        let cap = sites.shape().dims()[1] - 1;
        // A channel slice (or any view with contiguous images) is read in place, as the dense conv reads it.
        let xd;
        let (xb, at) = match conv_input(&x) {
            Some(at) => (self.buf(&x)?, at),
            None => {
                xd = self.dense(&x)?;
                (xd.raw(), ConvInput { offset: 0, n_stride: d.c_in * d.h * d.w })
            }
        };
        let (wd, sd) = (self.dense(&w)?, self.dense(&sites)?);
        let bd = bias.as_ref().map(|b| self.dense(b)).transpose()?;
        let len = d.n * d.c_out * cap;
        let out = self.alloc(dtype, len)?;
        let bias = bd.as_ref().map(|b| b.raw());
        match conv::conv2d_sites(&self.ctx, &d, (xb, at), wd.raw(), bias, sd.raw(), cap, out.raw()) {
            Err(GpuError::Unsupported(_)) => return Err(unsupported()),
            r => r.core()?,
        }
        if !to_map {
            return Ok(self.wrap(dtype, len, out));
        }
        let full = d.n * d.c_out * oh * ow;
        let map = self.alloc(dtype, full)?;
        let dims = (d.n, d.c_out, cap, oh * ow);
        detect_sites::scatter_sites(&self.ctx, out.raw(), sd.raw(), map.raw(), dims, kdt(dtype)).core()?;
        Ok(self.wrap(dtype, full, map))
    }

    pub(super) fn class_sites(&self, cls: View<'_>, p: &ClassSites) -> Result<(Storage, Storage, Storage)> {
        let (b, nc, h, w) = nchw(cls.shape());
        let (cap, fcap) = (p.cap, (9 * p.cap).min(h * w));
        let l = SiteLists { b, h, w, nc, cap, fcap };
        let cd = self.dense(&cls)?;
        let u = CoreDType::U32;
        let lens = [b * (1 + cap), b * 2 * cap, b * (1 + fcap)];
        let [sites, recs, field] = lens.map(|n| self.alloc(u, n));
        let (sites, recs, field, mark) = (sites?, recs?, field?, self.alloc(u, b * h * w)?);
        let bufs = (sites.raw(), recs.raw(), field.raw(), mark.raw());
        detect_sites::class_sites(&self.ctx, cd.raw(), &l, p.conf, kdt(cls.storage.dtype()), bufs).core()?;
        Ok((self.wrap(u, lens[0], sites), self.wrap(u, lens[1], recs), self.wrap(u, lens[2], field)))
    }

    pub(super) fn decode_sites(&self, levels: &[SiteLevel<'_>], reg_max: usize) -> Result<Storage> {
        let dtype = levels[0].raw.storage.dtype();
        let b = levels[0].raw.shape().dims()[0];
        let mut dense = Vec::with_capacity(levels.len());
        for l in levels {
            dense.push((self.dense(&l.raw)?, self.dense(&l.sites)?, self.dense(&l.recs)?));
        }
        let dl: Vec<DecodeLevel<'_>> = levels
            .iter()
            .zip(&dense)
            .map(|(l, (raw, sites, recs))| DecodeLevel {
                raw: raw.raw(),
                sites: sites.raw(),
                recs: recs.raw(),
                cap: l.sites.shape().dims()[1] - 1,
                w: l.width,
                stride: l.stride,
                start: l.start,
            })
            .collect();
        let slots: usize = dl.iter().map(|l| l.cap).sum();
        let len = b * (HEADER_WORDS + RECORD_WORDS * slots);
        let out = self.alloc(CoreDType::U32, len)?;
        detect_sites::decode_sites(&self.ctx, &dl, b, reg_max, kdt(dtype), out.raw()).core()?;
        Ok(self.wrap(CoreDType::U32, len, out))
    }
}
