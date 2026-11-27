//! Quantised forward graph for FOLO v2-Iso (`FoloV2Model`), mirroring [`FoloV2Model::forward_features`]
//! with zero-copy `FoloBlock` parts, C4 `SPD(2×2)` on `P2`, `HyperContext` at `P5`, and relocated `FullPadGate`
//! applied strictly on `f32` head inputs (`P3_head`, `P4_head`) after `neck.down3`/`neck.down4` taps.

use folo_core::Tensor;
use folo_nn::Result;

use crate::model::v2::{FoloBlock, FoloV2Model, HyperContext};

use super::{Exec, NameBuf, run_conv, sppf};

struct FoloBlockParts<A> {
    y: A,
    mid_tap: Option<A>,
    last: A,
}

/// [`FoloBlock::forward_cat`] body: `cv1` over `xs`, `reps` chain on its second half (keeping `mid_tap` when
/// `m >= 3`), returning `y`, `mid_tap`, and `last` for `cv2`.
fn folo_block_parts<E: Exec>(
    e: &mut E,
    nb: &mut NameBuf,
    b: &FoloBlock,
    xs: &[&E::A],
) -> Result<FoloBlockParts<E::A>> {
    let l2 = nb.push_field("cv2");
    let cv2_f32 = e.wants_f32(nb.as_str(), &b.cv2);
    nb.restore(l2);

    let first_rep_f32 = match b.reps.items().first() {
        Some(first) => {
            let lr = nb.push_rep(0);
            let ok = e.wants_f32_rep(nb.as_str(), first);
            nb.restore(lr);
            ok
        }
        None => true,
    };

    let l1 = nb.push_field("cv1");
    let y = run_conv(e, nb.as_str(), &b.cv1, xs, None, cv2_f32 && first_rep_f32)?;
    nb.restore(l1);

    if !cv2_f32 {
        e.ensure_u8(&y)?;
    }

    let m = b.reps.len();
    let mid_idx = if m >= 3 { Some(m / 2 - 1) } else { None };
    let first = e.narrow(&y, b.c_mid, b.c_mid)?;
    let mut cur: Option<E::A> = None;
    let mut mid_tap: Option<E::A> = None;

    for (j, rep) in b.reps.iter().enumerate() {
        let next_f32 = if j + 1 < m {
            let lr = nb.push_rep(j + 1);
            let next_rep_f32 = e.wants_f32_rep(nb.as_str(), &b.reps.items()[j + 1]);
            nb.restore(lr);
            next_rep_f32 && (mid_idx != Some(j) || cv2_f32)
        } else {
            cv2_f32
        };
        let src = match (&cur, &mid_tap) {
            (Some(c), _) => c,
            (None, Some(mt)) => mt,
            (None, None) => &first,
        };
        let lr = nb.push_rep(j);
        let next = e.rep_conv(nb.as_str(), rep, src, next_f32)?;
        nb.restore(lr);
        if mid_idx == Some(j) {
            mid_tap = Some(next);
            cur = None;
        } else {
            cur = Some(next);
        }
    }
    drop(first);
    let last = cur.expect("FoloBlock has m >= 1 and mid_idx < m - 1");
    Ok(FoloBlockParts { y, mid_tap, last })
}

fn folo_block<E: Exec>(
    e: &mut E,
    nb: &mut NameBuf,
    b: &FoloBlock,
    xs: &[&E::A],
    out_f32: bool,
) -> Result<E::A> {
    let FoloBlockParts { y, mid_tap, last } = folo_block_parts(e, nb, b, xs)?;
    let l2 = nb.push_field("cv2");
    let res = match &mid_tap {
        Some(mid) => run_conv(e, nb.as_str(), &b.cv2, &[&y, mid, &last], None, out_f32),
        None => run_conv(e, nb.as_str(), &b.cv2, &[&y, &last], None, out_f32),
    };
    nb.restore(l2);
    res
}

fn folo_block_f32<E: Exec>(e: &mut E, nb: &mut NameBuf, b: &FoloBlock, xs: &[&E::A]) -> Result<Tensor> {
    let FoloBlockParts { y, mid_tap, last } = folo_block_parts(e, nb, b, xs)?;
    let l2 = nb.push_field("cv2");
    let res = match &mid_tap {
        Some(mid) => e.conv_out_f32(nb.as_str(), &b.cv2, &[&y, mid, &last]),
        None => e.conv_out_f32(nb.as_str(), &b.cv2, &[&y, &last]),
    };
    nb.restore(l2);
    res
}

fn hyper_context<E: Exec>(
    e: &mut E,
    nb: &mut NameBuf,
    h_ctx: &HyperContext,
    x: &E::A,
    out_f32: bool,
) -> Result<(E::A, Tensor)> {
    let [b, _, h, w] = e.dims(x);
    let lav = nb.push_field("av_proj");
    let av = e.conv_f32(nb.as_str(), &h_ctx.av_proj, x)?;
    nb.restore(lav);

    let (y_ctx, g_scene) = e.hyper_attend_and_scene(h_ctx, &av, b, h, w)?;
    let lin = nb.push_field("out_proj.in");
    let pin = e.quantize(nb.as_str(), &y_ctx)?;
    nb.restore(lin);

    let lo = nb.push_field("out_proj");
    let out = run_conv(e, nb.as_str(), &h_ctx.out_proj, &[&pin], Some(x), out_f32)?;
    nb.restore(lo);
    Ok((out, g_scene))
}

/// Backbone and SPD-PAN neck of `model` (`FoloV2Model`) on `x` (`[B, in_channels, H, W]`, `f32`, already
/// checked) through `e`: the head's three `f32` inputs `[P3_head, P4_head, out5]`.
pub fn features_v2<E: Exec>(model: &FoloV2Model, e: &mut E, x: &Tensor) -> Result<[Tensor; 3]> {
    let (bb, nk) = (&model.backbone, &model.neck);
    let mut nb = NameBuf::new();

    let mut y = e.stem_v2("backbone.stem.conv", &bb.stem, x)?;

    let down2_out_f32 = match bb.stage2.items().first() {
        Some(first) => e.wants_f32_rep("backbone.stage2.rep0", first),
        None => e.wants_f32("backbone.down3", &bb.down3) && e.wants_f32("neck.p2_proj", &nk.p2_proj),
    };
    y = run_conv(e, "backbone.down2", &bb.down2, &[&y], None, down2_out_f32)?;

    let down3_f32 = e.wants_f32("backbone.down3", &bb.down3);
    let p2_proj_f32 = e.wants_f32("neck.p2_proj", &nk.p2_proj);
    let n_s2 = bb.stage2.len();
    nb.set("backbone.stage2");
    for (j, rep) in bb.stage2.iter().enumerate() {
        let next_f32 = if j + 1 < n_s2 {
            let lr = nb.push_rep(j + 1);
            let ok = e.wants_f32_rep(nb.as_str(), &bb.stage2.items()[j + 1]);
            nb.restore(lr);
            ok
        } else {
            down3_f32 && p2_proj_f32
        };
        let lr = nb.push_rep(j);
        y = e.rep_conv(nb.as_str(), rep, &y, next_f32)?;
        nb.restore(lr);
    }
    let p2 = y;

    y = run_conv(
        e,
        "backbone.down3",
        &bb.down3,
        &[&p2],
        None,
        e.wants_f32("backbone.stage3.cv1", &bb.stage3.cv1),
    )?;

    let p3_f32 = e.wants_f32("backbone.down4", &bb.down4) && e.wants_f32("neck.out3.cv1", &nk.out3.cv1);
    nb.set("backbone.stage3");
    let p3 = folo_block(e, &mut nb, &bb.stage3, &[&y], p3_f32)?;

    y = run_conv(
        e,
        "backbone.down4",
        &bb.down4,
        &[&p3],
        None,
        e.wants_f32("backbone.stage4.cv1", &bb.stage4.cv1),
    )?;

    let p4_f32 = e.wants_f32("backbone.down5", &bb.down5) && e.wants_f32("neck.td4.cv1", &nk.td4.cv1);
    nb.set("backbone.stage4");
    let p4 = folo_block(e, &mut nb, &bb.stage4, &[&y], p4_f32)?;

    y = run_conv(
        e,
        "backbone.down5",
        &bb.down5,
        &[&p4],
        None,
        e.wants_f32("backbone.stage5.cv1", &bb.stage5.cv1),
    )?;

    nb.set("backbone.stage5");
    y = folo_block(e, &mut nb, &bb.stage5, &[&y], e.wants_f32("backbone.sppf.cv1", &bb.sppf.cv1))?;

    let hyper_in_f32 = e.wants_f32("backbone.hyper.av_proj", &bb.hyper.av_proj)
        && e.wants_f32("backbone.hyper.out_proj", &bb.hyper.out_proj);
    nb.set("backbone.sppf");
    y = sppf(e, &mut nb, &bb.sppf, &y, hyper_in_f32)?;

    let td4_f32 = e.wants_f32("neck.td4.cv1", &nk.td4.cv1);
    let out5_f32 = e.wants_f32("neck.out5.cv1", &nk.out5.cv1);
    nb.set("backbone.hyper");
    let (p5, g_scene) = hyper_context(e, &mut nb, &bb.hyper, &y, td4_f32 && out5_f32)?;
    drop(y);

    let up11 = e.upsample2_for(&p5, td4_f32)?;

    let out3_f32 = e.wants_f32("neck.out3.cv1", &nk.out3.cv1);
    let out4_f32 = e.wants_f32("neck.out4.cv1", &nk.out4.cv1);
    nb.set("neck.td4");
    let td4 = folo_block(e, &mut nb, &nk.td4, &[&up11, &p4], out3_f32 && out4_f32)?;
    drop(up11);
    drop(p4);

    let up14 = e.upsample2_for(&td4, out3_f32)?;
    let [p2_tl, p2_tr, p2_bl, p2_br] = e.space_to_depth2_parts(&p2, p2_proj_f32)?;
    drop(p2);
    let p2_spd = run_conv(e, "neck.p2_proj", &nk.p2_proj, &[&p2_tl, &p2_tr, &p2_bl, &p2_br], None, out3_f32)?;
    drop((p2_tl, p2_tr, p2_bl, p2_br));

    nb.set("neck.out3");
    let out3_raw =
        folo_block(e, &mut nb, &nk.out3, &[&up14, &p3, &p2_spd], e.wants_f32("neck.down3", &nk.down3))?;
    drop((up14, p3, p2_spd));

    let nd3 = run_conv(e, "neck.down3", &nk.down3, &[&out3_raw], None, out4_f32)?;
    // Dequantise P3 while `out3_raw` is hot in L2 cache, then apply relocated FullPAD gate on f32 head input.
    let p3_raw = e.dequantize(&out3_raw)?;
    drop(out3_raw);
    let p3_head = match &nk.pad_p3 {
        Some(gate) => e.fullpad_gate(gate, &p3_raw, &g_scene)?,
        None => p3_raw,
    };

    nb.set("neck.out4");
    let out4_raw = folo_block(e, &mut nb, &nk.out4, &[&nd3, &td4], e.wants_f32("neck.down4", &nk.down4))?;
    drop((nd3, td4));

    let nd4 = run_conv(e, "neck.down4", &nk.down4, &[&out4_raw], None, out5_f32)?;
    let p4_raw = e.dequantize(&out4_raw)?;
    drop(out4_raw);
    let p4_head = match &nk.pad_p4 {
        Some(gate) => e.fullpad_gate(gate, &p4_raw, &g_scene)?,
        None => p4_raw,
    };

    nb.set("neck.out5");
    let out5 = folo_block_f32(e, &mut nb, &nk.out5, &[&nd4, &p5])?;
    drop((nd4, p5, g_scene));
    Ok([p3_head, p4_head, out5])
}
