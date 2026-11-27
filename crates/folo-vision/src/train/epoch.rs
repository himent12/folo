//! One training epoch (docs/MODEL.md §8.3): batches, forward, loss, backward into the accumulator, the
//! end-of-window optimizer step, `step` events, and the wall-clock accounting behind their ETA.

use std::time::Instant;

use folo_api::{LogLevel, RunEvent};
use folo_nn::{BatchNorm2d, GradScaler, LiveFiniteCheck, Module, RunningStats, ScaleUpdate};

use super::feed;
use super::{BatchSource, MAX_AMP_SKIPS, MAX_NONFINITE_BATCHES, Trainer};
use crate::error::{Error, Result};
use crate::loss::{
    BoxLossKind, LossConfig, LossItems, decode_device_loss_values, detection_loss_for_distill_with_wiou_mean,
    detection_loss_with_wiou_mean, device_loss_supported, encode_detection_loss_device, o2m_weight,
    step_loss, targets_from_normalized,
};
use crate::val::batch_tensor;

/// RAII guard that restores each backbone parameter's prior `requires_grad` flag when `train_epoch` exits
/// (on completion, stop, or error), preserving any parameter a caller already froze.
struct RestoreGrad(Vec<(folo_core::Tensor, bool)>);

impl Drop for RestoreGrad {
    fn drop(&mut self) {
        for (p, prev) in &self.0 {
            p.set_requires_grad(*prev);
        }
    }
}

impl Trainer {
    /// One epoch of §8.3. Returns the mean reported loss over its finite batches, or `None` if the stop flag
    /// interrupted it.
    pub(super) fn train_epoch(&mut self, e: u32, clock: &mut Clock) -> Result<Option<folo_api::Loss>> {
        self.model.train();
        let _backbone_grad_guard = if e < self.cfg.freeze_backbone_epochs {
            self.model.eval_backbone();
            let saved = self
                .model
                .backbone_parameters()
                .into_iter()
                .map(|p| {
                    let prev = p.requires_grad();
                    p.set_requires_grad(false);
                    (p, prev)
                })
                .collect();
            Some(RestoreGrad(saved))
        } else {
            None
        };
        // Stage-2 Mosaic-Off BN-Freeze safeguard (P3): keep BN in eval mode when mosaic is closed.
        let mosaic_closed = self.is_mosaic_closed(e);
        if self.cfg.bn_freeze_close_mosaic && mosaic_closed {
            self.model.for_each_module(&mut |_, m| {
                if let Some(bn) = (m as &dyn std::any::Any).downcast_ref::<BatchNorm2d>() {
                    bn.eval();
                }
            });
        }
        // MAL gamma warmup ramp over warmup_epochs (safeguard #38).
        let epoch_loss_cfg = self.epoch_loss_cfg(e);
        let need_kd_asg = self.cfg.distill.as_ref().is_some_and(|d| d.bckd_gamma > 0.0 || d.gt_verify);
        self.acc.zero();
        let s = self.cfg.imgsz;
        let (nb, acc_n) = (self.nb, self.accumulate);
        let decay = self.opt.groups().iter().position(|g| g.name == "decay").unwrap_or(0);
        let mut sum = [0f64; 3];
        // The reported distillation terms `[cls, dfl]` of each finite batch (§8.9).
        let mut kd_items: Vec<[f32; 2]> = Vec::new();
        let (mut finite_batches, mut window_finite) = (0usize, false);
        let (epoch_start, mut epoch_images) = (Instant::now(), 0usize);
        let skipped_before = self.scaler.as_ref().map_or(0, GradScaler::skipped_steps);
        let ctx = folo_gpu::context_for(self.device);
        self.model.set_o2o_detach(e >= self.cfg.o2o_attached_epochs);
        let mut source = match self.prefetched.take() {
            Some((pe, source)) if pe == e => source,
            _ => self.start_epoch(e)?,
        };
        let mut k = 0usize;
        // A batch's wall time starts when the previous one ends, so the ETA counts the wait for the loader too:
        // with mosaic on large JPEGs the loader, not the step, often sets the pace.
        let phase = self.mosaic_phase(e);
        if !RunningStats::all_finite_eval(&self.model)? {
            return Err(Error::Failed(format!(
                "epoch {}: frozen BatchNorm running statistics contain non-finite values; check loaded weights or \
                 checkpoint",
                e + 1
            )));
        }
        let mut snapshot = RunningStats::capture_training(&self.model)?;
        let mut snapshot_dirty = false;
        let mut batch_start = Instant::now();
        for item in source.by_ref() {
            let (mut batch, staged) = item?;
            let received = Instant::now();
            let n = batch.len();
            let targets = targets_from_normalized(&batch.targets, n, s, s)?;
            let x = match (&ctx, staged) {
                (Some(c), Some(st)) => feed::staged_tensor(c, self.device, &st, batch.shape)?,
                _ => batch_tensor(&mut batch, self.device, folo_core::DType::F32)?,
            };
            let lr = self.sched.lr_at(self.sched.step(), &self.opt.groups()[decay]);

            if snapshot_dirty {
                snapshot.recapture()?;
                snapshot_dirty = false;
            }
            // Distillation (§8.9): the frozen teacher sees the same batch, without autograd, before the student's
            // forward, so its activations are gone before the student's graph is built.
            let teacher_batch = match &self.teacher {
                Some(t) => Some(t.batch(&x)?),
                None => None,
            };
            let det_out = {
                let _amp = self.scaler.is_some().then(folo_core::autocast);
                self.model.forward(&x)?
            };
            drop(x);
            let iou_raw = det_out.iou_raw;
            let out = det_out.head;
            // Mixed precision: the loss (assigner, BCE, CIoU + DFL) runs in f32 on f32 copies of the f16 head.
            let f32_head = |t: &folo_core::Tensor| t.to_dtype(folo_core::DType::F32);
            let (box_raw, cls_raw) = (f32_head(&out.box_raw)?, f32_head(&out.cls_raw)?);
            let o2o_cfg = epoch_loss_cfg.o2o_branch(self.cfg.o2o_keeps_tal);
            let can_pipeline = ctx.is_some()
                && teacher_batch.is_none()
                && iou_raw.is_none()
                && device_loss_supported(&box_raw, &epoch_loss_cfg)
                && out.o2o.as_ref().is_none_or(|(b, _)| device_loss_supported(b, &o2o_cfg));

            let (finite, items, kd, wiou_batch_mean, grads_ready) = if can_pipeline {
                let c = ctx.as_ref().expect("checked");
                let (o2m_dev, _) = encode_detection_loss_device(
                    &box_raw,
                    &cls_raw,
                    &out.anchors,
                    &targets,
                    &epoch_loss_cfg,
                )?;
                drop((box_raw, cls_raw));
                let has_o2o = out.o2o.is_some();
                let o2o_dev = match &out.o2o {
                    Some((b, cl)) => {
                        let (ob, oc) = (f32_head(b)?, f32_head(cl)?);
                        Some(encode_detection_loss_device(&ob, &oc, &out.anchors, &targets, &o2o_cfg)?.0)
                    }
                    None => None,
                };
                let w_m = o2m_weight(e, self.cfg.epochs);
                let total = match &o2o_dev {
                    Some(o) => o2m_dev.total.mul_scalar(w_m)?.add(&o.total.mul_scalar(1.0 - w_m)?)?,
                    None => o2m_dev.total.clone(),
                };
                let bn_probe = snapshot.prepare_live_finite_check()?;
                let mut dl_tensors: Vec<&folo_core::Tensor> = Vec::with_capacity(3);
                dl_tensors.push(&o2m_dev.stats);
                if let Some(o) = &o2o_dev {
                    dl_tensors.push(&o.stats);
                }
                if let LiveFiniteCheck::DeviceScalar(ref s) = bn_probe {
                    dl_tensors.push(s);
                }
                // Submit forward + loss + BN finiteness + staging copies in one batch without blocking, then
                // encode and flush backward immediately so the GPU never idles waiting on host stat readbacks.
                let pending = folo_gpu::download_many_deferred(c, &dl_tensors)?;
                let scaled_total = match &self.scaler {
                    // The scale is a power of two, so the scaled loss (and every gradient before f16 rounding) is
                    // exact.
                    Some(sc) => total.mul_scalar(sc.scale() as f32)?,
                    None => total,
                };
                let grads = scaled_total.backward()?;
                drop(scaled_total);
                drop(o2m_dev);
                drop(o2o_dev);
                drop(out);
                c.flush();

                let mut dls = pending.into_iter();
                let o2m_st = dls.next().expect("o2m stats").finish()?.to_vec::<f32>()?;
                let (mut total_value, mut items, _, _, _) = decode_device_loss_values(&o2m_st);
                if has_o2o {
                    let o2o_st = dls.next().expect("o2o stats").finish()?.to_vec::<f32>()?;
                    let (o_val, o_items, _, _, _) = decode_device_loss_values(&o2o_st);
                    let w_o = 1.0 - w_m;
                    let mix = |a: f32, b: f32| w_m * a + w_o * b;
                    total_value = mix(total_value, o_val);
                    items = LossItems {
                        box_: mix(items.box_, o_items.box_),
                        cls: mix(items.cls, o_items.cls),
                        dfl: mix(items.dfl, o_items.dfl),
                    };
                }
                let bn_finite = match bn_probe {
                    LiveFiniteCheck::Ready(ok) => ok,
                    LiveFiniteCheck::DeviceScalar(_) => {
                        dls.next().expect("bn scalar").finish()?.item::<f32>()?.is_finite()
                    }
                };
                let finite = items.is_finite() && total_value.is_finite() && bn_finite;
                (finite, items, None, None, Some(grads))
            } else {
                let need_asg = need_kd_asg || iou_raw.is_some();
                let (mut loss, mut kd_asg) = detection_loss_for_distill_with_wiou_mean(
                    &box_raw,
                    &cls_raw,
                    &out.anchors,
                    &targets,
                    &epoch_loss_cfg,
                    need_asg,
                    self.wiou_iou_mean,
                )?;
                let kd_needs_host_asg = need_kd_asg
                    && teacher_batch.as_ref().is_some_and(super::teacher::TeacherBatch::is_host_targets);
                let mut cached_host_asg = None;
                let iou_loss = match (&iou_raw, &kd_asg) {
                    (Some(iou_t), Some(asg)) if loss.is_finite() => {
                        let host_asg = asg.as_host(n, out.num_anchors(), out.num_classes, &targets)?;
                        let l = crate::model::v2::iou_quality_loss(
                            &f32_head(iou_t)?,
                            &box_raw,
                            &out.anchors,
                            host_asg.as_ref(),
                            out.reg_max,
                            1.0,
                        )?;
                        if kd_needs_host_asg && matches!(asg, crate::loss::DistillAssignment::Device(_)) {
                            cached_host_asg = Some(host_asg.into_owned());
                        }
                        Some(l)
                    }
                    _ => None,
                };
                // On the one-to-many outputs, over the supervised loss's normaliser; part of the one-to-many
                // loss below. Skip distillation when the supervised loss is already non-finite (e.g. early AMP
                // forward overflow, where `kd_asg` is `None`) so the bad-batch BN rollback & skip handler below
                // runs cleanly.
                let kd = match (&teacher_batch, &self.cfg.distill) {
                    (Some(t), Some(d)) if loss.is_finite() => {
                        let asg_for_kd = if need_kd_asg {
                            if t.is_host_targets()
                                && matches!(kd_asg, Some(crate::loss::DistillAssignment::Device(_)))
                            {
                                let host = match cached_host_asg.take() {
                                    Some(h) => h,
                                    None => kd_asg
                                        .as_ref()
                                        .expect("device assignment")
                                        .as_host(n, out.num_anchors(), out.num_classes, &targets)?
                                        .into_owned(),
                                };
                                kd_asg = Some(crate::loss::DistillAssignment::Host(host));
                            }
                            kd_asg.as_ref()
                        } else {
                            None
                        };
                        Some(t.loss(
                            &box_raw,
                            &cls_raw,
                            &out.anchors,
                            loss.target_score_sum,
                            d,
                            asg_for_kd,
                        )?)
                    }
                    _ => None,
                };
                drop(kd_asg);
                drop(teacher_batch);
                drop(iou_raw);
                drop((box_raw, cls_raw));
                // Dual assignment (MODEL.md §6.6): the one-to-one branch's own loss, weighted in by epoch.
                let o2o = match &out.o2o {
                    Some((b, c)) => {
                        let (box_raw, cls_raw) = (f32_head(b)?, f32_head(c)?);
                        Some(detection_loss_with_wiou_mean(
                            &box_raw,
                            &cls_raw,
                            &out.anchors,
                            &targets,
                            &o2o_cfg,
                            self.wiou_iou_mean,
                        )?)
                    }
                    None => None,
                };
                let w_m = o2m_weight(e, self.cfg.epochs);
                // `w_m · (L_o2m + L_distill) + (1 − w_m) · L_o2o`; the reported terms stay unweighted by `w_m` (§8.9).
                loss = step_loss(loss, kd.as_ref(), o2o.as_ref().map(|o| (o, w_m)), n)?;
                let bn_probe = snapshot.prepare_live_finite_check()?;
                let (iou_finite, bn_finite) = match (iou_loss, bn_probe, &ctx) {
                    (Some(l_iou), LiveFiniteCheck::DeviceScalar(bn_s), Some(c)) => {
                        let mut host = folo_gpu::download_many(c, &[&l_iou, &bn_s])?;
                        let bn_ok = host.pop().expect("bn scalar").item::<f32>()?.is_finite();
                        let v = host.pop().expect("iou scalar").item::<f32>()?;
                        let iou_ok = v.is_finite();
                        if iou_ok {
                            loss.total = loss.total.add(&l_iou)?;
                            loss.total_value += v;
                        }
                        (iou_ok, bn_ok)
                    }
                    (iou_loss, bn_probe, _) => {
                        let mut iou_ok = true;
                        if let Some(l_iou) = iou_loss {
                            let v = l_iou.item::<f32>()?;
                            iou_ok = v.is_finite();
                            if iou_ok {
                                loss.total = loss.total.add(&l_iou)?;
                                loss.total_value += v;
                            }
                        }
                        let bn_ok = match bn_probe {
                            LiveFiniteCheck::Ready(ok) => ok,
                            LiveFiniteCheck::DeviceScalar(s) => s.item::<f32>()?.is_finite(),
                        };
                        (iou_ok, bn_ok)
                    }
                };
                let kd = kd.map(|kd| [kd.cls, kd.dfl]);
                let kd_finite = kd.is_none_or(|v| v.iter().all(|x| x.is_finite()));
                let finite = loss.is_finite() && kd_finite && iou_finite && bn_finite;
                let items = loss.items;
                let wiou_batch_mean = loss.mean_iou_loss;
                let grads = if finite {
                    let g = match &self.scaler {
                        // The scale is a power of two, so the scaled loss (and every gradient before f16 rounding) is
                        // exact.
                        Some(s) => loss.total.mul_scalar(s.scale() as f32)?.backward()?,
                        None => loss.total.backward()?,
                    };
                    Some(g)
                } else {
                    None
                };
                drop(loss);
                drop(out);
                (finite, items, kd, wiou_batch_mean, grads)
            };

            if finite {
                snapshot_dirty = !snapshot.is_empty();
                if epoch_loss_cfg.box_loss == BoxLossKind::WiouV3
                    && let Some(m) = wiou_batch_mean
                    && m.is_finite()
                {
                    let mom = epoch_loss_cfg.wiou_momentum;
                    self.wiou_iou_mean = ((1.0 - mom) * self.wiou_iou_mean + mom * m).max(1e-4);
                }
                if let Some(v) = kd {
                    kd_items.push(v);
                }
                self.acc.add_owned(grads_ready.expect("finite batch produced gradients"))?;
                window_finite = true;
                self.consecutive_bad = 0;
                finite_batches += 1;
                sum[0] += items.box_ as f64;
                sum[1] += items.cls as f64;
                sum[2] += items.dfl as f64;
            } else {
                drop(grads_ready);
                snapshot.restore()?;
                self.consecutive_bad += 1;
                let consecutive_bad = self.consecutive_bad;
                let kd_note = kd.map_or(String::new(), |[c, d]| format!(" (distillation cls {c}, dfl {d})"));
                self.emitter.log(
                    LogLevel::Warn,
                    format!(
                        "epoch {} batch {}: non-finite loss {:?}{kd_note}; skipped ({consecutive_bad} in a row)",
                        e + 1,
                        k + 1,
                        items
                    ),
                );
                if consecutive_bad >= MAX_NONFINITE_BATCHES {
                    return Err(Error::Failed(format!(
                        "{MAX_NONFINITE_BATCHES} consecutive non-finite batches (epoch {}, batch {}); lower lr0 or \
                         check the data",
                        e + 1,
                        k + 1
                    )));
                }
            }
            if (k + 1).is_multiple_of(acc_n) || k + 1 == nb {
                if window_finite {
                    self.optimizer_step(e, k)?;
                }
                self.acc.zero();
                window_finite = false;
            }

            epoch_images += n;
            clock.batches[phase] += 1;
            clock.wall[phase] += batch_start.elapsed().as_secs_f64();
            clock.step[phase] += received.elapsed().as_secs_f64();
            batch_start = Instant::now();
            // The previous epoch's event, once its files are written.
            self.poll_write()?;
            if finite {
                let elapsed = epoch_start.elapsed().as_secs_f64().max(1e-9);
                let step = folo_api::Step {
                    epoch: e + 1,
                    step: e as u64 * nb as u64 + k as u64 + 1,
                    lr: lr as f32,
                    loss: loss_json(&items),
                    imgs_per_s: (epoch_images as f64 / elapsed) as f32,
                    eta_s: self.eta(e, k, clock),
                    steps_per_epoch: nb as u64,
                    epochs: self.cfg.epochs,
                };
                self.emitter.step(RunEvent::Step(step), k + 1 == nb);
            }
            k += 1;
            if self.stopping() {
                return Ok(None);
            }
        }
        drop(source);
        // Start the next epoch now: its first batches are ready when validation and the checkpoints are done.
        // The batches are the same whenever an epoch starts (§8.3: everything is seeded by the epoch). Not after
        // the last epoch or once a stop is requested; a failure here only costs the head start (the epoch starts
        // its loader itself), not the trained epoch's validation and checkpoint.
        if k == nb && e + 1 < self.cfg.epochs && !self.stopping() {
            match self.start_epoch(e + 1) {
                Ok(source) => self.prefetched = Some((e + 1, source)),
                Err(err) => self.emitter.log(
                    LogLevel::Warn,
                    format!(
                        "epoch {}'s loader could not start early ({}); it starts with the epoch",
                        e + 2,
                        err.summary()
                    ),
                ),
            }
        }
        if k != nb {
            return Err(Error::Failed(format!("epoch {} delivered {k} of {nb} batches", e + 1)));
        }
        if finite_batches == 0 {
            return Err(Error::Failed(format!("epoch {} had no finite batch", e + 1)));
        }
        if let Some(s) = &self.scaler {
            self.emitter.log(
                LogLevel::Info,
                format!(
                    "epoch {}: loss scale {}, {} optimizer steps skipped for gradient overflow ({} in the run)",
                    e + 1,
                    s.scale(),
                    s.skipped_steps() - skipped_before,
                    s.skipped_steps()
                ),
            );
        }
        if !kd_items.is_empty() {
            self.emitter.log(LogLevel::Info, distill_line(e, &kd_items));
        }
        let f = finite_batches as f64;
        let (b, c, d) = ((sum[0] / f) as f32, (sum[1] / f) as f32, (sum[2] / f) as f32);
        Ok(Some(folo_api::Loss { box_: b, cls: c, dfl: d, total: b + c + d }))
    }

    /// Starts epoch `e` of the loader. On a folo-gpu device the loader delivers u8 pixels and a feeder thread
    /// stages each batch while the previous step runs; the step then only encodes the copy and the conversion.
    fn start_epoch(&self, e: u32) -> Result<BatchSource> {
        let iter = self.loader.epoch(e);
        Ok(match folo_gpu::context_for(self.device) {
            Some(c) => Box::new(feed::Feeder::new(iter, c)?.map(|r| r.map(|(b, s)| (b, Some(s))))),
            None => Box::new(iter.map(|r| r.map(|b| (b, None)).map_err(Error::from))),
        })
    }

    /// §8.3's end-of-window step: schedule, clip, optimizer, EMA, `t += 1`. With mixed precision (§8.8) the
    /// gradients are unscaled before clipping, and a window whose gradients are not all finite skips the step
    /// (`t`, the optimizer and the EMA unchanged) and backs the loss scale off: one readback of a single flag.
    /// `(e, k)` name the window's last batch in the log.
    fn is_mosaic_closed(&self, e: u32) -> bool {
        let a = &self.cfg.augment;
        a.mosaic > 0.0 && a.close_mosaic_epochs(self.cfg.epochs) > 0 && !a.mosaic_active(e, self.cfg.epochs)
    }

    fn epoch_loss_cfg(&self, e: u32) -> LossConfig {
        let mut c = self.loss_cfg;
        if c.mal_gamma > 0.0 && self.cfg.transfer {
            let warm = self.cfg.warmup_epochs.ceil() as u32;
            if warm > 0 && e < warm {
                c.mal_gamma *= (e + 1) as f32 / warm as f32;
            }
        }
        c
    }

    fn apply_schedule(&mut self, e: u32) {
        self.sched.apply(self.opt.as_mut());
        // Scale `bias.backbone`'s warmup start by `backbone_lr_scale` so backbone biases do not start at the
        // unscaled `warmup_bias_lr` while `decay.backbone` and `bn.backbone` are scaled.
        if self.cfg.backbone_lr_scale != 1.0 {
            let c = self.sched.contract();
            let t = self.sched.step();
            if t < c.warmup_iters && c.warmup_bias_lr != 0.0 {
                let frac = t as f64 / c.warmup_iters as f64;
                let shift = (1.0 - frac) * c.warmup_bias_lr * (1.0 - self.cfg.backbone_lr_scale);
                for g in self.opt.groups_mut() {
                    if g.name == "bias.backbone" {
                        g.lr = (g.lr - shift).max(0.0);
                    }
                }
            }
        }
        if self.cfg.cls_lr_close_scale != 1.0 && self.is_mosaic_closed(e) {
            for g in self.opt.groups_mut() {
                if g.name.ends_with(".cls") {
                    g.lr *= self.cfg.cls_lr_close_scale;
                }
            }
        }
    }

    pub(super) fn optimizer_step(&mut self, e: u32, k: usize) -> Result<()> {
        if let Some(scaler) = self.scaler.as_ref() {
            let inv_scale = scaler.inv_scale();
            let finite = if let Some(ctx) = folo_gpu::context_for(self.device) {
                if self.cfg.clip_grad_norm > 0.0 {
                    if let Some(plan) =
                        self.acc.unscale_clip_device_prepare(inv_scale, self.cfg.clip_grad_norm)?
                    {
                        // Submit the norm reduction and finiteness readback before in-place scaling so the
                        // scaling pass and schedule update overlap with the GPU wait.
                        let dl = folo_gpu::download_deferred(&ctx, plan.finite_scalar())?;
                        self.acc.unscale_clip_device_apply(plan)?;
                        ctx.flush();
                        self.apply_schedule(e);
                        dl.finish()?.item::<f32>()?.is_finite()
                    } else {
                        self.apply_schedule(e);
                        self.acc.unscale_clip(inv_scale, self.cfg.clip_grad_norm)?
                    }
                } else if let Some(flag) = self.acc.unscale_prepare_device(inv_scale)? {
                    let dl = folo_gpu::download_deferred(&ctx, &flag)?;
                    self.apply_schedule(e);
                    dl.finish()?.to_vec::<u32>()?[0] == 0
                } else {
                    self.apply_schedule(e);
                    self.acc.unscale_clip(inv_scale, self.cfg.clip_grad_norm)?
                }
            } else {
                self.apply_schedule(e);
                self.acc.unscale_clip(inv_scale, self.cfg.clip_grad_norm)?
            };
            let scaler = self.scaler.as_mut().expect("checked");
            let scale = scaler.scale();
            if scaler.update(!finite) == ScaleUpdate::BackedOff {
                let (streak, now) = (scaler.skip_streak() as usize, scaler.scale());
                self.emitter.log(
                    LogLevel::Info,
                    format!(
                        "epoch {} batch {}: gradient overflow at loss scale {scale}; optimizer step skipped, loss \
                         scale {now} ({} skipped in the run)",
                        e + 1,
                        k + 1,
                        scaler.skipped_steps()
                    ),
                );
                if streak >= MAX_AMP_SKIPS {
                    return Err(Error::Failed(format!(
                        "{MAX_AMP_SKIPS} consecutive optimizer steps had non-finite gradients at every loss scale \
                         down to {now} (epoch {}, batch {}); train without amp, lower lr0 or check the data",
                        e + 1,
                        k + 1
                    )));
                }
                return Ok(());
            }
        } else {
            self.apply_schedule(e);
            if self.cfg.clip_grad_norm > 0.0 {
                let norm = self.acc.clip_grad_norm(self.cfg.clip_grad_norm)?;
                // On GPU `clip_grad_norm` clips on-device without reading back and returns `NaN`; on CPU it returns
                // the exact global L2 norm, so a non-finite norm skips the step before corrupting weights or state.
                if self.device.is_cpu() && !norm.is_finite() {
                    self.emitter.log(
                        LogLevel::Warn,
                        format!(
                            "epoch {} batch {}: non-finite gradient norm ({norm}); optimizer step skipped",
                            e + 1,
                            k + 1
                        ),
                    );
                    return Ok(());
                }
            }
        }
        self.opt.step(&self.acc)?;
        if let Some(ema) = &mut self.ema {
            let lf = self.sched.lf(self.sched.step());
            ema.update_with_lf(&self.model, lf)?;
        }
        self.sched.advance();
        Ok(())
    }

    /// Remaining seconds after batch `k` of epoch `e` ([`eta_s`]), to 0.1 s (`70.2`, not `70.1722331406015`).
    fn eta(&self, e: u32, k: usize, clock: &Clock) -> f64 {
        (eta_s(clock, self.nb, e, k, self.cfg.epochs, |ep| self.mosaic_phase(ep)) * 10.0).round() / 10.0
    }

    /// The [`Clock`] phase of epoch `e`: 0 while mosaic is on, 1 once it is closed (or when it is off).
    fn mosaic_phase(&self, e: u32) -> usize {
        let a = &self.cfg.augment;
        usize::from(!(a.mosaic > 0.0 && a.mosaic_active(e, self.cfg.epochs)))
    }
}

fn loss_json(i: &LossItems) -> folo_api::Loss {
    folo_api::Loss { box_: i.box_, cls: i.cls, dfl: i.dfl, total: i.total() }
}

/// The epoch's distillation log line (docs/MODEL.md §8.9): the mean reported terms (`cls_gain · L_kc` and
/// `dfl_gain · L_kd`, without `B`) over its finite batches, and over its first and last quarter of them, so a log
/// shows whether they fall within the epoch.
pub(super) fn distill_line(e: u32, items: &[[f32; 2]]) -> String {
    let mean = |s: &[[f32; 2]]| {
        let n = s.len().max(1) as f64;
        let sum = s.iter().fold([0f64; 2], |a, v| [a[0] + v[0] as f64, a[1] + v[1] as f64]);
        format!("cls {:.5} dfl {:.5}", sum[0] / n, sum[1] / n)
    };
    let q = items.len().div_ceil(4);
    format!(
        "epoch {}: distillation {} over {} batches (first quarter {}, last quarter {})",
        e + 1,
        mean(items),
        items.len(),
        mean(&items[..q]),
        mean(&items[items.len() - q..])
    )
}

/// Wall-clock accounting for the ETA, per mosaic phase (index 0: mosaic on, 1: closed or off), because the two
/// can differ several times over: on 1080p frames a mosaic epoch of folo-n took 24 s and a closed one 10 s.
#[derive(Default)]
pub(super) struct Clock {
    pub(super) batches: [usize; 2],
    /// Wall time of the batches, the wait for the loader included.
    pub(super) wall: [f64; 2],
    /// Time from receiving each batch to the end of its step: what the batches cost when the loader keeps up.
    pub(super) step: [f64; 2],
    /// Seconds of a validation, from the recent epochs ([`Clock::ended_epoch`]): it grows as the model learns (more
    /// detections to match and draw), so the mean since the start trails it.
    pub(super) val: Option<f64>,
    /// The same for an epoch's work besides its batches and validation (starting the loader, writing checkpoints and
    /// samples): on a small dataset more than half of each epoch.
    pub(super) other: Option<f64>,
}

impl Clock {
    /// Records a finished epoch's validation and other seconds, the newest with half the weight.
    pub(super) fn ended_epoch(&mut self, val: f64, other: f64) {
        let lean = |old: Option<f64>, new: f64| Some(old.map_or(new, |o| (o + new) / 2.0));
        self.val = lean(self.val, val);
        self.other = lean(self.other, other);
    }

    /// Expected seconds per batch in `phase`: its measured wall time once it has run. Before the closed phase
    /// starts, the mosaic phase's step time, since without mosaic the loader keeps up and the step sets the pace;
    /// before any batch, 0.
    fn per_batch(&self, phase: usize) -> f64 {
        let other = 1 - phase;
        if self.batches[phase] > 0 {
            self.wall[phase] / self.batches[phase] as f64
        } else if self.batches[other] > 0 {
            let time = if phase == 1 { self.step[other] } else { self.wall[other] };
            time / self.batches[other] as f64
        } else {
            0.0
        }
    }
}

/// Remaining seconds of a run after batch `k` (0-based) of epoch `e` (0-based) of `epochs`, with `nb` batches per
/// epoch and `phase(epoch)` its [`Clock`] phase: each batch left at its phase's [`Clock::per_batch`], plus the
/// validations and each epoch's other work left at their recent time (a validation is a tenth of an epoch before the
/// first).
pub(super) fn eta_s(
    clock: &Clock,
    nb: usize,
    e: u32,
    k: usize,
    epochs: u32,
    phase: impl Fn(u32) -> usize,
) -> f64 {
    let train = (nb - k - 1) as f64 * clock.per_batch(phase(e))
        + (e + 1..epochs).map(|ep| nb as f64 * clock.per_batch(phase(ep))).sum::<f64>();
    let per_val = clock.val.unwrap_or_else(|| 0.1 * clock.per_batch(phase(e)) * nb as f64);
    let per_other = clock.other.unwrap_or(0.0);
    train + (epochs - e) as f64 * (per_val + per_other)
}
