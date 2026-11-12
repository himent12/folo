//! One training epoch (docs/MODEL.md §8.3): batches, forward, loss, backward into the accumulator, the
//! end-of-window optimizer step, `step` events, and the wall-clock accounting behind their ETA.

use std::time::Instant;

use folo_api::{LogLevel, RunEvent};
use folo_nn::{GradScaler, Module, RunningStats, ScaleUpdate};

use super::feed;
use super::{BatchSource, MAX_AMP_SKIPS, MAX_NONFINITE_BATCHES, Trainer};
use crate::error::{Error, Result};
use crate::loss::{LossConfig, LossItems, detection_loss, dual_loss, o2m_weight, targets_from_normalized};
use crate::val::batch_tensor;

impl Trainer {
    /// One epoch of §8.3. Returns the mean reported loss over its finite batches, or `None` if the stop flag
    /// interrupted it.
    pub(super) fn train_epoch(&mut self, e: u32, clock: &mut Clock) -> Result<Option<folo_api::Loss>> {
        self.model.train();
        self.acc.zero();
        let s = self.cfg.imgsz;
        let (nb, acc_n) = (self.nb, self.accumulate);
        let decay = self.opt.groups().iter().position(|g| g.name == "decay").unwrap_or(0);
        let mut sum = [0f64; 3];
        let (mut finite_batches, mut window_finite) = (0usize, false);
        let (epoch_start, mut epoch_images) = (Instant::now(), 0usize);
        let skipped_before = self.scaler.as_ref().map_or(0, GradScaler::skipped_steps);
        let ctx = folo_gpu::context_for(self.device);
        let mut source = match self.prefetched.take() {
            Some((pe, source)) if pe == e => source,
            _ => self.start_epoch(e)?,
        };
        let mut k = 0usize;
        // A batch's wall time starts when the previous one ends, so the ETA counts the wait for the loader too:
        // with mosaic on large JPEGs the loader, not the step, often sets the pace.
        let phase = self.mosaic_phase(e);
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

            let snapshot = RunningStats::capture(&self.model)?;
            let out = {
                let _amp = self.scaler.is_some().then(folo_core::autocast);
                self.model.forward(&x)?
            };
            drop(x);
            // Mixed precision: the loss (assigner, BCE, CIoU + DFL) runs in f32 on f32 copies of the f16 head.
            let f32_head = |t: &folo_core::Tensor| t.to_dtype(folo_core::DType::F32);
            let (box_raw, cls_raw) = (f32_head(&out.box_raw)?, f32_head(&out.cls_raw)?);
            let mut loss = detection_loss(&box_raw, &cls_raw, &out.anchors, &targets, &self.loss_cfg)?;
            drop((box_raw, cls_raw));
            if let Some((b, c)) = &out.o2o {
                // Dual assignment (MODEL.md §6.6): the one-to-one branch's own loss, weighted in by epoch.
                let (box_raw, cls_raw) = (f32_head(b)?, f32_head(c)?);
                let o2o = detection_loss(
                    &box_raw,
                    &cls_raw,
                    &out.anchors,
                    &targets,
                    &LossConfig { tal: crate::assign::TalConfig::o2o(), ..self.loss_cfg },
                )?;
                loss = dual_loss(&loss, &o2o, o2m_weight(e, self.cfg.epochs))?;
            }
            let finite = loss.is_finite() && RunningStats::all_finite(&self.model)?;
            let items = loss.items;
            if finite {
                let grads = match &self.scaler {
                    // The scale is a power of two, so the scaled loss (and every gradient before f16 rounding) is
                    // exact.
                    Some(s) => loss.total.mul_scalar(s.scale() as f32)?.backward()?,
                    None => loss.total.backward()?,
                };
                drop(loss);
                drop(out);
                self.acc.add(&grads)?;
                window_finite = true;
                self.consecutive_bad = 0;
                finite_batches += 1;
                sum[0] += items.box_ as f64;
                sum[1] += items.cls as f64;
                sum[2] += items.dfl as f64;
            } else {
                drop(loss);
                drop(out);
                snapshot.restore()?;
                self.consecutive_bad += 1;
                let consecutive_bad = self.consecutive_bad;
                self.emitter.log(
                    LogLevel::Warn,
                    format!(
                        "epoch {} batch {}: non-finite loss {:?}; skipped ({consecutive_bad} in a row)",
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
    fn optimizer_step(&mut self, e: u32, k: usize) -> Result<()> {
        self.sched.apply(self.opt.as_mut());
        if let Some(scaler) = &mut self.scaler {
            let finite = self.acc.unscale_clip(scaler.inv_scale(), self.cfg.clip_grad_norm)?;
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
        } else if self.cfg.clip_grad_norm > 0.0 {
            self.acc.clip_grad_norm(self.cfg.clip_grad_norm)?;
        }
        self.opt.step(&self.acc)?;
        if let Some(ema) = &mut self.ema {
            ema.update(&self.model)?;
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
