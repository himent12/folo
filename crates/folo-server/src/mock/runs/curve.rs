//! Deterministic metric curves: losses, learning rate and validation metrics by step and epoch.

use folo_api::{BestEpoch, ClassMetrics, Epoch, Loss, Step, TrainConfig, ValMetrics};

use crate::util::noise;

use super::{Curve, arch_spec, r3, r4};

impl Curve {
    /// Curves for `config` with `spe` steps per epoch converging to `quality`.
    pub fn new(seed: u64, quality: f32, nc: usize, config: &TrainConfig, spe: u64) -> Curve {
        let scale = (640.0 / config.imgsz.max(32) as f32).powi(2);
        Curve {
            seed,
            quality,
            nc,
            epochs: config.epochs,
            spe,
            lr0: config.lr0,
            ips: arch_spec(config.arch).2 * scale,
            batch: config.batch,
            close_mosaic: config.augment.close_mosaic,
        }
    }

    /// Total steps.
    pub fn total_steps(&self) -> u64 {
        self.epochs as u64 * self.spe
    }

    fn epoch_of(&self, step: u64) -> u32 {
        step.div_ceil(self.spe).max(1) as u32
    }

    /// Learning rate at a 1-based global step: linear warmup over 3 epochs, then cosine to `lr0 / 100`.
    pub fn lr(&self, step: u64) -> f32 {
        let warm = (3.min(self.epochs.div_ceil(2)) as u64 * self.spe).max(1);
        let s = step.saturating_sub(1);
        if s < warm {
            self.lr0 * (s + 1) as f32 / warm as f32
        } else {
            let t = (s - warm) as f32 / (self.total_steps() - warm).max(1) as f32;
            self.lr0 * (0.01 + 0.99 * 0.5 * (1.0 + (std::f32::consts::PI * t).cos()))
        }
    }

    fn base_loss(&self, e: f32) -> (f32, f32, f32) {
        let q = self.quality;
        let tau = (self.epochs as f32 * 0.15).max(1.5);
        let p = e / self.epochs as f32;
        let mut boxl = 0.95 + 0.35 * (1.0 - q) + 1.5 * (-e / tau).exp() - 0.15 * p;
        let mut cls = 0.42 + 0.6 * (1.0 - q) + 2.8 * (-e / (0.7 * tau)).exp() - 0.1 * p;
        let mut dfl = 0.88 + 0.15 * (1.0 - q) + 0.65 * (-e / tau).exp() - 0.05 * p;
        if e > (self.epochs.saturating_sub(self.close_mosaic)) as f32 {
            boxl *= 0.93;
            cls *= 0.9;
            dfl *= 0.96;
        }
        (boxl, cls, dfl)
    }

    /// Training loss of one batch.
    pub fn step_loss(&self, step: u64) -> Loss {
        let e = step as f32 / self.spe as f32;
        let (b, c, d) = self.base_loss(e);
        let n = |k: u64| 1.0 + 0.055 * noise(&[self.seed, step, k]);
        let (b, c, d) = (b * n(1), c * n(2), d * n(3));
        Loss { box_: r4(b), cls: r4(c), dfl: r4(d), total: r4(b + c + d) }
    }

    /// Step record with a given ETA.
    pub fn step(&self, step: u64, eta_s: f64) -> Step {
        let warm = (step as f32 / 8.0).min(1.0);
        let ips = self.ips * (0.6 + 0.4 * warm) * (1.0 + 0.03 * noise(&[self.seed, step, 9]));
        let lr = (self.lr(step) as f64 * 1e7).round() / 1e7;
        let imgs_per_s = (ips * 10.0).round() / 10.0;
        Step {
            epoch: self.epoch_of(step),
            step,
            lr: lr as f32,
            loss: self.step_loss(step),
            imgs_per_s,
            eta_s,
            steps_per_epoch: self.spe,
            epochs: self.epochs,
        }
    }

    /// Simulated wall time of one epoch in a "real" (uncompressed) run, including validation.
    pub fn real_epoch_s(&self) -> f64 {
        self.spe as f64 * self.batch as f64 / self.ips as f64 * 1.12
    }

    /// Validation mAP@0.5:0.95 after epoch `e`.
    fn map50_95(&self, e: u32) -> f32 {
        let tau = self.epochs as f32 * 0.14 + 1.0;
        let x = e as f32;
        let mut m = self.quality * (1.0 - (-x / tau).exp()).powf(1.6);
        if e > self.epochs.saturating_sub(self.close_mosaic) {
            m += 0.012 * self.quality;
        }
        let jitter = 0.014 * (1.0 - 0.6 * x / self.epochs as f32);
        r4((m + jitter * noise(&[self.seed, e as u64, 20])).clamp(0.0, 0.99))
    }

    /// Validation metrics after epoch `e`.
    pub fn val(&self, e: u32) -> ValMetrics {
        let to50 = |m: f32| (1.0 - (1.0 - m).powf(1.6)).clamp(0.0, 0.995);
        let pr = |m50: f32, k: u64| {
            let p = (m50 * 1.04 + 0.02 * noise(&[self.seed, e as u64, k])).clamp(0.0, 0.995);
            let r = (m50 * 0.93 + 0.02 * noise(&[self.seed, e as u64, k + 1])).clamp(0.0, 0.995);
            (p, r)
        };
        let m = self.map50_95(e);
        // Per-class mAP50-95 deviates from the mean by zero-mean factors (scaled down so no class passes 0.99), so
        // the overall numbers are the class means, as the engine reports them (docs/METRICS.md).
        let raw: Vec<f32> = (0..self.nc as u64).map(|c| 0.16 * noise(&[self.seed, c, 30])).collect();
        let mean_dev = raw.iter().sum::<f32>() / raw.len().max(1) as f32;
        let dev: Vec<f32> = raw.iter().map(|d| d - mean_dev).collect();
        let room = dev
            .iter()
            .filter(|&&d| d > 0.0)
            .map(|d| (0.99 - m).max(0.0) / (m.max(1e-6) * d))
            .fold(1.0, f32::min);
        let per_class: Vec<ClassMetrics> = dev
            .iter()
            .enumerate()
            .map(|(c, d)| {
                let cm = (m * (1.0 + room * d)).clamp(0.0, 0.99);
                let c50 = to50(cm);
                let (precision, recall) = pr(c50, 40 + 2 * c as u64);
                ClassMetrics {
                    class: c as u32,
                    map50: r4(c50),
                    map50_95: r4(cm),
                    precision: r4(precision),
                    recall: r4(recall),
                }
            })
            .collect();
        let mean = |f: fn(&ClassMetrics) -> f32| {
            r4(per_class.iter().map(f).sum::<f32>() / per_class.len().max(1) as f32)
        };
        let m50 = if per_class.is_empty() { to50(m) } else { mean(|c| c.map50) };
        let (precision, recall) =
            if per_class.is_empty() { pr(m50, 21) } else { (mean(|c| c.precision), mean(|c| c.recall)) };
        // The best-F1 confidence on the 0.001 grid: low while the model is unsure, settling near 0.3-0.45.
        let conf = 0.12 + 0.35 * m50 + 0.02 * noise(&[self.seed, e as u64, 22]);
        ValMetrics {
            map50: r4(m50),
            map50_95: r4(m),
            precision: r4(precision),
            recall: r4(recall),
            per_class,
            best_f1_conf: Some(r3(conf.clamp(0.001, 0.999))),
        }
    }

    /// Epoch record.
    pub fn epoch(&self, e: u32, duration_s: f64) -> Epoch {
        let first = (e as u64 - 1) * self.spe + 1;
        let n = self.spe as f32;
        let mut sum = Loss { box_: 0.0, cls: 0.0, dfl: 0.0, total: 0.0 };
        for s in first..first + self.spe {
            let l = self.step_loss(s);
            sum.box_ += l.box_ / n;
            sum.cls += l.cls / n;
            sum.dfl += l.dfl / n;
            sum.total += l.total / n;
        }
        let train_loss =
            Loss { box_: r4(sum.box_), cls: r4(sum.cls), dfl: r4(sum.dfl), total: r4(sum.total) };
        Epoch { epoch: e, train_loss, val: self.val(e), duration_s }
    }

    /// Best epoch among `1..=done`.
    pub fn best(&self, done: u32) -> Option<BestEpoch> {
        (1..=done)
            .map(|e| BestEpoch { epoch: e, map50_95: self.map50_95(e) })
            .max_by(|a, b| a.map50_95.total_cmp(&b.map50_95).then(b.epoch.cmp(&a.epoch)))
    }
}
