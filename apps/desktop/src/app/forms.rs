//! The New run and Generate synthetic forms: their state, the choices they offer, and how typed values
//! parse and carry over from the run a form was made from.
use crate::api::*;
use crate::input::TextInput;
use folo_ui::{Entity, Subscription};
/// Close mosaic: a whole number of epochs, at most the run's length.
pub fn parse_close(s: &str, epochs: f32) -> Option<u32> {
    s.trim().parse::<u32>().ok().filter(|v| (*v as f32) <= epochs)
}

/// Close mosaic's default for a run of `epochs`: the trainer's 10 under its own cap (half the run),
/// as Studio's New run has it.
pub fn auto_close_mosaic(epochs: u32) -> u32 {
    capped_close_mosaic(10, epochs)
}

impl NewRunForm {
    /// While close mosaic is still the form's default, it follows the epoch count: the text its
    /// field should show now, when the value moved.
    pub fn follow_epochs(&mut self) -> Option<String> {
        if !self.close_auto || self.epochs_invalid {
            return None;
        }
        let v = auto_close_mosaic(self.epochs.round() as u32);
        self.close_invalid = false;
        if self.close_mosaic.round() as u32 == v {
            return None;
        }
        self.close_mosaic = v as f32;
        Some(v.to_string())
    }

    /// Validate the close-mosaic text against the current epoch count.
    pub fn check_close(&mut self, s: &str) {
        match parse_close(s, self.epochs) {
            Some(v) => {
                self.close_mosaic = v as f32;
                self.close_invalid = false;
            }
            None => self.close_invalid = true,
        }
    }

    /// Picks optimizer `i` of [`OPTIMIZERS`] (also settles an unknown prefilled one).
    pub fn pick_optimizer(&mut self, i: usize) {
        self.optimizer = i;
        self.optimizer_unknown = false;
    }

    /// Prefills the optimizer from a run's config: an optimizer this build does not know (a newer server's)
    /// selects nothing and keeps the form from starting until one is picked, instead of becoming Auto.
    pub fn prefill_optimizer(&mut self, o: Optimizer) {
        match optimizer_index(o) {
            Some(i) => self.pick_optimizer(i),
            None => {
                self.optimizer = 0;
                self.optimizer_unknown = true;
            }
        }
    }

    /// Why the form cannot start for its optimizer, if it cannot.
    pub fn optimizer_blocks(&self) -> Option<&'static str> {
        self.optimizer_unknown
            .then_some("Pick an optimizer first: the run this was made from used an unknown one")
    }
}

/// The index of `o` in [`OPTIMIZERS`], `None` for one the picker does not offer (`Optimizer::Unknown`).
pub fn optimizer_index(o: Optimizer) -> Option<usize> {
    OPTIMIZERS.iter().position(|x| *x == o)
}

/// Image sizes and class sets offered for synthetic datasets.
pub const SYNTH_IMGSZ: [u32; 3] = [256, 320, 640];
pub const SYNTH_CLASSES: [&[&str]; 3] = [
    &["circle", "square", "triangle"],
    &["circle", "square", "triangle", "star", "hexagon"],
    &["circle", "square", "triangle", "star", "ring", "cross", "hexagon", "heart"],
];

/// Images in a synthetic dataset: a whole number in 1..=100000 (the server's limits).
pub fn parse_images(s: &str) -> Option<u64> {
    s.trim().replace([',', '_', ' '], "").parse::<u64>().ok().filter(|v| (1..=100_000).contains(v))
}

pub struct SynthForm {
    /// Which opening of the dialog this is: a late answer only touches its own dialog.
    pub id: u64,
    pub name: Entity<TextInput>,
    pub images: Entity<TextInput>,
    pub imgsz: usize,
    pub classes: usize,
    pub submitting: bool,
    pub error: Option<String>,
}

/// The name New run suggests: the dataset, the architecture's letter and the run's number
/// ("shapes-n-4").
pub fn suggested_name(dataset: &str, arch: Arch, n: usize) -> String {
    format!("{dataset}-{}-{n}", arch.as_str().trim_start_matches("folo-"))
}

pub struct NewRunForm {
    pub name: Entity<TextInput>,
    /// The name the form suggested while the field still holds it: it follows the dataset and the
    /// architecture until the name is typed (then `None`).
    pub name_suggested: Option<String>,
    pub dataset: usize,
    pub arch: usize,
    pub epochs: f32,
    pub batch: usize,
    pub imgsz: usize,
    pub device: usize,
    pub optimizer: usize,
    /// Prefilled from a run whose optimizer this build does not know: no optimizer is selected, and the form
    /// does not start until one is picked ([`NewRunForm::optimizer_blocks`]).
    pub optimizer_unknown: bool,
    pub lr0: f32,
    pub mosaic: f32,
    pub mixup: f32,
    pub fliplr: f32,
    /// Epochs at the end trained without mosaic.
    pub close_mosaic: f32,
    /// Scale jitter (± fraction).
    pub scale: f32,
    /// Why some values differ from the run this form was prefilled from.
    pub note: Option<String>,
    /// The run this form was prefilled from: settings the form doesn't show (weights, seed, colour and
    /// translate jitter) carry over unchanged.
    pub base: Option<TrainConfig>,
    /// Exact entry next to the sliders; invalid text keeps the last good value and shows a hint.
    pub epochs_input: Entity<TextInput>,
    pub lr_input: Entity<TextInput>,
    pub epochs_invalid: bool,
    pub lr_invalid: bool,
    pub close_input: Entity<TextInput>,
    pub close_invalid: bool,
    /// Close mosaic is the form's default and follows the epoch count (`follow_epochs`) until it
    /// is set by hand.
    pub close_auto: bool,
    /// Mixed-precision training (f16 on tensor cores). Sent only for a GPU device.
    pub amp: bool,
    /// The NMS-free end-to-end head (`head: o2o`); off is the default one-to-many head.
    pub e2e: bool,
    /// The Advanced group (head, optimizer, mixed precision, augmentation) is open.
    pub advanced: bool,
    /// Run on (servers with `cloud`): `None` trains on this machine, else on this cloud connection.
    pub run_on: Option<String>,
    /// The connection's offer to train on.
    pub offer: Option<String>,
    /// Stop after, hours (`compute.max_hours`): it follows the estimate until it is typed.
    pub max_hours: f32,
    pub hours_input: Entity<TextInput>,
    pub hours_invalid: bool,
    pub hours_edited: bool,
    /// What the server said about the compute choice, on the control it is about.
    pub compute_error: Option<(crate::cloud::ComputeField, String)>,
    /// New run from a cloud run's config: its offer, chosen once the connection's offers are in.
    pub want_offer: Option<String>,
    pub(super) _subs: Vec<Subscription>,
    pub submitting: bool,
    pub error: Option<ApiError>,
}

/// Epochs as typed: a whole number in 1..=10000 (the server's limit).
pub fn parse_epochs(s: &str) -> Option<u32> {
    s.trim().parse::<u32>().ok().filter(|v| (1..=10_000).contains(v))
}

/// Learning rate as typed, scientific notation allowed (`1e-3`): 0 < lr < 10.
/// A learning rate for its input field: three decimals in the slider's range (0.010), and as
/// many digits as it needs below it (0.0001 doesn't round to 0.000).
pub fn format_lr(v: f32) -> String {
    if v >= 0.001 { format!("{v:.3}") } else { format!("{v}") }
}

pub fn parse_lr(s: &str) -> Option<f32> {
    s.trim().parse::<f32>().ok().filter(|v| v.is_finite() && *v > 0.0 && *v < 10.0)
}

pub const ARCHS: [Arch; 5] = Arch::ALL;
pub const BATCHES: [u32; 7] = [2, 4, 8, 16, 32, 64, 128];

/// The offered batch size closest to `v` without exceeding it (the smallest if `v` is below all).
pub fn batch_index(v: u32) -> usize {
    BATCHES.iter().rposition(|b| *b <= v).unwrap_or(0)
}

/// Batch for retrying an out-of-memory run: half, snapped to an offered size; None at the minimum.
pub fn retry_batch(v: u32) -> Option<u32> {
    (v > BATCHES[0]).then(|| BATCHES[batch_index(v / 2)])
}

/// Keep what the form can't edit from the run it was prefilled from.
pub fn carry_over(base: Option<&TrainConfig>, mut c: TrainConfig) -> TrainConfig {
    if let Some(b) = base {
        c.weights = b.weights.clone();
        c.seed = b.seed;
        c.augment.hsv = b.augment.hsv;
        c.augment.translate = b.augment.translate;
        // Not `amp` nor `head`: the form shows both (prefilled from the base).
    }
    c
}
pub const IMGSZ: [u32; 5] = [320, 480, 640, 800, 1024];
pub const DEVICES: [&str; 3] = ["auto", "gpu:0", "cpu"];
pub const OPTIMIZERS: [Optimizer; 4] = [Optimizer::Auto, Optimizer::Sgd, Optimizer::Adamw, Optimizer::Musgd];

#[cfg(test)]
mod retry_tests {
    use super::{BATCHES, batch_index, carry_over, retry_batch};
    use crate::api::{Augment, TrainConfig};

    #[test]
    fn oom_retries_halve_to_offered_sizes() {
        assert_eq!(retry_batch(16), Some(8));
        assert_eq!(retry_batch(8), Some(4));
        assert_eq!(retry_batch(4), Some(2));
        assert_eq!(retry_batch(2), None);
        assert_eq!(retry_batch(48), Some(16));
        // Whatever the retry picks, the form can show it exactly.
        for b in [2, 4, 8, 16, 32, 64, 128] {
            if let Some(to) = retry_batch(b) {
                assert_eq!(BATCHES[batch_index(to)], to);
            }
        }
        assert_eq!(BATCHES[batch_index(1)], 2);
        assert_eq!(BATCHES[batch_index(500)], 128);
    }

    #[test]
    fn hidden_settings_carry_over() {
        let base = TrainConfig {
            name: "a".into(),
            dataset_id: "ds".into(),
            arch: crate::api::Arch::S,
            epochs: 50,
            batch: 8,
            imgsz: 640,
            device: "auto".into(),
            optimizer: crate::api::Optimizer::Auto,
            lr0: 0.01,
            weights: Some("runs/a/best.folo".into()),
            augment: Augment { hsv: [0.1, 0.2, 0.3], translate: 0.25, ..Augment::default() },
            seed: 42,
            amp: true,
            head: Default::default(),
            compute: None,
        };
        let edited = TrainConfig {
            name: "a-2".into(),
            weights: None,
            seed: 0,
            augment: Augment::default(),
            // The form turned mixed precision off (the CPU, or the user): the base must not undo it.
            amp: false,
            ..base.clone()
        };
        let out = carry_over(Some(&base), edited.clone());
        assert_eq!(out.weights, base.weights);
        assert_eq!(out.seed, 42);
        assert_eq!(out.augment.hsv, [0.1, 0.2, 0.3]);
        assert_eq!(out.augment.translate, 0.25);
        assert_eq!(out.name, "a-2");
        assert!(!out.amp);
        // The form's head stands (it shows the base's and may change it).
        let o2o = TrainConfig { head: crate::api::Head::O2o, ..edited.clone() };
        assert_eq!(carry_over(Some(&base), o2o).head, crate::api::Head::O2o);
        // A fresh form keeps its own values.
        assert_eq!(carry_over(None, edited.clone()), edited);
    }
}

#[cfg(test)]
mod form_tests {
    use super::{format_lr, parse_epochs, parse_lr, suggested_name};

    #[test]
    fn suggested_names_follow_the_choices() {
        assert_eq!(suggested_name("shapes-ux", crate::api::Arch::S, 4), "shapes-ux-s-4");
        assert_eq!(suggested_name("shapes-ux", crate::api::Arch::N, 4), "shapes-ux-n-4");
        assert_eq!(suggested_name("bccd", crate::api::Arch::X, 12), "bccd-x-12");
    }

    #[test]
    fn typed_numbers_validate() {
        assert_eq!(parse_epochs(" 250 "), Some(250));
        assert_eq!(parse_epochs("0"), None);
        assert_eq!(parse_epochs("10001"), None);
        assert_eq!(parse_epochs("12.5"), None);
        assert_eq!(parse_lr("1e-3"), Some(0.001));
        assert_eq!(parse_lr("0.01"), Some(0.01));
        assert_eq!(parse_lr("0"), None);
        assert_eq!(parse_lr("abc"), None);
        assert_eq!(parse_lr("NaN"), None);
        // Small rates keep their digits, so what the field shows reads back.
        assert_eq!(format_lr(0.01), "0.010");
        assert_eq!(format_lr(0.0001), "0.0001");
        assert_eq!(parse_lr(&format_lr(0.0001)), Some(0.0001));
    }
}

#[cfg(test)]
mod tests {
    use super::parse_close;

    /// The New run optimizer picker offers every value a client can send, so a run prefilled from one that
    /// trained with MuSGD keeps it; an optimizer this build does not know has no index (the form then shows it
    /// as unknown and blocks Start until one is picked, rather than becoming Auto).
    #[test]
    fn optimizer_picker_offers_every_optimizer() {
        use super::{OPTIMIZERS, optimizer_index};
        use crate::api::Optimizer;
        for o in [Optimizer::Auto, Optimizer::Sgd, Optimizer::Adamw, Optimizer::Musgd] {
            assert_eq!(optimizer_index(o).map(|i| OPTIMIZERS[i]), Some(o), "{o:?}");
        }
        assert_eq!(optimizer_index(Optimizer::Unknown), None);
    }

    #[test]
    fn close_mosaic_capped_as_the_engine_runs_it() {
        // docs/DOGFOOD-3 S2: a 15-epoch run asked for 10 closes mosaic for the last 7.
        assert_eq!(crate::api::capped_close_mosaic(10, 15), 7);
        assert_eq!(crate::api::capped_close_mosaic(10, 100), 10);
        assert_eq!(crate::api::capped_close_mosaic(10, 1), 0);
    }

    #[test]
    fn close_mosaic_default_is_studios() {
        // 10, never more than half the run (Studio's autoCloseMosaic: 0, 1, 2, 10, 10).
        let auto: Vec<u32> = [1, 2, 5, 20, 100].into_iter().map(super::auto_close_mosaic).collect();
        assert_eq!(auto, [0, 1, 2, 10, 10]);
    }

    #[test]
    fn close_mosaic_bounded_by_epochs() {
        // 200 is invalid at 100 epochs and valid once epochs rise to 300 ...
        assert_eq!(parse_close("200", 100.), None);
        assert_eq!(parse_close("200", 300.), Some(200));
        // ... and a valid value turns invalid when epochs drop below it.
        assert_eq!(parse_close("20", 100.), Some(20));
        assert_eq!(parse_close("20", 10.), None);
        assert_eq!(parse_close(" 7 ", 10.), Some(7));
        assert_eq!(parse_close("x", 10.), None);
    }

    #[test]
    fn synthetic_image_count() {
        use super::parse_images;
        assert_eq!(parse_images("1,000"), Some(1000));
        assert_eq!(parse_images(" 240 "), Some(240));
        assert_eq!(parse_images("0"), None);
        assert_eq!(parse_images("100001"), None);
        assert_eq!(parse_images("ten"), None);
    }
}
