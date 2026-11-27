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

    /// The selected batch size: a preserved non-preset value from a cloned run, else the preset.
    pub fn batch_value(&self) -> u32 {
        self.custom_batch.unwrap_or(BATCHES[self.batch])
    }

    /// The selected image size: a preserved non-preset value from a cloned run, else the preset.
    pub fn imgsz_value(&self) -> u32 {
        self.custom_imgsz.unwrap_or(IMGSZ[self.imgsz])
    }

    /// The selected device: a preserved non-preset value (e.g. `gpu:1`) from a cloned run, else the preset.
    pub fn device_value(&self) -> &str {
        self.custom_device.as_deref().unwrap_or(DEVICES[self.device])
    }

    /// Picks preset device `i` of [`DEVICES`] (also clears any cloned custom device).
    pub fn pick_device(&mut self, i: usize) {
        self.device = i;
        self.custom_device = None;
    }

    /// Canonical architecture selected in the New run form (`folo-*` or `folo2-*`).
    pub fn model_arch(&self) -> crate::arch::ModelArch {
        crate::arch::ModelArch::new(self.arch_family, ARCHS[self.arch.min(ARCHS.len() - 1)])
    }
}

/// Parsed device string matching `folo_vision::train::config::resolve_device` (aliases, case,
/// and surrounding whitespace).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ParsedDevice {
    /// One of [`DEVICES`]: `0 = auto`, `1 = gpu:0`, `2 = cpu`.
    Preset(usize),
    /// A non-zero GPU adapter index (`gpu:<n>` or `cuda:<n>` for `n > 0`), canonicalized to `gpu:<n>`.
    Gpu(usize),
    /// An unrecognized device string preserved verbatim so the user sees it and picks a replacement.
    Unknown(String),
}

/// Parses a device string the same way the training backend does (`auto`/`""`, `cpu`,
/// `gpu`/`cuda`/`gpu:0`/`cuda:0`, `gpu:<n>`/`cuda:<n>`, case-insensitive, trimmed).
pub fn parse_device(raw: &str) -> ParsedDevice {
    match raw.trim().to_ascii_lowercase().as_str() {
        "auto" | "" => ParsedDevice::Preset(0),
        "gpu" | "cuda" => ParsedDevice::Preset(1),
        "cpu" => ParsedDevice::Preset(2),
        other => {
            let idx = other
                .strip_prefix("gpu:")
                .or_else(|| other.strip_prefix("cuda:"))
                .and_then(|n| n.parse::<usize>().ok());
            match idx {
                Some(0) => ParsedDevice::Preset(1),
                Some(i) => ParsedDevice::Gpu(i),
                None => ParsedDevice::Unknown(raw.trim().to_string()),
            }
        }
    }
}

/// GPU adapter index for a device string (`"gpu"`, `"cuda"`, `"gpu:1"`, `"CUDA:0"`).
pub fn gpu_index(device: &str) -> Option<usize> {
    match parse_device(device) {
        ParsedDevice::Preset(1) => Some(0),
        ParsedDevice::Gpu(i) => Some(i),
        _ => None,
    }
}

/// Human-readable label for a device string (`"auto"` -> `"Auto"`, `"gpu:1"` -> `"GPU 1"`).
pub fn device_label(device: &str) -> String {
    match parse_device(device) {
        ParsedDevice::Preset(0) => "Auto".into(),
        ParsedDevice::Preset(1) => "GPU 0".into(),
        ParsedDevice::Preset(2) => "CPU".into(),
        ParsedDevice::Preset(_) => device.to_string(),
        ParsedDevice::Gpu(i) => format!("GPU {i}"),
        ParsedDevice::Unknown(s) => s,
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

/// Images in a synthetic dataset: a whole number in 1..=100000 (the server's limits), plain
/// (`12000`) or grouped in thousands by `,`, `_` or a space (`12,000`). A separator anywhere else
/// is no number: `1,5` (a decimal comma) doesn't become 15. One leading `+` (as a spreadsheet may
/// show it) is a sign, not part of the first group.
pub fn parse_images(s: &str) -> Option<u64> {
    let t = s.trim();
    let t = t.strip_prefix('+').unwrap_or(t);
    if t.starts_with('+') {
        return None;
    }
    let groups: Vec<&str> = t.split([',', '_', ' ']).collect();
    if let [head, rest @ ..] = groups.as_slice()
        && !rest.is_empty()
        && !((1..=3).contains(&head.len()) && rest.iter().all(|g| g.len() == 3))
    {
        return None;
    }
    groups.concat().parse::<u64>().ok().filter(|v| (1..=100_000).contains(v))
}

/// What is wrong with the Generate synthetic form's name and image count as typed, each shown
/// under its field at once (and Generate waits until both are right).
pub fn synth_errors(name: &str, images: &str) -> (Option<&'static str>, Option<&'static str>) {
    (
        name.trim().is_empty().then_some("Give the dataset a name"),
        parse_images(images).is_none().then_some("A whole number from 1 to 100,000"),
    )
}

pub struct SynthForm {
    /// Which opening of the dialog this is: a late answer only touches its own dialog.
    pub id: u64,
    pub name: Entity<TextInput>,
    pub images: Entity<TextInput>,
    pub imgsz: usize,
    pub classes: usize,
    pub submitting: bool,
    /// The name sent with Generate (the field stays editable while the server works).
    pub sent_name: String,
    pub error: Option<ApiError>,
}

/// The longest run name the server takes (docs/API.md, `POST /api/runs`), as for a dataset's.
pub const MAX_RUN_NAME: usize = 64;

/// `base` and then `suffix`, `base` shortened (and dashes left at its cut dropped) so the whole fits
/// in [`MAX_RUN_NAME`] characters, as Studio's New run fits its names (`web/src/routes/runnew/name.ts`).
pub fn fit_run_name(base: &str, suffix: &str) -> String {
    let room = MAX_RUN_NAME.saturating_sub(suffix.chars().count());
    let cut: String = base.chars().take(room).collect();
    format!("{}{suffix}", cut.trim_end_matches('-'))
}

/// What is wrong with a typed run name, in Studio's words: empty, or longer than the server takes.
pub fn run_name_error(name: &str) -> Option<String> {
    let n = name.trim().chars().count();
    if n == 0 {
        Some("Give the run a name.".into())
    } else if n > MAX_RUN_NAME {
        Some(format!("At most {MAX_RUN_NAME} characters (this one has {n})."))
    } else {
        None
    }
}

/// The name New run suggests for any canonical architecture (`shapes-n-4` for v1,
/// `shapes-folo2-n-4` for v2), the dataset's part shortened to fit the length limit.
pub fn suggested_model_name(dataset: &str, arch: crate::arch::ModelArch, n: usize) -> String {
    fit_run_name(dataset, &format!("-{}-{n}", arch.name_suffix()))
}

/// The name New run suggests: the dataset, the architecture's letter and the run's number
/// ("shapes-n-4"), the dataset's part shortened to fit the length limit.
#[cfg(test)]
pub fn suggested_name(dataset: &str, arch: Arch, n: usize) -> String {
    suggested_model_name(dataset, crate::arch::ModelArch::v1(arch), n)
}

/// The name of a new run made from run `name` (New run from this run, Retry): `name-2`, or the next
/// number no run of `runs` is called, `name` shortened to fit the length limit.
pub fn again_run_name(name: &str, runs: &[Run]) -> String {
    (2..)
        .map(|n| fit_run_name(name, &format!("-{n}")))
        .find(|candidate| !runs.iter().any(|r| r.name == *candidate))
        .expect("a free number")
}

/// [`suggested_model_name`] numbered one past the run count, or the next number no run of `runs` is
/// called: after a delete the count alone names a run that exists, and with no rename a second run
/// (and its models) would keep that name.
pub fn free_model_run_name(dataset: &str, arch: crate::arch::ModelArch, runs: &[Run]) -> String {
    (runs.len() + 1..)
        .map(|n| suggested_model_name(dataset, arch, n))
        .find(|name| !runs.iter().any(|r| r.name == *name))
        .expect("one of len + 1 names is free")
}

/// [`suggested_name`] numbered one past the run count, or the next number no run of `runs` is called.
#[cfg(test)]
pub fn free_run_name(dataset: &str, arch: Arch, runs: &[Run]) -> String {
    free_model_run_name(dataset, crate::arch::ModelArch::v1(arch), runs)
}

pub struct NewRunForm {
    pub name: Entity<TextInput>,
    /// The name the form suggested while the field still holds it: it follows the dataset and the
    /// architecture until the name is typed (then `None`).
    pub name_suggested: Option<String>,
    /// The dataset's id: a deletion that shrinks the list doesn't move the choice to another one
    /// (gone itself, the form asks for another, [`FoloApp::nr_dataset_gone`]).
    ///
    /// [`FoloApp::nr_dataset_gone`]: crate::app::FoloApp::nr_dataset_gone
    pub dataset: String,
    pub arch_family: crate::arch::ArchFamily,
    pub arch: usize,
    pub epochs: f32,
    pub batch: usize,
    /// Non-preset batch size carried over from a cloned/retried run until another choice is picked.
    pub custom_batch: Option<u32>,
    pub imgsz: usize,
    /// Non-preset image size carried over from a cloned/retried run until another choice is picked.
    pub custom_imgsz: Option<u32>,
    pub device: usize,
    /// Non-preset device (e.g. `gpu:1`) carried over from a cloned/retried run until another choice is picked.
    pub custom_device: Option<String>,
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
    /// Start from: the model id (or published name) to fine-tune, `None` from scratch.
    pub weights: Option<String>,
    /// The picked weights' name, to name them if they are deleted while picked.
    pub weights_name: Option<String>,
    /// The run this form was prefilled from: settings the form doesn't show (seed, colour and
    /// translate jitter) carry over unchanged.
    pub base: Option<TrainConfig>,
    /// Run id this form was cloned/retried from, so a late HTTP arch probe can sync its family.
    pub source_run_id: Option<String>,
    /// Set once the user explicitly picks a family or scale in the dialog.
    pub arch_touched: bool,
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
        // Not `weights`: Start from shows them (prefilled from the base).
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
        // Start from shows the weights: the form's choice stands (here, from scratch).
        assert_eq!(out.weights, None);
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
    fn the_suggested_run_name_is_never_one_a_run_has() {
        use crate::api::{Arch, Backend, Run};
        let base = crate::api::mock::MockBackend::new(4200, true).runs().unwrap().remove(0);
        let runs = |names: &[&str]| -> Vec<Run> {
            names.iter().map(|n| Run { name: n.to_string(), ..base.clone() }).collect()
        };
        // Five runs left after -4 was deleted: the count says -6, which a run has.
        let left = runs(&["a", "b", "shapes-v-n-5", "shapes-v-n-6", "c"]);
        assert_eq!(super::free_run_name("shapes-v", Arch::N, &left), "shapes-v-n-7");
        // Free at the count: the count.
        assert_eq!(super::free_run_name("shapes-v", Arch::N, &runs(&["shapes-v-n-5"])), "shapes-v-n-2");
        assert_eq!(super::free_run_name("shapes-v", Arch::S, &left), "shapes-v-s-6");
        assert_eq!(super::free_run_name("shapes-v", Arch::S, &[]), "shapes-v-s-1");
    }

    #[test]
    fn every_suggested_name_fits_the_servers_limit() {
        use super::{MAX_RUN_NAME, again_run_name, fit_run_name, run_name_error};
        use crate::api::{Arch, Backend, Run};
        let long = "a-very-long-dataset-name-registered-from-a-deeply-nested-folder-of-images";
        let s = suggested_name(long, Arch::S, 12);
        assert_eq!(s.chars().count(), MAX_RUN_NAME);
        assert!(s.starts_with("a-very-long") && s.ends_with("-s-12"), "{s}");
        // A cut right after a dash leaves no double dash.
        assert_eq!(fit_run_name(&format!("{}-tail", "x".repeat(61)), "-2"), format!("{}-2", "x".repeat(61)));
        // Characters, not bytes.
        assert_eq!(fit_run_name(&"é".repeat(70), "").chars().count(), MAX_RUN_NAME);
        // New run from a run named at the limit: shortened, numbered past the names taken.
        let base = crate::api::mock::MockBackend::new(4200, true).runs().unwrap().remove(0);
        let at_limit = "r".repeat(MAX_RUN_NAME);
        let two = again_run_name(&at_limit, &[]);
        assert_eq!((two.chars().count(), two.ends_with("-2")), (MAX_RUN_NAME, true));
        let taken = vec![Run { name: two.clone(), ..base }];
        let three = again_run_name(&at_limit, &taken);
        assert!(three.ends_with("-3") && three.chars().count() == MAX_RUN_NAME, "{three}");
        assert_eq!(again_run_name("shapes-s-1", &[]), "shapes-s-1-2");
        // A typed name: Studio's words.
        assert_eq!(run_name_error("  ").as_deref(), Some("Give the run a name."));
        assert_eq!(run_name_error(&at_limit), None);
        assert_eq!(
            run_name_error(&format!("{at_limit}x")).as_deref(),
            Some("At most 64 characters (this one has 65).")
        );
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
        // Separators only between groups of three digits (#351): a decimal comma is no count.
        for ok in ["12000", "12,000", "100,000", "1_000", "1 000", "999"] {
            assert!(parse_images(ok).is_some(), "{ok}");
        }
        for bad in ["1,5", "1,0", "10,00", "1,0000", "1000,000", ",500", "1,000,", "1,,000"] {
            assert_eq!(parse_images(bad), None, "{bad}");
        }
        // A leading `+` is a sign, not part of the first group.
        assert_eq!(parse_images("+100,000"), Some(100_000));
        assert_eq!(parse_images("+1,000"), Some(1000));
        assert_eq!(parse_images("+12"), Some(12));
        for bad in ["+,100", "+", "++1", "+1,5", "+ 1,000"] {
            assert_eq!(parse_images(bad), None, "{bad}");
        }
        use super::synth_errors;
        assert_eq!(synth_errors("s", "1,5").1, Some("A whole number from 1 to 100,000"));
        assert_eq!(synth_errors("shapes", "1,000"), (None, None));
        assert_eq!(
            synth_errors("  ", "0"),
            (Some("Give the dataset a name"), Some("A whole number from 1 to 100,000"))
        );
        assert_eq!(synth_errors("s", "200000").1, Some("A whole number from 1 to 100,000"));
    }
}
