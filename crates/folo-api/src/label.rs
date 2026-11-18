//! Agent labeling (docs/API.md, *Agent labeling*): unlabeled images, boxes drawn by a coding-agent CLI on the
//! server's machine (Claude Code, Codex), and the YOLO dataset made from them. Everything under `/api/label`.

use serde::{Deserialize, Serialize};

use crate::{BBox, LogLine};

/// [`SystemResponse::features`](crate::SystemResponse::features): the `/api/label` endpoints exist.
pub const FEATURE_AGENT_LABELING: &str = "agent_labeling";

/// A coding-agent CLI FOLO can ask for labels.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "lowercase")]
pub enum LabelAgentId {
    /// Claude Code (`claude`).
    Claude,
    /// Codex (`codex`).
    Codex,
}

impl LabelAgentId {
    /// Every agent, in the order they are listed.
    pub const ALL: [LabelAgentId; 2] = [LabelAgentId::Claude, LabelAgentId::Codex];

    /// The serialised name (`claude`, `codex`).
    pub fn as_str(self) -> &'static str {
        match self {
            LabelAgentId::Claude => "claude",
            LabelAgentId::Codex => "codex",
        }
    }

    /// Parses a serialised name.
    pub fn parse(s: &str) -> Option<LabelAgentId> {
        LabelAgentId::ALL.into_iter().find(|v| v.as_str() == s)
    }

    /// The product name (`Claude Code`, `Codex`).
    pub fn display_name(self) -> &'static str {
        match self {
            LabelAgentId::Claude => "Claude Code",
            LabelAgentId::Codex => "Codex",
        }
    }
}

/// `GET /api/label/agents`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct LabelAgentList {
    /// The agents this server knows, installed or not.
    pub agents: Vec<LabelAgent>,
}

/// One agent CLI and whether the server found it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct LabelAgent {
    /// The agent.
    pub id: LabelAgentId,
    /// Product name, e.g. `Claude Code`.
    pub name: String,
    /// True when its CLI is installed on the server's machine and answered `--version`. Whether it is signed in
    /// is only known once a run calls it.
    pub available: bool,
    /// What `--version` printed, e.g. `2.1.302`; `null` when not available.
    pub version: Option<String>,
    /// Models to offer. [`StartLabeling::model`] takes any model name the CLI accepts, listed or not.
    pub models: Vec<LabelModel>,
    /// The model a run uses when it names none; `null` means the CLI's own default.
    pub default_model: Option<String>,
    /// The reasoning effort a run uses when it names none: the one that labels best for its cost with this
    /// agent.
    pub default_effort: LabelEffort,
    /// The agent offers a faster service tier that uses more of the account's usage ([`StartLabeling::fast`]):
    /// Codex's Fast (priority processing). Claude Code offers none to FOLO's calls.
    #[serde(default)]
    pub fast: bool,
    /// How to make the agent available (install, sign in); `null` when it is.
    pub hint: Option<String>,
}

/// A model of a [`LabelAgent`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct LabelModel {
    /// What the CLI takes as the model name.
    pub id: String,
    /// Display name.
    pub label: String,
}

/// `GET /api/label/jobs`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct LabelJobList {
    /// Labeling jobs, newest first.
    pub jobs: Vec<LabelJob>,
}

/// Whether an agent is working on a [`LabelJob`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "lowercase")]
pub enum LabelJobStatus {
    /// No agent run is active.
    #[default]
    Idle,
    /// An agent run is labeling images.
    Running,
    /// A stop was requested; the calls in flight are being ended.
    Stopping,
}

impl LabelJobStatus {
    /// True while a run is active (`running` or `stopping`).
    pub fn is_active(self) -> bool {
        self != LabelJobStatus::Idle
    }
}

/// A labeling job: a set of images, the classes to find in them, and the labels so far.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct LabelJob {
    /// Opaque id, e.g. `lb_3f9a`.
    pub id: String,
    /// Display name; the default name of the dataset made from the job.
    pub name: String,
    /// Class names; a box's `class` indexes this list.
    pub classes: Vec<String>,
    /// What the agent is told about the classes and the images, in the user's words (may be empty).
    pub instructions: String,
    /// RFC 3339 creation time.
    pub created: String,
    /// Whether an agent run is active.
    pub status: LabelJobStatus,
    /// Image counts.
    pub counts: LabelCounts,
    /// The active agent run, else the most recent one; `null` before the first.
    pub run: Option<LabelRun>,
    /// Token use of every run of the job.
    pub usage: LabelUsage,
    /// The dataset most recently made from the job; `null` before the first.
    pub dataset_id: Option<String>,
}

/// Image counts of a [`LabelJob`]. `pending + labeled + failed == images`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct LabelCounts {
    /// All images of the job.
    pub images: u64,
    /// Images no agent has labeled yet.
    pub pending: u64,
    /// Images with labels (possibly zero boxes: a background image).
    pub labeled: u64,
    /// Images whose last agent call failed.
    pub failed: u64,
    /// Boxes over all labeled images.
    pub boxes: u64,
}

/// How hard the agent's model thinks per call (the CLIs' reasoning effort).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "lowercase")]
pub enum LabelEffort {
    /// Fewest reasoning tokens.
    Low,
    /// Medium.
    Medium,
    /// Most.
    High,
}

impl LabelEffort {
    /// The serialised name (`low`, `medium`, `high`), which is also what both CLIs call it.
    pub fn as_str(self) -> &'static str {
        match self {
            LabelEffort::Low => "low",
            LabelEffort::Medium => "medium",
            LabelEffort::High => "high",
        }
    }
}

/// How an agent run ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "lowercase")]
pub enum LabelOutcome {
    /// Every image of the run was tried.
    Completed,
    /// Stopped on request.
    Stopped,
    /// Ended early because the agent kept failing (not installed, not signed in, a bad model name, ...); `error`
    /// says why.
    Failed,
}

/// One agent run over some images of a [`LabelJob`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct LabelRun {
    /// The agent.
    pub agent: LabelAgentId,
    /// The model it was asked for; `null` means the CLI's default.
    pub model: Option<String>,
    /// Reasoning effort.
    pub effort: LabelEffort,
    /// It asked for the faster service tier ([`StartLabeling::fast`]).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub fast: bool,
    /// RFC 3339 start time.
    pub started: String,
    /// RFC 3339 end time; `null` while the run is active.
    pub finished: Option<String>,
    /// Images the run set out to label.
    pub total: u64,
    /// Images labeled so far.
    pub done: u64,
    /// Images whose call failed so far.
    pub failed: u64,
    /// How it ended; `null` while the run is active.
    pub outcome: Option<LabelOutcome>,
    /// Why it failed (`outcome: failed`), else `null`.
    pub error: Option<String>,
    /// How to fix `error`, when there is something useful to say.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hint: Option<String>,
    /// Token use of this run.
    pub usage: LabelUsage,
}

/// Tokens the agent CLIs reported.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct LabelUsage {
    /// Agent calls made (one call labels a batch of images).
    pub calls: u64,
    /// Input tokens, cached ones included.
    pub input_tokens: u64,
    /// The part of `input_tokens` read from the provider's prompt cache.
    pub cached_input_tokens: u64,
    /// Output tokens, reasoning included.
    pub output_tokens: u64,
    /// The cost the CLIs reported, in US dollars (Claude Code reports one, at list prices even on a subscription;
    /// Codex reports none); `null` when no call reported one.
    pub cost_usd: Option<f64>,
}

impl LabelUsage {
    /// Adds `other` to `self`.
    pub fn add(&mut self, other: &LabelUsage) {
        self.calls += other.calls;
        self.input_tokens += other.input_tokens;
        self.cached_input_tokens += other.cached_input_tokens;
        self.output_tokens += other.output_tokens;
        if let Some(c) = other.cost_usd {
            self.cost_usd = Some(self.cost_usd.unwrap_or(0.0) + c);
        }
    }
}

/// `POST /api/label/jobs`: a new job.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct CreateLabelJob {
    /// Display name.
    pub name: String,
    /// Class names (1 to 80, distinct).
    pub classes: Vec<String>,
    /// What to tell the agent about the classes and the images.
    #[serde(default)]
    pub instructions: String,
    /// A folder of images on the server's machine to start with (as [`ImportLabelImages::path`]).
    #[serde(default)]
    pub path: Option<String>,
}

/// `PUT /api/label/jobs/{id}`: the fields to change (absent ones stay).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct UpdateLabelJob {
    /// New display name.
    #[serde(default)]
    pub name: Option<String>,
    /// New class list. Once the job has boxes, classes can be renamed and appended, not removed or reordered
    /// (boxes refer to them by position).
    #[serde(default)]
    pub classes: Option<Vec<String>>,
    /// New instructions.
    #[serde(default)]
    pub instructions: Option<String>,
}

/// JSON body of `POST /api/label/jobs/{id}/images`: add the images of a folder on the server's machine.
/// (Uploads use `multipart/form-data` with one `image` file part per image.)
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct ImportLabelImages {
    /// The folder (searched recursively) or a single image file. Its images are read in place, not copied.
    pub path: String,
}

/// Answer of `POST /api/label/jobs/{id}/images`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct LabelImagesAdded {
    /// The images added, in order.
    pub added: Vec<LabelImage>,
    /// Files left out, with the reason (not an image, a duplicate, the job is full).
    pub skipped: Vec<SkippedImage>,
    /// The job after the change.
    pub job: LabelJob,
}

/// A file `POST /api/label/jobs/{id}/images` did not add.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct SkippedImage {
    /// The file's name.
    pub file: String,
    /// Why it was left out.
    pub reason: String,
}

/// Where an image of a [`LabelJob`] stands.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "lowercase")]
pub enum LabelImageStatus {
    /// Not labeled yet.
    #[default]
    Pending,
    /// In an agent call right now (its previous labels, if any, are still in `boxes`).
    Labeling,
    /// Labeled (by an agent or by hand).
    Labeled,
    /// The last agent call for it failed; `error` says why.
    Failed,
}

impl LabelImageStatus {
    /// Every status.
    pub const ALL: [LabelImageStatus; 4] = [
        LabelImageStatus::Pending,
        LabelImageStatus::Labeling,
        LabelImageStatus::Labeled,
        LabelImageStatus::Failed,
    ];

    /// The serialised name.
    pub fn as_str(self) -> &'static str {
        match self {
            LabelImageStatus::Pending => "pending",
            LabelImageStatus::Labeling => "labeling",
            LabelImageStatus::Labeled => "labeled",
            LabelImageStatus::Failed => "failed",
        }
    }

    /// Parses a serialised name.
    pub fn parse(s: &str) -> Option<LabelImageStatus> {
        LabelImageStatus::ALL.into_iter().find(|v| v.as_str() == s)
    }
}

/// Who drew an image's current boxes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "lowercase")]
pub enum LabelSource {
    /// An agent.
    Agent,
    /// A person (`PUT /api/label/jobs/{id}/images/{image_id}`).
    Manual,
}

/// One image of a [`LabelJob`] with its labels.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct LabelImage {
    /// Opaque id, unique within the job and never reused.
    pub id: String,
    /// File name, unique within the job; the dataset keeps it.
    pub file: String,
    /// Width in pixels (after EXIF orientation).
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Where it stands.
    pub status: LabelImageStatus,
    /// Its labels, pixel `xyxy` (empty until labeled; a labeled image may have none).
    pub boxes: Vec<BBox>,
    /// Who drew `boxes`; `null` until labeled.
    pub source: Option<LabelSource>,
    /// Why the last agent call for it failed (`status: failed`), else `null`.
    pub error: Option<String>,
    /// A person checked its labels: saved boxes drawn by hand, or marked them right. Only a `labeled` image is
    /// reviewed; an agent that labels it again clears it.
    #[serde(default)]
    pub reviewed: bool,
}

/// `GET /api/label/jobs/{id}/images`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct LabelImagePage {
    /// Images matching the filter (before `offset` / `limit`).
    pub total: u64,
    /// This page, in the order the images were added.
    pub images: Vec<LabelImage>,
}

/// `PUT /api/label/jobs/{id}/images/{image_id}`: labels drawn by hand.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct UpdateLabelImage {
    /// The image's boxes, pixel `xyxy`, replacing what it has; absent: they stay (to mark them reviewed).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub boxes: Option<Vec<BBox>>,
    /// Marks the image reviewed or not; absent: reviewed when `boxes` are given, else unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reviewed: Option<bool>,
}

/// Which images an agent run labels.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "lowercase")]
pub enum LabelScope {
    /// The images without labels: `pending` and `failed` ones (the default).
    #[default]
    Unlabeled,
    /// Every image, replacing the labels they have (hand-drawn ones too).
    All,
}

/// `POST /api/label/jobs/{id}/start`: ask an agent to label images of the job.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct StartLabeling {
    /// The agent to ask.
    pub agent: LabelAgentId,
    /// Its model; absent or `null` uses [`LabelAgent::default_model`].
    #[serde(default)]
    pub model: Option<String>,
    /// Reasoning effort; absent or `null` uses [`LabelAgent::default_effort`].
    #[serde(default)]
    pub effort: Option<LabelEffort>,
    /// Ask for the agent's faster service tier, which uses more of the account's usage; only for an agent whose
    /// [`LabelAgent::fast`] is true (otherwise `400 invalid_body`).
    #[serde(default)]
    pub fast: bool,
    /// Which images to label, unless `image_ids` names them.
    #[serde(default)]
    pub scope: LabelScope,
    /// Exactly these images (whatever their status), instead of `scope`.
    #[serde(default)]
    pub image_ids: Option<Vec<String>>,
    /// An extra instruction for this run only, e.g. what the previous labels got wrong.
    #[serde(default)]
    pub note: String,
    /// Images per agent call (1 to 16); absent uses the agent's default (Claude Code 4, Codex 8).
    #[serde(default)]
    pub batch: Option<u32>,
    /// Agent calls in flight at once (1 to 8, default 3).
    #[serde(default)]
    pub parallel: Option<u32>,
}

/// `POST /api/label/jobs/{id}/dataset`: make a dataset from the job's labeled images.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct LabelDatasetRequest {
    /// Dataset name; defaults to the job's.
    #[serde(default)]
    pub name: Option<String>,
    /// Share of the images that goes to the `val` split, in `0..=0.5`.
    #[serde(default = "default_val_fraction")]
    pub val_fraction: f32,
    /// Seed of the train / val shuffle.
    #[serde(default)]
    pub seed: u64,
}

impl Default for LabelDatasetRequest {
    fn default() -> Self {
        LabelDatasetRequest { name: None, val_fraction: default_val_fraction(), seed: 0 }
    }
}

fn default_val_fraction() -> f32 {
    0.2
}

/// One event of `GET /api/label/jobs/{id}/events`. On the wire the SSE `event:` field is [`LabelEvent::name`] and
/// `data:` is the JSON of the payload alone.
#[derive(Clone, Debug, PartialEq)]
pub enum LabelEvent {
    /// `job`: the job after a change (status, counts, run progress, usage).
    Job(Box<LabelJob>),
    /// `image`: an image after a change (added, sent to the agent, labeled, failed, edited).
    Image(LabelImage),
    /// `removed`: an image was deleted.
    Removed(RemovedImage),
    /// `reload` (payload `{}`): more changed than the stream carries (many images added at once, or this
    /// subscriber fell behind): load the job's images again.
    Reload,
    /// `log`: a line of the job's log.
    Log(LogLine),
}

/// SSE `removed` payload.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct RemovedImage {
    /// The deleted image's id.
    pub id: String,
}

impl LabelEvent {
    /// The SSE event name.
    pub fn name(&self) -> &'static str {
        match self {
            LabelEvent::Job(_) => "job",
            LabelEvent::Image(_) => "image",
            LabelEvent::Removed(_) => "removed",
            LabelEvent::Reload => "reload",
            LabelEvent::Log(_) => "log",
        }
    }

    /// The JSON `data:` payload.
    pub fn data_json(&self) -> String {
        let r = match self {
            LabelEvent::Job(p) => serde_json::to_string(p),
            LabelEvent::Image(p) => serde_json::to_string(p),
            LabelEvent::Removed(p) => serde_json::to_string(p),
            LabelEvent::Reload => Ok("{}".to_string()),
            LabelEvent::Log(p) => serde_json::to_string(p),
        };
        r.expect("event payloads always serialize")
    }

    /// Parses an SSE event from its `event:` name and `data:` JSON. Returns `None` for unknown event names (new
    /// event types may be added; clients should skip them) and an error for malformed data.
    pub fn from_sse(name: &str, data: &str) -> Option<Result<LabelEvent, serde_json::Error>> {
        Some(match name {
            "job" => serde_json::from_str(data).map(LabelEvent::Job),
            "image" => serde_json::from_str(data).map(LabelEvent::Image),
            "removed" => serde_json::from_str(data).map(LabelEvent::Removed),
            "reload" => Ok(LabelEvent::Reload),
            "log" => serde_json::from_str(data).map(LabelEvent::Log),
            _ => return None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn start_defaults_and_wire_names() {
        let s: StartLabeling = serde_json::from_str(r#"{"agent":"codex"}"#).unwrap();
        assert_eq!(s.agent, LabelAgentId::Codex);
        assert_eq!((s.effort, s.scope), (None, LabelScope::Unlabeled));
        assert!(s.model.is_none() && s.image_ids.is_none() && s.batch.is_none() && s.note.is_empty());
        assert!(serde_json::from_str::<StartLabeling>(r#"{"agent":"gemini"}"#).is_err());
        for a in LabelAgentId::ALL {
            assert_eq!(LabelAgentId::parse(a.as_str()), Some(a));
            assert_eq!(serde_json::to_value(a).unwrap(), a.as_str());
        }
        for s in LabelImageStatus::ALL {
            assert_eq!(LabelImageStatus::parse(s.as_str()), Some(s));
            assert_eq!(serde_json::to_value(s).unwrap(), s.as_str());
        }
        let d: LabelDatasetRequest = serde_json::from_str("{}").unwrap();
        assert_eq!(d, LabelDatasetRequest::default());
        assert_eq!(d.val_fraction, 0.2);
    }

    #[test]
    fn usage_adds_and_keeps_an_unknown_cost_unknown() {
        let mut u = LabelUsage::default();
        u.add(&LabelUsage {
            calls: 1,
            input_tokens: 10,
            cached_input_tokens: 4,
            output_tokens: 3,
            cost_usd: None,
        });
        assert_eq!((u.calls, u.input_tokens, u.cached_input_tokens, u.output_tokens), (1, 10, 4, 3));
        assert_eq!(u.cost_usd, None);
        u.add(&LabelUsage { calls: 1, cost_usd: Some(0.25), ..Default::default() });
        u.add(&LabelUsage { calls: 1, cost_usd: Some(0.5), ..Default::default() });
        assert_eq!((u.calls, u.cost_usd), (3, Some(0.75)));
        // `cost_usd` is sent as `null`, not left out.
        let v = serde_json::to_value(LabelUsage::default()).unwrap();
        assert!(v.as_object().unwrap().contains_key("cost_usd") && v["cost_usd"].is_null(), "{v}");
    }

    #[test]
    fn events_round_trip_by_name() {
        let image = LabelImage {
            id: "7".into(),
            file: "a.jpg".into(),
            width: 640,
            height: 480,
            status: LabelImageStatus::Labeled,
            boxes: vec![BBox { class: 1, x1: 1.0, y1: 2.0, x2: 30.0, y2: 40.0 }],
            source: Some(LabelSource::Agent),
            error: None,
            reviewed: true,
        };
        // An older server sends no `reviewed`: nobody checked the image.
        let mut old = serde_json::to_value(&image).unwrap();
        old.as_object_mut().unwrap().remove("reviewed");
        assert!(!serde_json::from_value::<LabelImage>(old).unwrap().reviewed);
        let ev = LabelEvent::Image(image);
        assert_eq!(ev.name(), "image");
        assert_eq!(LabelEvent::from_sse(ev.name(), &ev.data_json()).unwrap().unwrap(), ev);
        let ev = LabelEvent::Removed(RemovedImage { id: "7".into() });
        assert_eq!(LabelEvent::from_sse("removed", &ev.data_json()).unwrap().unwrap(), ev);
        assert_eq!(LabelEvent::from_sse("reload", "{}").unwrap().unwrap(), LabelEvent::Reload);
        assert_eq!((LabelEvent::Reload.name(), LabelEvent::Reload.data_json().as_str()), ("reload", "{}"));
        assert!(LabelEvent::from_sse("later", "{}").is_none());
        assert!(LabelEvent::from_sse("job", "{").unwrap().is_err());
    }
}
