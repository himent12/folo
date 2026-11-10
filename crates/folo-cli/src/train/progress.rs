//! What a run shows while it trains (a live view or JSON lines) and the summary at its end.

use std::io::{IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use folo_api::{RunEvent, RunStatus};
use folo_vision::TrainConfig;
use folo_vision::train::{TrainSummary, event_record};
use serde_json::json;

use crate::cli::JsonEvents;
use crate::output::Out;

/// The event consumer: JSON lines with `--json` (plus one-line epoch summaries on stderr), otherwise a live view.
/// `nb` (batches per epoch) is filled in once the trainer exists.
pub(super) fn progress_sink(
    out: Out,
    records: JsonEvents,
    epochs: u32,
    nb: Arc<AtomicU64>,
) -> folo_vision::train::EventSink {
    let tty = std::io::stdout().is_terminal();
    let mut view = Progress { out, records, epochs, nb, tty, header: false, live: false };
    Box::new(move |e: &RunEvent| view.event(e))
}

struct Progress {
    out: Out,
    /// `--json-events`.
    records: JsonEvents,
    epochs: u32,
    nb: Arc<AtomicU64>,
    tty: bool,
    header: bool,
    /// A `\r` progress line is on screen.
    live: bool,
}

const BAR: usize = 24;

impl Progress {
    fn event(&mut self, e: &RunEvent) {
        if self.out.json {
            if !(self.records == JsonEvents::Epoch && matches!(e, RunEvent::Step(_))) {
                self.out.record(&event_record(e));
            }
            if let RunEvent::Epoch(ep) = e {
                eprintln!(
                    "epoch {}/{}  loss {:.3}  mAP50 {:.3}  mAP50-95 {:.3}  {:.1}s",
                    ep.epoch, self.epochs, ep.train_loss.total, ep.val.map50, ep.val.map50_95, ep.duration_s
                );
            }
            return;
        }
        match e {
            RunEvent::Step(s) => {
                let nb = self.nb.load(Ordering::Relaxed).max(1);
                let k = s.step - (s.epoch as u64 - 1) * nb;
                if !self.tty && k != nb {
                    return;
                }
                let filled = ((k as f64 / nb.max(1) as f64) * BAR as f64).round() as usize;
                let line = format!(
                    "epoch {:>3}/{} [{}{}] {:>4}/{nb}  box {:.3}  cls {:.3}  dfl {:.3}  {:>6.1} img/s  ETA {}",
                    s.epoch,
                    self.epochs,
                    "#".repeat(filled.min(BAR)),
                    "-".repeat(BAR - filled.min(BAR)),
                    k,
                    s.loss.box_,
                    s.loss.cls,
                    s.loss.dfl,
                    s.imgs_per_s,
                    duration(s.eta_s)
                );
                let mut stdout = std::io::stdout().lock();
                if self.tty {
                    let _ = write!(stdout, "\r\x1b[2K{line}");
                    self.live = true;
                } else {
                    let _ = writeln!(stdout, "{line}");
                }
                let _ = stdout.flush();
            }
            RunEvent::Epoch(ep) => {
                self.end_line();
                if !self.header {
                    println!(
                        "{:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>9} {:>7}",
                        "epoch", "box", "cls", "dfl", "P", "R", "mAP50", "mAP50-95", "time"
                    );
                    self.header = true;
                }
                let l = &ep.train_loss;
                let v = &ep.val;
                println!(
                    "{:>7} {:>7.3} {:>7.3} {:>7.3} {:>7.3} {:>7.3} {:>7.3} {:>9.3} {:>6.1}s",
                    format!("{}/{}", ep.epoch, self.epochs),
                    l.box_,
                    l.cls,
                    l.dfl,
                    v.precision,
                    v.recall,
                    v.map50,
                    v.map50_95,
                    ep.duration_s
                );
            }
            RunEvent::Log(l) => {
                self.end_line();
                let tag = match l.level {
                    folo_api::LogLevel::Warn => "warning: ",
                    folo_api::LogLevel::Error => "error: ",
                    _ => "",
                };
                println!("{tag}{}", l.message);
            }
            // Queue and cloud events come from a server's runs, never from this trainer.
            RunEvent::Status(_) | RunEvent::Queue(_) | RunEvent::Cloud(_) => {}
        }
    }

    fn end_line(&mut self) {
        if self.live {
            println!();
            self.live = false;
        }
    }
}

fn duration(secs: f64) -> String {
    let s = secs.max(0.0).round() as u64;
    if s >= 3600 {
        format!("{}h{:02}m", s / 3600, (s / 60) % 60)
    } else {
        format!("{}m{:02}s", s / 60, s % 60)
    }
}

pub(super) fn finish(
    s: &TrainSummary,
    run_dir: &Path,
    cfg: &TrainConfig,
    resolved: &folo_api::ResolvedTrain,
    out: Out,
) {
    let path = |p: &Option<PathBuf>| p.as_ref().map(folo_api::json::path);
    out.record(&json!({
        "event": "done",
        "status": s.status,
        "run_dir": folo_api::json::path(run_dir),
        "epochs_completed": s.epochs_completed,
        "epochs": cfg.epochs,
        "early_stopped": s.early_stopped,
        "best": s.best.map(|(epoch, fitness)| json!({ "epoch": epoch, "map50_95": fitness })),
        "best_metrics": s.best_metrics,
        "weights": { "best": path(&s.best_weights), "last": path(&s.last) },
        "resolved": resolved,
    }));
    let verb = match s.status {
        RunStatus::Stopped => "Stopped",
        _ if s.early_stopped => "Stopped early",
        _ => "Finished",
    };
    out.human(format!("{verb} after {} of {} epochs.", s.epochs_completed, cfg.epochs));
    if let (Some((epoch, _)), Some(m)) = (s.best, &s.best_metrics) {
        out.human(format!(
            "Best epoch {epoch}: mAP50 {:.3}  mAP50-95 {:.3}  P {:.3}  R {:.3}{}",
            m.map50,
            m.map50_95,
            m.precision,
            m.recall,
            m.best_f1_conf.map_or(String::new(), |c| format!(" (at conf {c:.3})"))
        ));
        if !m.per_class.is_empty() {
            out.human(format!("  {:<16} {:>7} {:>9} {:>7} {:>7}", "class", "mAP50", "mAP50-95", "P", "R"));
            let names = read_classes(run_dir);
            for c in &m.per_class {
                let name = names.get(c.class as usize).cloned().unwrap_or_else(|| c.class.to_string());
                out.human(format!(
                    "  {:<16} {:>7.3} {:>9.3} {:>7.3} {:>7.3}",
                    name, c.map50, c.map50_95, c.precision, c.recall
                ));
            }
        }
    }
    if let Some(b) = &s.best_weights {
        out.human(format!("Weights: {} (and last.folo)", folo_api::json::path(b)));
    }
}

fn read_classes(run_dir: &Path) -> Vec<String> {
    std::fs::read_to_string(run_dir.join("config.json"))
        .ok()
        .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
        .and_then(|v| serde_json::from_value(v["classes"].clone()).ok())
        .unwrap_or_default()
}
