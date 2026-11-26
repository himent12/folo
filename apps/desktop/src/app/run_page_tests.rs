//! A run's page after the run: its whole history once it ends, and a cloud run's phases and device
//! as they happened.

use std::sync::Arc;

use super::tests::{drive, open, open_with};
use crate::api::mock::MockBackend;
use crate::api::{CloudPhase, RunStatus};
use crate::cloud::{PhaseState, panel_cost, panel_warning, phase_states};
use crate::store::Load;

#[test]
fn a_run_watched_live_shows_its_whole_history_once_it_ends() {
    let (mut app, window) = open();
    let id = drive(&mut app, window, |this, _, cx| {
        let id = this.store.runs.ready().unwrap().iter().find(|r| !r.status.is_active()).unwrap().id.clone();
        this.ensure_run_data(&id, cx);
        id
    });
    let all = drive(&mut app, window, |this, _, _| this.store.api.metrics(&id).unwrap().steps.len());
    // What the stream gave a page watching it live: every fifth step (the server sends ~10 a second).
    drive(&mut app, window, |this, _, cx| {
        let Some(Load::Ready(d)) = this.store.run_data.get_mut(&id) else { panic!("loaded") };
        let thinned: Vec<_> = d.metrics.steps.iter().step_by(5).cloned().collect();
        d.metrics.steps = thinned;
        this.run_ended(&id, RunStatus::Completed, cx);
    });
    drive(&mut app, window, |this, _, _| {
        let Some(Load::Ready(d)) = this.store.run_data.get(&id) else { panic!("still shown") };
        assert_eq!(d.metrics.steps.len(), all, "the file's every step, not the stream's");
    });
}

#[test]
fn a_cloud_run_that_failed_uploading_did_not_train() {
    let (mut app, window) = open_with(Arc::new(MockBackend::with_cloud_runs(4200, true)));
    drive(&mut app, window, |this, _, _| {
        use PhaseState::*;
        let mut r = this.store.runs.ready().unwrap().iter().find(|r| r.cloud.is_some()).unwrap().clone();
        let mut c = r.cloud.clone().unwrap();
        let marks = |c: &crate::api::RunCloud, r: &crate::api::Run| -> Vec<PhaseState> {
            phase_states(c, r, false).into_iter().map(|(_, s)| s).collect()
        };
        // Released after the upload failed: syncing failed, training and collecting never reached.
        c.phase = CloudPhase::Released;
        c.ended_in = Some(CloudPhase::Syncing);
        (r.status, r.error_code) = (RunStatus::Failed, Some("cloud_sync_failed".into()));
        assert_eq!(marks(&c, &r), [Done, Done, Failed, Skipped, Skipped, Done, Done]);
        // The copy back failed: collecting is where.
        c.ended_in = Some(CloudPhase::Collecting);
        assert_eq!(marks(&c, &r), [Done, Done, Done, Done, Failed, Done, Done]);
        // The instance went away while its weights were copied: the collection failed, not training.
        r.error_code = Some("cloud_instance_lost".into());
        assert_eq!(marks(&c, &r), [Done, Done, Done, Done, Failed, Done, Done]);
        // Training failed on the worker, its weights were collected: training failed.
        r.error_code = Some("oom".into());
        assert_eq!(marks(&c, &r), [Done, Done, Done, Failed, Done, Done, Done]);
        // Stopped while it booted: booting skipped, the rest never reached.
        c.ended_in = Some(CloudPhase::Booting);
        (r.status, r.error_code) = (RunStatus::Stopped, None);
        assert_eq!(marks(&c, &r), [Done, Skipped, Skipped, Skipped, Skipped, Done, Done]);
        // A server that doesn't say where it ended, or a completed run: every phase done.
        c.ended_in = None;
        assert!(marks(&c, &r).iter().all(|s| *s == Done));
        // Live: done up to the current one.
        c.phase = CloudPhase::Training;
        assert_eq!(marks(&c, &r), [Done, Done, Done, Now, Next, Next, Next]);
        // Its device until the worker reports one: the GPU once, not "<gpu> at <provider>".
        let provider = this.provider_name(&c.provider);
        r.device = format!("{} at {provider}", c.offer.gpu);
        let said = this.cloud_device(&r, &c);
        assert_eq!(said, format!("{provider} · {}", crate::ui::device_label(&c.offer.gpu)));
        assert!(!said.contains(" at "), "{said}");
    });
}

#[test]
fn a_history_asked_for_before_the_end_is_replaced_by_the_final_one() {
    // The page opened just before the run ended: its history request was answered then (held
    // here), and lands after the end.
    let api = Arc::new(MockBackend::new(4200, true));
    let (mut app, window) = open_with(api.clone());
    let held = api.hold("metrics run_52e8");
    app.update(|cx| window.update(cx, |this, _, cx| this.ensure_run_data("run_52e8", cx)).unwrap());
    app.run_until_held(1);
    held.wait_held();
    let key = ("run_52e8".to_string(), "metrics");
    let (asked, final_read) = app.update(|cx| {
        window
            .update(cx, |this, _, cx| {
                let asked = this.history_tickets.get(&key).copied();
                this.run_ended("run_52e8", RunStatus::Completed, cx);
                (asked, this.history_tickets.get(&key).copied())
            })
            .unwrap()
    });
    assert_ne!(
        asked, final_read,
        "the end asks for the history again, and the earlier answer no longer counts"
    );
    drop(held);
    app.run_until_idle();
    drive(&mut app, window, |this, _, _| {
        // The final read installed it (the earlier answer's ticket no longer counts).
        let Some(Load::Ready(_)) = this.store.run_data.get("run_52e8") else { panic!("loaded") };
    });
}

#[test]
fn a_final_read_that_fails_does_not_leave_the_history_loading() {
    let api = Arc::new(MockBackend::new(4200, true));
    let (mut app, window) = open_with(api.clone());
    let held = api.hold("metrics run_52e8");
    app.update(|cx| window.update(cx, |this, _, cx| this.ensure_run_data("run_52e8", cx)).unwrap());
    app.run_until_held(1);
    held.wait_held();
    // The final read fails (the server went away for a moment).
    api.fail_metrics_once();
    app.update(|cx| {
        window.update(cx, |this, _, cx| this.run_ended("run_52e8", RunStatus::Completed, cx)).unwrap()
    });
    drop(held);
    app.run_until_idle();
    drive(&mut app, window, |this, _, _| {
        assert!(
            matches!(this.store.run_data.get("run_52e8"), Some(Load::Failed(_))),
            "a failure, not loading for good"
        );
    });
}

#[test]
fn a_history_that_failed_to_load_draws_its_failure_in_per_class_with_the_retry() {
    let (mut app, window) = open();
    drive(&mut app, window, |this, w, cx| this.navigate(crate::app::Route::Run("run_52e8".into()), w, cx));
    app.frame(window.into());
    app.run_until_idle();
    drive(&mut app, window, |this, _, _| {
        assert!(!this.focus.borrow().contains_key("per-class-retry"), "loaded: the classes, no retry");
        let down = crate::api::ApiError::new("server_unreachable", "Could not reach folo serve");
        this.store.run_data.insert("run_52e8".into(), Load::Failed(down));
    });
    app.frame(window.into());
    drive(&mut app, window, |this, _, _| {
        let drawn = this.focus.borrow();
        assert!(drawn.contains_key("run-data-retry"), "the charts say it");
        assert!(drawn.contains_key("per-class-retry"), "and Per class, not skeletons for good");
    });
}

#[test]
fn a_released_runs_instance_not_confirmed_terminated_says_it_may_still_be_billed() {
    use PhaseState::*;
    // RunPod hasn't confirmed the termination of the training run's instance (the demo's +trouble).
    let (mut app, window) = open_with(Arc::new(MockBackend::with_cloud_runs(4200, true).cloud_trouble()));
    drive(&mut app, window, |this, _, cx| this.ensure_cloud(cx));
    drive(&mut app, window, |this, _, _| {
        let r = this.store.runs.ready().unwrap().iter().find(|r| r.id == "run_c2e1").unwrap().clone();
        let c = r.cloud.clone().unwrap();
        assert_eq!(c.phase, CloudPhase::Released);
        let i = this.released_instance(&c).expect("its instance in the ledger").clone();
        assert!(i.status.is_alive(), "not confirmed terminated");
        // Releasing is still under way, released still to come.
        let marks: Vec<_> = phase_states(&c, &r, true).into_iter().map(|(_, s)| s).collect();
        assert_eq!(&marks[5..], [Now, Next]);
        // The words promise no more than the app knows, and lead to the instance.
        let (_, words, to_settings) = panel_warning(&c, true, false, 0).unwrap();
        assert_eq!(words, "Termination not confirmed: the instance may still be billed.");
        assert!(to_settings);
        // Its cost is the instance's own, still counting.
        assert_eq!(
            panel_cost(&c, Some(&i), false),
            format!("{} so far", crate::ui::fmt_cost(i.cost_usd.unwrap()))
        );
        // Confirmed: final, no warning, every phase done.
        let mut gone = i.clone();
        gone.status = crate::api::CloudInstanceStatus::Terminated;
        assert_eq!(panel_cost(&c, Some(&gone), false), crate::ui::fmt_cost(i.cost_usd.unwrap()));
        assert!(panel_warning(&c, false, false, 0).is_none());
        // The ledger couldn't be read: that is said, not "terminated".
        let (_, words, _) = panel_warning(&c, false, true, 0).unwrap();
        assert_eq!(words, "Could not check that the instance is terminated.");
        // …and its cost stays provisional: the run's own, "so far", not a final figure.
        let usd = crate::ui::fmt_cost(c.cost_usd.unwrap());
        assert_eq!(panel_cost(&c, None, true), format!("{usd} so far"));
        assert_eq!(panel_cost(&c, None, false), usd, "a row the ledger no longer keeps: final");
        // A machine of yours (no price) is not billed either way.
        let mut own = c.clone();
        own.offer.price_per_hour = None;
        assert_eq!(panel_cost(&own, None, true), "not billed");
    });
}

#[test]
fn a_cloud_run_whose_worker_is_silent_says_so_with_the_time() {
    let (mut app, window) = open_with(Arc::new(MockBackend::with_cloud_runs(4200, true).cloud_silent()));
    drive(&mut app, window, |this, _, _| {
        let r = this.store.runs.ready().unwrap().iter().find(|r| r.id == "run_c2e1").unwrap().clone();
        let c = r.cloud.clone().unwrap();
        assert_eq!(c.phase, CloudPhase::Training);
        let since = c.unanswered_since.as_deref().and_then(crate::api::time::parse).expect("silent");
        let (mark, words, to_settings) = panel_warning(&c, false, false, since).unwrap();
        assert_eq!(mark, "icons/wifi-off.svg");
        assert!(!to_settings);
        let at = crate::api::time::local_deadline(since, since);
        assert_eq!(
            words,
            format!(
                "The worker has not answered since {at}. FOLO keeps asking; the run may still be training or waiting there."
            )
        );
        // A silent worker comes first: it is what the run is waiting on.
        assert_eq!(panel_warning(&c, true, true, since).unwrap().0, "icons/wifi-off.svg");
    });
}

#[test]
fn a_cloud_run_the_instance_list_had_not_got_is_read_again_until_a_read_covers_it() {
    let api = Arc::new(MockBackend::with_cloud_runs(4200, true).cloud_trouble());
    let (mut app, window) = open_with(api.clone());
    let cloud = |this: &crate::app::FoloApp| {
        this.store.runs.ready().unwrap().iter().find(|r| r.id == "run_c2e1").unwrap().cloud.clone().unwrap()
    };
    // The list as read before these runs existed (they started elsewhere afterwards): no live
    // instance, so the 5 s refresh that follows live ones never fired for them.
    drive(&mut app, window, |this, _, _| {
        this.cloud.instances = Load::Ready(Vec::new());
        assert!(!this.instance_alive());
        assert!(this.instances_missing(), "the runs' instances are to be read");
        // Meanwhile nothing confirms the release: its cost and Releasing stay provisional, without
        // an alarm while no read has failed.
        assert!(this.released_instance_unverified(&cloud(this)));
        assert!(!this.released_instance_unknown(&cloud(this)));
    });
    // That read fails: still missing, still provisional, read again at the next tick.
    api.fail_instances_once();
    drive(&mut app, window, |this, _, cx| this.refresh_instances(cx));
    drive(&mut app, window, |this, _, _| {
        assert!(this.instances_missing(), "a failed read covers nothing");
        assert!(this.released_instance_unknown(&cloud(this)), "and it is said");
    });
    // The next one answers: the instance is there, alive, its termination unconfirmed.
    drive(&mut app, window, |this, _, cx| this.refresh_instances(cx));
    drive(&mut app, window, |this, _, _| {
        let i = this.released_instance(&cloud(this)).expect("found after a read that worked");
        assert!(i.status.is_alive());
        assert!(!this.released_instance_unverified(&cloud(this)) && !this.instances_missing());
        // An id a successful read covered but the ledger no longer keeps (ended over 7 days ago)
        // is final, not read again forever.
        this.cloud.instances = Load::Ready(Vec::new());
        assert!(!this.instances_missing());
        assert!(!this.released_instance_unverified(&cloud(this)));
    });
}

#[test]
fn an_instance_list_answer_older_than_the_one_shown_is_dropped() {
    let api = Arc::new(MockBackend::with_cloud_runs(4200, true).cloud_trouble());
    let (mut app, window) = open_with(api.clone());
    // A first read is held out; a second one answers and is shown.
    let held = api.hold("instances");
    app.update(|cx| window.update(cx, |this, _, cx| this.refresh_instances(cx)).unwrap());
    app.run_until_held(1);
    held.wait_held();
    app.update(|cx| window.update(cx, |this, _, cx| this.refresh_instances(cx)).unwrap());
    app.run_until_held(1);
    // What the newer answer showed, told apart from the older one's: emptied here.
    app.update(|cx| {
        window
            .update(cx, |this, _, _| {
                assert!(this.cloud.instances.ready().is_some_and(|l| !l.is_empty()));
                this.cloud.instances = Load::Ready(Vec::new());
            })
            .unwrap()
    });
    // The older answer lands last: dropped, not put back over the newer list.
    drop(held);
    app.run_until_idle();
    drive(&mut app, window, |this, _, _| {
        assert!(this.cloud.instances.ready().is_some_and(|l| l.is_empty()), "the older answer was installed");
    });
}

#[test]
fn a_released_run_reads_provisional_while_the_instance_list_loads() {
    let (mut app, window) = open_with(Arc::new(MockBackend::with_cloud_runs(4200, true).cloud_trouble()));
    drive(&mut app, window, |this, _, _| {
        this.cloud.instances = Load::Loading;
        let c = this
            .store
            .runs
            .ready()
            .unwrap()
            .iter()
            .find(|r| r.id == "run_c2e1")
            .unwrap()
            .cloud
            .clone()
            .unwrap();
        assert!(this.released_instance_unverified(&c), "nothing confirms the release yet");
        assert!(!this.released_instance_unknown(&c), "but no read failed: no alarm");
        let usd = crate::ui::fmt_cost(c.cost_usd.unwrap());
        assert_eq!(panel_cost(&c, None, true), format!("{usd} so far"));
    });
}

#[test]
fn an_earlier_epochs_samples_landing_after_leaving_the_page_are_dropped() {
    // A past epoch was picked on the run page, and the page was left before its samples came.
    let api = Arc::new(MockBackend::new(4200, true));
    let (mut app, window) = open_with(api.clone());
    let held = api.hold("samples run_52e8");
    app.update(|cx| {
        window
            .update(cx, |this, _, cx| {
                this.sample_epoch = Some(5);
                this.ensure_samples("run_52e8", cx);
            })
            .unwrap()
    });
    app.run_until_held(1);
    held.wait_held();
    app.update(|cx| window.update(cx, |this, w, cx| this.navigate(crate::app::Route::Runs, w, cx)).unwrap());
    drop(held);
    app.run_until_idle();
    drive(&mut app, window, |this, _, _| {
        assert_eq!(this.sample_epoch, None, "the next run page starts on the latest epoch");
        assert!(
            !this.store.samples.contains_key("run_52e8"),
            "epoch 5's late answer must not stand in for the latest"
        );
    });
}

#[test]
fn validation_samples_start_at_every_prediction_the_server_keeps() {
    // x1-parity-8: Studio starts the Min confidence slider at 0.25 (the server keeps predictions >= 0.25);
    // Desktop started at 0.3 and drew fewer boxes on the same sample.
    let (mut app, window) = open();
    drive(&mut app, window, |this, _, _| {
        assert_eq!(this.sample_conf, crate::screens::runs::SAMPLE_CONF.0);
        assert_eq!(this.sample_conf, 0.25);
    });
}

#[test]
fn log_search_filters_and_copy_log_copies_only_matching_lines() {
    use crate::api::{LogLevel, LogLine, RunLogs};
    use crate::app::{LogFilter, Route};

    let (mut app, window) = open();
    let id = "run_52e8";
    drive(&mut app, window, |this, w, cx| this.navigate(Route::Run(id.into()), w, cx));
    app.frame(window.into());
    app.run_until_idle();

    let sample_lines = vec![
        LogLine { t: "2026-11-10T10:00:00Z".into(), level: LogLevel::Info, message: "loaded dataset".into() },
        LogLine {
            t: "2026-11-10T10:00:01Z".into(),
            level: LogLevel::Warn,
            message: "slow dataloader".into(),
        },
        LogLine {
            t: "2026-11-10T10:00:02Z".into(),
            level: LogLevel::Error,
            message: "CUDA out of memory".into(),
        },
        LogLine {
            t: "2026-11-10T10:00:03Z".into(),
            level: LogLevel::Info,
            message: "memory check ok".into(),
        },
    ];
    drive(&mut app, window, |this, _, cx| {
        this.store.install_logs(id, Ok(RunLogs { lines: sample_lines.clone(), total: 4 }));
        cx.notify();
    });
    app.frame(window.into());

    // Unfiltered copy copies all 4 lines.
    drive(&mut app, window, |this, _, cx| {
        this.copy_log(id, cx);
        let clip = cx.read_from_clipboard().and_then(|c| c.text()).unwrap_or_default();
        assert_eq!(clip.lines().count(), 4);
        assert!(clip.contains("loaded dataset"));
        assert!(clip.contains("CUDA out of memory"));
        assert_eq!(this.toasts.last().unwrap().title.as_ref(), "Copied 4 log lines");
    });

    // Search "memory" matches 2 lines (Error + Info); copy copies only those 2.
    drive(&mut app, window, |this, _, cx| {
        this.log_search.update(cx, |i, cx| i.set_text("MEMORY", cx));
        this.copy_log(id, cx);
        let clip = cx.read_from_clipboard().and_then(|c| c.text()).unwrap_or_default();
        assert_eq!(clip.lines().count(), 2);
        assert!(clip.contains("CUDA out of memory"));
        assert!(clip.contains("memory check ok"));
        assert!(!clip.contains("loaded dataset"));
        assert_eq!(this.toasts.last().unwrap().title.as_ref(), "Copied 2 matching log lines");
    });

    // Combine Error level + search "memory" -> 1 matching line.
    drive(&mut app, window, |this, _, cx| {
        this.log_filter = LogFilter::Error;
        this.copy_log(id, cx);
        let clip = cx.read_from_clipboard().and_then(|c| c.text()).unwrap_or_default();
        assert_eq!(clip.lines().count(), 1);
        assert!(clip.contains("ERR CUDA out of memory"));
        assert_eq!(this.toasts.last().unwrap().title.as_ref(), "Copied 1 matching log line");
    });

    // Unmatched search shows Clear filters control and disables copy.
    drive(&mut app, window, |this, _, cx| {
        this.log_search.update(cx, |i, cx| i.set_text("no-such-token", cx));
        this.focus.borrow_mut().clear();
        cx.notify();
    });
    app.frame(window.into());
    drive(&mut app, window, |this, _, _| {
        let drawn = this.focus.borrow();
        assert!(drawn.contains_key("log-clear-filters"), "filtered-empty log offers Clear filters");
        assert!(!drawn.contains_key("log-copy"), "copy button is disabled when 0 lines match");
    });

    // Navigating away clears both log_filter and log_search.
    drive(&mut app, window, |this, w, cx| this.navigate(Route::Runs, w, cx));
    drive(&mut app, window, |this, _, cx| {
        assert_eq!(this.log_filter, LogFilter::All);
        assert_eq!(this.log_search.read(cx).text(), "");
    });
}

#[test]
fn empty_run_log_hides_clear_and_copy_and_reset_clears_log_search_on_backend_switch() {
    use crate::api::RunLogs;
    use crate::app::{LogFilter, Route, ServerMode};

    let (mut app, window) = open();
    let id = "run_52e8";
    drive(&mut app, window, |this, w, cx| this.navigate(Route::Run(id.into()), w, cx));
    app.frame(window.into());
    app.run_until_idle();

    drive(&mut app, window, |this, _, cx| {
        this.store.install_logs(id, Ok(RunLogs { lines: Vec::new(), total: 0 }));
        this.focus.borrow_mut().clear();
        cx.notify();
    });
    app.frame(window.into());
    drive(&mut app, window, |this, _, cx| {
        {
            let drawn = this.focus.borrow();
            assert!(!drawn.contains_key("log-clear-filters"), "0-line log is not a filtered empty state");
            assert!(!drawn.contains_key("log-copy"), "0-line log has no copy button");
        }
        // Set search + filter and verify set_backend resets both.
        this.log_filter = LogFilter::Warn;
        this.log_search.update(cx, |i, cx| i.set_text("spike", cx));
        this.set_backend(Arc::new(MockBackend::new(4200, true)), ServerMode::Mock, cx);
        assert_eq!(this.log_filter, LogFilter::All);
        assert_eq!(this.log_search.read(cx).text(), "");
    });
}

#[test]
fn failed_run_samples_and_logs_offer_retry_and_recover() {
    use crate::api::ApiError;
    use crate::app::Route;
    use crate::store::Load;
    use folo_ui::{KeyDownEvent, Keystroke, PlatformInput};

    let (mut app, window) = open();
    let id = "run_52e8";
    drive(&mut app, window, |this, w, cx| this.navigate(Route::Run(id.into()), w, cx));
    app.frame(window.into());
    app.run_until_idle();

    // Inject transient load failures for both samples and logs while holding new fetches so the
    // failure panels render cleanly.
    drive(&mut app, window, |this, _, cx| {
        this.hold_loading = true;
        this.store
            .samples
            .insert(id.into(), Load::Failed(ApiError::new("server_unreachable", "samples timeout")));
        this.store.logs.insert(id.into(), Load::Failed(ApiError::new("server_unreachable", "logs timeout")));
        this.focus.borrow_mut().clear();
        cx.notify();
    });
    app.frame(window.into());
    drive(&mut app, window, |this, _, _| {
        let drawn = this.focus.borrow();
        assert!(drawn.contains_key("run-samples-retry"), "failed samples card offers Retry button");
        assert!(drawn.contains_key("run-logs-retry"), "failed logs card offers Retry button");
    });

    // Activating Retry on samples reloads them and settles back to Ready.
    drive(&mut app, window, |this, w, cx| {
        this.hold_loading = false;
        this.focus_key("run-samples-retry", w, cx);
    });
    app.frame(window.into());
    app.simulate_input(
        window.into(),
        PlatformInput::KeyDown(KeyDownEvent {
            keystroke: Keystroke::parse("enter").unwrap(),
            is_held: false,
        }),
    );
    app.frame(window.into());
    app.run_until_idle();
    app.frame(window.into());
    drive(&mut app, window, |this, _, _| {
        assert!(
            this.store.samples.get(id).is_some_and(|l| l.ready().is_some()),
            "samples Retry reloaded from backend"
        );
    });

    // Activating Retry on logs reloads them and settles back to Ready.
    drive(&mut app, window, |this, w, cx| this.focus_key("run-logs-retry", w, cx));
    app.frame(window.into());
    app.simulate_input(
        window.into(),
        PlatformInput::KeyDown(KeyDownEvent {
            keystroke: Keystroke::parse("enter").unwrap(),
            is_held: false,
        }),
    );
    app.frame(window.into());
    app.run_until_idle();
    app.frame(window.into());
    drive(&mut app, window, |this, _, _| {
        assert!(
            this.store.logs.get(id).is_some_and(|l| l.ready().is_some()),
            "logs Retry reloaded from backend"
        );
    });
}

#[test]
fn switching_to_another_cached_runs_logs_scrolls_to_bottom() {
    use crate::api::{LogLevel, LogLine, RunLogs};
    use crate::app::Route;
    use folo_ui::px;

    let (mut app, window) = open();
    let (run_a, run_b) = drive(&mut app, window, |this, _, _| {
        let runs = this.store.runs.ready().unwrap();
        (runs[0].id.clone(), runs[1].id.clone())
    });
    let mk_logs = |prefix: &str, n: usize| RunLogs {
        lines: (0..n)
            .map(|i| LogLine {
                t: format!("2026-11-10T10:{:02}:{:02}Z", i / 60, i % 60),
                level: LogLevel::Info,
                message: format!("{prefix} line {i}"),
            })
            .collect(),
        total: n as u64,
    };

    // Pre-populate both runs' logs so switching from A to B does not change `store.log_rev`.
    drive(&mut app, window, |this, w, cx| {
        this.store.install_logs(&run_a, Ok(mk_logs("A", 3)));
        this.store.install_logs(&run_b, Ok(mk_logs("B", 120)));
        this.navigate(Route::Run(run_a.clone()), w, cx);
    });
    for _ in 0..3 {
        app.frame(window.into());
        app.run_until_idle();
    }

    // Switch to already-cached run B (much taller log): must follow and scroll to its bottom.
    drive(&mut app, window, |this, w, cx| {
        this.navigate(Route::Run(run_b.clone()), w, cx);
    });
    for _ in 0..4 {
        app.frame(window.into());
        app.run_until_idle();
    }
    drive(&mut app, window, |this, _, _| {
        let max = this.log_scroll.max_offset().height;
        let y = -this.log_scroll.offset().y;
        assert!(max > px(100.), "run B logs overflow the scroll box: max={max:?}");
        assert!(this.log_follow, "switching runs re-enables log follow");
        assert!(max - y < px(8.), "run B logs scrolled to bottom (y={y:?}, max={max:?})");
    });
}

#[test]
fn validation_samples_wait_for_first_evaluation_and_recover_from_failure_on_epoch() {
    use crate::api::{ApiError, Epoch, Loss, RunEvent, RunStatus, ValMetrics};
    use crate::app::Route;
    use crate::store::Load;

    let (mut app, window) = open();
    // Pick a running run and reset it to mid-epoch 1 before any evaluation has finished.
    let id = drive(&mut app, window, |this, _, _| {
        let r = this
            .store
            .runs
            .ready_mut()
            .unwrap()
            .iter_mut()
            .find(|r| r.status == RunStatus::Running)
            .expect("demo has a running run");
        r.epoch = 1;
        r.best = None;
        let id = r.id.clone();
        this.store.run_data.insert(
            id.clone(),
            Load::Ready(crate::store::RunData::new(
                crate::api::RunMetrics { steps: Vec::new(), epochs: Vec::new() },
                r.steps_per_epoch,
            )),
        );
        this.store.samples.remove(&id);
        this.store.samples_stale.remove(&id);
        id
    });

    // Opening the run mid-epoch 1 must show the waiting state and NOT fire an epoch-0 sample fetch.
    drive(&mut app, window, |this, w, cx| this.navigate(Route::Run(id.clone()), w, cx));
    app.frame(window.into());
    app.run_until_idle();
    app.frame(window.into());
    drive(&mut app, window, |this, _, _| {
        let r = this.store.run(&id).unwrap();
        assert_eq!(this.evaluated_epoch(r), 0, "no epoch evaluated yet");
        assert!(
            !this.store.samples.contains_key(&id),
            "must not request samples before the first evaluation completes"
        );
    });

    // Even if a prior sample fetch failed, an arriving Epoch event clears the failure and loads samples.
    drive(&mut app, window, |this, _, cx| {
        this.store
            .samples
            .insert(id.clone(), Load::Failed(ApiError::new("not_found", "no samples for epoch 0")));
        this.store.apply(
            &id,
            RunEvent::Epoch(Epoch {
                epoch: 1,
                train_loss: Loss { box_: 1.0, cls: 1.0, dfl: 1.0, total: 3.0 },
                val: ValMetrics {
                    map50: 0.5,
                    map50_95: 0.3,
                    precision: 0.6,
                    recall: 0.5,
                    best_f1_conf: Some(0.4),
                    per_class: Vec::new(),
                },
                duration_s: 1.0,
            }),
        );
        cx.notify();
    });
    app.frame(window.into());
    app.run_until_idle();
    app.frame(window.into());
    drive(&mut app, window, |this, _, _| {
        assert!(
            this.store.samples.get(&id).is_some_and(|l| l.ready().is_some()),
            "Epoch event cleared failed samples and automatically loaded latest samples"
        );
    });
}

#[test]
fn live_sample_refresh_keeps_ready_tiles_mounted_without_placeholder_jump() {
    use crate::api::{Epoch, Loss, RunEvent, RunStatus, ValMetrics};
    use crate::app::Route;

    let api = Arc::new(MockBackend::new(4200, true));
    let (mut app, window) = open_with(api.clone());
    let id = drive(&mut app, window, |this, w, cx| {
        let id = this
            .store
            .runs
            .ready()
            .unwrap()
            .iter()
            .find(|r| r.status == RunStatus::Running)
            .expect("demo has a running run")
            .id
            .clone();
        this.navigate(Route::Run(id.clone()), w, cx);
        id
    });
    for _ in 0..3 {
        app.frame(window.into());
        app.run_until_idle();
    }
    drive(&mut app, window, |this, _, _| {
        assert!(this.store.samples.get(&id).is_some_and(|l| l.ready().is_some()), "initial samples loaded");
    });

    // Hold the next samples fetch and deliver a new live Epoch event.
    let hold_key: &'static str = Box::leak(format!("samples {id}").into_boxed_str());
    let held = api.hold(hold_key);
    app.update(|cx| {
        window
            .update(cx, |this, _, cx| {
                this.store.apply(
                    &id,
                    RunEvent::Epoch(Epoch {
                        epoch: 99,
                        train_loss: Loss { box_: 0.5, cls: 0.5, dfl: 0.5, total: 1.5 },
                        val: ValMetrics {
                            map50: 0.8,
                            map50_95: 0.55,
                            precision: 0.82,
                            recall: 0.78,
                            best_f1_conf: Some(0.45),
                            per_class: Vec::new(),
                        },
                        duration_s: 1.0,
                    }),
                );
                cx.notify();
            })
            .unwrap();
    });
    app.frame(window.into());
    app.run_until_held(1);
    held.wait_held();
    app.frame(window.into());

    // While the refresh is in flight, existing Ready samples must remain mounted (never Load::Loading).
    app.update(|cx| {
        window
            .update(cx, |this, _, _| {
                assert!(
                    this.store.samples.get(&id).is_some_and(|l| l.ready().is_some()),
                    "live refresh keeps ready tiles visible instead of replacing with 8 skeletons"
                );
                assert!(this.focus.borrow().contains_key("smp-0"), "sample tiles stay mounted");
            })
            .unwrap();
    });

    drop(held);
    app.run_until_idle();
    app.frame(window.into());
    drive(&mut app, window, |this, _, _| {
        assert!(this.store.samples.get(&id).is_some_and(|l| l.ready().is_some()));
        assert!(!this.store.samples_stale.contains(&id));
    });
}

#[test]
fn evaluated_epoch_fallbacks_in_flight_failure_and_refresh_error_retry() {
    use crate::api::{ApiError, BestEpoch, Epoch, Loss, RunEvent, RunStatus, StatusEvent, ValMetrics};
    use crate::app::Route;
    use crate::store::Load;

    let (mut app, window) = open();
    // 1. Verify `evaluated_epoch` across `run_data` Ready and absent/Loading fallback states.
    drive(&mut app, window, |this, _, _| {
        let mut r = this.store.runs.ready().unwrap()[0].clone();
        this.store.run_data.remove(&r.id);
        this.store.samples.remove(&r.id);

        // Loaded `run_data` takes precedence even if `r.epoch` is still 0 in a stale run snapshot.
        r.status = RunStatus::Running;
        r.epoch = 0;
        this.store.run_data.insert(
            r.id.clone(),
            Load::Ready(crate::store::RunData::new(
                crate::api::RunMetrics {
                    steps: Vec::new(),
                    epochs: vec![Epoch {
                        epoch: 1,
                        train_loss: Loss { box_: 1.0, cls: 1.0, dfl: 1.0, total: 3.0 },
                        val: ValMetrics {
                            map50: 0.5,
                            map50_95: 0.3,
                            precision: 0.6,
                            recall: 0.5,
                            best_f1_conf: Some(0.4),
                            per_class: Vec::new(),
                        },
                        duration_s: 1.0,
                    }],
                },
                10,
            )),
        );
        assert_eq!(this.evaluated_epoch(&r), 1);
        this.store.run_data.remove(&r.id);

        // Checkpoint-seeded queued/active run at epoch 0 with inherited `best`: must be 0.
        r.status = RunStatus::Queued;
        r.epoch = 0;
        r.best = Some(BestEpoch { epoch: 5, map50_95: 0.4 });
        assert_eq!(this.evaluated_epoch(&r), 0);

        // Failed mid-epoch 1 without `run_data`: must be 0 (`r.epoch` is the epoch it failed in).
        r.status = RunStatus::Failed;
        r.epoch = 1;
        r.best = None;
        assert_eq!(this.evaluated_epoch(&r), 0);

        // Stopped run without `run_data`: `r.epoch` is the last finished epoch (0 if stopped in epoch 1).
        r.status = RunStatus::Stopped;
        r.epoch = 0;
        assert_eq!(this.evaluated_epoch(&r), 0);
        r.epoch = 3;
        assert_eq!(this.evaluated_epoch(&r), 3);

        // Completed run while `run_data` is loading: uses `r.epoch` (not lower `r.best.epoch`).
        r.status = RunStatus::Completed;
        r.epoch = 10;
        r.best = Some(BestEpoch { epoch: 1, map50_95: 0.4 });
        assert_eq!(this.evaluated_epoch(&r), 10);

        // Active run on epoch 10 while `run_data` reloads on stream recovery: keeps epoch >= 9.
        r.status = RunStatus::Running;
        r.epoch = 10;
        r.best = Some(BestEpoch { epoch: 1, map50_95: 0.4 });
        assert_eq!(this.evaluated_epoch(&r), 9);
    });

    // 2. If an in-flight `Load::Loading` sample fetch fails AFTER a newer `RunEvent::Epoch`
    //    marked it stale, `ensure_samples` consumes `samples_stale` and recovers automatically.
    let id = drive(&mut app, window, |this, w, cx| {
        let id = this
            .store
            .runs
            .ready()
            .unwrap()
            .iter()
            .find(|r| r.status == RunStatus::Running)
            .unwrap()
            .id
            .clone();
        this.navigate(Route::Run(id.clone()), w, cx);
        id
    });
    app.frame(window.into());
    app.run_until_idle();
    drive(&mut app, window, |this, _, cx| {
        this.store.runs.ready_mut().unwrap().iter_mut().find(|r| r.id == id).unwrap().epoch = 0;
        this.store.samples.insert(id.clone(), Load::Loading);
        this.store.apply(
            &id,
            RunEvent::Epoch(Epoch {
                epoch: 5,
                train_loss: Loss { box_: 0.5, cls: 0.5, dfl: 0.5, total: 1.5 },
                val: ValMetrics {
                    map50: 0.7,
                    map50_95: 0.45,
                    precision: 0.7,
                    recall: 0.7,
                    best_f1_conf: Some(0.4),
                    per_class: Vec::new(),
                },
                duration_s: 1.0,
            }),
        );
        assert_eq!(this.store.run(&id).unwrap().epoch, 5, "Epoch event advances Run.epoch");
        assert!(this.store.samples_stale.contains(&id));
        // Simulate the older in-flight request failing after Epoch arrived.
        this.store.samples.insert(id.clone(), Load::Failed(ApiError::new("not_found", "stale epoch failed")));
        cx.notify();
    });
    app.frame(window.into());
    app.run_until_idle();
    app.frame(window.into());
    drive(&mut app, window, |this, _, _| {
        assert!(
            this.store.samples.get(&id).is_some_and(|l| l.ready().is_some()),
            "stale failed samples automatically recovered after newer Epoch"
        );
    });

    // 3. Terminal Status arriving while a background Latest refresh is in flight queues a retry.
    drive(&mut app, window, |this, _, cx| {
        this.store.samples_refreshing.insert(id.clone());
        this.store.apply(&id, RunEvent::Status(StatusEvent::new(RunStatus::Completed, None)));
        assert!(
            this.store.samples_stale.contains(&id),
            "terminal status during in-flight refresh marks samples stale for auto-retry"
        );
        cx.notify();
    });
    app.frame(window.into());
    app.run_until_idle();
    app.frame(window.into());
    drive(&mut app, window, |this, _, _| {
        assert!(!this.store.samples_stale.contains(&id));
        assert!(!this.store.samples_refreshing.contains(&id));
        assert!(!this.store.samples_refresh_error.contains_key(&id));
        assert!(this.store.samples.get(&id).is_some_and(|l| l.ready().is_some()));
    });

    // 4. A failed background Latest refresh keeps Ready tiles mounted AND surfaces Retry.
    drive(&mut app, window, |this, _, cx| {
        this.store
            .samples_refresh_error
            .insert(id.clone(), ApiError::new("server_unreachable", "refresh timed out"));
        this.focus.borrow_mut().clear();
        cx.notify();
    });
    app.frame(window.into());
    drive(&mut app, window, |this, _, cx| {
        let drawn = this.focus.borrow();
        assert!(drawn.contains_key("smp-0"), "existing tiles stay mounted during refresh error");
        assert!(drawn.contains_key("run-samples-retry"), "caption surfaces inline Retry button");
        drop(drawn);
        this.retry_samples(&id, cx);
    });
    app.run_until_idle();
    app.frame(window.into());
    drive(&mut app, window, |this, _, _| {
        assert!(!this.store.samples_refresh_error.contains_key(&id), "Retry cleared refresh error");
        assert!(this.store.samples.get(&id).is_some_and(|l| l.ready().is_some()));
    });
}
