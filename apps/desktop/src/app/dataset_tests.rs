//! Datasets as a person meets them: a large one's page, one with no usable image, generating one in
//! the background.

use std::sync::Arc;

use super::ServerMode;
use super::data::IMAGE_PAGE;
use super::tests::{drive, open_with};
use crate::api::mock::MockBackend;
use crate::api::{Split, SplitsExt};
use crate::ui::Tone;

#[test]
fn a_dataset_page_holds_every_image_past_a_hundred_thousand() {
    // COCO-sized: the demo's shapes-1k with 100,000 more val images after its own.
    let api = Arc::new(MockBackend::new(4200, true).with_more_val_images("ds_8f2c", 100_000));
    let (mut app, window) = open_with(api);
    drive(&mut app, window, |this, _, cx| this.ensure_dataset("ds_8f2c", cx));
    drive(&mut app, window, |this, _, _| {
        let ds = this.store.dataset.get("ds_8f2c").and_then(|l| l.ready()).cloned().expect("loaded");
        let images = this.store.images.get("ds_8f2c").and_then(|l| l.ready()).expect("listed");
        assert_eq!(images.len() as u64, ds.splits.total(), "the grid lists every image");
        this.ds_split = Some(Split::Val);
        assert_eq!(this.filtered_indices("ds_8f2c").len() as u64, ds.splits.val, "the last ones too");
    });
}

#[test]
fn a_large_list_shows_its_first_page_while_the_rest_arrives() {
    let api = Arc::new(MockBackend::new(4200, true).with_more_val_images("ds_8f2c", 25_000));
    let (mut app, window) = open_with(api.clone());
    let len = |this: &super::FoloApp| {
        this.store.images.get("ds_8f2c").and_then(|l| l.ready()).map_or(0, |v| v.len())
    };
    // The first page lands; the second is held at the server.
    let (first, second) = (api.hold("images"), api.hold("images"));
    app.update(|cx| window.update(cx, |this, _, cx| this.ensure_dataset("ds_8f2c", cx)).unwrap());
    app.run_until_held(1);
    drop(first);
    // Until the first page is applied and the second asked for (bounded by the condition, not
    // a time: a call that never comes fails `run_until_held`).
    while !second.is_held() {
        app.run_until_held(1);
    }
    // (Plain updates while it is held: `drive` waits for every call.)
    app.update(|cx| {
        window
            .update(cx, |this, _, cx| {
                assert_eq!(len(this), IMAGE_PAGE as usize, "the grid has the first page");
                assert!(this.images_pending("ds_8f2c"), "and knows more is coming");
                // The list reloaded meanwhile (a change elsewhere): the old load's pages are
                // dropped, never appended twice.
                this.store.images.remove("ds_8f2c");
                this.ensure_dataset("ds_8f2c", cx);
            })
            .unwrap()
    });
    drop(second);
    app.run_until_idle();
    drive(&mut app, window, |this, _, _| {
        let total = this.store.dataset.get("ds_8f2c").and_then(|l| l.ready()).map(|d| d.splits.total());
        assert_eq!(Some(len(this) as u64), total);
        assert!(!this.images_pending("ds_8f2c"));
    });
}

#[test]
fn a_page_that_fails_after_the_first_is_asked_for_again() {
    let api = Arc::new(MockBackend::new(4200, true).with_more_val_images("ds_8f2c", 25_000));
    let (mut app, window) = open_with(api.clone());
    api.fail_images_once_from(IMAGE_PAGE);
    drive(&mut app, window, |this, _, cx| this.ensure_dataset("ds_8f2c", cx));
    drive(&mut app, window, |this, _, _| {
        let got = this.store.images.get("ds_8f2c").and_then(|l| l.ready()).map(|v| v.len());
        assert_eq!(got, Some(IMAGE_PAGE as usize));
        assert!(this.images_pending("ds_8f2c"), "still loading, not passed off as the whole list");
    });
    app.advance_clock(std::time::Duration::from_secs(5));
    drive(&mut app, window, |this, _, _| {
        let total = this.store.dataset.get("ds_8f2c").and_then(|l| l.ready()).map(|d| d.splits.total());
        let got = this.store.images.get("ds_8f2c").and_then(|l| l.ready()).map(|v| v.len() as u64);
        assert_eq!(got, total);
        assert!(!this.images_pending("ds_8f2c"));
    });
}

#[test]
fn a_retry_ends_with_the_list_it_was_for() {
    let api = Arc::new(MockBackend::new(4200, true).with_more_val_images("ds_8f2c", 25_000));
    let (mut app, window) = open_with(api.clone());
    api.fail_images_once_from(IMAGE_PAGE);
    drive(&mut app, window, |this, _, cx| this.ensure_dataset("ds_8f2c", cx));
    // The dataset removed (here or elsewhere) while its page waits to be asked again.
    drive(&mut app, window, |this, _, _| {
        assert!(this.images_pending("ds_8f2c"));
        this.store.images.remove("ds_8f2c");
    });
    // Asked again, it would fail again (a removed dataset is a 404) and wait for another turn.
    api.fail_images_once_from(IMAGE_PAGE);
    app.advance_clock(std::time::Duration::from_secs(5));
    drive(&mut app, window, |this, _, _| {
        assert!(!this.images_pending("ds_8f2c"), "the retry ends instead of asking forever");
        assert!(!this.store.images.contains_key("ds_8f2c"), "nothing comes back for a dropped list");
    });
}

#[test]
fn a_sample_past_the_first_page_opens_on_itself() {
    let api = Arc::new(MockBackend::new(4200, true).with_more_val_images("ds_8f2c", 25_000));
    let (mut app, window) = open_with(api.clone());
    drive(&mut app, window, |this, _, cx| this.ensure_dataset("ds_8f2c", cx));
    let last = drive(&mut app, window, |this, _, _| {
        this.store.dataset.get("ds_8f2c").and_then(|l| l.ready()).map(|d| d.splits.total()).unwrap()
    });
    let want = format!("img_{last}");
    drive(&mut app, window, |this, _, _| this.store.images.remove("ds_8f2c"));
    // The first page in, the second held: the inspector waits for its image.
    let (first, second) = (api.hold("images"), api.hold("images"));
    app.update(|cx| {
        window
            .update(cx, |this, w, cx| this.open_sample("ds_8f2c".into(), want.clone(), "run_x".into(), w, cx))
            .unwrap()
    });
    app.run_until_held(1);
    drop(first);
    while !second.is_held() {
        app.run_until_held(1);
    }
    app.frame(window.into());
    app.update(|cx| {
        window
            .update(cx, |this, _, _| {
                let i = this.inspector.as_ref().expect("still open");
                assert_eq!(i.want.as_deref(), Some(want.as_str()), "waiting, not on image 0");
            })
            .unwrap()
    });
    drop(second);
    app.run_until_idle();
    app.frame(window.into());
    drive(&mut app, window, |this, _, _| {
        let i = this.inspector.as_ref().expect("open");
        assert!(i.want.is_none());
        assert_eq!(this.filtered_image("ds_8f2c", i.index).map(|m| m.id.clone()), Some(want.clone()));
    });
}

#[test]
fn a_folder_with_no_usable_image_is_added_with_a_warning() {
    let (mut app, window) = open_with(Arc::new(MockBackend::new(4200, true)));
    drive(&mut app, window, |this, _, cx| {
        let mut ds = this.store.datasets.ready().unwrap()[0].clone();
        ds.id = "ds_0000".into();
        ds.name = "broken".into();
        ds.splits = Default::default();
        this.dataset_added(ds, "Added", cx);
        let t = this.toasts.last().expect("a toast");
        assert_eq!((t.tone, t.title.as_ref()), (Tone::Warning, "broken has no usable images"));
        assert!(t.action.is_some(), "with Open, to its page");
    });
}

#[test]
fn closing_generate_while_it_works_says_it_goes_on() {
    let (mut app, window) = open_with(Arc::new(MockBackend::new(4200, true)));
    let toasts = |app: &mut _| drive(app, window, |this, _, _| this.toasts.len());
    let before = toasts(&mut app);
    // Closed idle: nothing to say.
    drive(&mut app, window, |this, w, cx| {
        this.server_mode = ServerMode::Local;
        this.open_synth(w, cx);
        this.dismiss(w, cx);
    });
    assert_eq!(toasts(&mut app), before);
    // Esc while the server generates: the dialog goes, and a toast says the job goes on.
    drive(&mut app, window, |this, w, cx| {
        this.open_synth(w, cx);
        let f = this.synth.as_mut().unwrap();
        (f.submitting, f.sent_name) = (true, "shapes-sent".into());
        // Edited after Generate: the toast names what was sent.
        f.name.update(cx, |i, cx| i.set_text("other", cx));
        this.dismiss(w, cx);
        assert!(this.synth.is_none());
        let t = this.toasts.last().expect("a toast");
        assert_eq!(t.title.as_ref(), "Generating shapes-sent in the background");
    });
    assert_eq!(toasts(&mut app), before + 1);
}

#[test]
fn a_dataset_page_has_its_tabs_and_one_toolbar_and_statistics_holds_the_strip() {
    let (mut app, window) = super::tests::open();
    drive(&mut app, window, |this, w, cx| this.navigate(super::Route::Dataset("ds_3b91".into()), w, cx));
    let images = drawn_keys(&mut app, window);
    // Images / Statistics as underline tabs; the split, the class picker (a select, not a chip per
    // class), Boxes, S / M / L (not a slider) and Fit / Fill on one toolbar.
    for k in ["ds-tab-0", "ds-tab-1", "ds-split", "ds-class", "ds-boxes", "ds-size", "ds-fit"] {
        assert!(images.iter().any(|x| x == k), "{k}: {images:?}");
    }
    assert!(!images.iter().any(|k| k == "ds-tile" || k.starts_with("ds-class-")), "{images:?}");
    // The empty-images and broken-labels tiles moved into Statistics with the rest of the strip.
    assert!(!images.iter().any(|k| k == "ds-empty-tile"), "{images:?}");
    drive(&mut app, window, |this, _, cx| {
        this.ds_tab = super::DsTab::Statistics;
        cx.notify();
    });
    let stats = drawn_keys(&mut app, window);
    assert!(stats.iter().any(|k| k == "ds-empty-tile"), "Show them stays: {stats:?}");
    // The health summary is a Tab stop (its focus tip lists every finding).
    assert!(stats.iter().any(|k| k == "ds-health"), "{stats:?}");
    // "Show them": the empty-images chip takes the class picker's place on the toolbar.
    drive(&mut app, window, |this, _, cx| {
        this.ds_tab = super::DsTab::Images;
        this.ds_empty_only = true;
        cx.notify();
    });
    let empty = drawn_keys(&mut app, window);
    assert!(empty.iter().any(|k| k == "ds-empty-chip"), "{empty:?}");
}

/// x1-parity-7 and its review: an Overview card says when its server can't read the dataset, in
/// Studio's words with the server's reason, and follows the folder moved away after the card was
/// read, then put back (a visit to a page with cards reads each dataset again). Its page offers
/// Retry, Remove from FOLO and the list, not Connection settings.
#[test]
fn a_dataset_card_follows_its_folder_away_and_back() {
    let api = Arc::new(MockBackend::new(4200, true));
    let (mut app, window) = open_with(api.clone());
    let (id, format) = drive(&mut app, window, |this, _, _| {
        let d = this.store.datasets.ready().unwrap()[0].clone();
        (d.id.clone(), crate::api::Label::label(&d.format))
    });
    // The card as the Overview draws it, once its reads have answered.
    let card = |app: &mut folo_ui::HeadlessApp| {
        drawn_keys(app, window);
        drive(app, window, |this, _, _| {
            let d = this.store.dataset_summary(&id).cloned().expect("listed");
            this.dataset_card_badge(&d)
        })
    };
    // What the Datasets list painted (its cards on top of the page, inside the window; the
    // Overview's are below the fold): its warning-toned badges (their fill), counted in the scene.
    let warning_badges = |app: &mut folo_ui::HeadlessApp| {
        let fill = drive(app, window, |this, w, cx| {
            this.navigate(super::Route::Datasets, w, cx);
            crate::theme::theme(cx).warning_muted
        });
        drawn_keys(app, window);
        drawn_keys(app, window);
        app.scene(window.into())
            .primitives
            .iter()
            .filter(|p| matches!(p, folo_ui::Primitive::Quad(q) if q.background.solid == fill))
            .count()
    };
    let go = |app: &mut folo_ui::HeadlessApp, route: super::Route| {
        drive(app, window, |this, w, cx| this.navigate(route, w, cx));
    };
    go(&mut app, super::Route::Overview);
    assert_eq!(card(&mut app), (format, Tone::Neutral, None));
    let healthy = warning_badges(&mut app);

    // The folder moved away: the next visit says so, with the server's reason for the tip.
    api.break_dataset(&id, "path_not_found", "Dataset folder not found: D:/data/shapes");
    go(&mut app, super::Route::Datasets);
    go(&mut app, super::Route::Overview);
    let (label, tone, why) = card(&mut app);
    assert_eq!((label, tone), ("Folder missing", Tone::Warning));
    assert_eq!(why.as_deref(), Some("Dataset folder not found: D:/data/shapes"));
    // The card itself paints it: one warning badge more than the healthy page had.
    assert_eq!(warning_badges(&mut app), healthy + 1, "the card draws its warning badge");

    // Its page: the dataset's own ways on, not the connection panel.
    go(&mut app, super::Route::Dataset(id.clone()));
    let keys = drawn_keys(&mut app, window);
    assert!(keys.iter().any(|k| k == "ds-problem-retry"), "{keys:?}");
    assert!(keys.iter().any(|k| k == "ds-problem-delete"), "{keys:?}");
    assert!(!keys.iter().any(|k| k == "ds-detail-retry-settings"), "{keys:?}");

    // Put back: Retry on its page reads it again, and the card is healthy again.
    api.mend_dataset(&id);
    drive(&mut app, window, |this, _, cx| {
        this.ds_recheck.insert(id.clone());
        cx.notify();
    });
    drawn_keys(&mut app, window);
    drive(&mut app, window, |this, _, _| {
        assert!(this.store.dataset.get(&id).is_some_and(|l| l.ready().is_some()), "read again");
    });
    go(&mut app, super::Route::Overview);
    assert_eq!(card(&mut app), (format, Tone::Neutral, None));
    assert_eq!(warning_badges(&mut app), healthy, "and paints no warning");

    // A folder that no longer reads as a dataset: "Can't read".
    api.break_dataset(&id, "invalid_dataset", "no images found");
    go(&mut app, super::Route::Datasets);
    go(&mut app, super::Route::Overview);
    assert_eq!(card(&mut app).0, "Can't read");

    // What the user reads on the card (review round 3): its keyboard-focus tip names the server's
    // reason. The glyphs the Datasets list paints with the card focused, less those without, for two
    // reasons that differ by a few words: the tip is longer by exactly those words.
    let tip_glyphs = |app: &mut folo_ui::HeadlessApp, why: &str| -> i64 {
        api.break_dataset(&id, "path_not_found", why);
        go(app, super::Route::Overview);
        go(app, super::Route::Datasets);
        let glyphs = |app: &mut folo_ui::HeadlessApp, focused: bool| {
            drive(app, window, |this, w, cx| {
                this.kb = focused;
                this.focus_key(&format!("ds-card-{id}"), w, cx);
                cx.notify();
            });
            drawn_keys(app, window);
            let scene = app.scene(window.into());
            scene.primitives.iter().filter(|p| matches!(p, folo_ui::Primitive::MonoSprite(_))).count() as i64
        };
        let plain = glyphs(app, false);
        glyphs(app, true) - plain
    };
    let short = "Dataset folder not found: D:/data/shapes";
    let extra = " (moved to the recycle bin)";
    let (tip_short, tip_long) =
        (tip_glyphs(&mut app, short), tip_glyphs(&mut app, &format!("{short}{extra}")));
    assert!(tip_short > short.len() as i64 / 2, "the focused card shows its tip: {tip_short} glyphs");
    let words = extra.chars().filter(|c| !c.is_whitespace()).count() as i64;
    assert_eq!(tip_long - tip_short, words, "the tip reads the server's reason");
}

/// s1-datasets-6: a dataset whose folder is gone kept adding its last known image count to the
/// Datasets header, and its card showed that count as if current. Studio leaves it out of the total
/// and says "(last known)" on the card.
#[test]
fn a_dataset_whose_folder_is_gone_counts_only_as_last_known() {
    let api = Arc::new(MockBackend::new(4200, true));
    let (mut app, window) = open_with(api.clone());
    let (id, all, its) = drive(&mut app, window, |this, w, cx| {
        this.navigate(super::Route::Datasets, w, cx);
        let l = this.store.datasets.ready().unwrap();
        let all: u64 = l.iter().map(|d| d.splits.total()).sum();
        (l[0].id.clone(), all, l[0].splits.total())
    });
    drawn_keys(&mut app, window);
    let facts = |app: &mut folo_ui::HeadlessApp| {
        drive(app, window, |this, _, _| {
            let l = this.store.datasets.ready().unwrap().clone();
            let d = l.iter().find(|d| d.id == id).unwrap();
            (
                this.datasets_subtitle(&l),
                crate::screens::overview::card_facts(d, this.dataset_unreadable(&id)),
            )
        })
    };
    let (header, card) = facts(&mut app);
    assert!(header.ends_with(&crate::ui::counted(all, "image", "images")), "{header}");
    assert!(!card.contains("last known"), "{card}");

    api.break_dataset(&id, "path_not_found", "Dataset folder not found: D:/data/shapes");
    drive(&mut app, window, |this, w, cx| this.navigate(super::Route::Overview, w, cx));
    drive(&mut app, window, |this, w, cx| this.navigate(super::Route::Datasets, w, cx));
    drawn_keys(&mut app, window);
    let (header, card) = facts(&mut app);
    assert!(header.ends_with(&crate::ui::counted(all - its, "image", "images")), "{header}");
    assert!(
        card.starts_with(&format!("{} (last known) · ", crate::ui::counted(its, "image", "images"))),
        "{card}"
    );
}

/// Review round 2 of x1-parity-7: two reads of a dataset on their way at once (two Retries, two
/// visits) that answer in reverse order. The later read saw the folder as it is now: its answer
/// stands, whichever way the folder went in between.
#[test]
fn an_older_dataset_read_answering_last_never_overwrites_a_newer_one() {
    for first_broken in [true, false] {
        let api = Arc::new(MockBackend::new(4200, true));
        let (mut app, window) = open_with(api.clone());
        let id = drive(&mut app, window, |this, _, cx| {
            let id = this.store.datasets.ready().unwrap()[0].id.clone();
            this.ensure_dataset(&id, cx);
            id
        });
        // Every card's dataset read once, so the held read below is ours.
        drawn_keys(&mut app, window);
        let read_again = |app: &mut folo_ui::HeadlessApp| {
            app.update(|cx| {
                window
                    .update(cx, |this, _, cx| {
                        this.ds_recheck.insert(id.clone());
                        this.ensure_dataset(&id, cx);
                    })
                    .unwrap()
            });
        };
        let set = |broken: bool| {
            if broken {
                api.break_dataset(&id, "path_not_found", "Dataset folder not found");
            } else {
                api.mend_dataset(&id);
            }
        };
        // Read A is answered as the folder was, then held back.
        set(first_broken);
        let first = api.hold("dataset");
        read_again(&mut app);
        app.run_until_held(1);
        first.wait_held();
        // The folder changes; read B answers first.
        set(!first_broken);
        read_again(&mut app);
        app.run_until_held(1);
        // (A plain update while A is held: `drive` waits for every call.)
        let now = |app: &mut folo_ui::HeadlessApp| {
            app.update(|cx| {
                window
                    .update(cx, |this, _, _| this.store.dataset.get(&id).map(|l| l.error().is_some()))
                    .unwrap()
            })
        };
        assert_eq!(now(&mut app), Some(!first_broken), "read B's answer is shown");
        // A lands last: dropped.
        drop(first);
        app.run_until_idle();
        assert_eq!(now(&mut app), Some(!first_broken), "read A answered late (broken first: {first_broken})");
    }
}

/// Review round 2 of x1-parity-7: a dataset read again whose answer differs from what the app read
/// before (a repaired copy put back with an image fewer) lists its images again: the grid and the
/// split tabs count the same images, and nothing derived from the old files stays.
#[test]
fn a_dataset_read_again_with_other_contents_lists_its_images_again() {
    let api = Arc::new(MockBackend::new(4200, true));
    let (mut app, window) = open_with(api.clone());
    let id = drive(&mut app, window, |this, _, cx| {
        let id = this.store.datasets.ready().unwrap()[0].id.clone();
        this.ensure_dataset(&id, cx);
        id
    });
    // What the app holds from before: one image more than the folder has now, and label issues.
    let total = drive(&mut app, window, |this, _, _| {
        let real = this.store.dataset.get(&id).and_then(|l| l.ready()).cloned().expect("read");
        let mut old = real.clone();
        old.splits.train += 1;
        this.store.dataset.insert(id.clone(), crate::store::Load::Ready(old));
        let list = this.store.images.get_mut(&id).and_then(|l| l.ready_mut()).expect("listed");
        let extra = list.last().cloned().expect("an image");
        list.push(extra);
        this.store
            .issues
            .insert(id.clone(), crate::store::Load::Failed(crate::api::ApiError::new("old", "old")));
        real.splits.total()
    });
    // A visit to a page with cards (the app opens on the Overview: the list is another page).
    drive(&mut app, window, |this, w, cx| this.navigate(super::Route::Datasets, w, cx));
    drawn_keys(&mut app, window);
    drawn_keys(&mut app, window);
    drive(&mut app, window, |this, _, _| {
        let d = this.store.dataset.get(&id).and_then(|l| l.ready()).expect("read again");
        assert_eq!(d.splits.total(), total, "the dataset as it is now");
        let images = this.store.images.get(&id).and_then(|l| l.ready()).map(|l| l.len() as u64);
        assert_eq!(images, Some(total), "its image list read again, not the old one kept");
        assert!(
            this.store.issues.get(&id).is_none_or(|l| l.error().is_none_or(|e| e.code != "old")),
            "its old label issues dropped"
        );
    });
}

/// Review round 3 of x1-parity-7: a dataset's issue check still on its way when its files change
/// (the app forgets what it read of them, as a read with other contents does) answers after the
/// check asked for since. The newer check's counts stand: the old file errors never come back.
#[test]
fn an_issue_check_of_files_since_changed_never_comes_back() {
    let api = Arc::new(MockBackend::new(4200, true));
    let (mut app, window) = open_with(api.clone());
    // A dataset whose old files had broken labels (the mock counts them as errors).
    let id = "ds_3b91".to_string();
    drawn_keys(&mut app, window);
    let errors = |app: &mut folo_ui::HeadlessApp| {
        app.update(|cx| {
            window
                .update(cx, |this, _, _| this.store.issues.get(&id).and_then(|l| l.ready()).map(|c| c.error))
                .unwrap()
        })
    };
    // Check A counts the old files, and is held.
    let old = api.hold("issues");
    app.update(|cx| {
        window
            .update(cx, |this, _, cx| {
                this.store.issues.remove(&id);
                this.ensure_issues(&id, cx);
            })
            .unwrap()
    });
    app.run_until_held(1);
    old.wait_held();
    // The files change: what was read of them is forgotten, and check B is asked for.
    app.update(|cx| {
        window
            .update(cx, |this, _, cx| {
                this.forget_dataset_contents(&id, cx);
                this.ensure_issues(&id, cx);
            })
            .unwrap()
    });
    app.run_until_held(1);
    assert!(errors(&mut app).is_some(), "check B answered");
    // B's answer is the repaired files': no errors.
    app.update(|cx| {
        window
            .update(cx, |this, _, _| {
                let clean =
                    crate::api::IssueCounts { error: 0, warning: 0, info: 0, by_kind: Default::default() };
                this.store.issues.insert(id.clone(), crate::store::Load::Ready(clean));
            })
            .unwrap()
    });
    // A lands last, with the old files' errors: dropped.
    drop(old);
    app.run_until_idle();
    assert_eq!(errors(&mut app), Some(0), "the old check's errors came back");
}

/// Review round 3 of x1-parity-7, on lane x4's attempts: image A is asked for, the dataset's files
/// change (its ids are positions: `img_1` may be another picture now), the same id is asked for
/// again (B), and A answers first. A's answer is dropped; the slot waits for B's, and B's fills it.
#[test]
fn a_load_from_before_a_forget_never_fills_the_slot_asked_for_again() {
    use crate::store::{ImgKind, ImgSlot};
    let api = Arc::new(MockBackend::new(4200, true));
    let (mut app, window) = open_with(api.clone());
    drawn_keys(&mut app, window);
    let id = "ds_3b91".to_string();
    let slot = |app: &mut folo_ui::HeadlessApp| {
        app.update(|cx| {
            window
                .update(cx, |this, _, _| {
                    let key = crate::store::ImgKey {
                        dataset: id.clone(),
                        image: "img_1".into(),
                        kind: ImgKind::Thumb,
                    };
                    match this.store.images_px.get(&key) {
                        Some(ImgSlot::Ready(_)) => "ready",
                        Some(ImgSlot::Failed) => "failed",
                        Some(_) => "loading",
                        None => "none",
                    }
                })
                .unwrap()
        })
    };
    let ask = |app: &mut folo_ui::HeadlessApp| {
        app.update(|cx| {
            window
                .update(cx, |this, _, cx| {
                    this.forget_dataset_contents(&id, cx);
                    assert!(this.image(&id, "img_1", ImgKind::Thumb, cx).is_none());
                })
                .unwrap()
        })
    };
    let (a, b) = (api.hold("pixels ds_3b91 img_1"), api.hold("pixels ds_3b91 img_1"));
    ask(&mut app);
    app.run_until_held(1);
    a.wait_held();
    ask(&mut app);
    app.run_until_held(2);
    b.wait_held();
    drop(a);
    app.run_until_held(1);
    assert_eq!(slot(&mut app), "loading", "the old picture filled the slot asked for again");
    drop(b);
    app.run_until_idle();
    assert_eq!(slot(&mut app), "ready", "the new picture");
}

/// Lane d6's reconnect sweep (`forget_unreachable_loads`) and this lane's recheck on navigation
/// (`ds_recheck`) together: a dataset whose read failed while the server was away, shown on a page
/// with cards meanwhile, is read once when shown after the reconnect, not by both.
#[test]
fn a_dataset_that_failed_while_away_is_read_once_after_the_reconnect() {
    let (mut app, window) = open_with(Arc::new(MockBackend::new(4200, true)));
    drawn_keys(&mut app, window);
    let id = "ds_3b91".to_string();
    drive(&mut app, window, |this, _, cx| {
        let away = crate::api::ApiError::new("server_unreachable", "Could not reach folo serve");
        this.store.dataset.insert(id.clone(), crate::store::Load::Failed(away));
        this.ds_recheck.insert(id.clone());
        this.reconnected(cx);
        let before = this.ds_read_seq;
        this.ensure_dataset(&id, cx);
        assert_eq!(this.ds_read_seq, before + 1, "one read, not the recheck's and a fresh one's");
        assert!(!this.ds_recheck.contains(&id));
    });
    drive(&mut app, window, |this, _, _| {
        assert!(this.store.dataset.get(&id).is_some_and(|l| l.ready().is_some()), "read, and in");
    });
}

/// The focus keys a page draws, once what it asked for has come.
fn drawn_keys(app: &mut folo_ui::HeadlessApp, window: folo_ui::WindowHandle<super::FoloApp>) -> Vec<String> {
    app.frame(window.into());
    app.run_until_idle();
    app.frame(window.into());
    drive(app, window, |this, _, _| this.focus.borrow().keys().map(|k| k.to_string()).collect())
}

#[test]
fn the_health_summary_asks_for_the_check_even_when_every_label_parses() {
    let (mut app, window) = super::tests::open();
    // A dataset whose labels all parse: the check can still find unreadable images or warnings.
    let id = drive(&mut app, window, |this, _, _| {
        let ids: Vec<String> = this.store.datasets.ready().unwrap().iter().map(|d| d.id.clone()).collect();
        ids.into_iter().find(|id| id != "ds_3b91").unwrap()
    });
    drive(&mut app, window, |this, w, cx| this.navigate(super::Route::Dataset(id.clone()), w, cx));
    drawn_keys(&mut app, window);
    drawn_keys(&mut app, window);
    drive(&mut app, window, |this, _, _| {
        let stats = this.store.dataset.get(&id).and_then(|l| l.ready()).and_then(|d| d.stats.clone());
        assert_eq!(stats.map(|s| s.broken_labels), Some(0), "{id}");
        assert!(this.store.issues.contains_key(&id), "the check's counts were asked for");
    });
}
