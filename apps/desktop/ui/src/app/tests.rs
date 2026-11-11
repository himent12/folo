//! Runtime tests: entities, effects, subscriptions, globals, executors and quit on the headless platform.

use std::{
    cell::{Cell, RefCell},
    rc::Rc,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering::SeqCst},
    },
    time::Duration,
};

use crate::{
    App, AppContext, Context, Entity, EventEmitter, Global, HeadlessApp, HeadlessOptions, Subscription,
    TextBackend, WeakEntity,
};

fn app() -> HeadlessApp {
    HeadlessApp::new(HeadlessOptions { text: TextBackend::Fixed, ..Default::default() })
}

/// Counts drops, to see when an entity's state is released.
struct Tracked(Rc<Cell<usize>>);

impl Drop for Tracked {
    fn drop(&mut self) {
        self.0.set(self.0.get() + 1);
    }
}

#[derive(Default)]
struct Counter {
    count: usize,
    log: Vec<String>,
    subs: Vec<Subscription>,
}

struct Bumped(usize);

impl EventEmitter<Bumped> for Counter {}

#[test]
fn entities_read_update_and_weak_handles() {
    let mut app = app();
    let counter = app.update(|cx| cx.new(|_| Counter::default()));
    let weak = counter.downgrade();
    let result = app.update(|cx| {
        counter.update(cx, |c, cx| {
            c.count += 2;
            assert_eq!(cx.entity_id(), cx.entity().entity_id());
            c.count * 10
        })
    });
    assert_eq!(result, 20);
    assert_eq!(app.read(|cx| counter.read(cx).count), 2);
    assert_eq!(app.read(|cx| counter.read_with(cx, |c, _| c.count)), 2);
    assert!(weak.upgrade().is_some());
    assert_eq!(app.update(|cx| weak.update(cx, |c, _| c.count)).unwrap(), 2);
}

#[test]
fn dropping_the_last_handle_releases_state_at_the_flush() {
    let mut app = app();
    let drops = Rc::new(Cell::new(0));
    let entity = app.update(|cx| cx.new(|_| Tracked(drops.clone())));
    let weak = entity.downgrade();
    let clone = entity.clone();
    drop(entity);
    assert!(weak.upgrade().is_some(), "a clone still holds it");
    drop(clone);
    assert!(weak.upgrade().is_none(), "no strong handle: upgrade fails at once");
    assert_eq!(drops.get(), 0, "state lives until the flush");
    app.update(|_| {});
    assert_eq!(drops.get(), 1);
    assert!(app.update(|cx| weak.update(cx, |_, _| ())).is_err());
}

#[test]
fn releasing_an_entity_releases_the_entities_it_owned() {
    let mut app = app();
    let drops = Rc::new(Cell::new(0));
    let inner = app.update(|cx| cx.new(|_| Tracked(drops.clone())));
    let inner_weak = inner.downgrade();
    let outer = app.update(|cx| cx.new(|_| (inner, Tracked(drops.clone()))));
    drop(outer);
    app.update(|_| {});
    assert_eq!(drops.get(), 2);
    assert!(inner_weak.upgrade().is_none());
}

#[test]
#[should_panic(expected = "already being updated")]
fn re_entrant_update_panics() {
    let mut app = app();
    let counter = app.update(|cx| cx.new(|_| Counter::default()));
    app.update(|cx| counter.update(cx, |_, cx| counter.update(cx, |_, _| ())));
}

#[test]
fn notify_runs_observers_once_per_flush_and_emit_reaches_subscribers() {
    let mut app = app();
    let source = app.update(|cx| cx.new(|_| Counter::default()));
    let watcher = app.update(|cx| {
        cx.new(|cx: &mut Context<Counter>| {
            let mut c = Counter::default();
            c.subs.push(cx.observe(&source, |this, _, _| this.log.push("notified".into())));
            c.subs.push(cx.subscribe(&source, |this, _, event: &Bumped, _| {
                this.log.push(format!("bumped {}", event.0))
            }));
            c
        })
    });
    app.update(|cx| {
        source.update(cx, |_, cx| {
            cx.notify();
            cx.notify();
            cx.emit(Bumped(7));
        })
    });
    assert_eq!(app.read(|cx| watcher.read(cx).log.clone()), ["notified", "bumped 7"]);
}

#[test]
fn a_subscription_does_not_see_the_effect_cycle_it_was_made_in() {
    let mut app = app();
    let source = app.update(|cx| cx.new(|_| Counter::default()));
    let seen = Rc::new(Cell::new(0));
    let sub = app.update(|cx| {
        source.update(cx, |_, cx| cx.notify());
        let seen = seen.clone();
        cx.observe(&source, move |_, _| seen.set(seen.get() + 1))
    });
    assert_eq!(seen.get(), 0, "the notify queued before subscribing is not seen");
    app.update(|cx| source.update(cx, |_, cx| cx.notify()));
    assert_eq!(seen.get(), 1);
    drop(sub);
    app.update(|cx| source.update(cx, |_, cx| cx.notify()));
    assert_eq!(seen.get(), 1, "dropping the subscription unsubscribes");
}

#[test]
fn detached_subscriptions_live_until_the_emitter_is_released() {
    let mut app = app();
    let source = app.update(|cx| cx.new(|_| Counter::default()));
    let seen = Rc::new(Cell::new(0));
    app.update(|cx| {
        let seen = seen.clone();
        cx.subscribe(&source, move |_, _: &Bumped, _| seen.set(seen.get() + 1)).detach();
    });
    app.update(|cx| source.update(cx, |_, cx| cx.emit(Bumped(1))));
    assert_eq!(seen.get(), 1);
    let id = source.entity_id();
    drop(source);
    app.update(|_| {});
    assert!(!app.read(|cx| cx.event_listeners.has(&id)), "released with the emitter");
}

#[test]
fn observer_dropped_while_its_key_runs_is_removed() {
    let mut app = app();
    let source = app.update(|cx| cx.new(|_| Counter::default()));
    let seen = Rc::new(Cell::new(0));
    let slot: Rc<RefCell<Option<Subscription>>> = Rc::default();
    let sub = app.update(|cx| {
        let (seen, slot) = (seen.clone(), slot.clone());
        cx.observe(&source, move |_, _| {
            seen.set(seen.get() + 1);
            slot.borrow_mut().take();
        })
    });
    *slot.borrow_mut() = Some(sub);
    app.update(|cx| source.update(cx, |_, cx| cx.notify()));
    app.update(|cx| source.update(cx, |_, cx| cx.notify()));
    assert_eq!(seen.get(), 1, "the callback dropped its own subscription");
}

#[test]
fn effects_flush_in_order_and_defer_runs_after_queued_effects() {
    let mut app = app();
    let log: Rc<RefCell<Vec<&str>>> = Rc::default();
    let source = app.update(|cx| cx.new(|_| Counter::default()));
    let _sub = app.update(|cx| {
        let log = log.clone();
        cx.observe(&source, move |_, cx| {
            log.borrow_mut().push("observer");
            let log = log.clone();
            cx.defer(move |_| log.borrow_mut().push("deferred by observer"));
        })
    });
    app.update(|cx| {
        let log2 = log.clone();
        cx.defer(move |_| log2.borrow_mut().push("defer 1"));
        source.update(cx, |_, cx| cx.notify());
        let log3 = log.clone();
        cx.defer(move |_| log3.borrow_mut().push("defer 2"));
        log.borrow_mut().push("update body");
    });
    assert_eq!(*log.borrow(), ["update body", "defer 1", "observer", "defer 2", "deferred by observer"]);
}

struct Theme(u32);

impl Global for Theme {}

#[test]
fn globals_set_update_and_observe() {
    let mut app = app();
    app.update(|cx| cx.set_global(Theme(1)));
    let seen = Rc::new(Cell::new(0));
    let _sub = app.update(|cx| {
        let seen = seen.clone();
        cx.observe_global::<Theme>(move |cx| seen.set(cx.global::<Theme>().0))
    });
    let read = app.update(|cx| {
        cx.update_global::<Theme, _>(|theme, cx| {
            theme.0 = 5;
            assert!(!cx.has_global::<Theme>(), "leased out during the update");
            theme.0
        })
    });
    assert_eq!(read, 5);
    assert_eq!(seen.get(), 5);
    app.update(|cx| cx.global_mut::<Theme>().0 = 9);
    assert_eq!(seen.get(), 9);
    assert_eq!(app.read(|cx| cx.read_global(|t: &Theme, _| t.0)), 9);
}

#[test]
fn foreground_tasks_run_when_pumped_and_background_tasks_on_threads() {
    let mut app = app();
    let counter = app.update(|cx| cx.new(|_| Counter::default()));
    let task = app.update(|cx| {
        counter.update(cx, |_, cx| {
            cx.spawn(async move |this: WeakEntity<Counter>, cx| {
                let value = cx
                    .background_executor()
                    .spawn(async { std::thread::current().name().map(String::from) })
                    .await;
                this.update(cx, |c, cx| {
                    c.log.push(value.unwrap_or_default());
                    cx.notify();
                })
            })
        })
    });
    task.detach();
    assert!(app.read(|cx| counter.read(cx).log.is_empty()), "nothing runs until pumped");
    app.run_until_idle();
    assert_eq!(app.read(|cx| counter.read(cx).log.clone()), ["folo-ui-headless"]);
}

#[test]
fn timers_fire_only_when_the_clock_passes_them() {
    let mut app = app();
    let fired = Rc::new(RefCell::new(Vec::new()));
    let start = app.now();
    for ms in [300u64, 100, 200] {
        let fired = fired.clone();
        app.update(|cx| {
            cx.spawn(async move |cx| {
                cx.background_executor().timer(Duration::from_millis(ms)).await;
                fired.borrow_mut().push(ms);
            })
            .detach()
        });
    }
    app.run_until_idle();
    assert!(fired.borrow().is_empty());
    app.advance_clock(Duration::from_millis(150));
    assert_eq!(*fired.borrow(), [100]);
    app.advance_clock(Duration::from_millis(200));
    assert_eq!(*fired.borrow(), [100, 200, 300]);
    assert_eq!(app.now() - start, Duration::from_millis(350));
}

#[test]
fn a_periodic_timer_fires_once_per_period_while_advancing() {
    let mut app = app();
    let ticks = Rc::new(Cell::new(0));
    let ticks2 = ticks.clone();
    app.update(|cx| {
        cx.spawn(async move |cx| {
            loop {
                cx.background_executor().timer(Duration::from_millis(100)).await;
                ticks2.set(ticks2.get() + 1);
            }
        })
        .detach()
    });
    app.advance_clock(Duration::from_millis(1050));
    assert_eq!(ticks.get(), 10);
}

#[test]
fn dropping_a_task_cancels_it() {
    let mut app = app();
    let ran = Rc::new(Cell::new(false));
    let ran2 = ran.clone();
    let task = app.update(|cx| {
        cx.spawn(async move |cx| {
            cx.background_executor().timer(Duration::from_millis(10)).await;
            ran2.set(true);
        })
    });
    drop(task);
    app.advance_clock(Duration::from_millis(20));
    assert!(!ran.get());
}

#[test]
fn awaiting_a_task_gives_its_value() {
    let mut app = app();
    let out = Rc::new(Cell::new(0));
    let out2 = out.clone();
    app.update(|cx| {
        let inner = cx.spawn(async |_| 21);
        cx.spawn(async move |_| out2.set(inner.await * 2)).detach();
    });
    app.run_until_idle();
    assert_eq!(out.get(), 42);
}

#[test]
fn a_task_woken_after_the_app_is_gone_does_not_panic() {
    let mut app = app();
    let dispatcher = app.dispatcher().clone();
    let result: Rc<RefCell<Option<anyhow::Result<()>>>> = Rc::default();
    let result2 = result.clone();
    app.update(|cx| {
        cx.spawn(async move |cx| {
            cx.background_executor().timer(Duration::from_millis(10)).await;
            *result2.borrow_mut() = Some(cx.update(|_| ()));
        })
        .detach()
    });
    app.run_until_idle();
    drop(app);
    dispatcher.advance_clock(Duration::from_millis(20));
    // The app (and its tasks' queue owner) is gone: the task either ran and saw `Err`, or was dropped.
    assert!(result.borrow().as_ref().is_none_or(|r| r.is_err()));
}

#[test]
#[should_panic(expected = "still running")]
fn run_until_idle_fails_loudly_on_a_stuck_background_job() {
    let mut app = app();
    app.dispatcher().set_idle_timeout(Duration::from_millis(50));
    app.update(|cx| {
        cx.background_executor().spawn(async { std::thread::sleep(Duration::from_millis(400)) }).detach()
    });
    app.run_until_idle();
}

#[test]
#[should_panic(expected = "a background job panicked: boom")]
fn a_panicking_background_job_fails_run_until_idle() {
    let mut app = app();
    app.update(|cx| cx.background_executor().spawn(async { panic!("boom") }).detach());
    app.run_until_idle();
}

#[test]
fn quit_runs_app_quit_callbacks_once() {
    let mut app = app();
    let quits = Rc::new(Cell::new(0));
    let counter: Entity<Counter> = app.update(|cx| {
        let quits = quits.clone();
        cx.new(move |cx: &mut Context<Counter>| {
            let mut c = Counter::default();
            c.subs.push(cx.on_app_quit(move |this, _| {
                this.count += 1;
                quits.set(quits.get() + 1);
                async {}
            }));
            c
        })
    });
    app.update(|cx| cx.quit());
    app.update(|cx| cx.quit());
    assert_eq!(quits.get(), 1);
    assert_eq!(app.read(|cx| counter.read(cx).count), 1);
    assert!(app.platform().quit_requested.get());
}

#[test]
fn background_spawn_from_any_context_and_async_reads() {
    let mut app = app();
    let counter = app.update(|cx| cx.new(|_| Counter { count: 3, ..Default::default() }));
    let hits = Arc::new(AtomicUsize::new(0));
    let out = Rc::new(Cell::new(0));
    let (hits2, out2, counter2) = (hits.clone(), out.clone(), counter.clone());
    app.update(|cx| {
        cx.spawn(async move |cx| {
            cx.background_spawn(async move { hits2.fetch_add(1, SeqCst) }).await;
            let entity = cx.new(|_| Counter { count: 4, ..Default::default() }).unwrap();
            let a = counter2.read_with(cx, |c, _| c.count).unwrap();
            let b = entity.read_with(cx, |c, _| c.count).unwrap();
            out2.set(a * b);
        })
        .detach()
    });
    app.run_until_idle();
    assert_eq!((hits.load(SeqCst), out.get()), (1, 12));
}

#[test]
fn global_actions_dispatch_capture_then_bubble() {
    crate::actions!(test_runtime, [Ping]);
    let mut app = app();
    let log: Rc<RefCell<Vec<&str>>> = Rc::default();
    let (l1, l2) = (log.clone(), log.clone());
    app.update(|cx| {
        cx.on_action(move |_: &Ping, _| l1.borrow_mut().push("first"));
        cx.on_action(move |_: &Ping, cx: &mut App| {
            l2.borrow_mut().push("second");
            cx.propagate();
        });
    });
    app.update(|cx| {
        cx.dispatch_global_action(&Ping, crate::DispatchPhase::Capture);
        assert!(cx.propagate_event, "capture handlers of `on_action` do nothing");
        cx.dispatch_global_action(&Ping, crate::DispatchPhase::Bubble);
        assert!(!cx.propagate_event, "the first (later-registered-last) handler stopped it");
    });
    assert_eq!(*log.borrow(), ["second", "first"], "bubble runs the last registered first");
}

#[test]
fn quit_from_an_update_of_an_entity_with_an_app_quit_handler() {
    let mut app = app();
    let counter: Entity<Counter> = app.update(|cx| {
        cx.new(|cx: &mut Context<Counter>| {
            let mut c = Counter::default();
            c.subs.push(cx.on_app_quit(|this, _| {
                this.count += 1;
                async {}
            }));
            c
        })
    });
    // The handler leases `counter`, which this update holds: shutdown waits for the update to end.
    app.update(|cx| counter.update(cx, |_, cx| cx.quit()));
    assert_eq!(app.read(|cx| counter.read(cx).count), 1);
    assert!(app.platform().quit_requested.get());
}

/// A root view that registers an `on_app_quit` handler, as the app's does.
struct QuitView {
    quits: Rc<Cell<usize>>,
    _sub: Option<Subscription>,
}

impl crate::Render for QuitView {
    fn render(&mut self, _: &mut crate::Window, _: &mut Context<Self>) -> impl crate::IntoElement {
        crate::div()
    }
}

#[test]
fn closing_the_last_window_quits_through_on_window_closed() {
    // The app's path: `on_window_closed(|cx| cx.quit())` and the platform closes the window. Shutdown runs
    // once; the root view's own handler does not, since the view is released with its window (as on GPUI,
    // whose shutdown runs after the loop has ended).
    let mut app = app();
    let (view_quits, app_quits) = (Rc::new(Cell::new(0)), Rc::new(Cell::new(0)));
    app.update(|cx| {
        let view_quits = view_quits.clone();
        cx.open_window(crate::WindowOptions::default(), move |_, cx| {
            cx.new(move |cx: &mut Context<QuitView>| {
                let sub = cx.on_app_quit(|this, _| {
                    this.quits.set(this.quits.get() + 1);
                    async {}
                });
                QuitView { quits: view_quits, _sub: Some(sub) }
            })
        })
        .expect("open a window");
        let app_quits = app_quits.clone();
        cx.on_app_quit(move |_| {
            app_quits.set(app_quits.get() + 1);
            async {}
        })
        .detach();
        cx.on_window_closed(|cx| cx.quit()).detach();
    });
    let close = app.platform().windows.borrow()[0].callbacks.borrow_mut().close.take();
    close.expect("the window registered on_close")();
    assert_eq!((app_quits.get(), view_quits.get()), (1, 0));
    assert!(app.read(|cx| cx.windows().is_empty()));
    assert!(app.platform().quit_requested.get());
}

#[test]
fn quit_polls_every_app_quit_future_together() {
    // The first future waits for what the second sends: awaited one after another, neither finishes.
    let mut app = app();
    let (tx, rx) = crate::oneshot::channel::<()>();
    let done = Rc::new(Cell::new(false));
    app.update(|cx| {
        let (mut rx, done) = (Some(rx), done.clone());
        cx.on_app_quit(move |_| {
            let (rx, done) = (rx.take(), done.clone());
            async move {
                if let Some(rx) = rx {
                    done.set(rx.await.is_ok());
                }
            }
        })
        .detach();
        let mut tx = Some(tx);
        cx.on_app_quit(move |_| {
            let tx = tx.take();
            async move {
                if let Some(tx) = tx {
                    tx.send(()).ok();
                }
            }
        })
        .detach();
    });
    app.update(|cx| cx.quit());
    assert!(done.get(), "the first quit future saw the second's signal");
}

#[test]
fn a_subscription_dropped_after_its_callback_subscribed_again_is_removed() {
    // The callback adds an observer of the same emitter, then drops its own subscription.
    let mut app = app();
    let source = app.update(|cx| cx.new(|_| Counter::default()));
    let seen = Rc::new(Cell::new(0));
    let slot: Rc<RefCell<Option<Subscription>>> = Rc::default();
    let added: Rc<RefCell<Vec<Subscription>>> = Rc::default();
    let sub = app.update(|cx| {
        let (seen, slot, added) = (seen.clone(), slot.clone(), added.clone());
        cx.observe(&source, move |source, cx| {
            seen.set(seen.get() + 1);
            added.borrow_mut().push(cx.observe(&source, |_, _| {}));
            let original = slot.borrow_mut().take();
            drop(original);
        })
    });
    *slot.borrow_mut() = Some(sub);
    app.update(|cx| source.update(cx, |_, cx| cx.notify()));
    app.update(|cx| source.update(cx, |_, cx| cx.notify()));
    assert_eq!(seen.get(), 1, "the dropped subscription's callback no longer runs");
    drop(added.take());
    assert!(!app.read(|cx| cx.observers.has(&source.entity_id())), "nothing is left for the emitter");
}

#[test]
fn dropping_the_app_drops_its_queued_tasks_and_dispatcher() {
    let mut app = app();
    let dispatcher = Arc::downgrade(app.dispatcher());
    let drops = Rc::new(Cell::new(0));
    // A task waiting on a timer, and one queued and never pumped: both runnables sit in the dispatcher's
    // queues, and each one's scheduler holds the dispatcher.
    app.update(|cx| {
        let tracked = Tracked(drops.clone());
        cx.spawn(async move |cx| {
            cx.background_executor().timer(Duration::from_secs(1)).await;
            drop(tracked);
        })
        .detach()
    });
    app.run_until_idle();
    let tracked = Tracked(drops.clone());
    app.update(|cx| cx.spawn(async move |_| drop(tracked)).detach());
    drop(app);
    assert_eq!(drops.get(), 2, "both tasks' state is dropped with the app");
    assert!(dispatcher.upgrade().is_none(), "nothing keeps the dispatcher alive");
}

#[test]
fn a_main_thread_task_woken_off_the_main_thread_after_the_app_is_gone_does_not_panic() {
    let mut app = app();
    let (tx, rx) = crate::oneshot::channel::<()>();
    app.update(|cx| {
        cx.spawn(async move |cx| {
            cx.background_spawn(async move { rx.await.ok() }).await;
        })
        .detach()
    });
    app.run_until_idle();
    drop(app);
    // Waking the background task from another thread cancels it there, which wakes the main-thread task
    // awaiting it on that thread: that task must not be dropped there (async-task panics).
    let sender = std::thread::spawn(move || tx.send(()).ok());
    assert!(sender.join().is_ok());
}
