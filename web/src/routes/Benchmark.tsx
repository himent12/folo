// One vision benchmark (`#/benchmarks/<id>`): its plan (editable here while no entry is done or under way), its
// entries and its loops. The parts are in ./bench/.
import ArrowLeft from '$icons/arrow-left';
import Copy from '$icons/copy';
import Download from '$icons/download';
import GitCompare from '$icons/git-compare-arrows';
import Pencil from '$icons/pencil';
import Plus from '$icons/plus';
import Repeat from '$icons/repeat';
import Sparkles from '$icons/sparkles';
import Square from '$icons/square';
import Trash from '$icons/trash-2';
import WifiOff from '$icons/wifi-off';
import { computed, css, effect, For, MediaQuery, onCleanup, onDocument, onWindow, signal, untrack } from 'fw';
import { ApiError, hub } from '$lib/api/index';
import { LiveBench } from '$lib/api/liveBench';
import { lastKnown, Now } from '$lib/api/freshness';
import { Resource } from '$lib/api/resource';
import type { BenchEntry, BenchLoop, BenchPlan } from '$lib/api/types';
import { clonePlan, samePlan, editedPlanError, planCondition, unreadableWeight } from '$lib/bench/plan';
import { isEntryActive, isLoopActive, planEditable } from '$lib/bench/rules';
import { media } from '$lib/design/tokens';
import { ago, date, num } from '$lib/format';
import { isTyping } from '$lib/keys';
import { href, router } from '$lib/router';
import Page from '$lib/shell/Page';
import { ownCreateAction, setCrumbs } from '$lib/shell/crumbs';
import { pageCommands, type PageCommand } from '$lib/shell/pageCommands';
import { gChord } from '$lib/shell/shortcuts';
import { exportTableCsv, exportTableJson } from '$lib/tableExport';
import Badge from '$lib/ui/Badge';
import Button from '$lib/ui/Button';
import Dialog from '$lib/ui/Dialog';
import ErrorState from '$lib/ui/ErrorState';
import { DialogStatus } from '$lib/ui/feedback';
import Gate from '$lib/ui/Gate';
import Ident from '$lib/ui/Ident';
import InfoButton from '$lib/ui/InfoButton';
import Menu, { type MenuItem } from '$lib/ui/Menu';
import Meta from '$lib/ui/Meta';
import Note from '$lib/ui/Note';
import { isOffline, OFFLINE_REASON } from '$lib/ui/offline';
import Skeleton from '$lib/ui/Skeleton';
import Tabs from '$lib/ui/Tabs';
import { toast } from '$lib/ui/toast';
import { AgentChoice } from './label/agentChoice';
import AddEntryDialog from './bench/AddEntryDialog';
import { useAddQuery } from './bench/addQuery';
import DesignDialog from './bench/DesignDialog';
import EditBenchDialog from './bench/EditBenchDialog';
import { leaderboardExport } from './bench/export';
import Leaderboard from './bench/Leaderboard.tsx';
import { boardOrder, comparePath, compareHint, compareReason, keepSelection } from './bench/leaderboard';
import LoopsTab from './bench/LoopsTab';
import NewLoopDialog from './bench/NewLoopDialog';
import PlanTab from './bench/PlanTab';
import { cannotBench } from './bench/support';
import Unsupported from './bench/Unsupported';
import { classesText, entriesText, FINGERPRINT_HELP, FROZEN, imagesText, shortFingerprint, sourceKind, sourcePath } from './bench/words';
import { tickingNow } from './runs/when';

type Tab = 'leaderboard' | 'conditions' | 'loops';

export default function Benchmark() {
  // Relative times ("made 3m ago") tick while the page is open (runs/when).
  const clock = tickingNow();
  const id = computed(() => router.route.params.id ?? '');
  const tab = computed<Tab>(() => {
    const t = router.route.query.get('tab');
    return t === 'conditions' || t === 'loops' ? t : 'leaderboard';
  });
  // `?tab=loops&loop=…` (the loops' first links) opens the loop's own page.
  effect(() => {
    const l = router.route.query.get('loop');
    if (l && id.get()) router.go(`/benchmarks/${encodeURIComponent(id.get())}/loops/${encodeURIComponent(l)}`, { replace: true });
  });

  const system = new Resource((s) => hub.api.system(s));
  const catalog = new Resource((s) => hub.api.benchCatalog(s));
  const agents = new Resource((s) => hub.api.labelAgents(s));
  const models = new Resource((s) => hub.api.models(s));
  const datasets = new Resource((s) => hub.api.datasets(s));
  const jobs = new Resource((s) => hub.api.labelJobs(s));
  const live = new LiveBench(() => id.get() || null);
  const choice = new AgentChoice(() => agents.data);
  const bench = computed(() => live.bench);
  const unsupported = computed(() => cannotBench(system.data, live.error));

  // The header's Add entry is this page's create action: the top bar's New run steps aside.
  ownCreateAction();
  effect(() => {
    setCrumbs({ label: 'Benchmarks', path: '/benchmarks' }, { label: bench.get()?.name ?? id.get() });
  });

  /** Replaces query params (null removes one) without adding history entries. */
  function setQuery(patch: Record<string, string | null>) {
    const q = new URLSearchParams(router.route.query);
    for (const [k, v] of Object.entries(patch)) {
      if (v === null) q.delete(k);
      else q.set(k, v);
    }
    const s = q.toString();
    router.go(`/benchmarks/${encodeURIComponent(id.peek())}${s ? `?${s}` : ''}`, { replace: true });
  }

  // ---------- the plan: editable here while no entry is done or under way, saved with Save plan ----------
  const draft = signal<BenchPlan>({ conditions: [] });
  // Fixed once an entry finished or is under way (failed and stopped ones graded nothing to keep comparable).
  const editable = computed(() => !!bench.get() && planEditable(live.entries));
  const dirty = computed(() => !!bench.get() && editable.get() && !samePlan(draft.get(), bench.get()!.plan));
  // A plan from the server (the first load, a save, another tab's change) replaces the editor's unless it has edits;
  // one saved elsewhere under the edits keeps them and says so (Save plan would replace it, and nothing else would tell).
  const planKey = computed(() => {
    const b = bench.get();
    return b ? `${b.id}:${JSON.stringify(b.plan)}` : '';
  });
  // The weight fields as typed (PlanTab): one emptied keeps the draft at the last number typed, so every save checks
  // them, the plan's Save plan and the leave dialog's alike. Another benchmark, Discard and a plan replaced from the
  // server start afresh; a condition turned off drops its field, and one that holds no number stays so (with its error)
  // across edits of other conditions. A number that is no longer the draft's weight (changed from outside the field) goes.
  const typed = signal<Record<string, string>>({});
  const hasUnreadable = computed(() => editable.get() && draft.get().conditions.some((c) => unreadableWeight(typed.get()[c.condition])));
  const unsaved = computed(() => dirty.get() || hasUnreadable.get());
  const saving = signal(false);
  const elsewhere = signal(false);
  // The plan this page's Save plan has sent and not yet heard back on: an update holding it is the page's own, any other
  // one that comes while it saves is another tab's (told by what it holds, not by when it comes).
  let sent: BenchPlan | null = null;
  let shownKey = '';
  let shownPlan: BenchPlan | null = null;
  effect(() => {
    const key = planKey.get();
    if (!key || key === shownKey) return;
    untrack(() => {
      const b = bench.get()!;
      const hadEdits = shownPlan !== null && (!samePlan(draft.get(), shownPlan) || hasUnreadable.get() || sent !== null);
      if (!shownKey.startsWith(`${b.id}:`) || !hadEdits) {
        draft.set(clonePlan(b.plan));
        // Replaced from the server: what was typed is for a plan no longer shown.
        typed.set({});
        elsewhere.set(false);
      } else if (dirty.get() && !(sent && samePlan(b.plan, sent))) elsewhere.set(true);
      shownKey = key;
      shownPlan = clonePlan(b.plan);
    });
  });
  effect(() => {
    if (!dirty.get()) untrack(() => elsewhere.set(false));
  });
  effect(() => {
    void id.get();
    untrack(() => typed.set({}));
  });
  effect(() => {
    const d = draft.get();
    untrack(() => {
      const t = typed.get();
      const kept = Object.entries(t).filter(([c, v]) => {
        const w = planCondition(d, c)?.weight;
        return w !== undefined && (unreadableWeight(v) || Number(v) === w);
      });
      if (kept.length !== Object.keys(t).length) typed.set(Object.fromEntries(kept));
    });
  });
  const planWhy = computed(() => (editable.get() ? editedPlanError(draft.get(), typed.get(), catalog.data?.conditions) : null));
  let mounted = true;
  onCleanup(() => (mounted = false));
  async function savePlan(): Promise<boolean> {
    const b = bench.peek();
    if (!b || saving.peek()) return false;
    const why = planWhy.peek();
    if (why) {
      // page-level: both Save plan buttons are held while this fails; the leave dialog closes on a refused save (answer).
      toast.error(new Error(why), 'Could not save the plan');
      return false;
    }
    saving.set(true);
    const benchId = b.id;
    const epoch = hub.epoch;
    const plan = clonePlan(draft.peek());
    sent = plan;
    try {
      // The answer holds the plan as the server keeps it (a weight rounded to a 32-bit float, a reason trimmed).
      const saved = (await live.act(() => hub.api.updateBench(benchId, { plan }))).plan;
      if (hub.epoch !== epoch || hub.mode === 'connecting') return false;
      const now = bench.peek();
      if (!mounted || !now || now.id !== benchId) {
        // page-level: the benchmark page unmounted or switched benchmarks while the save was in flight.
        toast.success('Plan saved', `${plan.conditions.length} conditions. It is fixed once the first entry runs.`);
        return false;
      }
      if (!samePlan(now.plan, saved)) {
        // Another tab saved after this save landed (its event came before this answer): the server holds that plan, so
        // the edits stay edits and the page says so, here and at every Save plan.
        if (dirty.peek()) elsewhere.set(true);
        // page-level: the save did land, then was replaced; Plan saved would say the wrong thing.
        toast.info('Plan saved, then changed elsewhere', 'Another tab or window saved its plan after yours. Save plan replaces it with yours; Discard shows it.');
        // Not saved as it stands: the leave dialog stays on the page rather than leave the edits behind.
        return false;
      }
      // Server now holds our own saved plan: clear any prior conflict and mark it seen so the upcoming planKey flush
      // does not mistake our save for another tab's update after `sent` clears in `finally`.
      elsewhere.set(false);
      shownKey = `${benchId}:${JSON.stringify(now.plan)}`;
      shownPlan = clonePlan(now.plan);
      if (!samePlan(draft.peek(), plan) || hasUnreadable.peek()) {
        // Edits were made while this save was in flight: keep them in the draft and leave them marked unsaved.
        // page-level: the plan's own Save, outside any dialog.
        toast.info('Earlier plan saved', 'Edits made while saving are still unsaved.');
        return false;
      }
      // Shown as kept: a field rounded by the server shows the number it holds.
      draft.set(clonePlan(saved));
      typed.set({});
      // page-level: the plan's own Save, outside any dialog.
      toast.success('Plan saved', `${plan.conditions.length} conditions. It is fixed once the first entry runs.`);
      return true;
    } catch (e) {
      if (hub.epoch !== epoch || hub.mode === 'connecting') return false;
      // page-level: the plan's own Save, or the leave dialog closes on a refused save in answer.
      toast.error(e, 'Could not save the plan');
      return false;
    } finally {
      sent = null;
      saving.set(false);
    }
  }
  const discard = () => {
    const b = bench.peek();
    if (!b || saving.peek()) return;
    draft.set(clonePlan(b.plan));
    typed.set({});
  };

  // Leaving with unsaved edits asks first, whichever way (a link, Back, the palette); a change of tab does not.
  const leaving = signal<{ proceed: () => void } | null>(null);
  effect(() => {
    const off = router.guard((to, proceed) => {
      if (!unsaved.get() || to.path === router.route.path) return true;
      leaving.set({ proceed });
      return false;
    });
    return () => void off();
  });
  async function answer(keep: 'save' | 'discard') {
    if (saving.peek()) return;
    const ask = leaving.peek();
    const startPath = router.location.path;
    if (keep === 'save' && !(await savePlan())) {
      if (mounted) leaving.set(null);
      return;
    }
    if (keep === 'discard') discard();
    if (mounted) leaving.set(null);
    // The editor is clean now, so the guard lets it through (unless the page unmounted or moved on).
    if (mounted && router.location.path === startPath) {
      queueMicrotask(() => {
        if (mounted && router.location.path === startPath) ask?.proceed();
      });
    }
  }
  onWindow('beforeunload', (e: BeforeUnloadEvent) => {
    if (unsaved.peek()) e.preventDefault();
  });

  // ---------- entries ----------
  const selected = signal<string[]>([]);
  effect(() => {
    const sel = selected.get();
    const kept = keepSelection(sel, live.entries);
    if (kept.length !== sel.length) untrack(() => selected.set(kept));
  });
  const comparing = computed(() => selected.get().length >= 2);
  function clearSelection() {
    selected.set([]);
    (document.getElementById(`bn-tab-${tab.peek()}`) ?? document.querySelector<HTMLElement>('h1[data-route-focus]'))?.focus();
  }
  // Escape clears Compare selection before global Escape navigates back to /benchmarks (never intercepting G-chord or popovers).
  const hasOpenPopover = () => {
    try {
      return !!document.querySelector('[popover]:popover-open');
    } catch {
      return false;
    }
  };
  onDocument('keydown', (e) => {
    const onCheckbox = e.target instanceof HTMLInputElement && e.target.type === 'checkbox';
    if (e.key !== 'Escape' || e.repeat || e.defaultPrevented || e.isComposing || e.ctrlKey || e.metaKey || e.altKey || e.shiftKey || (!onCheckbox && isTyping(e))) return;
    if (tab.peek() !== 'leaderboard' || !selected.peek().length) return;
    if (gChord.peek() || hasOpenPopover() || document.querySelector('dialog[open], [aria-expanded="true"][aria-haspopup], [role="menu"], #folo-tooltip[data-show], .info-pop.open')) return;
    e.preventDefault();
    selected.set([]);
    queueMicrotask(() => {
      if (!document.activeElement || document.activeElement === document.body) {
        document.querySelector<HTMLElement>('h1[data-route-focus]')?.focus();
      }
    });
  });
  /** The entries as the leaderboard shows them (one graded on an earlier version after the current ones): Compare's order. */
  const board = computed(() => {
    const b = bench.get();
    return b ? boardOrder(live.entries, b.fingerprint) : live.entries;
  });
  let closeAdd: () => boolean = () => true;
  const { addOpen, addKind, openAdd } = useAddQuery(
    () => setQuery({ add: null }),
    () => closeAdd(),
  );
  function started(e: BenchEntry) {
    live.putEntry(e, live.mark());
    if (tab.peek() !== 'leaderboard') setQuery({ tab: null });
    // page-level: the dialog has closed (AddEntryDialog closes before it calls this).
    toast.success(
      e.status === 'done' ? 'Entry scored' : 'Entry added',
      e.status === 'done' && e.score
        ? `${e.name} · score ${e.score.total.toFixed(1)}`
        : `${e.name} · ${e.status === 'queued' ? (e.contestant.kind === 'agent' ? 'starting' : 'waiting for the GPU') : 'running'}`,
    );
  }

  const stopping = signal<BenchEntry | null>(null);
  const stopBusy = signal(false);
  const stopStatus = new DialogStatus();
  async function stopEntry() {
    const b = bench.peek();
    const e = stopping.peek();
    if (!b || !e || stopBusy.peek()) return;
    const targetId = e.id;
    stopBusy.set(true);
    const mark = live.mark();
    try {
      live.putEntry(await hub.api.stopBenchEntry(b.id, targetId), mark);
      if (mounted && stopping.peek()?.id === targetId) stopping.set(null);
    } catch (err) {
      if (mounted && stopping.peek()?.id === targetId) stopStatus.error(err, 'Could not stop the entry');
      // page-level: the stop confirm is closed or unmounted.
      else toast.error(err, 'Could not stop the entry');
    } finally {
      stopBusy.set(false);
    }
  }
  const removing = signal<BenchEntry | null>(null);
  const removeBusy = signal(false);
  const removeStatus = new DialogStatus();
  async function removeEntry() {
    const b = bench.peek();
    const e = removing.peek();
    if (!b || !e || removeBusy.peek()) return;
    const targetId = e.id;
    removeBusy.set(true);
    try {
      await hub.api.deleteBenchEntry(b.id, targetId);
      live.dropEntry(targetId);
      if (mounted && removing.peek()?.id === targetId) removing.set(null);
      // page-level: its confirm dialog has closed on the line above.
      toast.success(`Deleted ${e.name}`, 'Its result is gone from the leaderboard. The model and its files were not touched.');
    } catch (err) {
      if (mounted && removing.peek()?.id === targetId) removeStatus.error(err, 'Could not delete the entry');
      // page-level: the delete confirm is closed or unmounted.
      else toast.error(err, 'Could not delete the entry');
    } finally {
      removeBusy.set(false);
    }
  }

  // ---------- the benchmark ----------
  const editOpen = signal(false);
  const designOpen = signal(false);
  const deleteOpen = signal(false);
  const deleteBusy = signal(false);
  const deleteStatus = new DialogStatus();
  // The server's own word (`Bench.active`, exactly when DELETE answers 409 bench_active), and what the page has heard since.
  const busyWhy = computed(() =>
    bench.get()?.active || live.entries.some(isEntryActive) || live.loops.some(isLoopActive) ? 'Stop its running entries and loops first' : null,
  );

  // ---------- loops ----------
  const loopOpen = signal(false);
  function loopStarted(l: BenchLoop) {
    live.putLoop(l, live.mark());
    // page-level: the dialog has closed.
    toast.success(`Loop ${l.name} started`, 'Its first generation is under way.', {
      label: 'Open loop',
      run: () => router.go(`/benchmarks/${encodeURIComponent(id.peek())}/loops/${encodeURIComponent(l.id)}`),
    });
  }
  async function remove() {
    const b = bench.peek();
    if (!b || deleteBusy.peek()) return;
    const startPath = router.location.path;
    deleteBusy.set(true);
    try {
      await hub.api.deleteBench(b.id);
      if (mounted) deleteOpen.set(false);
      toast.success(`Deleted ${b.name}`, 'Its images, entries and loops are gone. Models, runs and datasets were not touched.');
      if (mounted && router.location.path === startPath) router.go('/benchmarks', { replace: true });
    } catch (e) {
      if (mounted && deleteOpen.peek()) deleteStatus.error(e, 'Could not delete the benchmark');
      else toast.error(e, 'Could not delete the benchmark');
    } finally {
      deleteBusy.set(false);
    }
  }
  async function copyFingerprint() {
    const b = bench.peek();
    if (!b) return;
    // page-level: the header's ⋯ and the fingerprint's own button, outside any dialog.
    try {
      await navigator.clipboard.writeText(b.fingerprint);
      toast.success('Fingerprint copied', b.fingerprint);
    } catch {
      toast.error(new Error('The browser blocked clipboard access.'), 'Could not copy the fingerprint');
    }
  }

  // ---------- the header ----------
  const phone = new MediaQuery(media.phone, false);
  const now = new Now(() => live.lost !== null);
  const offline = computed(() => isOffline(live.error) || isOffline(live.resyncError));
  const addWhy = computed(() => (offline.get() ? OFFLINE_REASON : unsaved.get() ? 'Save or discard the plan’s changes first' : null));
  const designWhy = computed(() => (!bench.get() ? null : !editable.get() ? FROZEN : null));
  /** The source, linked while the server still has it; a benchmark keeps its own copy either way. */
  const sourceKnown = computed(() => {
    const b = bench.get();
    if (!b) return null;
    const s = b.source;
    if (s.kind === 'dataset') return datasets.data ? datasets.data.some((d) => d.id === s.dataset_id) : null;
    return jobs.data ? jobs.data.some((j) => j.id === s.job_id) : jobs.error ? false : null;
  });
  const hasEntries = computed(() => live.entries.length > 0);
  function exportBoard(format: 'csv' | 'json') {
    const b = bench.peek();
    const entries = board.peek();
    if (!b || !entries.length) return;
    const payload = leaderboardExport(b, entries, agents.data);
    const ok =
      format === 'csv'
        ? exportTableCsv(payload.filenameBase, payload.headers, payload.rows)
        : exportTableJson(payload.filenameBase, payload.json);
    // page-level: triggered from the header overflow menu or command palette, outside any dialog.
    if (ok) toast.success(`Exported ${format.toUpperCase()}`, `${entries.length} ${entries.length === 1 ? 'entry' : 'entries'} from ${b.name}.`);
  }

  const menu = computed<MenuItem[]>(() => [
    { label: 'Edit name and description', icon: Pencil, onclick: () => editOpen.set(true) },
    { label: 'Copy fingerprint', icon: Copy, onclick: () => void copyFingerprint() },
    ...(hasEntries.get()
      ? [
          { label: 'Export leaderboard CSV', icon: Download, onclick: () => exportBoard('csv') },
          { label: 'Export leaderboard JSON', icon: Download, onclick: () => exportBoard('json') },
        ]
      : []),
    { label: 'Delete benchmark', icon: Trash, danger: true, onclick: () => deleteOpen.set(true) },
  ]);

  pageCommands((): PageCommand[] => {
    const b = bench.get();
    if (!b) return [];
    const out: PageCommand[] = [];
    if (!addWhy.get())
      out.push({ id: 'bench-add', label: 'Add entry', match: ['New entry', 'Benchmark a model'], icon: Plus, run: () => !addWhy.peek() && openAdd('model') });
    if (comparing.get()) out.push({ id: 'bench-compare', label: 'Compare selected entries', icon: GitCompare, run: () => router.go(comparePath(b.id, selected.peek(), board.peek())) });
    if (editable.get())
      out.push({ id: 'bench-design', label: 'Design the plan with an agent', icon: Sparkles, run: () => (setQuery({ tab: 'conditions' }), designOpen.set(true)) });
    for (const e of live.entries.filter((x) => x.status === 'running' || x.status === 'queued'))
      out.push({ id: `bench-stop-${e.id}`, label: `Stop entry ${e.name}`, icon: Square, run: () => stopping.set(e) });
    if (!offline.get())
      out.push({ id: 'bench-loop-new', label: 'New self-improvement loop', match: ['New loop'], icon: Repeat, run: () => (setQuery({ tab: 'loops' }), loopOpen.set(true)) });
    if (hasEntries.get()) {
      out.push({ id: 'bench-export-csv', label: 'Export leaderboard CSV', icon: Download, run: () => exportBoard('csv') });
      out.push({ id: 'bench-export-json', label: 'Export leaderboard JSON', icon: Download, run: () => exportBoard('json') });
    }
    out.push({ id: 'bench-fingerprint', label: 'Copy fingerprint', icon: Copy, run: () => void copyFingerprint() });
    return out;
  });

  const b = () => bench.get()!;
  const addButton = (why: () => string | undefined) => (
    <Button
      variant={comparing.get() ? 'secondary' : 'primary'}
      icon={Plus}
      disabled={!!why()}
      aria-describedby={why()}
      onclick={() => !addWhy.peek() && openAdd('model')}
    >
      Add entry
    </Button>
  );

  return (
    <>
      {unsupported.get() ? (
        <Page title="Benchmarks">
          <Unsupported />
        </Page>
      ) : live.error && !bench.get() ? (
        <Page
          title="Benchmark"
          actions={() => (
            <Button icon={ArrowLeft} href={href('/benchmarks')}>
              All benchmarks
            </Button>
          )}
        >
          {live.error instanceof ApiError && live.error.status === 404 ? (
            <ErrorState error={live.error} title="Benchmark not found" />
          ) : (
            <ErrorState error={live.error} title="Could not load this benchmark" retry={() => live.retry()} />
          )}
        </Page>
      ) : !bench.get() || live.loading ? (
        // Until the lists load too: the stream's first `bench` event can come first, and an empty list is no board.
        <div class="sk-page" aria-busy="true" aria-label="Loading benchmark">
          <div class="sk-head">
            <Skeleton w="min(240px, 60%)" h="24px" />
            <Skeleton w="min(420px, 90%)" h="14px" />
          </div>
          <div class="sk-head">
            <Skeleton w="min(500px, 90%)" h="12px" />
            <Skeleton w="min(380px, 70%)" h="12px" />
          </div>
          <div class="sk-tabs">
            <Skeleton w="96px" h="14px" />
            <Skeleton w="88px" h="14px" />
            <Skeleton w="56px" h="14px" />
          </div>
          {/* The leaderboard's frame: its heading, then rows of a name over its facts, a score and the axes. */}
          <div class="sk-board">
            <div class="sk-title">
              <Skeleton w="96px" h="14px" />
            </div>
            <For each={[0, 1, 2, 3, 4]} key={(i) => i}>
              {() => (
                <div class="sk-row">
                  <span class="sk-name">
                    <Skeleton w="140px" h="12px" />
                    <Skeleton w="100px" h="10px" />
                  </span>
                  <Skeleton w="40px" h="16px" />
                  <Skeleton w="66px" h="20px" />
                  <span class="sk-plot">
                    <Skeleton h="6px" />
                  </span>
                </div>
              )}
            </For>
          </div>
        </div>
      ) : (
        <>
          <Page
            title={b().name}
            inlineActions
            lead={() => (
              <>
                {live.connection === 'reconnecting' ? (
                  <Badge tone="warning">
                    <WifiOff size={12} /> {live.lost !== null ? lastKnown('Reconnecting', live.lost, now.current) : 'Reconnecting'}
                  </Badge>
                ) : null}
              </>
            )}
            sub={() => (
              <Meta>
                <span>
                  {sourceKnown.get() ? (
                    <a class="link" href={href(sourcePath(b().source))}>
                      {b().source_name}
                    </a>
                  ) : (
                    <>
                      <span title={sourceKnown.get() === false ? 'No longer on this server; the benchmark keeps its own copy of the images.' : undefined}>
                        {b().source_name}
                      </span>
                      {sourceKnown.get() === false ? <Note text="deleted since" /> : null}
                    </>
                  )}
                </span>
                <span>{sourceKind(b().source)}</span>
                <span class="num">{imagesText(b().images)}</span>
                <span class="num">{num(b().boxes)} boxes</span>
                <span class="num">{classesText(b().classes.length)}</span>
                <span class="faint" title={date(b().created)}>
                  made {ago(b().created, clock.get())}
                </span>
              </Meta>
            )}
            subPhone={() => (
              <span class="num">
                {imagesText(b().images)} · {entriesText(b().entries)}
              </span>
            )}
            actions={() => (
              <>
                {phone.current ? (
                  <>
                    <Gate reason={addWhy.get()}>{addButton}</Gate>
                    <Menu label="More benchmark actions" items={menu.get()} />
                  </>
                ) : (
                  <>
                    {live.entries.length >= 2 ? (
                      <>
                        {tab.get() === 'leaderboard' ? (
                          <span class="cmp-hint" role="status" data-gate-why>
                            {compareHint(selected.get().length)}
                          </span>
                        ) : null}
                        {selected.get().length > 0 ? (
                          <Button variant="ghost" size="sm" onclick={clearSelection} aria-label="Clear selection">
                            Clear
                          </Button>
                        ) : null}
                        <Gate reason={compareReason(selected.get().length)}>
                          {(why) => (
                            <Button
                              variant={comparing.get() ? 'primary' : 'secondary'}
                              icon={GitCompare}
                              disabled={!!why()}
                              aria-describedby={why()}
                              href={comparing.get() ? href(comparePath(b().id, selected.get(), board.get())) : undefined}
                            >
                              Compare{selected.get().length ? <span class="num count">{selected.get().length}</span> : null}
                            </Button>
                          )}
                        </Gate>
                      </>
                    ) : null}
                    {addWhy.get() && addWhy.get() !== OFFLINE_REASON ? (
                      <span class="cmp-hint" data-gate-why>
                        {addWhy.get()}
                      </span>
                    ) : null}
                    <Gate reason={addWhy.get()}>{addButton}</Gate>
                    <Menu label="More benchmark actions" items={menu.get()} />
                  </>
                )}
              </>
            )}
          >
            <div class="about">
              {b().description.trim() ? (
                <p class="desc">{b().description}</p>
              ) : (
                <p class="desc none">
                  No description yet. Say where the camera is and what makes the job hard: an agent designing the plan reads it.{' '}
                  <button type="button" class="inline" onclick={() => editOpen.set(true)}>
                    Add a description
                  </button>
                </p>
              )}
              <div class="fp">
                <InfoButton label="What the fingerprint is for" trigger={() => <>Fingerprint</>} text={FINGERPRINT_HELP} />
                <span title={b().fingerprint}>
                  <Ident value={shortFingerprint(b().fingerprint)} />
                </span>
                <Button variant="ghost" size="sm" icon={Copy} aria-label="Copy the fingerprint" onclick={copyFingerprint} />
              </div>
            </div>
            <div class="tabs">
              <Tabs
                label="Benchmark views"
                idPrefix="bn-tab"
                tabs={[
                  { value: 'leaderboard', label: 'Leaderboard', meta: String(live.entries.length) },
                  { value: 'conditions', label: 'Conditions', meta: String(b().plan.conditions.length) },
                  { value: 'loops', label: 'Loops', meta: String(live.loops.length) },
                ]}
                value={tab.get()}
                onchange={(t) => setQuery({ tab: t === 'leaderboard' ? null : t })}
              />
            </div>
            <div id="bn-tab-panel" role="tabpanel" aria-labelledby={`bn-tab-${tab.get()}`} class="panel">
              {tab.get() === 'conditions' ? (
                <>
                  {catalog.data ? (
                    <PlanTab
                      bench={b()}
                      editable={editable.get()}
                      catalog={catalog.data}
                      agents={agents.data}
                      bind:draft={draft}
                      bind:typed={typed}
                      error={planWhy.get()}
                      elsewhere={elsewhere.get()}
                      dirty={dirty.get()}
                      hasUnreadable={hasUnreadable.get()}
                      saving={saving.get()}
                      onsave={() => void savePlan()}
                      ondiscard={discard}
                      ondesign={() => designOpen.set(true)}
                      designWhy={designWhy.get()}
                    />
                  ) : catalog.error ? (
                    <ErrorState error={catalog.error} title="Could not load the conditions" retry={() => catalog.refresh()} />
                  ) : (
                    <Skeleton h="480px" r="var(--radius-lg)" />
                  )}
                </>
              ) : tab.get() === 'loops' ? (
                <LoopsTab
                  bench={b()}
                  loops={live.loops}
                  entries={live.entries}
                  datasets={datasets.data}
                  jobs={jobs.data}
                  onnew={() => loopOpen.set(true)}
                  offline={offline.get()}
                />
              ) : (
                <Leaderboard
                  bench={b()}
                  entries={live.entries}
                  catalog={catalog.data}
                  agents={agents.data}
                  bind:selected={selected}
                  lost={live.lost !== null}
                  addWhy={addWhy.get()}
                  onadd={() => !addWhy.peek() && openAdd('model')}
                  onstop={(e) => stopping.set(e)}
                  ondelete={(e) => removing.set(e)}
                  oncompare={() => router.go(comparePath(b().id, selected.peek(), board.peek()))}
                />
              )}
            </div>
          </Page>
          <AddEntryDialog
            bind:open={addOpen}
            bind:kind={addKind}
            bench={b()}
            catalog={catalog.data}
            entries={live.entries}
            models={models.data}
            modelsError={models.error}
            agents={agents.data}
            agentsError={agents.error}
            retryAgents={() => agents.refresh()}
            choice={choice}
            blockedReason={addWhy.get()}
            registerClose={(fn) => (closeAdd = fn)}
            onstarted={started}
          />
          {catalog.data ? (
            <DesignDialog
              bind:open={designOpen}
              bench={b()}
              catalog={catalog.data}
              current={draft.get()}
              agents={agents.data}
              agentsError={agents.error}
              retryAgents={() => agents.refresh()}
              choice={choice}
              onuse={(p) => {
                // A proposal replaces the plan: fields typed for the old one (an emptied weight) go with it.
                typed.set({});
                draft.set(clonePlan(p));
              }}
            />
          ) : null}
          <NewLoopDialog
            bind:open={loopOpen}
            bench={b()}
            models={models.data}
            datasets={datasets.data}
            jobs={jobs.data}
            entries={live.entries}
            onstarted={loopStarted}
          />
          <EditBenchDialog bind:open={editOpen} bench={b()} onsave={(change) => live.act(() => hub.api.updateBench(b().id, change))} />
          <Dialog
            open={!!stopping.get()}
            onclose={() => stopping.set(null)}
            onbeforeclose={() => !stopBusy.peek()}
            title={`Stop ${stopping.get()?.name ?? 'the entry'}?`}
            description="It ends what is in flight. A stopped entry has no score: what it ran so far is not kept."
            size="sm"
            status={stopStatus}
            footer={() => (
              <>
                <Button disabled={stopBusy.get()} onclick={() => !stopBusy.peek() && stopping.set(null)}>Keep running</Button>
                <Button variant="destructive" icon={Square} loading={stopBusy.get()} onclick={stopEntry}>
                  Stop entry
                </Button>
              </>
            )}
          />
          <Dialog
            open={!!removing.get()}
            onclose={() => removing.set(null)}
            onbeforeclose={() => !removeBusy.peek()}
            title={`Delete ${removing.get()?.name ?? 'the entry'}?`}
            description="Removes the entry and its result from the leaderboard. The model, its files and the benchmark's images are not touched."
            size="sm"
            status={removeStatus}
            footer={() => (
              <>
                <Button disabled={removeBusy.get()} onclick={() => !removeBusy.peek() && removing.set(null)}>Cancel</Button>
                <Button variant="destructive" icon={Trash} loading={removeBusy.get()} onclick={removeEntry}>
                  Delete entry
                </Button>
              </>
            )}
          />
          <Dialog
            bind:open={deleteOpen}
            onbeforeclose={() => !deleteBusy.peek()}
            title={`Delete ${b().name}?`}
            description="Removes the benchmark, its copy of the images, its entries and its loops. Models, runs and datasets (a loop's included) are not touched."
            size="sm"
            status={deleteStatus}
            footer={() => (
              <>
                {busyWhy.get() ? (
                  <p class="why" data-gate-why>
                    {busyWhy.get()}
                  </p>
                ) : null}
                <Button disabled={deleteBusy.get()} onclick={() => !deleteBusy.peek() && deleteOpen.set(false)}>Cancel</Button>
                <Gate reason={busyWhy.get()}>
                  {(why) => (
                    <Button variant="destructive" icon={Trash} loading={deleteBusy.get()} disabled={!!why()} aria-describedby={why()} onclick={remove}>
                      Delete benchmark
                    </Button>
                  )}
                </Gate>
              </>
            )}
          />
          <Dialog
            open={!!leaving.get()}
            title="Save the plan?"
            description={`${b().name}'s plan has changes you have not saved.${
              elsewhere.get() ? ' It was also saved from another tab or window while you edited: Save plan replaces that with yours; Discard keeps that one.' : ''
            }`}
            size="sm"
            onclose={() => leaving.set(null)}
            onbeforeclose={() => !saving.peek()}
            footer={() => (
              <>
                <Button disabled={saving.get()} onclick={() => answer('discard')}>Discard changes</Button>
                {planWhy.get() ? (
                  <p class="why" data-gate-why>
                    {planWhy.get()}
                  </p>
                ) : null}
                <Gate reason={planWhy.get() ? 'Fix the plan first' : null}>
                  {(why) => (
                    <Button variant="primary" loading={saving.get()} disabled={!!why()} aria-describedby={why()} onclick={() => answer('save')}>
                      Save plan
                    </Button>
                  )}
                </Gate>
              </>
            )}
          />
        </>
      )}
    </>
  );
}

css`
  .count {
    font-weight: var(--weight-regular);
    margin-left: var(--space-1);
  }
  .cmp-hint {
    align-self: center;
    white-space: nowrap;
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
  }
  .link {
    color: var(--text-primary);
    text-decoration: underline;
    text-decoration-color: var(--line-strong);
    text-underline-offset: 3px;
  }
  /* What the benchmark stands for, then its fingerprint: one band under the header. */
  .about {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: var(--space-3) var(--space-8);
    margin-top: calc(-1 * var(--space-2));
  }
  .desc {
    max-inline-size: var(--measure-dialog);
    color: var(--text-secondary);
  }
  .desc.none {
    color: var(--text-tertiary);
  }
  .inline {
    color: var(--text-primary);
    text-decoration: underline;
    text-decoration-color: var(--line-strong);
    text-underline-offset: 3px;
  }
  .fp {
    flex-shrink: 0;
    display: flex;
    align-items: center;
    gap: var(--space-2);
    height: var(--control-md);
    font-size: var(--type-sm-size);
    color: var(--text-secondary);
  }
  .tabs {
    margin-bottom: calc(var(--space-1) * -1);
  }
  .panel {
    display: grid;
    gap: var(--space-5);
    min-width: 0;
  }
  .why {
    margin-right: auto;
    align-self: center;
    font-size: var(--type-sm-size);
    color: var(--text-secondary);
  }
  @media (min-width: 768px) and (max-width: 1023px) {
    .cmp-hint {
      display: none;
    }
  }
  @media (max-width: 1023px) {
    .about {
      flex-direction: column;
      gap: var(--space-2);
    }
  }
  .sk-page {
    max-width: calc(var(--size-content-max) + 2 * var(--space-6));
    margin: 0 auto;
    padding: var(--space-6);
    display: grid;
    gap: var(--space-5);
  }
  .sk-head {
    display: grid;
    gap: var(--space-2);
  }
  .sk-tabs {
    display: flex;
    gap: var(--space-5);
    padding-bottom: var(--space-3);
    border-bottom: 1px solid var(--line-subtle);
  }
  .sk-board {
    border: 1px solid var(--line-panel);
    border-radius: var(--radius-lg);
    background: var(--bg-panel);
  }
  .sk-title {
    display: flex;
    align-items: center;
    height: 40px;
    padding: 0 var(--space-4);
  }
  .sk-row {
    display: grid;
    grid-template-columns: minmax(0, 1fr) 40px 66px minmax(0, 200px);
    align-items: center;
    gap: var(--space-6);
    height: 48px;
    padding: 0 var(--space-4);
    border-top: 1px solid var(--line-subtle);
  }
  .sk-name {
    display: grid;
    gap: var(--space-2);
  }
  @media (max-width: 767px) {
    .sk-row {
      grid-template-columns: minmax(0, 1fr) 40px;
      height: 56px;
    }
    .sk-row > :global(:nth-child(3)),
    .sk-plot {
      display: none;
    }
  }
  @media (max-width: 767px) {
    .sk-page {
      padding: var(--space-4);
    }
    .why {
      flex-basis: 100%;
    }
  }
`;
