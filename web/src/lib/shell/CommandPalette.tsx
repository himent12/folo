/** Ctrl/Cmd+K: jump anywhere, run any action, find any run, dataset or model. */
import Activity from '$icons/activity';
import Box from '$icons/box';
import GitCompare from '$icons/git-compare-arrows';
import Images from '$icons/images';
import Keyboard from '$icons/keyboard';
import PanelLeft from '$icons/panel-left';
import Plus from '$icons/plus';
import ScanSearch from '$icons/scan-search';
import Search from '$icons/search';
import SettingsIcon from '$icons/settings';
import SunMoon from '$icons/sun-moon';
import Tags from '$icons/tags';
import Target from '$icons/target';
import X from '$icons/x';
import { bindable, computed, css, Dynamic, effect, For, Index, signal } from 'fw';
import { hub } from '$lib/api/index';
import type { Bench, Dataset, Model, Run, RunStatus } from '$lib/api/types';
import { failureCause } from '$lib/oom';
import { prefs } from '$lib/prefs';
import { router } from '$lib/router';
import { scoreContext, scoreText } from '$lib/scoreSummary';
import { rememberFocus } from '$lib/ui/focus';
import Ident from '$lib/ui/Ident';
import Kbd from '$lib/ui/Kbd';
import { keyPrefs } from '$lib/ui/keyPrefs';
import { backdropClose } from '$lib/ui/backdrop';
import Meta from '$lib/ui/Meta';
import { statusWord } from '$lib/ui/StatusBadge';
import type { IconType } from '$lib/ui/types';
import { highlight, rank } from './commands';
import { mainNav, settingsNav } from './nav';
import { pageCommandList, recentIds, rememberRecent } from './pageCommands';
import { touchOnly as isTouchOnly } from './shortcuts';
import { sidebar } from './sidebarState';

interface Item {
  id: string;
  group: string;
  label: string;
  /** Facts about the entity, one meta item each; `{ ident }` is an identifier (an architecture), set mono (ui/Ident). */
  hint?: string | (string | { ident: string })[];
  /**
   * The hint restates what the label already says (an action's description), where other hints are facts about the
   * entity (status, model, size). Phones, short of width, leave it out instead of cutting it to a fragment.
   */
  describes?: boolean;
  keys?: string;
  icon: IconType;
  /** Other phrasings it answers to (commands.rank). */
  match?: string[];
  /** An action about one run, dataset or model: offered once the query names it. */
  scoped?: boolean;
  /** A Recent entry: the id of the item it repeats (what is remembered when it is chosen). */
  origin?: string;
  /** A run's status, from the API's enum: what filters read, never the hint's display word. */
  status?: RunStatus;
  run: () => void;
}

/** Settings' sections (routes/Settings: `?section=`), with what each holds, as people may look for it. */
const SETTINGS_SECTIONS: { id: string; label: string; match: string[] }[] = [
  { id: 'connection', label: 'Connection', match: ['Server URL', 'Server address', 'folo serve'] },
  { id: 'cloud', label: 'Cloud', match: ['Cloud compute', 'Connect a cloud', 'RunPod', 'Rented GPU'] },
  { id: 'appearance', label: 'Appearance', match: ['Theme', 'Dark theme', 'Light theme'] },
  { id: 'units', label: 'Units', match: ['Memory units', 'GiB', 'GB'] },
  { id: 'shortcuts', label: 'Keyboard shortcuts', match: ['Single-key shortcuts'] },
  { id: 'system', label: 'System', match: ['GPU', 'Device', 'Driver', 'Throughput'] },
  { id: 'about', label: 'About', match: ['Version'] },
];

export interface CommandPaletteProps {
  open?: boolean;
  onshortcuts: () => void;
}

export default function CommandPalette(props: CommandPaletteProps) {
  const open = bindable(props, 'open', false);
  const touchOnly = isTouchOnly();

  let el!: HTMLDialogElement;
  const input = signal<HTMLInputElement>();
  const list = signal<HTMLUListElement>();
  const query = signal('');
  const index = signal(0);
  const runs = signal<Run[]>([]);
  const datasets = signal<Dataset[]>([]);
  const models = signal<Model[]>([]);
  /** Benchmarks, on servers that have them (a server without answers 501 or 404: none listed). */
  const benches = signal<Bench[]>([]);
  const benchesOk = signal(false);
  /** The items chosen last (ids), read on open: with no query they head the list, under Recent. */
  const recent = signal<string[]>([]);

  // A press outside the box closes it, unless it is the second click of the double-click that opened it (ui/backdrop).
  const backdrop = backdropClose(() => el, () => el.close());
  // Focus returns to the opener on close; a command that navigates then hands it to the new page (App).
  let restore: (() => void) | null = null;
  effect(() => {
    if (open.get() && !el.open) {
      restore = rememberFocus();
      el.showModal();
      backdrop.opened();
      query.set('');
      index.set(0);
      recent.set(recentIds());
      queueMicrotask(() => input.peek()?.focus());
      const api = hub.api;
      api.runs().then((r) => runs.set(r), () => {});
      api.datasets().then((d) => datasets.set(d), () => {});
      api.models().then((m) => models.set(m), () => {});
      api.benches().then(
        (b) => (benches.set(b), benchesOk.set(true)),
        () => (benches.set([]), benchesOk.set(false)),
      );
    } else if (!open.get() && el.open) el.close();
  });

  const go = (path: string) => () => router.go(path);
  const ACTIVE: RunStatus[] = ['running', 'stopping', 'queued'];
  /** How long typing must pause before the result count is announced. */
  const ANNOUNCE_MS = 500;

  const items = computed((): Item[] => {
    // What the page on screen offers (shell/pageCommands): first, under This page.
    const page: Item[] = pageCommandList.items.map((c) => ({
      id: `page-${c.id}`,
      group: 'This page',
      label: c.label,
      match: c.match,
      keys: c.keys,
      icon: c.icon,
      run: c.run,
    }));
    const nav: Item[] = [...mainNav, settingsNav].map((n) => ({
      id: `nav-${n.id}`,
      group: 'Go to',
      label: n.label,
      keys: n.keys,
      icon: n.icon,
      run: go(n.path),
    }));
    const actions: Item[] = [
      { id: 'new-run', group: 'Actions', label: 'New training run', keys: 'N', icon: Plus, run: go('/runs/new') },
      { id: 'compare', group: 'Actions', label: 'Compare runs', icon: GitCompare, run: go('/runs/compare') },
      // Agent labeling, under Datasets (no nav item of its own).
      { id: 'label', group: 'Actions', label: 'Label images', match: ['Labeling jobs', 'Claude Code', 'Codex'], icon: Tags, run: go('/datasets/label') },
      { id: 'label-new', group: 'Actions', label: 'New labeling job', icon: Plus, run: go('/datasets/label?new=1') },
      // Vision benchmarks: offered where the server has them (its list answered).
      ...(benchesOk.get()
        ? [{ id: 'bench-new', group: 'Actions', label: 'New benchmark', match: ['Benchmark a model', 'Vision benchmark'], icon: Plus, run: go('/benchmarks?new=1') }]
        : []),
      {
        id: 'theme',
        group: 'Actions',
        label: `Switch theme (now ${prefs.theme})`,
        match: ['Toggle theme', 'Dark theme', 'Light theme'],
        icon: SunMoon,
        run: () => prefs.cycleTheme(),
      },
      // Touch-only devices have no keys to look up, and a phone or tablet held in the hand no sidebar to fold.
      ...(touchOnly
        ? []
        : [
            {
              id: 'sidebar',
              group: 'Actions',
              label: sidebar.collapsed ? 'Expand sidebar' : 'Collapse sidebar',
              keys: '[',
              icon: PanelLeft,
              run: () => sidebar.toggle(),
            },
            { id: 'shortcuts', group: 'Actions', label: 'Keyboard shortcuts', keys: '?', icon: Keyboard, run: () => props.onshortcuts() },
          ]),
    ];
    // Settings' sections, by name and by what they hold ("cloud", "theme", "server URL"): for a query only.
    const settings: Item[] = SETTINGS_SECTIONS.filter((x) => !(touchOnly && x.id === 'shortcuts')).map((x) => ({
      id: `settings-${x.id}`,
      group: 'Settings',
      label: x.label,
      match: x.match,
      scoped: true,
      icon: SettingsIcon,
      run: go(`/settings?section=${x.id}`),
    }));
    const r: Item[] = runs.get().map((x) => ({
      id: `run-${x.id}`,
      group: 'Runs',
      label: x.name,
      hint: [failureCause(x) ? 'Failed, out of GPU memory' : statusWord(x), { ident: x.config.arch }, `epoch ${x.epoch}/${x.epochs}`],
      icon: Activity,
      status: x.status,
      run: go(`/runs/${x.id}`),
    }));
    const d: Item[] = datasets.get().map((x) => ({
      id: `ds-${x.id}`,
      group: 'Datasets',
      label: x.name,
      hint: [`${x.classes.length} classes`, `${(x.splits.train + x.splits.val + x.splits.test).toLocaleString('en-US')} images`],
      icon: Images,
      run: go(`/datasets/${x.id}`),
    }));
    // A model's score with what it is on (lib/scoreSummary): "mAP50-95 0.604 on fasteners", "mAP50-95 0.701 · split unknown".
    const runById = new Map(runs.get().map((x) => [x.id, x]));
    const dsById = new Map(datasets.get().map((x) => [x.id, x]));
    const b: Item[] = benches.get().map((x) => ({
      id: `bench-${x.id}`,
      group: 'Benchmarks',
      label: x.name,
      hint: [`${x.images.toLocaleString('en-US')} images`, `${x.entries} ${x.entries === 1 ? 'entry' : 'entries'}`, ...(x.best ? [`best ${x.best.score.toFixed(1)}`] : [])],
      icon: Target,
      run: go(`/benchmarks/${x.id}`),
    }));
    const m: Item[] = models.get().map((x) => ({
      id: `model-${x.id}`,
      group: 'Models',
      label: x.name,
      hint: [{ ident: x.arch }, x.metrics ? scoreText(x.metrics.map50_95, scoreContext(x, runById, dsById)) : 'not scored'],
      icon: Box,
      run: go(`/models/${x.id}`),
    }));
    // Actions that take what the query names: train on a dataset, try a model. Ranked with the entities themselves.
    const onEntity: Item[] = [
      ...datasets.get().map((x) => ({
        id: `act-train-${x.id}`,
        group: 'Actions',
        label: `New run on ${x.name}`,
        hint: 'Train a detector on this dataset',
        describes: true,
        match: [x.name, `Train ${x.name}`],
        scoped: true,
        icon: Plus,
        run: go(`/runs/new?dataset=${encodeURIComponent(x.id)}`),
      })),
      ...models.get().map((x) => ({
        id: `act-try-${x.id}`,
        group: 'Actions',
        label: `Open ${x.name} in Playground`,
        hint: 'Run it on an image',
        describes: true,
        match: [x.name, `Playground ${x.name}`, `Try ${x.name}`],
        scoped: true,
        icon: ScanSearch,
        run: go(`/playground?model=${encodeURIComponent(x.id)}`),
      })),
    ];
    // With no query: navigation, the actions and the runs in flight. A filter reads the status enum, never the hint's
    // display word ("Running"): round 26's lowercase comparison against the sentence-case label dropped every run.
    const idle = (i: Item) => i.group === 'Go to' || i.group === 'Actions' || (i.status !== undefined && ACTIVE.includes(i.status));
    return [
      ...page,
      ...nav,
      ...actions,
      ...settings,
      ...onEntity,
      ...r.map((i) => ({ ...i, scoped: !idle(i) })),
      ...d.map((i) => ({ ...i, scoped: true })),
      ...m.map((i) => ({ ...i, scoped: true })),
      ...b.map((i) => ({ ...i, scoped: true })),
    ];
  });

  // No query: this page's commands, then Recent (the last items chosen, wherever they live), then the usual list.
  const results = computed(() => {
    const q = query.get();
    const all = items.get();
    if (q.trim()) return rank(all, q);
    const byId = new Map(all.map((i) => [i.id, i]));
    const rec = recent
      .get()
      .map((id) => byId.get(id))
      .filter((i): i is Item => !!i)
      .map((i) => ({ ...i, id: `recent-${i.id}`, group: 'Recent', scoped: false, origin: i.id }));
    const ranked = rank(all, '');
    return [...ranked.filter((i) => i.group === 'This page'), ...rec, ...ranked.filter((i) => i.group !== 'This page')];
  });

  const grouped = computed(() => {
    const out: { group: string; items: { item: Item; n: number }[] }[] = [];
    // Groups in the order of their best result, except that an action about an entity ("New run on shapes-five") never
    // puts its group above the entities' own: those groups take their place from their other results, or come last.
    const all = results.get();
    for (const item of [...all.filter((i) => !i.describes), ...all.filter((i) => i.describes)])
      if (!out.some((x) => x.group === item.group)) out.push({ group: item.group, items: [] });
    all.forEach((item, n) => out.find((x) => x.group === item.group)!.items.push({ item, n }));
    // Keep numbering in visual order.
    let n = 0;
    for (const g of out) for (const it of g.items) it.n = n++;
    return out;
  });

  const flat = computed(() => grouped.get().flatMap((g) => g.items.map((x) => x.item)));

  // The result count, said politely once typing pauses ("5 results", "No results"), so a screen reader hears what the
  // query found without leaving the field (focus stays in the input). Not said at open, where nothing was searched.
  const said = signal('');
  effect(() => {
    const q = query.get().trim();
    const n = flat.get().length;
    if (!open.get() || !q) return void said.set('');
    const t = setTimeout(() => said.set(n === 0 ? 'No results' : n === 1 ? '1 result' : `${n} results`), ANNOUNCE_MS);
    return () => clearTimeout(t);
  });

  effect(() => {
    void query.get();
    index.set(0);
  });

  effect(() => {
    if (!open.get()) return;
    const elx = list.get()?.querySelector<HTMLElement>(`[data-n="${index.get()}"]`);
    elx?.scrollIntoView({ block: 'nearest' });
  });

  function choose(item: Item | undefined) {
    if (!item) return;
    if (item.group !== 'This page') rememberRecent(item.origin ?? item.id);
    el.close();
    item.run();
  }

  function onkeydown(e: KeyboardEvent) {
    const all = flat.peek();
    if (e.key === 'ArrowDown' || (e.key === 'n' && e.ctrlKey)) {
      e.preventDefault();
      // The selection wraps at the ends, so the last item is one key from the first.
      index.set(all.length ? (index.peek() + 1) % all.length : 0);
    } else if (e.key === 'ArrowUp' || (e.key === 'p' && e.ctrlKey)) {
      e.preventDefault();
      index.set(all.length ? (index.peek() - 1 + all.length) % all.length : 0);
    } else if (e.key === 'Enter') {
      e.preventDefault();
      choose(all[index.peek()]);
    } else if (e.key === 'Home' && e.ctrlKey) index.set(0);
    else if (e.key === 'End' && e.ctrlKey) index.set(all.length - 1);
  }

  const activeId = () => {
    const it = flat.get()[index.get()];
    return it ? `pi-${it.id}` : undefined;
  };

  return (
    <dialog
      ref={(e) => (el = e)}
      class="palette"
      aria-label="Command palette"
      onclose={() => {
        open.set(false);
        restore?.();
        restore = null;
      }}
      onpointerdown={backdrop.onpointerdown}
      onclick={backdrop.onclick}
    >
      <div class="box">
        <div class="search">
          <Search size={16} strokeWidth={1.75} aria-hidden="true" />
          <input
            ref={input}
            bind:value={query}
            onkeydown={onkeydown}
            placeholder="Search runs, datasets, models, or type a command…"
            role="combobox"
            aria-label="Search commands, runs, datasets and models"
            aria-expanded="true"
            aria-controls="palette-list"
            aria-activedescendant={activeId()}
            aria-autocomplete="list"
            spellcheck="false"
            autocomplete="off"
          />
          {!touchOnly ? (
            <span class="esc">
              <Kbd keys="Escape" subtle />
            </span>
          ) : null}
          <button class="close" type="button" aria-label="Close" onclick={() => el.close()}>
            <X size={18} />
          </button>
        </div>
        <ul id="palette-list" class="scroll-shade" role="listbox" ref={list} aria-label="Results">
          <For
            each={grouped.get()}
            key={(g) => g.group}
            fallback={
              <li class="none" role="presentation">
                {`No matches for “${query.get()}”`}
              </li>
            }
          >
            {(g) => (
              <>
                <li role="presentation" class="group group-label">
                  {g().group}
                </li>
                <For each={g().items} key={(x) => x.item.id}>
                  {(x) => {
                    const on = () => x().n === index.get();
                    // The keys are the combobox's (arrows and Enter on the input, aria-activedescendant): an option never
                    // takes focus, so it has no key handler of its own; the pointer can also choose it.
                    return (
                      <li
                        id={`pi-${x().item.id}`}
                        role="option"
                        aria-selected={on()}
                        class="item"
                        class:on={on()}
                        data-n={x().n}
                        onpointermove={() => index.set(x().n)}
                        onclick={() => choose(x().item)}
                      >
                        <Dynamic component={x().item.icon} size={16} strokeWidth={1.75} aria-hidden="true" />
                        {/* The name and its hint share one block: the hint sits right of the name while both fit and drops,
                            whole, to a line under it when they do not (always on phones), so no fact in it is cut
                            (DESIGN_DIRECTION §2.3). */}
                        <span class="main">
                          <span class="label">
                            <Index each={highlight(query.get(), x().item.label)}>{(part) => <>{part().m ? <mark>{part().s}</mark> : part().s}</>}</Index>
                          </span>
                          {hint(x)}
                        </span>
                        {/* Every hint here is a single key: none while those are off (Settings). */}
                        {x().item.keys && keyPrefs.singleKeys && !touchOnly ? <Kbd keys={x().item.keys ?? ''} subtle /> : null}
                      </li>
                    );
                  }}
                </For>
              </>
            )}
          </For>
        </ul>
        <p class="sr-only" role="status" aria-live="polite" data-testid="palette-count">
          {said.get()}
        </p>
        {/* The keys that drive the list, for a keyboard: a touch-only device has none. */}
        {!touchOnly ? (
          <footer>
            <span>
              <Kbd keys="ArrowUp" subtle />
              <Kbd keys="ArrowDown" subtle /> move
            </span>
            <span>
              <Kbd keys="Enter" subtle /> open
            </span>
            <span>
              <Kbd keys="Escape" subtle /> close
            </span>
          </footer>
        ) : null}
      </div>
    </dialog>
  );
}

/** A row's hint: a sentence, or facts as meta items (an identifier set mono). */
function hint(x: () => { item: Item; n: number }) {
  const h = () => x().item.hint;
  return (
    <>
      {typeof h() === 'string' ? (
        <span class="hint" class:describes={x().item.describes ?? false}>
          {h() as string}
        </span>
      ) : h() ? (
        <span class="hint num">
          <Meta>
            <Index each={h() as (string | { ident: string })[]}>
              {(part) => <>{typeof part() === 'string' ? <span>{part() as string}</span> : <Ident value={(part() as { ident: string }).ident} />}</>}
            </Index>
          </Meta>
        </span>
      ) : null}
    </>
  );
}

css`
  .palette {
    width: min(640px, calc(100vw - 24px));
    margin: 12vh auto auto;
    padding: 0;
    border: 0;
    border-radius: var(--radius-xl);
    background: var(--bg-overlay);
    color: var(--text-primary);
    box-shadow: var(--shadow-lg);
    overflow: hidden;
  }
  .palette[open] {
    animation: enter var(--dur-base) var(--ease-out);
  }
  .palette::backdrop {
    background: var(--bg-scrim);
  }
  .box {
    display: flex;
    flex-direction: column;
    max-height: min(560px, 70vh);
  }
  /* The search row and the footer keep their height whatever the list holds: only the list gives (shell F9). */
  .search {
    flex-shrink: 0;
    display: flex;
    align-items: center;
    gap: var(--space-3);
    height: 52px;
    padding: 0 var(--space-4);
    border-bottom: 1px solid var(--line-subtle);
    color: var(--text-tertiary);
  }
  input {
    flex: 1;
    min-width: 0;
    height: 100%;
    border: 0;
    background: none;
    outline: none;
    font-size: var(--type-md-size);
    color: var(--text-primary);
  }
  input::placeholder {
    color: var(--text-tertiary);
  }
  input:focus-visible {
    outline: none;
  }
  /* Touch: the query at 16 px (type.lg), under which iOS zooms the page into the field (sourceRules.test.ts). */
  @media (pointer: coarse) {
    input {
      font-size: var(--type-lg-size);
    }
  }
  /* The results scroll with the dialog's edge shades (base.css .scroll-shade); 8 px under the last item, so the end of
     the list reads as its end, clear of the footer. */
  ul {
    flex: 1;
    min-height: 0;
    padding: var(--space-2);
    scroll-padding: var(--space-8);
  }
  /* Type and ink: the shared group label (base.css .group-label). */
  .group {
    padding: var(--space-2) var(--space-2) var(--space-1);
  }
  .group:not(:first-child) {
    margin-top: var(--space-1);
  }
  /* A long name wraps to a second line (the row grows) instead of losing its end; the hint gives way first. */
  .item {
    display: flex;
    align-items: center;
    gap: var(--space-3);
    min-height: 36px;
    padding: var(--space-2) 12px;
    border-radius: var(--radius-md);
    color: var(--text-secondary);
    cursor: pointer;
  }
  .item.on {
    background: var(--bg-active);
    color: var(--text-primary);
  }
  .item > :global(svg) {
    flex-shrink: 0;
  }
  .main {
    flex: 1;
    min-width: 0;
    display: flex;
    flex-wrap: wrap;
    align-items: baseline;
    column-gap: var(--space-3);
  }
  .label {
    min-width: 0;
    overflow-wrap: anywhere;
  }
  /* Matched characters: ink and weight only. Lime is not decoration (it means focus, live or the one action). */
  mark {
    background: none;
    color: var(--text-primary);
    font-weight: var(--weight-semibold);
  }
  /* Right of the name while it fits; else a line of its own under it, its items wrapping whole (ui/Meta). */
  .hint {
    margin-left: auto;
    min-width: 0;
    max-width: 100%;
    text-align: right;
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-tertiary);
  }
  .none {
    padding: var(--space-8) var(--space-4);
    text-align: center;
    color: var(--text-tertiary);
  }
  footer {
    flex-shrink: 0;
    display: flex;
    align-items: center;
    gap: var(--space-4);
    height: 36px;
    padding: 0 var(--space-4);
    border-top: 1px solid var(--line-subtle);
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
  }
  footer span {
    display: inline-flex;
    align-items: center;
    gap: 6px;
  }
  @keyframes enter {
    from {
      opacity: 0;
      transform: translateY(-6px) scale(0.985);
    }
  }
  .close {
    display: none;
  }
  @media (pointer: coarse) {
    .item {
      min-height: 44px;
    }
  }
  @media (max-width: 767px) {
    .esc {
      display: none;
    }
    .close {
      display: grid;
      place-items: center;
      width: 44px;
      height: 44px;
      margin-right: calc(-1 * var(--space-3));
      border-radius: var(--radius-md);
      color: var(--text-secondary);
    }
    .palette {
      margin-top: var(--space-3);
    }
    footer {
      display: none;
    }
    .hint.describes {
      display: none;
    }
    /* Phones: the hint is the row's second line, under the name. */
    .main {
      flex-direction: column;
      align-items: stretch;
    }
    .hint {
      margin-left: 0;
      text-align: left;
    }
  }
`;
