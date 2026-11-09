<script lang="ts">
  import { sidebar } from './sidebarState.svelte';
  /** Ctrl/Cmd+K: jump anywhere, run any action, find any run, dataset or model. */
  import Search from '@lucide/svelte/icons/search';
  import X from '@lucide/svelte/icons/x';
  import Plus from '@lucide/svelte/icons/plus';
  import SunMoon from '@lucide/svelte/icons/sun-moon';
  import PanelLeft from '@lucide/svelte/icons/panel-left';
  import Activity from '@lucide/svelte/icons/activity';
  import Images from '@lucide/svelte/icons/images';
  import Box from '@lucide/svelte/icons/box';
  import { scoreContext, scoreText } from '$lib/scoreSummary';
  import GitCompare from '@lucide/svelte/icons/git-compare-arrows';
  import Keyboard from '@lucide/svelte/icons/keyboard';
  import ScanSearch from '@lucide/svelte/icons/scan-search';
  import { hub } from '$lib/api/index.svelte';
  import { failureCause } from '$lib/oom';
  import type { Dataset, Model, Run } from '$lib/api/types';
  import { prefs } from '$lib/prefs.svelte';
  import { router } from '$lib/router.svelte';
  import Ident from '$lib/ui/Ident.svelte';
  import Kbd from '$lib/ui/Kbd.svelte';
  import Meta from '$lib/ui/Meta.svelte';
  import { keyPrefs } from '$lib/ui/keyPrefs.svelte';
  import type { IconType } from '$lib/ui/types';
  import { rememberFocus } from '$lib/ui/focus';
  import { highlight, rank } from './commands';
  import { mainNav, settingsNav } from './nav';
  import { touchOnly as isTouchOnly } from './shortcuts';
  import { pageCommandList, recentIds, rememberRecent } from './pageCommands.svelte';

  let { open = $bindable(false), onshortcuts }: { open?: boolean; onshortcuts: () => void } = $props();
  const touchOnly = isTouchOnly();

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
    run: () => void;
  }

  let el: HTMLDialogElement;
  let input: HTMLInputElement | undefined = $state();
  let list: HTMLUListElement | undefined = $state();
  let query = $state('');
  let index = $state(0);
  let runs = $state.raw<Run[]>([]);
  let datasets = $state.raw<Dataset[]>([]);
  let models = $state.raw<Model[]>([]);
  /** The items chosen last (ids), read on open: with no query they head the list, under Recent. */
  let recent = $state.raw<string[]>([]);

  // Focus returns to the opener on close; a command that navigates then hands it to the new page (App).
  let restore: (() => void) | null = null;
  $effect(() => {
    if (open && !el.open) {
      restore = rememberFocus();
      el.showModal();
      query = '';
      index = 0;
      recent = recentIds();
      queueMicrotask(() => input?.focus());
      const api = hub.api;
      api.runs().then((r) => (runs = r), () => {});
      api.datasets().then((d) => (datasets = d), () => {});
      api.models().then((m) => (models = m), () => {});
    } else if (!open && el.open) el.close();
  });

  const go = (path: string) => () => router.go(path);

  const items = $derived.by((): Item[] => {
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
      {
        id: 'theme',
        group: 'Actions',
        label: `Switch theme (now ${prefs.theme})`,
        match: ['Toggle theme', 'Dark theme', 'Light theme'],
        icon: SunMoon,
        run: () => prefs.cycleTheme(),
      },
      {
        id: 'sidebar',
        group: 'Actions',
        label: sidebar.collapsed ? 'Expand sidebar' : 'Collapse sidebar',
        keys: '[',
        icon: PanelLeft,
        run: () => sidebar.toggle(),
      },
      // Touch-only devices have no keys to look up.
      ...(touchOnly ? [] : [{ id: 'shortcuts', group: 'Actions', label: 'Keyboard shortcuts', keys: '?', icon: Keyboard, run: onshortcuts }]),
    ];
    const r: Item[] = runs.map((x) => ({
      id: `run-${x.id}`,
      group: 'Runs',
      label: x.name,
      hint: [failureCause(x) ? 'failed, out of GPU memory' : x.status, { ident: x.config.arch }, `epoch ${x.epoch}/${x.epochs}`],
      icon: Activity,
      run: go(`/runs/${x.id}`),
    }));
    const d: Item[] = datasets.map((x) => ({
      id: `ds-${x.id}`,
      group: 'Datasets',
      label: x.name,
      hint: [`${x.classes.length} classes`, `${(x.splits.train + x.splits.val + x.splits.test).toLocaleString('en-US')} images`],
      icon: Images,
      run: go(`/datasets/${x.id}`),
    }));
    // A model's score with what it is on (lib/scoreSummary): "mAP 0.604 on fasteners", "mAP 0.701 · split unknown".
    const runById = new Map(runs.map((x) => [x.id, x]));
    const dsById = new Map(datasets.map((x) => [x.id, x]));
    const m: Item[] = models.map((x) => ({
      id: `model-${x.id}`,
      group: 'Models',
      label: x.name,
      hint: [{ ident: x.arch }, x.metrics ? scoreText(x.metrics.map50_95, scoreContext(x, runById, dsById)) : 'not scored'],
      icon: Box,
      run: go(`/models/${x.id}`),
    }));
    // Actions that take what the query names: train on a dataset, try a model. Ranked with the entities themselves.
    const onEntity: Item[] = [
      ...datasets.map((x) => ({
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
      ...models.map((x) => ({
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
    // With no query: navigation, the actions and the runs in flight.
    const idle = (i: Item) => i.group === 'Go to' || i.group === 'Actions' || (i.group === 'Runs' && (i.hint?.[0] === 'running' || i.hint?.[0] === 'queued'));
    return [...page, ...nav, ...actions, ...onEntity, ...r.map((i) => ({ ...i, scoped: !idle(i) })), ...d.map((i) => ({ ...i, scoped: true })), ...m.map((i) => ({ ...i, scoped: true }))];
  });

  // No query: this page's commands, then Recent (the last items chosen, wherever they live), then the usual list.
  const results = $derived.by(() => {
    if (query.trim()) return rank(items, query);
    const byId = new Map(items.map((i) => [i.id, i]));
    const rec = recent
      .map((id) => byId.get(id))
      .filter((i): i is Item => !!i)
      .map((i) => ({ ...i, id: `recent-${i.id}`, group: 'Recent', scoped: false, origin: i.id }));
    const all = rank(items, '');
    return [...all.filter((i) => i.group === 'This page'), ...rec, ...all.filter((i) => i.group !== 'This page')];
  });

  const grouped = $derived.by(() => {
    const out: { group: string; items: { item: Item; n: number }[] }[] = [];
    results.forEach((item, n) => {
      const g = out.find((x) => x.group === item.group);
      if (g) g.items.push({ item, n });
      else out.push({ group: item.group, items: [{ item, n }] });
    });
    // Keep numbering in visual order.
    let n = 0;
    for (const g of out) for (const it of g.items) it.n = n++;
    return out;
  });

  const flat = $derived(grouped.flatMap((g) => g.items.map((x) => x.item)));

  $effect(() => {
    void query;
    index = 0;
  });

  $effect(() => {
    if (!open) return;
    const elx = list?.querySelector<HTMLElement>(`[data-n="${index}"]`);
    elx?.scrollIntoView({ block: 'nearest' });
  });

  function choose(item: Item | undefined) {
    if (!item) return;
    if (item.group !== 'This page') rememberRecent(item.origin ?? item.id);
    el.close();
    item.run();
  }

  function onkeydown(e: KeyboardEvent) {
    if (e.key === 'ArrowDown' || (e.key === 'n' && e.ctrlKey)) {
      e.preventDefault();
      // The selection wraps at the ends, so the last item is one key from the first.
      index = flat.length ? (index + 1) % flat.length : 0;
    } else if (e.key === 'ArrowUp' || (e.key === 'p' && e.ctrlKey)) {
      e.preventDefault();
      index = flat.length ? (index - 1 + flat.length) % flat.length : 0;
    } else if (e.key === 'Enter') {
      e.preventDefault();
      choose(flat[index]);
    } else if (e.key === 'Home' && e.ctrlKey) index = 0;
    else if (e.key === 'End' && e.ctrlKey) index = flat.length - 1;
  }
</script>

<dialog
  bind:this={el}
  class="palette"
  aria-label="Command palette"
  onclose={() => {
    open = false;
    restore?.();
    restore = null;
  }}
  onclick={(e) => {
    if (e.target === el) el.close();
  }}
>
  <div class="box">
    <div class="search">
      <Search size={16} strokeWidth={1.75} aria-hidden="true" />
      <input
        bind:this={input}
        bind:value={query}
        {onkeydown}
        placeholder="Search runs, datasets, models, or type a command…"
        role="combobox"
        aria-label="Search commands, runs, datasets and models"
        aria-expanded="true"
        aria-controls="palette-list"
        aria-activedescendant={flat[index] ? `pi-${flat[index]?.id}` : undefined}
        aria-autocomplete="list"
        spellcheck="false"
        autocomplete="off"
      />
      <span class="esc"><Kbd keys="Escape" subtle /></span>
      <button class="close" type="button" aria-label="Close" onclick={() => el.close()}><X size={18} /></button>
    </div>
    <ul id="palette-list" role="listbox" bind:this={list} aria-label="Results">
      {#each grouped as g (g.group)}
        <li role="presentation" class="group">{g.group}</li>
        {#each g.items as { item, n } (item.id)}
          <li
            id="pi-{item.id}"
            role="option"
            aria-selected={n === index}
            class="item"
            class:on={n === index}
            data-n={n}
            onpointermove={() => (index = n)}
            onclick={() => choose(item)}
            onkeydown={() => {}}
          >
            <item.icon size={16} strokeWidth={1.75} aria-hidden="true" />
            <!-- The name and its hint share one block: the hint sits right of the name while both fit and drops, whole, to
                 a line under it when they do not (always on phones), so no fact in it is cut (DESIGN_DIRECTION §2.3). -->
            <span class="main">
              <span class="label">
                {#each highlight(query, item.label) as part, k (k)}{#if part.m}<mark>{part.s}</mark>{:else}{part.s}{/if}{/each}
              </span>
              {#if typeof item.hint === 'string'}<span class="hint" class:describes={item.describes}>{item.hint}</span>
              {:else if item.hint}<span class="hint num"
                  ><Meta>{#each item.hint as part, k (k)}{#if typeof part === 'string'}<span>{part}</span>{:else}<Ident value={part.ident} />{/if}{/each}</Meta></span
                >{/if}
            </span>
            <!-- Every hint here is a single key: none while those are off (Settings). -->
            {#if item.keys && keyPrefs.singleKeys}<Kbd keys={item.keys} subtle />{/if}
          </li>
        {/each}
      {:else}
        <li class="none" role="presentation">No matches for “{query}”</li>
      {/each}
    </ul>
    <footer>
      <span><Kbd keys="ArrowUp" subtle /><Kbd keys="ArrowDown" subtle /> move</span>
      <span><Kbd keys="Enter" subtle /> open</span>
      <span><Kbd keys="Escape" subtle /> close</span>
    </footer>
  </div>
</dialog>

<style>
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
  .search {
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
  ul {
    flex: 1;
    overflow-y: auto;
    padding: var(--space-2);
    scroll-padding: var(--space-8);
  }
  .group {
    padding: var(--space-2) var(--space-2) var(--space-1);
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
    font-weight: var(--weight-medium);
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
</style>
