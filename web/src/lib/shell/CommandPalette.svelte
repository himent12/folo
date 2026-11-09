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
  import GitCompare from '@lucide/svelte/icons/git-compare-arrows';
  import Keyboard from '@lucide/svelte/icons/keyboard';
  import CornerDownLeft from '@lucide/svelte/icons/corner-down-left';
  import { hub } from '$lib/api/index.svelte';
  import { failureCause } from '$lib/oom';
  import type { Dataset, Model, Run } from '$lib/api/types';
  import { prefs } from '$lib/prefs.svelte';
  import { router } from '$lib/router.svelte';
  import Kbd from '$lib/ui/Kbd.svelte';
  import type { IconType } from '$lib/ui/types';
  import { rememberFocus } from '$lib/ui/focus';
  import { fuzzy, highlight } from './commands';
  import { mainNav, settingsNav } from './nav';

  let { open = $bindable(false), onshortcuts }: { open?: boolean; onshortcuts: () => void } = $props();

  interface Item {
    id: string;
    group: string;
    label: string;
    hint?: string;
    keys?: string;
    icon: IconType;
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

  // Focus returns to the opener on close; a command that navigates then hands it to the new page (App).
  let restore: (() => void) | null = null;
  $effect(() => {
    if (open && !el.open) {
      restore = rememberFocus();
      el.showModal();
      query = '';
      index = 0;
      queueMicrotask(() => input?.focus());
      const api = hub.api;
      api.runs().then((r) => (runs = r), () => {});
      api.datasets().then((d) => (datasets = d), () => {});
      api.models().then((m) => (models = m), () => {});
    } else if (!open && el.open) el.close();
  });

  const go = (path: string) => () => router.go(path);

  const items = $derived.by((): Item[] => {
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
      { id: 'shortcuts', group: 'Actions', label: 'Keyboard shortcuts', keys: '?', icon: Keyboard, run: onshortcuts },
    ];
    const r: Item[] = runs.map((x) => ({
      id: `run-${x.id}`,
      group: 'Runs',
      label: x.name,
      hint: `${failureCause(x) ? 'failed, out of GPU memory' : x.status} · ${x.config.arch} · epoch ${x.epoch}/${x.epochs}`,
      icon: Activity,
      run: go(`/runs/${x.id}`),
    }));
    const d: Item[] = datasets.map((x) => ({
      id: `ds-${x.id}`,
      group: 'Datasets',
      label: x.name,
      hint: `${x.classes.length} classes · ${(x.splits.train + x.splits.val + x.splits.test).toLocaleString('en-US')} images`,
      icon: Images,
      run: go(`/datasets/${x.id}`),
    }));
    const m: Item[] = models.map((x) => ({
      id: `model-${x.id}`,
      group: 'Models',
      label: x.name,
      hint: `${x.arch} · mAP ${x.metrics?.map50_95.toFixed(3) ?? '—'}`,
      icon: Box,
      run: go(`/models/${x.id}`),
    }));
    return [...nav, ...actions, ...r, ...d, ...m];
  });

  const results = $derived.by(() => {
    if (!query.trim()) return items.filter((i) => i.group === 'Go to' || i.group === 'Actions' || (i.group === 'Runs' && /running|queued/.test(i.hint ?? '')));
    return items
      .map((i) => ({ i, s: Math.max(fuzzy(query, i.label), fuzzy(query, `${i.group} ${i.label}`) - 20) }))
      .filter((x) => x.s >= 0)
      .sort((a, b) => b.s - a.s)
      .slice(0, 40)
      .map((x) => x.i);
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
    el.close();
    item.run();
  }

  function onkeydown(e: KeyboardEvent) {
    if (e.key === 'ArrowDown' || (e.key === 'n' && e.ctrlKey)) {
      e.preventDefault();
      index = Math.min(flat.length - 1, index + 1);
    } else if (e.key === 'ArrowUp' || (e.key === 'p' && e.ctrlKey)) {
      e.preventDefault();
      index = Math.max(0, index - 1);
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
            <span class="label">
              {#each highlight(query, item.label) as part, k (k)}{#if part.m}<mark>{part.s}</mark>{:else}{part.s}{/if}{/each}
            </span>
            {#if item.hint}<span class="hint">{item.hint}</span>{/if}
            {#if item.keys}<Kbd keys={item.keys} subtle />{/if}
          </li>
        {/each}
      {:else}
        <li class="none" role="presentation">No matches for “{query}”</li>
      {/each}
    </ul>
    <footer>
      <span><Kbd keys="ArrowUp" subtle /><Kbd keys="ArrowDown" subtle /> move</span>
      <span><CornerDownLeft size={12} /> open</span>
      <span class="brand">FOLO</span>
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
  .item {
    display: flex;
    align-items: center;
    gap: var(--space-3);
    height: 36px;
    padding: 0 12px;
    border-radius: var(--radius-md);
    color: var(--text-secondary);
    cursor: pointer;
  }
  .item.on {
    background: var(--bg-active);
    color: var(--text-primary);
  }
  .item.on :global(svg) {
    color: var(--accent-text);
  }
  .label {
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  /* Matched characters: ink and weight only. Lime is not decoration (it means focus, live or the one action). */
  mark {
    background: none;
    color: var(--text-primary);
    font-weight: var(--weight-semibold);
  }
  .hint {
    flex: 1;
    min-width: 0;
    text-align: right;
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .label + :global(.kbd) {
    margin-left: auto;
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
  .brand {
    margin-left: auto;
    font-weight: var(--weight-semibold);
    letter-spacing: 0.04em;
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
      height: 44px;
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
  }
</style>
