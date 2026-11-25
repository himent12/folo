/** Ctrl/Cmd+K: jump anywhere, run any action, find any run, dataset or model. */
import Search from '$icons/search';
import X from '$icons/x';
import { bindable, computed, css, Dynamic, effect, For, Index, onCleanup, signal } from 'fw';
import { hub } from '$lib/api/index';
import { LABEL_FEATURE } from '$lib/label';
import { router } from '$lib/router';
import { backdropClose } from '$lib/ui/backdrop';
import { rememberFocus } from '$lib/ui/focus';
import Ident from '$lib/ui/Ident';
import Kbd from '$lib/ui/Kbd';
import { keyPrefs } from '$lib/ui/keyPrefs';
import Meta from '$lib/ui/Meta';
import Spinner from '$lib/ui/Spinner';
import { pageCommandList, recentIds, rememberRecent } from '../pageCommands';
import { touchOnly as isTouchOnly } from '../shortcuts';
import { createPaletteData } from './data';
import { buildPaletteItems } from './providers';
import { groupResults, highlightItem, rank } from './scoring';
import type { CommandPaletteProps, Item } from './types';

export type { CommandPaletteProps } from './types';

/** How long typing must pause before the result count is announced. */
const ANNOUNCE_MS = 500;

export default function CommandPalette(props: CommandPaletteProps) {
  const open = bindable(props, 'open', false);
  const touchOnly = isTouchOnly();

  let el!: HTMLDialogElement;
  const input = signal<HTMLInputElement>();
  const list = signal<HTMLUListElement>();
  const query = signal('');
  /**
   * The highlighted item, by id (null: none yet, the first). A list that changes under it (a health answer's features, a
   * list that loads late) keeps it highlighted; one that drops it highlights the first, so Enter runs only what is
   * highlighted.
   */
  const selected = signal<string | null>(null);
  /** The items chosen last (ids), read on open: with no query they head the list, under Recent. */
  const recent = signal<string[]>([]);

  /** Pointer guard: ignore stationary pointermove coordinates triggered when keyboard ArrowDown/ArrowUp scrolls the list. */
  let lastPointer: { x: number; y: number } | null = null;
  let keyboardNav = false;

  const data = createPaletteData({
    open: () => open.get(),
    isDialogOpen: () => !!el?.open,
    query,
    recent,
  });

  const closePalette = () => {
    data.cancelFetches();
    data.clearPending();
    if (el?.open) el.close();
    if (open.peek()) open.set(false);
  };

  // A press outside the box closes it, unless it is the second click of the double-click that opened it (ui/backdrop).
  const backdrop = backdropClose(() => el, closePalette);
  // Focus returns to the opener on close; a command that navigates or opens another dialog keeps its own focus.
  let restore: (() => void) | null = null;

  onCleanup(() => {
    const fn = restore;
    restore = null;
    if (el?.open) el.close();
    fn?.();
  });

  effect(() => {
    if (open.get()) {
      const epoch = hub.epoch;
      const opening = !el.open;
      if (!opening && !data.isStaleEpoch(epoch)) return;
      if (opening) {
        restore = rememberFocus();
        el.showModal();
        backdrop.opened();
        lastPointer = null;
        keyboardNav = false;
        const seed = props.takeInitialQuery?.() ?? '';
        query.set(seed);
        selected.set(null);
        recent.set(recentIds());
        queueMicrotask(() => {
          const inp = input.peek();
          if (!inp || !el.open) return;
          inp.focus();
          if (inp.value.length) inp.setSelectionRange(inp.value.length, inp.value.length);
        });
      }
      data.startFetch(epoch);
    } else {
      closePalette();
    }
  });

  const items = computed((): Item[] =>
    buildPaletteItems({
      pageCommands: pageCommandList.items,
      runs: data.runs.get(),
      datasets: data.datasets.get(),
      models: data.models.get(),
      benches: data.benches.get(),
      benchesOk: data.benchesOk.get(),
      labelJobs: data.labelJobs.get(),
      loops: data.loops.get(),
      cloud: data.cloud.get(),
      lacksLabel: hub.lacks(LABEL_FEATURE),
      touchOnly,
      onshortcuts: props.onshortcuts,
    }),
  );

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

  const grouped = computed(() => groupResults(results.get()));
  const flat = computed(() => grouped.get().flatMap((g) => g.items.map((x) => x.item)));

  // The result count, said politely once typing pauses ("5 results", "No results"), so a screen reader hears what the
  // query found without leaving the field (focus stays in the input). Not said at open, where nothing was searched.
  const said = signal('');
  effect(() => {
    const q = query.get().trim();
    const n = flat.get().length;
    const loading = data.loadingEntities.get();
    const failed = data.failedEntityList.get();
    if (!open.get() || !q) return void said.set('');
    const t = setTimeout(() => {
      if (n === 0) {
        if (loading) said.set('Searching…');
        else said.set(failed.length ? `No results; could not load ${failed.join(', ')}` : 'No results');
      } else said.set(n === 1 ? '1 result' : `${n} results`);
    }, ANNOUNCE_MS);
    return () => clearTimeout(t);
  });

  /** Where the highlighted item is in `flat`. */
  const index = computed(() => Math.max(0, flat.get().findIndex((i) => i.id === selected.get())));
  effect(() => {
    void query.get();
    keyboardNav = true;
    selected.set(null);
  });
  // The first item is highlighted by its id too, at open, on a new query and when the highlighted one goes, so a row
  // that loads above it later does not take the highlight; one that comes back later does not take it back.
  effect(() => {
    const all = flat.get();
    const id = selected.get();
    if (!all.some((i) => i.id === id)) selected.set(all[0]?.id ?? null);
  });

  effect(() => {
    if (!open.get()) return;
    const elx = list.get()?.querySelector<HTMLElement>(`[data-n="${index.get()}"]`);
    elx?.scrollIntoView({ block: 'nearest' });
  });

  function choose(item: Item | undefined) {
    if (!item) return;
    if (item.group !== 'This page') rememberRecent(item.origin ?? item.id);
    // Take ownership of focus restoration before closing so an async native dialog `close` event cannot steal focus
    // back after a command navigates or opens another dialog (e.g. Keyboard shortcuts, Stop run).
    const restoreFocus = restore;
    restore = null;
    closePalette();
    const beforePath = router.location.path;
    const beforeQuery = router.location.query.toString();
    item.run();
    const navigated = router.location.path !== beforePath || router.location.query.toString() !== beforeQuery;
    if (restoreFocus && !navigated && !document.querySelector('dialog[open]')) restoreFocus();
  }

  function onItemPointerMove(e: PointerEvent, id: string) {
    if (lastPointer && e.clientX === lastPointer.x && e.clientY === lastPointer.y) return;
    const hadPointer = lastPointer !== null;
    lastPointer = { x: e.clientX, y: e.clientY };
    if (!hadPointer && keyboardNav) return;
    keyboardNav = false;
    selected.set(id);
  }

  function onkeydown(e: KeyboardEvent) {
    const all = flat.peek();
    const to = (n: number) => {
      keyboardNav = true;
      selected.set(all[n]?.id ?? null);
    };
    if (e.key === 'ArrowDown' || (e.key === 'n' && e.ctrlKey)) {
      e.preventDefault();
      // The selection wraps at the ends, so the last item is one key from the first.
      to(all.length ? (index.peek() + 1) % all.length : 0);
    } else if (e.key === 'ArrowUp' || (e.key === 'p' && e.ctrlKey)) {
      e.preventDefault();
      to(all.length ? (index.peek() - 1 + all.length) % all.length : 0);
    } else if (e.key === 'Enter') {
      e.preventDefault();
      choose(all[index.peek()]);
    } else if (e.key === 'Home' && e.ctrlKey) {
      e.preventDefault();
      to(0);
    } else if (e.key === 'End' && e.ctrlKey) {
      e.preventDefault();
      to(Math.max(0, all.length - 1));
    }
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
        if (el?.open) return;
        data.cancelFetches();
        data.clearPending();
        open.set(false);
        restore?.();
        restore = null;
      }}
      onpointerdown={backdrop.onpointerdown}
      onmousedown={backdrop.onmousedown}
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
          {query.get() ? (
            <button
              class="clear"
              type="button"
              aria-label="Clear search"
              onclick={() => {
                query.set('');
                input.peek()?.focus();
              }}
            >
              <X size={14} aria-hidden="true" />
            </button>
          ) : null}
          {!touchOnly ? (
            <span class="esc">
              <Kbd keys="Escape" subtle />
            </span>
          ) : null}
          <button class="close" type="button" aria-label="Close" onclick={closePalette}>
            <X size={18} />
          </button>
        </div>
        <ul id="palette-list" class="scroll-shade" role="listbox" ref={list} aria-label="Results">
          <For
            each={grouped.get()}
            key={(g) => g.group}
            fallback={
              <li class="none" role="presentation">
                {data.loadingEntities.get() ? (
                  <span class="loading-msg">
                    <Spinner size={12} /> Searching…
                  </span>
                ) : data.failedEntityList.get().length ? (
                  `No matches for “${query.get()}” — could not load ${data.failedEntityList.get().join(', ')}.`
                ) : (
                  `No matches for “${query.get()}”`
                )}
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
                        onpointermove={(e) => onItemPointerMove(e, x().item.id)}
                        onclick={() => choose(x().item)}
                      >
                        <Dynamic component={x().item.icon} size={16} strokeWidth={1.75} aria-hidden="true" />
                        {/* The name and its hint share one block: the hint sits right of the name while both fit and drops,
                            whole, to a line under it when they do not (always on phones), so no fact in it is cut
                            (DESIGN_DIRECTION §2.3). */}
                        <span class="main">
                          <span class="label">
                            <Index each={highlightItem(query.get(), x().item)}>{(part) => <>{part().m ? <mark>{part().s}</mark> : part().s}</>}</Index>
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
  .loading-msg {
    display: inline-flex;
    align-items: center;
    gap: var(--space-2);
  }
  .clear {
    display: grid;
    place-items: center;
    flex-shrink: 0;
    width: 24px;
    height: 24px;
    border-radius: var(--radius-sm);
    color: var(--text-tertiary);
    cursor: pointer;
    transition:
      background-color var(--dur-fast) var(--ease-standard),
      color var(--dur-fast) var(--ease-standard);
  }
  .clear:hover {
    background: var(--bg-hover);
    color: var(--text-primary);
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
    .clear {
      width: 44px;
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
