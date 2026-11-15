/**
 * An overflow menu: a "⋯" button that opens a short list of actions (links or buttons). Keyboard (WAI-ARIA menu
 * button): Enter, Space or ArrowDown on the button opens it on the first item (ArrowUp on the last); in the list,
 * the arrows wrap, Home and End go to the ends, typing a letter moves to the next item starting with it (typeahead),
 * Escape closes it back to the button, Tab closes it and moves on. Roving tabindex: the item last focused is the one
 * in the tab order.
 * On phones it is the shared bottom sheet (ui/Dialog, as the loss view's options): the items as full-width 44 px
 * rows under the sheet's title, never a popover that lands on the text beside its button (round 24's queued run).
 * Tab there moves between the sheet's Close and that item (the arrows move among the items).
 */
import Ellipsis from '$icons/ellipsis';
import { computed, createUniqueId, css, defaults, Dynamic, effect, For, MediaQuery, signal, tick } from 'fw';
import { media } from '$lib/design/tokens';
import Button from './Button';
import Dialog from './Dialog';
import type { IconType } from './types';

export interface MenuItem {
  label: string;
  icon?: IconType;
  href?: string;
  onclick?: () => void;
  danger?: boolean;
}

export interface MenuProps {
  label: string;
  items: MenuItem[];
  variant?: 'secondary' | 'ghost';
  /** A row's menu inside a box that clips (a table that scrolls sideways): the list is placed on the viewport. */
  fixed?: boolean;
}

export default function Menu(props: MenuProps) {
  const p = defaults(props, { variant: 'secondary', fixed: false });
  const uid = createUniqueId();
  const open = signal(false);
  /**
   * The items in the order shown: a destructive item (Delete, Remove from queue) always last, under a hairline, so
   * every detail page's ⋯ ends the same way whatever order a caller gives (DESIGN_DIRECTION §13).
   */
  const shown = computed(() => [...p.items.filter((i) => !i.danger), ...p.items.filter((i) => i.danger)]);
  const root = signal<HTMLElement>();
  const list = signal<HTMLElement>();
  /** The item in the tab order (roving tabindex): the one focused last. */
  const active = signal(0);
  const phone = new MediaQuery(media.phone, false);
  /** `fixed`: where the list sits, under the trigger's right edge. */
  const at = signal<{ top: number; right: number; max: number | null } | null>(null);

  function place() {
    const r = root.get()?.getBoundingClientRect();
    if (!r) return;
    // Under the button; above it when the list (measured once shown) would cross the window's bottom edge and fits
    // above: a row near the bottom would otherwise put its last items, Delete among them, out of reach (the page may not
    // scroll, and scrolling the page closes a placed list). Where it fits on neither side (a short window), it takes
    // the side with more room, capped at the window's edge, and scrolls inside itself.
    const el = list.get();
    // Its whole height (border included), not the capped one.
    const h = el ? el.scrollHeight + el.offsetHeight - el.clientHeight : 0;
    const below = r.bottom + 6;
    const roomBelow = window.innerHeight - 8 - below;
    const roomAbove = r.top - 6 - 8;
    let top = below;
    let max: number | null = null;
    if (h > roomBelow) {
      if (h <= roomAbove) top = r.top - 6 - h;
      else if (roomAbove > roomBelow) (top = 8), (max = Math.max(roomAbove, 44));
      else max = Math.max(roomBelow, 44);
    }
    at.set({ top, right: Math.max(8, window.innerWidth - r.right), max });
  }

  const entries = () => [...(list.get()?.querySelectorAll<HTMLElement>('[role="menuitem"]') ?? [])];

  async function show(where: 'first' | 'last' = 'first') {
    if (p.fixed) place();
    active.set(where === 'first' ? 0 : shown.get().length - 1);
    open.set(true);
    await tick();
    // Placed again now that the list has a height: above the button where it does not fit below.
    if (p.fixed) place();
    const all = entries();
    (where === 'first' ? all[0] : all.at(-1))?.focus();
  }
  function ontrigger(e: KeyboardEvent) {
    if (e.key !== 'ArrowDown' && e.key !== 'ArrowUp') return;
    e.preventDefault();
    void show(e.key === 'ArrowDown' ? 'first' : 'last');
  }
  // Typeahead: the letters typed within half a second, matched against the start of each label. A new search starts
  // after the focused item; a word being typed may stay on it; one letter pressed again cycles through its items.
  let typed = '';
  let typedAt = 0;
  function typeahead(key: string, all: HTMLElement[], i: number) {
    const now = Date.now();
    const k = key.toLowerCase();
    typed = now - typedAt < 500 && typed !== k ? typed + k : k;
    typedAt = now;
    const start = typed.length > 1 ? Math.max(0, i) : i + 1;
    const order = [...all.slice(start), ...all.slice(0, start)];
    order.find((el) => el.textContent?.trim().toLowerCase().startsWith(typed))?.focus();
  }
  function hide(refocus = true) {
    open.set(false);
    if (refocus) root.get()?.querySelector<HTMLElement>('button')?.focus();
  }
  async function choose(it: MenuItem) {
    // Focus goes back to the trigger first, so a dialog opened by the action returns focus there on close. The phone
    // sheet is modal: it closes (and the page leaves inert) before the trigger can take focus.
    hide(true);
    if (phone.current) {
      await tick();
      root.get()?.querySelector<HTMLElement>('button')?.focus();
    }
    it.onclick?.();
  }

  function onfocusin(e: FocusEvent) {
    const i = entries().indexOf(e.target as HTMLElement);
    if (i >= 0) active.set(i);
  }

  function onkeydown(e: KeyboardEvent) {
    const all = entries();
    const i = all.indexOf(document.activeElement as HTMLElement);
    if (e.key === 'Escape') {
      e.preventDefault();
      hide();
    } else if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
      e.preventDefault();
      const d = e.key === 'ArrowDown' ? 1 : -1;
      all[(i + d + all.length) % all.length]?.focus();
    } else if (e.key === 'Home' || e.key === 'End') {
      e.preventDefault();
      (e.key === 'Home' ? all[0] : all.at(-1))?.focus();
    } else if (e.key === 'Tab') {
      // The phone sheet is modal: Tab moves on to its Close, and from there back to the active item.
      if (!phone.current) hide(false);
    } else if (e.key.length === 1 && e.key !== ' ' && !e.metaKey && !e.ctrlKey && !e.altKey) {
      e.preventDefault();
      typeahead(e.key, all, i);
    }
  }

  effect(() => {
    if (!open.get() || phone.current) return;
    const away = (e: PointerEvent) => {
      const r = root.get();
      if (r && !r.contains(e.target as Node)) hide(false);
    };
    document.addEventListener('pointerdown', away);
    // A list placed on the viewport would drift from its row: scrolling or resizing closes it.
    const drift = (e: Event) => {
      if (!list.get()?.contains(e.target as Node)) hide(false);
    };
    if (p.fixed) {
      window.addEventListener('scroll', drift, true);
      window.addEventListener('resize', drift);
    }
    return () => {
      document.removeEventListener('pointerdown', away);
      window.removeEventListener('scroll', drift, true);
      window.removeEventListener('resize', drift);
    };
  });

  const entriesList = () => (
    <For each={shown.get()} key={(it) => it.label}>
      {(it, i) => {
        const tabindex = () => (i() === Math.min(active.get(), shown.get().length - 1) ? 0 : -1);
        const icon = () => (it().icon ? <Dynamic component={it().icon} size={16} strokeWidth={1.75} aria-hidden="true" /> : null);
        return (
          <>
            {it().danger && i() > 0 && !shown.get()[i() - 1]!.danger ? <div class="sep" role="separator"></div> : null}
            {it().href ? (
              <a role="menuitem" tabindex={tabindex()} href={it().href} class:danger={!!it().danger} onclick={() => hide(false)}>
                {icon()}
                {it().label}
              </a>
            ) : (
              <button role="menuitem" tabindex={tabindex()} class:danger={!!it().danger} onclick={() => choose(it())}>
                {icon()}
                {it().label}
              </button>
            )}
          </>
        );
      }}
    </For>
  );

  return (
    <>
      <div class="menu" ref={root}>
        <Button
          variant={p.variant}
          size={p.variant === 'ghost' ? 'sm' : undefined}
          icon={Ellipsis}
          aria-label={p.label}
          title={p.label}
          aria-haspopup="menu"
          aria-expanded={open.get()}
          aria-controls={`${uid}-list`}
          onclick={() => (open.get() ? hide() : show())}
          onkeydown={ontrigger}
        />
        {open.get() && !phone.current ? (
          <div
            class="list"
            class:fixed={p.fixed}
            id={`${uid}-list`}
            role="menu"
            aria-label={p.label}
            ref={list}
            style:top={p.fixed && at.get() ? `${at.get()!.top}px` : undefined}
            style:right={p.fixed && at.get() ? `${at.get()!.right}px` : undefined}
            style:max-height={p.fixed && at.get()?.max != null ? `${at.get()!.max}px` : undefined}
            onkeydown={onkeydown}
            onfocusin={onfocusin}
          >
            {entriesList()}
          </div>
        ) : null}
      </div>
      {/* Phones: the shared bottom sheet. Dialog returns focus to the button when it closes. */}
      {phone.current ? (
        <Dialog bind:open={open} title={p.label} size="sm">
          <div class="sheet" id={`${uid}-list`} role="menu" aria-label={p.label} ref={list} onkeydown={onkeydown} onfocusin={onfocusin}>
            {entriesList()}
          </div>
        </Dialog>
      ) : null}
    </>
  );
}

css`
  .menu {
    position: relative;
    display: inline-flex;
  }
  .list {
    position: absolute;
    top: calc(100% + 6px);
    right: 0;
    z-index: var(--z-overlay);
    min-width: 200px;
    padding: var(--space-1);
    background: var(--bg-overlay);
    border: 1px solid var(--line-default);
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow-md);
    display: grid;
    animation: in var(--dur-fast) var(--ease-out);
  }
  /* Capped by place() in a window too short for it: the rest scrolls inside the list (which keeps it open). */
  .list.fixed {
    position: fixed;
    overflow-y: auto;
    overscroll-behavior: contain;
  }
  [role='menuitem'] {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    height: 36px;
    padding: 0 var(--space-3);
    border-radius: var(--radius-md);
    color: var(--text-primary);
    text-align: left;
    white-space: nowrap;
  }
  [role='menuitem'] :global(svg) {
    color: var(--text-tertiary);
  }
  [role='menuitem']:hover {
    background: var(--bg-hover);
  }
  /* Keyboard focus is its own mark (an inset ring in the focus colour), not the faint hover fill. */
  [role='menuitem']:focus-visible {
    background: var(--bg-hover);
    outline: var(--size-focus-ring) solid var(--focus);
    outline-offset: -2px;
  }
  /* A destructive item is last, under a hairline, its icon neutral at rest like every other item's and in the danger
     hue only while it is pointed at or focused; the label stays ink, and red text is the confirm dialog's alone. */
  [role='menuitem'].danger:is(:hover, :focus-visible) :global(svg) {
    color: var(--danger-text);
  }
  .sep {
    height: 1px;
    margin: var(--space-1) 0;
    background: var(--line-subtle);
  }
  .sheet .sep {
    margin: var(--space-1) var(--space-3);
  }
  @media (pointer: coarse) {
    [role='menuitem'] {
      height: 44px;
    }
  }
  @keyframes in {
    from {
      opacity: 0;
      transform: translateY(-2px);
    }
  }
  /* The sheet's items run edge to edge of its body, as the loss view's options do. */
  .sheet {
    display: grid;
    margin: 0 calc(-1 * var(--space-3));
  }
  .sheet [role='menuitem'] {
    height: 44px;
  }
`;
