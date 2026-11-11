// fw port of InfoButton.svelte.
/**
 * The ⓘ: what a chart's terms, a measure or a set of numbers mean, one tap away. A real button: a tap, a click,
 * Enter or Space opens it (a tooltip alone shows on hover and focus only, so touch could never read it), with
 * `aria-expanded` and `aria-controls`, and Escape or a tap outside closes it.
 * - Default: a small popover with the definition (`text`, or `children` for a short list), in the top layer (a manual
 *   position next to the button, collision-aware like the tooltip), so it shows over a modal dialog too.
 * - `controls`: an inline disclosure the page renders itself under `open` (Model's metric help, a block under the KPI
 *   strip); the button toggles it and Escape or a tap outside both closes it.
 * - `chart`: an ⓘ over a chart. On phones its definition opens in place, between the chart's heading and its readout
 *   (the frame's ./InfoRegion: ui/Panel provides one), so it never covers the plot it explains; wider screens keep the
 *   anchored popover. Every ⓘ in a chart's header takes it (DESIGN_DIRECTION §2.12).
 * `trigger` puts words in place of the icon (New run's "30 steps / epoch", dotted underline).
 * **Short by rule** (DESIGN_DIRECTION §2.9, help): the definition is at most two lines at 13 px (`HELP_MAX`
 * characters, `InfoButton.svelte.test.ts` scans the literals); a formula, a method or a provenance goes in `how`, a
 * fold inside it ("How it's computed", or `howLabel`), closed at every opening.
 */
import Info from '$icons/info';
import { bindable, computed, createUniqueId, css, defaults, effect, Key, MediaQuery, signal, untrack } from 'fw';
import { media } from '$lib/design/tokens';
import { infoSlot } from './infoSlot';
import { boundsFor, placeTip, popovers, type Side } from './tooltip';

/** A definition's most characters: two lines at 13 px in the 320 px popover (DESIGN_DIRECTION §2.9, help). */
export const HELP_MAX = 100;

export interface InfoButtonProps {
  /** The button's accessible name: what it explains ("What the loss terms mean"). */
  label: string;
  text?: string;
  children?: JSX.Element;
  trigger?: JSX.Element;
  /** Id of an inline region the page shows while `open` (disclosure mode: no popover). */
  controls?: string;
  /** Over a chart: on phones the definition opens in place under the chart's heading, not over the plot. */
  chart?: boolean;
  open?: boolean;
  side?: Side;
  class?: string;
  /** The detail behind the short definition (a formula, how a figure is made, where weights come from), folded. */
  how?: string | (() => JSX.Element);
  howLabel?: string;
}

export default function InfoButton(props: InfoButtonProps) {
  const p = defaults(props, { chart: false, side: 'bottom', class: '', howLabel: "How it's computed" });
  const open = bindable(props, 'open', false);

  const uid = createUniqueId();
  const popId = `${uid}-info`;
  // A chart's ⓘ on a phone: a disclosure of the frame's in-place region (./infoSlot), filled while open.
  const slot = infoSlot();
  const phone = new MediaQuery(media.phone, false);
  const inPlace = computed(() => p.chart && !!slot && phone.current);
  const controls = computed(() => (inPlace.get() ? slot!.id : p.controls));
  const me = {};
  effect(() => {
    if (!slot) return;
    const release = () => untrack(() => slot.content?.owner === me && (slot.content = null));
    if (inPlace.get() && open.get()) slot.content = { owner: me, children: help };
    else release();
    return release;
  });
  let btn!: HTMLButtonElement;
  const pop = signal<HTMLElement | undefined>(undefined);
  const native = popovers();
  /** Puts the open popover next to its button (fixed coordinates, set before the frame paints). */
  function place() {
    const el = pop.peek();
    if (!el || !open.peek()) return;
    // Measured at its natural height (the open fold may have made it taller), keeping where its own scroll was.
    const scrolled = el.scrollTop;
    el.style.maxHeight = '';
    const r = btn.getBoundingClientRect();
    const t = el.getBoundingClientRect();
    // The tooltip's bounds, 12 px further in: a card keeps the page's 16 px gutter on a phone and a panel's padding.
    const b = boundsFor(btn, t.width + 24);
    const at = placeTip(r, t, p.side, { ...b, left: b.left + 12, right: b.right - 12 });
    // Taller than the room on either side: it keeps the side and scrolls within what fits (above: its foot stays by
    // the button).
    const over = Math.max(0, at.y + t.height - b.bottom, b.top - at.y);
    if (over > 0) {
      el.style.maxHeight = `${Math.floor(t.height - over)}px`;
      if (at.side === 'top') at.y += over;
    }
    el.scrollTop = scrolled;
    el.style.translate = `${at.x}px ${at.y}px`;
  }

  function show() {
    const el = pop.peek();
    if (!controls.peek() && native && el && !el.matches(':popover-open')) el.showPopover();
    open.set(true);
    place();
  }
  function hide() {
    const el = pop.peek();
    if (!controls.peek() && native && el?.matches(':popover-open')) el.hidePopover();
    open.set(false);
  }

  // A chart's ⓘ left open across the phone breakpoint: its in-place region is gone and the popover takes over, so the
  // popover is shown too (else the definition would vanish while aria-expanded still said it was open).
  effect(() => {
    const el = pop.get();
    if (!open.get() || controls.get() || !native || !el) return;
    untrack(() => {
      if (el.matches(':popover-open')) return;
      el.showPopover();
      place();
    });
  });

  // A press on the button while it is open closes it. The popover's light dismiss closes it on the same press before
  // the click arrives, so the state is read on pointerdown, not on click (else the click would open it again).
  let pressedOpen: boolean | null = null;
  function onclick() {
    const was = pressedOpen ?? open.peek();
    pressedOpen = null;
    if (was) hide();
    else show();
  }

  // The native popover closes itself (Escape, a tap outside): the state follows.
  function ontoggle(e: Event) {
    open.set((e as ToggleEvent).newState === 'open');
  }

  // While open: the popover follows its button through scrolls and resizes. Without the native popover (and for the
  // inline disclosure), Escape and a press outside close it here.
  effect(() => {
    if (!open.get()) return;
    const ctl = controls.get();
    // Scroll and resize events come at most once a frame: placing at once keeps the popover on its button.
    const follow = () => place();
    if (!ctl) {
      window.addEventListener('scroll', follow, true);
      window.addEventListener('resize', follow);
    }
    const own = ctl || !native;
    const region = () => (controls.peek() ? document.getElementById(controls.peek()!) : pop.peek());
    const onkey = (e: KeyboardEvent) => {
      if (e.key !== 'Escape') return;
      const r = region();
      const inside = btn.contains(document.activeElement) || !!r?.contains(document.activeElement);
      hide();
      if (inside) btn.focus();
      e.stopPropagation();
    };
    const ondown = (e: PointerEvent) => {
      const r = region();
      if (e.target instanceof Node && (btn.contains(e.target) || r?.contains(e.target))) return;
      hide();
    };
    if (own) {
      document.addEventListener('keydown', onkey, true);
      document.addEventListener('pointerdown', ondown, true);
    }
    return () => {
      window.removeEventListener('scroll', follow, true);
      window.removeEventListener('resize', follow);
      document.removeEventListener('keydown', onkey, true);
      document.removeEventListener('pointerdown', ondown, true);
    };
  });

  function help() {
    return (
      <>
        <span class="info-text">{props.children ? props.children : props.text}</span>{' '}
        {props.how ? (
          // Closed each time the definition opens: the short line is what most readers need.
          <Key when={open.get()}>
            {/* Opening the fold makes the popover taller: placed again, so it stays on screen (it scrolls past the viewport). */}
            <details class="info-how" ontoggle={place}>
              <summary>{p.howLabel}</summary>
              <span class="info-how-body">{typeof props.how === 'string' ? props.how : props.how}</span>
            </details>
          </Key>
        ) : null}
      </>
    );
  }

  return (
    <>
      <button
        ref={(el) => (btn = el)}
        type="button"
        class={`info-btn ${p.class}`}
        class:words={!!props.trigger}
        aria-label={props.trigger ? undefined : p.label}
        aria-expanded={open.get()}
        aria-controls={controls.get() ?? popId}
        onpointerdown={() => (pressedOpen = open.peek())}
        onclick={onclick}
      >
        {props.trigger ? (
          <>
            {props.trigger}
            <span class="sr-only">{`. ${p.label}`}</span>
          </>
        ) : (
          <Info size={14} aria-hidden="true" />
        )}
      </button>{' '}
      {!controls.get() ? (
        // Next to its button in the reading order; shown in the top layer.
        <span ref={pop} id={popId} class="info-pop" class:open={open.get() && !native} popover={native ? 'auto' : undefined} ontoggle={ontoggle}>
          {help()}
        </span>
      ) : null}
    </>
  );
}

css`
  .info-btn {
    position: relative;
    flex-shrink: 0;
    display: inline-grid;
    place-items: center;
    width: var(--control-sm);
    height: var(--control-sm);
    border-radius: var(--radius-sm);
    color: var(--text-tertiary);
  }
  .info-btn:hover,
  .info-btn[aria-expanded='true'] {
    background: var(--bg-hover);
    color: var(--text-primary);
  }
  /* Words as the trigger: the value keeps its own look, a dotted underline says there is more behind it. */
  .info-btn.words {
    display: inline;
    width: auto;
    height: auto;
    padding: 0;
    border-radius: 2px;
    color: inherit;
    font: inherit;
    text-decoration: underline dotted var(--text-tertiary);
    text-underline-offset: 3px;
    cursor: help;
  }
  .info-btn.words:hover,
  .info-btn.words[aria-expanded='true'] {
    background: none;
    text-decoration-color: var(--text-primary);
  }
  /* Touch: at least a 44 × 44 px target around the 24 px button. */
  @media (pointer: coarse) {
    .info-btn::after {
      content: '';
      position: absolute;
      inset: calc((var(--control-sm) - 44px) / 2);
    }
    .info-btn.words::after {
      inset: -12px -4px;
    }
  }
  /* The popover's UA box (centred, scrolling) is reset to a small card at fixed coordinates, a menu's surface. */
  .info-pop {
    position: fixed;
    inset: 0 auto auto 0;
    margin: 0;
    width: max-content;
    max-width: min(320px, calc(100vw - 32px));
    padding: var(--space-3);
    border: 1px solid var(--line-default);
    border-radius: var(--radius-lg);
    background: var(--bg-overlay);
    color: var(--text-secondary);
    box-shadow: var(--shadow-md);
    font-size: var(--type-base-size);
    line-height: var(--type-base-line);
    font-weight: var(--weight-regular);
    text-align: left;
    white-space: normal;
    overflow-y: auto;
    overscroll-behavior: contain;
    z-index: var(--z-tooltip);
  }
  .info-text {
    display: block;
    max-inline-size: var(--measure);
  }
  /* The detail, one tap further: a quiet disclosure under the definition, its body at the 12 px helper size. */
  .info-how {
    margin-top: var(--space-2);
  }
  .info-how summary {
    display: inline-flex;
    align-items: center;
    min-height: 24px;
    color: var(--text-primary);
    text-decoration: underline;
    text-decoration-color: var(--line-strong);
    text-underline-offset: 3px;
    cursor: pointer;
    list-style: none;
  }
  .info-how summary::-webkit-details-marker {
    display: none;
  }
  .info-how summary::after {
    content: ' ›';
    white-space: pre;
  }
  .info-how[open] summary::after {
    content: ' ⌄';
  }
  .info-how-body {
    display: block;
    margin-top: var(--space-1);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
  }
  @media (pointer: coarse) {
    .info-how summary {
      min-height: 44px;
    }
  }
  .info-pop:not(:popover-open):not(.open) {
    display: none;
  }
  /* Opening is the change worth showing: a 120 ms fade, as a menu opens (0 under reduced motion). */
  .info-pop:popover-open,
  .info-pop.open {
    animation: pop-in var(--dur-fast) var(--ease-out);
  }
  @keyframes pop-in {
    from {
      opacity: 0;
    }
  }
  .info-pop :global(b),
  .info-pop :global(dt) {
    color: var(--text-primary);
    font-weight: var(--weight-medium);
  }
`;
