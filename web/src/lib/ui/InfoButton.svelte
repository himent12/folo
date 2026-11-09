<script lang="ts">
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
   */
  import Info from '@lucide/svelte/icons/info';
  import { untrack, type Snippet } from 'svelte';
  import { MediaQuery } from 'svelte/reactivity';
  import { media } from '$lib/design/tokens';
  import { infoSlot } from './infoSlot.svelte';
  import { boundsFor, placeTip, popovers, type Side } from './tooltip';

  let {
    label,
    text,
    children,
    trigger,
    controls: controlsProp,
    chart = false,
    open = $bindable(false),
    side = 'bottom',
    class: klass = '',
  }: {
    /** The button's accessible name: what it explains ("What the loss terms mean"). */
    label: string;
    text?: string;
    children?: Snippet;
    trigger?: Snippet;
    /** Id of an inline region the page shows while `open` (disclosure mode: no popover). */
    controls?: string;
    /** Over a chart: on phones the definition opens in place under the chart's heading, not over the plot. */
    chart?: boolean;
    open?: boolean;
    side?: Side;
    class?: string;
  } = $props();

  const uid = $props.id();
  const popId = `${uid}-info`;
  // A chart's ⓘ on a phone: a disclosure of the frame's in-place region (./infoSlot), filled while open.
  const slot = chart ? infoSlot() : undefined;
  const phone = new MediaQuery(media.phone, false);
  const inPlace = $derived(!!slot && phone.current);
  const controls = $derived(inPlace ? slot!.id : controlsProp);
  const me = {};
  $effect(() => {
    if (!slot) return;
    const release = () => untrack(() => slot.content?.owner === me && (slot.content = null));
    if (inPlace && open) slot.content = { owner: me, text, children };
    else release();
    return release;
  });
  let btn: HTMLButtonElement;
  let pop = $state<HTMLElement>();
  const native = popovers();
  /** Puts the open popover next to its button (fixed coordinates, set before the frame paints). */
  function place() {
    if (!pop || !open) return;
    const r = btn.getBoundingClientRect();
    const t = pop.getBoundingClientRect();
    // The tooltip's bounds, 12 px further in: a card keeps the page's 16 px gutter on a phone and a panel's padding.
    const b = boundsFor(btn, t.width + 24);
    const p = placeTip(r, t, side, { ...b, left: b.left + 12, right: b.right - 12 });
    pop.style.translate = `${p.x}px ${p.y}px`;
  }

  function show() {
    if (!controls && native && pop && !pop.matches(':popover-open')) pop.showPopover();
    open = true;
    place();
  }
  function hide() {
    if (!controls && native && pop?.matches(':popover-open')) pop.hidePopover();
    open = false;
  }

  // A press on the button while it is open closes it. The popover's light dismiss closes it on the same press before
  // the click arrives, so the state is read on pointerdown, not on click (else the click would open it again).
  let pressedOpen: boolean | null = null;
  function onclick() {
    const was = pressedOpen ?? open;
    pressedOpen = null;
    if (was) hide();
    else show();
  }

  // The native popover closes itself (Escape, a tap outside): the state follows.
  function ontoggle(e: Event) {
    open = (e as ToggleEvent).newState === 'open';
  }

  // While open: the popover follows its button through scrolls and resizes. Without the native popover (and for the
  // inline disclosure), Escape and a press outside close it here.
  $effect(() => {
    if (!open) return;
    // Scroll and resize events come at most once a frame: placing at once keeps the popover on its button.
    const follow = () => place();
    if (!controls) {
      window.addEventListener('scroll', follow, true);
      window.addEventListener('resize', follow);
    }
    const own = controls || !native;
    const onkey = (e: KeyboardEvent) => {
      if (e.key !== 'Escape') return;
      const region = controls ? document.getElementById(controls) : pop;
      const inside = btn.contains(document.activeElement) || !!region?.contains(document.activeElement);
      hide();
      if (inside) btn.focus();
      e.stopPropagation();
    };
    const ondown = (e: PointerEvent) => {
      const region = controls ? document.getElementById(controls) : pop;
      if (e.target instanceof Node && (btn.contains(e.target) || region?.contains(e.target))) return;
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
</script>

<button
  bind:this={btn}
  type="button"
  class="info-btn {klass}"
  class:words={!!trigger}
  aria-label={trigger ? undefined : label}
  aria-expanded={open}
  aria-controls={controls ?? popId}
  onpointerdown={() => (pressedOpen = open)}
  {onclick}
>
  {#if trigger}{@render trigger()}<span class="sr-only">. {label}</span>{:else}<Info size={14} aria-hidden="true" />{/if}
</button>
{#if !controls}
  <!-- Next to its button in the reading order; shown in the top layer. -->
  <span
    bind:this={pop}
    id={popId}
    class="info-pop"
    class:open={open && !native}
    popover={native ? 'auto' : undefined}
    {ontoggle}
  >
    {#if children}{@render children()}{:else}{text}{/if}
  </span>
{/if}

<style>
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
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    font-weight: var(--weight-regular);
    text-align: left;
    white-space: normal;
    overflow: visible;
    z-index: var(--z-tooltip);
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
</style>
