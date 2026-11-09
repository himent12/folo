<script lang="ts" module>
  export type Smoothing = '0' | '0.6' | '0.9' | '0.97';
  export type Scale = 'lin' | 'log';

  /** The trigger's text: what the chart shows now, e.g. "EMA 0.9 · Log". */
  export function lossSummary(smoothing: Smoothing, scale: Scale): string {
    return `${smoothing === '0' ? 'Raw' : `EMA ${smoothing}`} · ${scale === 'log' ? 'Log' : 'Linear'}`;
  }
</script>

<script lang="ts">
  /**
   * The loss chart's one control: a compact button naming the current view ("EMA 0.9 · Log") that opens a small panel
   * with the two choices behind it, smoothing and y scale, as native radio groups (arrow keys move within a group, Tab
   * between them). Escape or a click outside closes it and focus returns to the button. On phones the panel is a bottom
   * sheet over a scrim, like the other phone menus, so it never floats over the chart's legend.
   */
  import ChevronDown from '@lucide/svelte/icons/chevron-down';
  import Check from '@lucide/svelte/icons/check';
  import { tick } from 'svelte';

  let { smoothing = $bindable(), scale = $bindable() }: { smoothing: Smoothing; scale: Scale } = $props();

  const uid = $props.id();
  let open = $state(false);
  let root = $state<HTMLElement>();
  let trigger = $state<HTMLButtonElement>();

  const SMOOTHING: { value: Smoothing; label: string }[] = [
    { value: '0', label: 'Raw' },
    { value: '0.6', label: 'EMA 0.6' },
    { value: '0.9', label: 'EMA 0.9' },
    { value: '0.97', label: 'EMA 0.97' },
  ];
  const SCALE: { value: Scale; label: string }[] = [
    { value: 'log', label: 'Log' },
    { value: 'lin', label: 'Linear' },
  ];

  async function show() {
    open = true;
    await tick();
    root?.querySelector<HTMLInputElement>('input:checked')?.focus();
  }
  function hide(refocus = true) {
    open = false;
    if (refocus) trigger?.focus();
  }
  // On the whole control, not just the panel: Escape also closes it from the trigger (after Shift+Tab back to it).
  function onkeydown(e: KeyboardEvent) {
    if (open && e.key === 'Escape') {
      e.preventDefault();
      e.stopPropagation();
      hide();
    }
  }
  // Tabbing out of the panel closes it.
  function onfocusout(e: FocusEvent) {
    if (open && root && !root.contains(e.relatedTarget as Node | null)) hide(false);
  }
  $effect(() => {
    if (!open) return;
    const away = (e: PointerEvent) => {
      if (root && !root.contains(e.target as Node)) hide(false);
    };
    document.addEventListener('pointerdown', away);
    return () => document.removeEventListener('pointerdown', away);
  });
</script>

{#snippet group(name: string, legend: string, options: { value: string; label: string }[], value: string, set: (v: string) => void)}
  <fieldset>
    <legend>{legend}</legend>
    {#each options as o (o.value)}
      <label class="opt" class:on={value === o.value}>
        <input type="radio" name="{uid}-{name}" value={o.value} checked={value === o.value} onchange={() => set(o.value)} />
        <span>{o.label}</span>
        {#if value === o.value}<Check size={14} aria-hidden="true" />{/if}
      </label>
    {/each}
  </fieldset>
{/snippet}

<!-- svelte-ignore a11y_no_static_element_interactions -->
<div class="lo" bind:this={root} {onfocusout} {onkeydown}>
  <button
    bind:this={trigger}
    type="button"
    class="trigger num"
    aria-haspopup="dialog"
    aria-expanded={open}
    aria-controls="{uid}-panel"
    aria-label="Loss view: {lossSummary(smoothing, scale)}"
    onclick={() => (open ? hide() : show())}
  >
    <span>{lossSummary(smoothing, scale)}</span>
    <ChevronDown size={14} strokeWidth={1.75} aria-hidden="true" />
  </button>
  {#if open}
    <!-- Phones: the scrim behind the sheet; a tap on it closes the sheet. -->
    <!-- svelte-ignore a11y_click_events_have_key_events, a11y_no_static_element_interactions (Escape closes it too) -->
    <div class="scrim" aria-hidden="true" onclick={() => hide()}></div>
    <div class="panel" id="{uid}-panel" role="dialog" aria-label="Loss view" tabindex="-1">
      <p class="sheet-title" aria-hidden="true">Loss view</p>
      {@render group('smoothing', 'Smoothing', SMOOTHING, smoothing, (v) => (smoothing = v as Smoothing))}
      {@render group('scale', 'Y scale', SCALE, scale, (v) => (scale = v as Scale))}
    </div>
  {/if}
</div>

<style>
  .lo {
    position: relative;
    display: inline-flex;
  }
  .trigger {
    display: inline-flex;
    align-items: center;
    gap: var(--space-1);
    height: var(--control-sm);
    padding: 0 6px 0 8px;
    border-radius: var(--radius-sm);
    background: var(--bg-raised);
    border: 1px solid var(--line-control);
    color: var(--text-primary);
    font-size: var(--type-sm-size);
    white-space: nowrap;
    transition: border-color var(--dur-fast) var(--ease-standard);
  }
  .trigger :global(svg) {
    color: var(--text-tertiary);
  }
  .trigger:hover {
    border-color: var(--text-tertiary);
  }
  .trigger:focus-visible {
    outline: none;
    border-color: var(--focus);
    box-shadow: 0 0 0 3px color-mix(in srgb, var(--focus) 18%, transparent);
  }
  .panel {
    position: absolute;
    top: calc(100% + 6px);
    right: 0;
    z-index: var(--z-overlay);
    display: grid;
    gap: var(--space-1);
    min-width: 176px;
    padding: var(--space-1);
    background: var(--bg-overlay);
    border: 1px solid var(--line-default);
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow-md);
    animation: in var(--dur-fast) var(--ease-out);
  }
  fieldset {
    display: grid;
    border: 0;
    padding: 0;
    margin: 0;
    min-width: 0;
  }
  /* A hairline between the groups, on the second one's caption (a fieldset's own border would run through it). */
  fieldset + fieldset legend {
    width: 100%;
    margin-top: var(--space-1);
    padding-top: var(--space-2);
    border-top: 1px solid var(--line-subtle);
  }
  legend {
    padding: var(--space-1) var(--space-3) var(--space-1);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-tertiary);
  }
  .opt {
    position: relative;
    display: flex;
    align-items: center;
    gap: var(--space-2);
    height: 28px;
    padding: 0 var(--space-3);
    border-radius: var(--radius-md);
    color: var(--text-secondary);
    cursor: pointer;
    font-variant-numeric: tabular-nums;
  }
  .opt span {
    flex: 1;
  }
  .opt input {
    position: absolute;
    opacity: 0;
    width: 1px;
    height: 1px;
    margin: 0;
  }
  .opt:hover {
    background: var(--bg-hover);
  }
  .opt:has(input:focus-visible) {
    outline: 2px solid var(--focus);
    outline-offset: -2px;
  }
  /* The chosen option of a radio group in a menu: its check mark, in ink (the lime bar is for selected rows in lists). */
  .opt.on {
    color: var(--text-primary);
  }
  .scrim,
  .sheet-title {
    display: none;
  }
  @media (pointer: coarse) {
    .trigger {
      height: 44px;
    }
    .opt {
      height: 44px;
    }
  }
  @keyframes in {
    from {
      opacity: 0;
      transform: translateY(-2px);
    }
  }
  @media (max-width: 767px) {
    /* Phones: a bottom sheet over a scrim (as Dialog's phone sheets), clear of the home indicator. */
    .scrim {
      display: block;
      position: fixed;
      inset: 0;
      z-index: var(--z-dialog);
      background: var(--bg-scrim);
      animation: fade var(--dur-slow) var(--ease-out);
    }
    .panel {
      position: fixed;
      top: auto;
      left: 0;
      right: 0;
      bottom: 0;
      z-index: var(--z-dialog);
      min-width: 0;
      padding: var(--space-2) var(--space-2) calc(var(--space-4) + env(safe-area-inset-bottom));
      border-width: 1px 0 0;
      border-radius: var(--radius-xl) var(--radius-xl) 0 0;
      box-shadow: var(--shadow-lg);
      animation: sheet var(--dur-slow) var(--ease-out);
    }
    .sheet-title {
      display: block;
      padding: var(--space-2) var(--space-3) var(--space-1);
      font-size: var(--type-md-size);
      line-height: var(--type-md-line);
      font-weight: var(--weight-medium);
      color: var(--text-primary);
    }
  }
  @keyframes fade {
    from {
      opacity: 0;
    }
  }
  @keyframes sheet {
    from {
      opacity: 0;
      transform: translateY(24px);
    }
  }
</style>
