// fw port of LossOptions.svelte.
import ChevronDown from '$icons/chevron-down';
import Check from '$icons/check';
import { bindable, createUniqueId, css, effect, For, MediaQuery, signal, tick } from 'fw';
import Dialog from '$lib/ui/Dialog';
import { media } from '$lib/design/tokens';

export type Smoothing = '0' | '0.6' | '0.9' | '0.98';
export type Scale = 'lin' | 'log';

/**
 * The smoothing a per-step loss opens with: an EMA weight is per step, so 0.9 averages ~10 steps, a fraction of an
 * epoch, and the line kept the batch-to-batch noise (round 27: "EMA 0.9 still looks raw"); 0.98 averages ~50, so the
 * trend and the terms' gaps read while the mosaic-off drop still shows within an epoch or two. The raw last step is
 * always stated beside it (the Loss tab's "raw · smoothed").
 */
export const DEFAULT_SMOOTHING: Smoothing = '0.98';

/** The trigger's text: what the chart shows now, e.g. "EMA 0.98 · Log". */
export function lossSummary(smoothing: Smoothing, scale: Scale): string {
  return `${smoothing === '0' ? 'Raw' : `EMA ${smoothing}`} · ${scale === 'log' ? 'Log' : 'Linear'}`;
}

export interface LossOptionsProps {
  smoothing: Smoothing;
  scale: Scale;
}

const SMOOTHING: { value: Smoothing; label: string }[] = [
  { value: '0', label: 'Raw' },
  { value: '0.6', label: 'EMA 0.6' },
  { value: '0.9', label: 'EMA 0.9' },
  { value: '0.98', label: 'EMA 0.98' },
];
const SCALE: { value: Scale; label: string }[] = [
  { value: 'log', label: 'Log' },
  { value: 'lin', label: 'Linear' },
];

/**
 * The loss chart's one control: a compact button naming the current view ("EMA 0.98 · Log") that opens a small panel
 * with the two choices behind it, smoothing and y scale, as native radio groups (arrow keys move within a group, Tab
 * between them). On wider screens the panel is a non-modal popover: Escape, Tab away or a click outside closes it
 * and Escape returns focus to the button. On phones it is the shared modal sheet (ui/Dialog: one dismiss, its Close,
 * focus held inside, the page inert, focus back on the button when it closes), each row with a radio mark. A choice
 * applies at once in both.
 * The trigger's focus ring is the global `:focus-visible` one (a tinted halo is under 3:1 on light panels).
 */
export default function LossOptions(props: LossOptionsProps) {
  const smoothing = bindable(props, 'smoothing');
  const scale = bindable(props, 'scale');

  const uid = createUniqueId();
  const open = signal(false);
  let root!: HTMLElement;
  let trigger!: HTMLButtonElement;
  const sheetBody = signal<HTMLElement>();
  const phone = new MediaQuery(media.phone, false);

  async function show() {
    open.set(true);
    await tick();
    // The chosen smoothing takes focus (in the sheet, once the dialog has opened and moved focus to its first control).
    (phone.current ? sheetBody.peek() : root)?.querySelector<HTMLInputElement>('input:checked')?.focus();
  }
  function hide(refocus = true) {
    open.set(false);
    if (refocus) trigger?.focus();
  }
  // On the whole control, not just the panel: Escape also closes it from the trigger (after Shift+Tab back to it).
  function onkeydown(e: KeyboardEvent) {
    if (open.peek() && e.key === 'Escape') {
      e.preventDefault();
      e.stopPropagation();
      hide();
    }
  }
  // Tabbing out of the popover closes it (the phone sheet is modal: it holds focus until Close or Done).
  function onfocusout(e: FocusEvent) {
    if (open.peek() && !phone.current && root && !root.contains(e.relatedTarget as Node | null)) hide(false);
  }
  effect(() => {
    if (!open.get() || phone.current) return;
    const away = (e: PointerEvent) => {
      if (root && !root.contains(e.target as Node)) hide(false);
    };
    document.addEventListener('pointerdown', away);
    return () => document.removeEventListener('pointerdown', away);
  });

  const group = (name: string, legend: string, options: { value: string; label: string }[], value: () => string, set: (v: string) => void) => (
    <fieldset>
      <legend class="group-label">{legend}</legend>
      <For each={options} key={(o) => o.value}>
        {(o) => (
          <label class="opt" class:on={value() === o().value}>
            <input type="radio" name={`${uid}-${name}`} value={o().value} checked={value() === o().value} onchange={() => set(o().value)} />
            {phone.current ? <span class="mark" aria-hidden="true"></span> : null}
            <span>{o().label}</span>
            {value() === o().value && !phone.current ? <Check size={14} aria-hidden="true" /> : null}
          </label>
        )}
      </For>
    </fieldset>
  );

  const groups = () => (
    <>
      {group('smoothing', 'Smoothing', SMOOTHING, () => smoothing.get(), (v) => smoothing.set(v as Smoothing))}
      {group('scale', 'Y scale', SCALE, () => scale.get(), (v) => scale.set(v as Scale))}
    </>
  );

  return (
    <>
      <div class="lo" ref={(e) => (root = e)} onfocusout={onfocusout} onkeydown={onkeydown}>
        <button
          ref={(e) => (trigger = e)}
          type="button"
          class="trigger num"
          aria-haspopup="dialog"
          aria-expanded={open.get()}
          aria-controls={`${uid}-panel`}
          aria-label={`Loss view: ${lossSummary(smoothing.get(), scale.get())}`}
          onclick={() => (open.peek() ? hide() : show())}
        >
          <span>{lossSummary(smoothing.get(), scale.get())}</span>
          <ChevronDown size={14} strokeWidth={1.75} aria-hidden="true" />
        </button>
        {open.get() && !phone.current ? (
          <div class="panel" id={`${uid}-panel`} role="dialog" aria-label="Loss view" tabindex={-1}>
            {groups()}
          </div>
        ) : null}
      </div>
      {/* Phones: the shared modal sheet. Dialog returns focus to the button when it closes. */}
      {phone.current ? (
        <Dialog bind:open={open} title="Loss view" size="sm">
          <div class="sheet" id={`${uid}-panel`} ref={sheetBody}>
            {groups()}
          </div>
        </Dialog>
      ) : null}
    </>
  );
}

css`
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
  /* Type and ink: the shared group label (base.css .group-label). */
  legend {
    padding: var(--space-1) var(--space-3) var(--space-1);
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
  /* The chosen option of a radio group in a menu: its check mark, in ink (the ink bar is for selected rows in lists). */
  .opt.on {
    color: var(--text-primary);
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
  /* The sheet's groups run edge to edge of its body, the options as full-width rows. */
  .sheet {
    display: grid;
    gap: var(--space-1);
    margin: 0 calc(-1 * var(--space-3));
  }
  /* The sheet's radio mark: a ring, filled in ink when chosen (neutral, like a checked row box). */
  .opt .mark {
    flex: none;
    display: grid;
    place-items: center;
    width: 18px;
    height: 18px;
    border: 1.5px solid var(--line-strong);
    border-radius: 50%;
  }
  .on .mark {
    border-color: var(--text-primary);
  }
  .on .mark::after {
    content: '';
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--text-primary);
  }
`;
