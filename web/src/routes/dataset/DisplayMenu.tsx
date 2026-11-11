// fw port of DisplayMenu.svelte.
/**
 * Phones: the image toolbar's "Display" button. It opens a small popover with the boxes switch and the thumbnail
 * size, so the toolbar stays one row; wider screens show both controls inline.
 */
import SlidersHorizontal from '$icons/sliders-horizontal';
import { createUniqueId, css, effect, signal, tick } from 'fw';
import Segmented from '$lib/ui/Segmented';
import Switch from '$lib/ui/Switch';
import type { Density } from '$lib/ui/TileGrid';

export interface DisplayMenuProps {
  boxes: boolean;
  density: Density;
  onboxes: (on: boolean) => void;
  ondensity: (d: Density) => void;
}

const SIZES: { value: Density; label: string }[] = [
  { value: 's', label: 'S' },
  { value: 'm', label: 'M' },
  { value: 'l', label: 'L' },
];

export default function DisplayMenu(props: DisplayMenuProps) {
  const uid = createUniqueId();
  const open = signal(false);
  const root = signal<HTMLElement>();
  const pop = signal<HTMLElement>();

  async function show() {
    open.set(true);
    await tick();
    pop.get()?.querySelector<HTMLElement>('input, button')?.focus();
  }
  function hide(refocus = true) {
    open.set(false);
    if (refocus) root.get()?.querySelector<HTMLElement>('.trigger')?.focus();
  }

  function onkeydown(e: KeyboardEvent) {
    if (e.key === 'Escape') {
      e.preventDefault();
      hide();
    }
  }

  effect(() => {
    if (!open.get()) return;
    const away = (e: PointerEvent) => {
      const r = root.get();
      if (r && !r.contains(e.target as Node)) hide(false);
    };
    document.addEventListener('pointerdown', away);
    return () => document.removeEventListener('pointerdown', away);
  });

  // Focus leaving the control (Tab past its last field) closes the popover; moving between its fields does not.
  return (
    <div
      class="display"
      ref={root}
      onfocusout={(e) => {
        const r = root.get();
        if (open.get() && !(r && e.relatedTarget instanceof Node && r.contains(e.relatedTarget))) hide(false);
      }}
    >
      <button
        type="button"
        class="trigger"
        aria-label="Display options"
        title="Display options"
        aria-haspopup="dialog"
        aria-expanded={open.get()}
        aria-controls={`${uid}-pop`}
        onclick={() => (open.get() ? hide() : show())}
      >
        <SlidersHorizontal size={14} strokeWidth={1.75} aria-hidden="true" />
      </button>
      {open.get() ? (
        <div class="pop" id={`${uid}-pop`} role="dialog" tabindex={-1} aria-label="Display options" ref={pop} onkeydown={onkeydown}>
          <Switch label="Boxes" hint="Draw the labelled boxes" checked={props.boxes} onchange={props.onboxes} />
          <div class="row">
            <span class="label">Thumbnail size</span>
            <Segmented label="Thumbnail size" size="sm" options={SIZES} value={props.density} onchange={props.ondensity} />
          </div>
        </div>
      ) : null}
    </div>
  );
}

css`
  .display {
    position: relative;
    display: inline-flex;
    flex-shrink: 0;
  }
  .trigger {
    display: grid;
    place-items: center;
    width: var(--control-sm);
    height: var(--control-sm);
    border-radius: var(--radius-sm);
    border: 1px solid var(--line-default);
    color: var(--text-secondary);
    transition:
      border-color var(--dur-fast) var(--ease-standard),
      color var(--dur-fast) var(--ease-standard);
  }
  .trigger:hover,
  .trigger[aria-expanded='true'] {
    color: var(--text-primary);
    border-color: var(--line-strong);
  }
  @media (pointer: coarse) {
    .trigger {
      width: 44px;
      height: 44px;
    }
  }
  .pop {
    position: absolute;
    top: calc(100% + 4px);
    right: 0;
    z-index: var(--z-overlay);
    width: min(280px, calc(100vw - 32px));
    padding: var(--space-3) var(--space-4);
    background: var(--bg-overlay);
    border: 1px solid var(--line-default);
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow-md);
    display: grid;
    gap: var(--space-3);
    animation: in var(--dur-fast) var(--ease-out);
  }
  .row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-3);
  }
  .label {
    color: var(--text-primary);
  }
  @keyframes in {
    from {
      opacity: 0;
      transform: translateY(-2px);
    }
  }
`;
