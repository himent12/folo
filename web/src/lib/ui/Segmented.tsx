/**
 * A segmented control (radio group): a filter or a view option (design/UNIFIED.md §2). One outlined track, 28 high; its
 * items in 12 px regular, the chosen one text.primary 500 on a raised fill with a hairline, a count after a label in
 * text.tertiary. A page's sections are underline tabs (ui/Tabs), not this. Arrow keys move the selection, as in native
 * radio groups.
 */
import { bindable, createUniqueId, css, defaults, Dynamic, For } from 'fw';
import { describedBy, getField } from './field';
import { tooltip } from './tooltip';
import type { IconType } from './types';

export interface SegmentedOption<T extends string> {
  value: T;
  label: string;
  icon?: IconType;
  /** Extra text after the label, e.g. a count. */
  meta?: string;
  /** Shown but not selectable (e.g. "coming soon"). */
  disabled?: boolean;
  /**
   * Why a disabled option is off: its tooltip, and its accessible description (`aria-describedby`), so the reason is
   * never pointer-only (pair it with a visible line the group is described by).
   */
  tip?: string;
}

export interface SegmentedProps<T extends string> {
  options: SegmentedOption<T>[];
  value: T;
  label: string;
  size?: 'sm' | 'md';
  iconOnly?: boolean;
  onchange?: (v: T) => void;
  id?: string;
  /** In error: a danger edge and focus ring (base.css); pair it with a Field error and `describedby`. */
  invalid?: boolean;
  /** More ids to describe it by; an enclosing Field's hint or error is added on its own (`ui/field`). */
  describedby?: string;
  /** The whole group off (a list toolbar over an empty or failed list keeps its place, DESIGN_DIRECTION §2.8). */
  disabled?: boolean;
}

export default function Segmented<T extends string>(props: SegmentedProps<T>) {
  const p = defaults(props, { size: 'md', iconOnly: false, invalid: false, disabled: false });
  const value = bindable(props, 'value');
  const field = getField();
  const uid = createUniqueId();
  let root!: HTMLDivElement;

  function select(v: T) {
    value.set(v);
    p.onchange?.(v);
  }

  function onkeydown(e: KeyboardEvent) {
    const options = p.options;
    const i = options.findIndex((o) => o.value === value.get());
    const n = options.length;
    // Step over disabled options in the direction of travel.
    const step = (from: number, d: number) => {
      for (let k = 1; k <= n; k++) {
        const j = (((from + d * k) % n) + n) % n;
        if (!options[j]?.disabled) return j;
      }
      return from;
    };
    let j = i;
    if (e.key === 'ArrowRight' || e.key === 'ArrowDown') j = step(i, 1);
    else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') j = step(i, -1);
    else if (e.key === 'Home') j = step(-1, 1);
    else if (e.key === 'End') j = step(n, -1);
    else return;
    e.preventDefault();
    const next = options[j];
    if (!next || next.disabled) return;
    select(next.value);
    root.querySelectorAll<HTMLButtonElement>('button')[j]?.focus();
  }

  const option = (o: () => SegmentedOption<T>, on: () => boolean) => (
    <button
      type="button"
      role="radio"
      aria-checked={on()}
      tabindex={on() ? 0 : -1}
      class:on={on()}
      class:icon-only={p.iconOnly && !!o().icon}
      disabled={p.disabled || o().disabled}
      aria-describedby={o().disabled && o().tip ? `${uid}-${o().value}-why` : undefined}
      onclick={() => select(o().value)}
      aria-label={p.iconOnly ? o().label : undefined}
      title={p.iconOnly ? o().label : undefined}
    >
      {o().icon ? <Dynamic component={o().icon} size={14} strokeWidth={1.75} aria-hidden="true" /> : null}
      {!p.iconOnly || !o().icon ? <span>{o().label}</span> : null}
      {o().meta ? <span class="meta">{o().meta}</span> : null}
    </button>
  );

  return (
    <div
      class={`seg ${p.size}`}
      id={p.id}
      role="radiogroup"
      aria-label={p.label}
      aria-invalid={p.invalid || field?.invalid || undefined}
      aria-describedby={describedBy(p.describedby, field?.describedby)}
      aria-disabled={p.disabled || undefined}
      ref={(el) => (root = el)}
      onkeydown={onkeydown}
      tabindex={-1}
    >
      <For each={p.options} key={(o) => o.value}>
        {(o) => {
          const on = () => o().value === value.get();
          // A disabled button takes no pointer events in some browsers: the wrapper takes the hover, as ui/Gate does.
          // The reason is also the option's description, for assistive technology (a tooltip is a pointer's alone).
          return (
            <>
              {o().disabled && o().tip ? (
                <>
                  <span class="tip" use:tooltip={o().tip}>
                    {option(o, on)}
                  </span>
                  <span class="sr-only" id={`${uid}-${o().value}-why`}>
                    {o().tip}
                  </span>
                </>
              ) : (
                option(o, on)
              )}
            </>
          );
        }}
      </For>
    </div>
  );
}

css`
  .seg {
    display: inline-flex;
    gap: 2px;
    padding: 2px;
    border-radius: var(--radius-md);
    background: var(--bg-inset);
    /* --seg-edge is set by an invalid group (base.css). */
    border: 1px solid var(--seg-edge, var(--line-control));
    max-width: 100%;
    justify-self: start;
    overflow-x: auto;
    scrollbar-width: none;
  }
  .seg:focus {
    outline: none;
  }
  /* An item fills the track's inner height: 28 less its 1 px edges and 2 px padding. */
  button {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
    height: calc(var(--control-md) - 6px);
    padding: 0 var(--space-3);
    border-radius: var(--radius-sm);
    border: 1px solid transparent;
    color: var(--text-secondary);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    font-weight: var(--weight-regular);
    white-space: nowrap;
    transition:
      background-color var(--dur-fast) var(--ease-standard),
      border-color var(--dur-fast) var(--ease-standard),
      color var(--dur-fast) var(--ease-standard);
  }
  .sm button {
    height: calc(var(--control-sm) - 6px);
    padding: 0 8px;
  }
  button.icon-only {
    width: calc(var(--control-md) - 6px);
    padding: 0;
  }
  .sm button.icon-only {
    width: calc(var(--control-sm) - 6px);
  }
  button:hover:not(:disabled, .on) {
    color: var(--text-primary);
    background: var(--bg-hover);
  }
  .tip {
    display: inline-flex;
    cursor: not-allowed;
  }
  .tip > button {
    pointer-events: none;
  }
  button:disabled,
  [aria-disabled='true'] button.on {
    color: var(--text-disabled);
  }
  button.on {
    background: var(--bg-raised);
    border-color: var(--line-control);
    color: var(--text-primary);
    font-weight: var(--weight-medium);
  }
  .meta {
    color: var(--text-tertiary);
    font-weight: var(--weight-regular);
    font-variant-numeric: tabular-nums;
  }
  button:focus-visible {
    outline-offset: 0;
  }
  @media (pointer: coarse) {
    button,
    .sm button {
      height: 44px;
      min-width: 44px;
      padding: 0 12px;
    }
    button.icon-only,
    .sm button.icon-only {
      width: 44px;
    }
  }
`;
