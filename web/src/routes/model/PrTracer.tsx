/**
 * "Trace a class": the classes as chips under the curves' readout row, at every width, so tracing never needs the
 * table. The traced ones first (in the order they were traced), in their hue, each a tap to remove; then the others
 * in the table's order, a tap to trace; with many classes the first 12 until a name is typed. The traced set is the
 * one the table shows.
 */
import Search from '$icons/search';
import X from '$icons/x';
import { bindable, computed, css, For, signal } from 'fw';
import { classStroke } from '$lib/design/classInk';
import Input from '$lib/ui/Input';
import { matchesClass } from './pr';

export interface PrTracerProps {
  /** The group's id (the Trace button's `aria-controls`). */
  id: string;
  open: boolean;
  /** Every class id, in the table's order. */
  order: number[];
  name: (i: number) => string;
  /** Many classes: a filter, and the first 12 until a name is typed. */
  many: boolean;
  selected?: number[];
}

export default function PrTracer(props: PrTracerProps) {
  const selected = bindable(props, 'selected', []);
  const query = signal('');
  const list = computed(() => {
    const s = selected.get();
    const q = query.get();
    const rest = props.order.filter((i) => !s.includes(i) && matchesClass(props.name(i), q));
    return [...s, ...(props.many && !q ? rest.slice(0, 12) : rest)];
  });
  const toggle = (i: number) => {
    const s = selected.peek();
    selected.set(s.includes(i) ? s.filter((x) => x !== i) : [...s, i]);
  };

  return (
    <div class="tracer" id={props.id} role="group" aria-label="Classes to trace" hidden={!props.open}>
      {props.many ? (
        <div class="filter">
          <Input bind:value={query} icon={Search} size="sm" type="search" placeholder="Find a class" aria-label="Find a class to trace" />
        </div>
      ) : null}
      <div class="t-chips">
        <For each={list.get()} key={(i) => i}>
          {(i) => {
            const on = () => selected.get().includes(i());
            return (
              <button
                type="button"
                class="t-chip"
                aria-pressed={on()}
                aria-label={on() ? `Stop tracing ${props.name(i())}` : `Trace ${props.name(i())}`}
                style={{ '--c': classStroke(i()) }}
                onclick={() => toggle(i())}
              >
                <span class="t-key" aria-hidden="true"></span>
                <span class="t-name">{props.name(i())}</span>
                {on() ? <X size={14} aria-hidden="true" /> : null}
              </button>
            );
          }}
        </For>
        {!list.get().length ? <span class="hint">No class matches “{query.get()}”</span> : null}
      </div>
    </div>
  );
}

css`
  /* Wrapping chips, each a 32 px toggle (44 on touch). A traced class is the selection rule (bg.active) with its hue
     as the key and a remove mark; the others a neutral hollow key (a class takes its hue once traced, as its curve
     does). */
  .tracer {
    display: grid;
    gap: var(--space-2);
    padding: 0 var(--space-4) var(--space-2);
  }
  .tracer[hidden] {
    display: none;
  }
  .filter {
    width: 100%;
  }
  .t-chips {
    display: flex;
    flex-wrap: wrap;
    gap: var(--space-2);
  }
  .t-chip {
    display: inline-flex;
    align-items: center;
    gap: var(--space-1-5);
    min-height: 32px;
    max-width: 100%;
    padding: 0 var(--space-3);
    border: 1px solid var(--line-default);
    border-radius: var(--radius-md);
    font-size: var(--type-sm-size);
    color: var(--text-secondary);
  }
  .t-chip[aria-pressed='true'] {
    padding-right: var(--space-2);
    background: var(--bg-active);
    border-color: transparent;
    color: var(--text-primary);
  }
  .t-chip :global(svg) {
    flex-shrink: 0;
    color: var(--text-tertiary);
  }
  .t-key {
    flex-shrink: 0;
    width: 10px;
    height: 10px;
    border: 2px solid var(--text-tertiary);
    border-radius: 50%;
  }
  .t-chip[aria-pressed='true'] .t-key {
    border-color: var(--c);
    background: var(--c);
  }
  .t-name {
    overflow-wrap: anywhere;
    text-align: left;
  }
  .hint {
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
  }
  @media (hover: hover) {
    .t-chip:hover {
      border-color: var(--line-strong);
      color: var(--text-primary);
    }
  }
  @media (pointer: coarse) {
    .t-chip {
      min-height: 44px;
    }
  }
`;
