/** Phones: the legend's management in a bottom sheet: each run with how it started, open and remove, then Add. */
import X from '$icons/x';
import ArrowUpRight from '$icons/arrow-up-right';
import { bindable, css, For } from 'fw';
import type { Run } from '$lib/api/types';
import { FROM_SCRATCH } from '$lib/lineage';
import { href } from '$lib/router';
import Ident from '$lib/ui/Ident';
import Dialog from '$lib/ui/Dialog';

export interface ManageSheetProps {
  managing?: boolean;
  chosen: Run[];
  startOf: (r: Run) => string | null;
  fill: (id: string) => string;
  SLOT_PATTERN: readonly (undefined | 'dash' | 'dot')[];
  slotOf: (id: string) => number;
  ids: string[];
  setIds: (next: string[]) => void;
  MAX: number;
  candidates: Run[];
  /** Add run…, as the page draws it. */
  adder: JSX.Element;
}

export default function ManageSheet(props: ManageSheetProps) {
  const managing = bindable(props, 'managing', false);

  // Phones: the legend's management in a bottom sheet: each run with how it started, open and remove, then Add.
  return (
    <Dialog bind:open={managing} title="Runs in this comparison" description={`Open a run, take it out, or add another (up to ${props.MAX}).`}>
      <ul class="m-list">
        <For each={props.chosen} key={(r) => r.id}>
          {(r) => {
            const from = () => props.startOf(r());
            return (
              <li class="m-row" style={{ '--c': props.fill(r().id) }}>
                <i aria-hidden="true" data-pattern={props.SLOT_PATTERN[props.slotOf(r().id)]}></i>
                <span class="m-what">
                  <span class="m-name">{r().name}</span>{' '}
                  <span class="m-from" class:tuned={!!from()}>
                    {from() ? (
                      <>
                        Fine-tuned from <Ident value={from()!} />
                      </>
                    ) : (
                      FROM_SCRATCH
                    )}
                  </span>
                </span>
                <a class="m-act" href={href(`/runs/${r().id}`)} aria-label={`Open ${r().name}`} title="Open run">
                  <ArrowUpRight size={16} aria-hidden="true" />
                </a>
                <button
                  type="button"
                  class="m-act"
                  aria-label={`Remove ${r().name}`}
                  title="Remove from the comparison"
                  onclick={() => props.setIds(props.ids.filter((id) => id !== r().id))}
                >
                  <X size={16} aria-hidden="true" />
                </button>
              </li>
            );
          }}
        </For>
      </ul>
      {props.ids.length < props.MAX && props.candidates.length ? <div class="m-add">{props.adder}</div> : null}
    </Dialog>
  );
}

css`
  i[data-pattern='dash'] {
    mask-image: repeating-linear-gradient(90deg, #000 0 5px, transparent 5px 7px);
    border-radius: 0;
  }
  i[data-pattern='dot'] {
    mask-image: repeating-linear-gradient(90deg, #000 0 3px, transparent 3px 5px);
    border-radius: 0;
  }
  /* The Edit sheet: one 56 px row per run (key, name over how it started, open, remove), then Add run. */
  .m-list {
    display: grid;
    margin: 0 calc(-1 * var(--space-2));
  }
  .m-row {
    display: grid;
    grid-template-columns: 16px minmax(0, 1fr) 44px 44px;
    align-items: center;
    column-gap: var(--space-2);
    min-height: 56px;
    padding: var(--space-1) var(--space-2);
  }
  .m-row + .m-row {
    border-top: 1px solid var(--line-subtle);
  }
  .m-row i {
    align-self: start;
    margin-top: 14px; /* grid-ok: on the name line, centred in the 56 px row */
  }
  .m-what {
    display: grid;
    min-width: 0;
  }
  .m-name {
    font-weight: var(--weight-medium);
    overflow-wrap: break-word;
  }
  .m-from {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-tertiary);
    overflow-wrap: anywhere;
  }
  .m-from.tuned {
    color: var(--text-secondary);
  }
  .m-from :global(.ident) {
    font-size: var(--type-sm-size);
  }
  .m-act {
    display: grid;
    place-items: center;
    width: 44px;
    height: 44px;
    border-radius: var(--radius-md);
    color: var(--text-secondary);
  }
  .m-act:hover {
    background: var(--bg-hover);
    color: var(--text-primary);
  }
  .m-add {
    padding-top: var(--space-3);
  }
  .m-row i {
    flex-shrink: 0;
    width: 16px;
    height: 4px;
    border-radius: 2px;
    background: var(--c);
    box-shadow: inset 0 0 0 1px var(--chart-swatch-edge);
  }
`;
