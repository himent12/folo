/** New run, section 1: the dataset, as cards from 768 px up and as a compact select with a one-line summary on phones. */
import Check from '$icons/check';
import { bindable, css, For, Index } from 'fw';
import type { Resource } from '$lib/api/resource';
import type { Dataset } from '$lib/api/types';
import ClassStrip from '$lib/detect/ClassStrip';
import { compact, num } from '$lib/format';
import Panel from '$lib/ui/Panel';
import Select from '$lib/ui/Select';
import Skeleton from '$lib/ui/Skeleton';
import Meta from '$lib/ui/Meta';
import { balanced } from '$lib/ui/balancedGrid';
import type { FormErrors } from './jump';

export interface DatasetSectionProps {
  datasets: Resource<Dataset[]>;
  datasetId: string;
  dataset: Dataset | undefined;
  errors: FormErrors;
}

export default function DatasetSection(props: DatasetSectionProps) {
  const datasetId = bindable(props, 'datasetId');

  return (
    <Panel title="1 · Dataset" id="sec-data">
      {!props.datasets.data ? (
        <div class="cards" use:balanced={{ n: 3, min: 184, gap: 8 }}>
          <Index each={[0, 1, 2]}>{() => <Skeleton h="88px" r="var(--radius-md)" />}</Index>
        </div>
      ) : (
        <>
          {/* Phones: a compact picker and a one-line summary; the cards are for wider screens. */}
          <div class="ds-pick">
            <Select
              id="dataset-pick"
              label="Dataset"
              bind:value={datasetId}
              invalid={!!props.errors.dataset}
              options={props.datasets.data!.map((d) => ({ value: d.id, label: d.name }))}
            />
            {props.dataset ? (
              <p class="ds-line num">
                <Meta>
                  <span>{num(props.dataset.splits.train)} train</span>
                  <span>{num(props.dataset.splits.val)} val</span>
                  <span>{props.dataset.classes.length} classes</span>
                  <span>{props.dataset.format}</span>
                </Meta>
              </p>
            ) : null}
          </div>
          <div
            class="cards"
            role="radiogroup"
            aria-label="Dataset"
            aria-invalid={props.errors.dataset ? 'true' : undefined}
            use:balanced={{ n: props.datasets.data!.length, min: 184, gap: 8 }}
          >
            <For each={props.datasets.data!} key={(d) => d.id}>
              {(d) => {
                const on = () => d().id === datasetId.get();
                // The label covers the card (a stretched label); the class balance, its own disclosure, sits above it.
                return (
                  <div class="card" class:on={on()}>
                    <label class="c-hit">
                      <input type="radio" name="dataset" value={d().id} bind:group={datasetId} class="sr-only" />
                      {/* Selection adds a mark and takes no fact away: the check before the name, the format kept. */}
                      <span class="c-top">
                        <span class="c-name">
                          {on() ? (
                            <span class="tick" aria-hidden="true">
                              <Check size={14} strokeWidth={2.25} />
                            </span>
                          ) : null}
                          {d().name}
                        </span>
                        <span class="c-fmt">{d().format}</span>
                      </span>
                      {/* Breaks only between the parts, and a wrapped line never ends on "·" (ui/Meta). */}
                      <span class="c-meta num">
                        <Meta>
                          <span>{compact(d().splits.train)} train</span>
                          <span>{compact(d().splits.val)} val</span>
                          <span>{d().classes.length} classes</span>
                        </Meta>
                      </span>
                    </label>
                    {d().stats ? (
                      <span class="c-strip">
                        <ClassStrip counts={d().stats!.class_counts} classes={d().classes} height={4} named />
                      </span>
                    ) : null}
                  </div>
                );
              }}
            </For>
          </div>
        </>
      )}
    </Panel>
  );
}

css`
  /* As many cards across as fit at 184 px, never one alone on a row (ui/balancedGrid sets --cols): four datasets are
     one row on a desktop and 2 × 2 on a laptop or tablet, never 3 + 1. */
  .cards {
    display: grid;
    grid-template-columns: repeat(var(--cols, 1), minmax(0, 1fr));
    gap: var(--space-2);
  }
  .card {
    position: relative;
    border-radius: var(--radius-md);
    border: 1px solid var(--line-default);
    background: var(--bg-raised);
    cursor: pointer;
    transition:
      border-color var(--dur-fast) var(--ease-standard),
      background-color var(--dur-fast) var(--ease-standard),
      box-shadow var(--dur-fast) var(--ease-standard);
  }
  /* The class strip on the card's bottom edge (the label takes the slack), so the strips of a row line up whatever
     each card's meta line wraps to (fasteners-mini's "2 classes" on a line of its own). */
  .card {
    display: grid;
    grid-template-rows: 1fr auto;
    gap: 6px;
    padding: var(--space-3);
  }
  .c-hit {
    align-content: start;
  }
  .card:hover {
    border-color: var(--line-strong);
  }
  /* Selection: a strong edge on a neutral fill and a small ink check; lime stays for the Start button and focus,
     never a selection state (DESIGN_DIRECTION §2.7). */
  .card.on {
    border-color: var(--line-strong);
    background: var(--bg-active);
  }
  .tick {
    display: grid;
    color: var(--text-primary);
  }
  .card .tick {
    display: inline-grid;
    margin-right: var(--space-1);
    vertical-align: -2px;
  }
  .card:has(input:focus-visible) {
    outline: 2px solid var(--focus);
    outline-offset: 2px;
  }
  .c-hit {
    display: grid;
    gap: 6px;
    cursor: pointer;
  }
  .c-hit::after {
    content: '';
    position: absolute;
    inset: 0;
    border-radius: inherit;
  }
  .c-strip {
    position: relative;
    z-index: 1;
  }
  .c-top {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: var(--space-2);
  }
  .c-name {
    font-weight: var(--weight-medium);
  }
  .c-fmt {
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
  }
  .c-meta {
    font-size: var(--type-sm-size);
    color: var(--text-secondary);
  }
  .ds-pick {
    display: none;
  }
  .ds-line {
    margin-top: var(--space-2);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
  }
  @media (max-width: 767px) {
    .ds-pick {
      display: block;
    }
    .ds-pick :global(select) {
      height: 44px;
    }
    .cards {
      display: none;
    }
  }
`;
