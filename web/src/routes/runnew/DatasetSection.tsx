/** New run, section 1: the dataset, as cards from 768 px up and as a compact select with a one-line summary on phones. */
import Check from '$icons/check';
import CircleAlert from '$icons/circle-alert';
import Info from '$icons/info';
import { bindable, css, For, Index } from 'fw';
import type { Resource } from '$lib/api/resource';
import Badge from '$lib/ui/Badge';
import { PROBLEM_LABEL, type ListedDataset } from '../datasets/problem';
import ClassStrip from '$lib/detect/ClassStrip';
import { compact, count, num } from '$lib/format';
import Panel from '$lib/ui/Panel';
import Select from '$lib/ui/Select';
import Skeleton from '$lib/ui/Skeleton';
import Meta from '$lib/ui/Meta';
import { balanced } from '$lib/ui/balancedGrid';
import type { FormErrors } from './jump';
import { noValNote } from './valSplit';


export interface DatasetSectionProps {
  datasets: Resource<ListedDataset[]>;
  datasetId: string;
  dataset: ListedDataset | undefined;
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
              options={props.datasets.data!.map((d) => ({ value: d.id, label: d.problem ? `${d.name} (${PROBLEM_LABEL[d.problem]})` : d.name }))}
            />
            {props.dataset && !props.dataset.problem ? (
              <p class="ds-line num">
                <Meta>
                  <span>{num(props.dataset.splits.train)} train</span>
                  <span>{num(props.dataset.splits.val)} val</span>
                  <span>{count(props.dataset.classes.length, 'class')}</span>
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
            aria-describedby={props.errors.dataset ? 'dataset-error' : undefined}
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
                        {/* A folder gone since it was registered: the Datasets list's word, and Start says why it waits. */}
                        {d().problem ? <Badge tone="warning">{PROBLEM_LABEL[d().problem!]}</Badge> : <span class="c-fmt">{d().format}</span>}
                      </span>
                      {/* Breaks only between the parts, and a wrapped line never ends on "·" (ui/Meta). */}
                      <span class="c-meta num">
                        <Meta>
                          <span>{compact(d().splits.train)} train</span>
                          <span>{compact(d().splits.val)} val</span>
                          <span>{count(d().classes.length, 'class')}</span>
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
          {/* Said at the picker, not only beside Start: a dataset with no train split cannot be trained on. */}
          {props.errors.dataset ? (
            <p class="ds-error" id="dataset-error" role="alert">
              <CircleAlert size={14} strokeWidth={1.75} aria-hidden="true" />
              <span>
                {props.errors.dataset}
                {props.dataset && !props.dataset.problem ? ' Training needs images in a train split: pick another dataset, or add training images to this one.' : ''}
              </span>
            </p>
          ) : null}
            {/* A train-only dataset trains, but its scores are on the training images (runnew/valSplit). */}
            {!props.errors.dataset && props.dataset && !props.dataset.problem && noValNote(props.dataset) ? (
              <p class="ds-note" id="dataset-note">
              <Info size={14} strokeWidth={1.75} aria-hidden="true" />
              <span>{noValNote(props.dataset)}</span>
            </p>
          ) : null}
        </>
      )}
    </Panel>
  );
}

css`
  /* As Field's error line: the danger icon and text, one readable measure. */
  .ds-error {
    display: flex;
    align-items: flex-start;
    gap: var(--space-1);
    margin-top: var(--space-2);
    max-inline-size: var(--measure);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--danger-text);
  }
  .ds-error > :global(svg) {
    flex-shrink: 0;
    margin-top: 1px;
  }
  /* A train-only dataset's note: the error line's geometry in secondary ink (it informs, Start stays on). */
  .ds-note {
    display: flex;
    align-items: flex-start;
    gap: var(--space-1);
    margin-top: var(--space-2);
    max-inline-size: var(--measure);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
  }
  .ds-note > :global(svg) {
    flex-shrink: 0;
    margin-top: 1px;
  }
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
