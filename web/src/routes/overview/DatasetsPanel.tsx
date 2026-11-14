/** Overview's Datasets: the first six as cards with their class balance (kept as it was until the shared dataset card lands). */
import ArrowRight from '$icons/arrow-right';
import Images from '$icons/images';
import Plus from '$icons/plus';
import { css, For } from 'fw';
import type { Dataset } from '$lib/api/types';
import type { ListedDataset } from '../datasets/problem';
import type { Resource } from '$lib/api/resource';
import ClassStrip from '$lib/detect/ClassStrip';
import { compact, count } from '$lib/format';
import { href } from '$lib/router';
import { balanced } from '$lib/ui/balancedGrid';
import Badge from '$lib/ui/Badge';
import Button from '$lib/ui/Button';
import EmptyState from '$lib/ui/EmptyState';
import ErrorState from '$lib/ui/ErrorState';
import Meta from '$lib/ui/Meta';
import Panel from '$lib/ui/Panel';
import Skeleton from '$lib/ui/Skeleton';
import { PROBLEM_LABEL } from '../datasets/problem';

const total = (d: Dataset) => d.splits.train + d.splits.val + d.splits.test;

export default function DatasetsPanel(props: { datasets: Resource<ListedDataset[]> }) {
  const datasets = props.datasets;
  const firstDatasets = { get: () => (datasets.data ?? []).slice(0, 6) };
  return (
    <Panel
      title="Datasets"
      meta={datasets.data ? String(datasets.data.length) : undefined}
      pad="none"
      id="datasets"
      actions={() => (
        <Button size="sm" variant="ghost" iconRight={ArrowRight} href={href('/datasets')}>
          All datasets
        </Button>
      )}
    >
      {datasets.error && !datasets.data ? (
        <ErrorState error={datasets.error} retry={() => datasets.refresh()} title="Could not load datasets" compact />
      ) : !datasets.data ? (
        <ul class="dslist" aria-busy="true" aria-label="Loading datasets" use:balanced={{ n: 3, min: 280 }}>
          <For each={[0, 1, 2]} key={(i) => i}>
            {() => (
              <li aria-hidden="true">
                <span class="d-sk">
                  <span class="d-row">
                    <span class="l-20">
                      <Skeleton w="96px" h="12px" />
                    </span>
                    <span class="l-20">
                      <Skeleton w="128px" h="10px" />
                    </span>
                  </span>{' '}
                  <Skeleton h="4px" r="2px" />
                </span>
              </li>
            )}
          </For>
        </ul>
      ) : datasets.data.length === 0 ? (
        <EmptyState
          icon={Images}
          title="No datasets registered"
          description="Register a YOLO or COCO folder, or generate a sample set of shapes."
          actions={() => (
            <Button size="sm" icon={Plus} href={href('/datasets?add=folder')}>
              Add dataset
            </Button>
          )}
        />
      ) : (
        <ul class="dslist" use:balanced={{ n: Math.min(6, datasets.data.length), min: 280 }}>
          <For each={firstDatasets.get()} key={(d) => d.id}>
            {(d) => (
              // The link covers the card (a stretched link); the class balance, its own disclosure, sits above it.
              <li class="d-card">
                <a href={href(`/datasets/${d().id}`)}>
                  <span class="d-row">
                    <span class="d-name truncate">{d().name}</span>{' '}
                    {/* A folder gone since it was registered: the Datasets list's word in place of its counts. */}
                    <span class="d-nums num">
                      {d().problem ? (
                        <Badge tone="warning">{PROBLEM_LABEL[d().problem!]}</Badge>
                      ) : (
                        <Meta line>
                          <span>{`${compact(total(d()))} ${total(d()) === 1 ? 'image' : 'images'}`}</span>
                          <span>{count(d().classes.length, 'class')}</span>
                        </Meta>
                      )}
                    </span>
                  </span>
                </a>{' '}
                {d().stats ? (
                  <span class="d-strip">
                    <ClassStrip counts={d().stats!.class_counts} classes={d().classes} height={4} named />
                  </span>
                ) : null}
              </li>
            )}
          </For>
        </ul>
      )}
    </Panel>
  );
}

css`
  /* Skeleton bars centred in the line box of the text they stand for (13/20 body, 12/16 meta). */
  .l-20 {
    display: flex;
    align-items: center;
    height: 20px;
  }
  .l-20.end {
    justify-content: flex-end;
  }
  /* Entries side by side wherever two or more fit, in balanced columns (ui/balancedGrid sets --cols): four datasets
     are 2 × 2, never 3 + 1 with one alone on a row. */
  .dslist {
    display: grid;
    grid-template-columns: repeat(var(--cols, 1), minmax(0, 1fr));
    column-gap: var(--space-4);
  }
  .dslist {
    padding: 0 var(--space-2) var(--space-2);
  }
  .d-card,
  .d-sk {
    position: relative;
    display: grid;
    gap: 6px;
    padding: var(--space-2);
    border-radius: var(--radius-md);
    transition: background-color var(--dur-fast) var(--ease-standard);
  }
  .d-card:has(> a:hover) {
    background: var(--bg-hover);
  }
  .d-card > a {
    display: block;
    min-width: 0;
  }
  /* The ring goes round the card the link covers. */
  .d-card:has(> a:focus-visible) {
    outline: 2px solid var(--focus);
    outline-offset: 2px;
  }
  .d-card > a:focus-visible {
    outline: none;
  }
  .d-card > a::after {
    content: '';
    position: absolute;
    inset: 0;
    border-radius: inherit;
  }
  .d-strip {
    position: relative;
    z-index: 1;
  }
  .d-row {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: var(--space-3);
    min-width: 0;
  }
  .d-name {
    font-weight: var(--weight-medium);
  }
  .d-nums {
    flex-shrink: 0;
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
  }
`;
