/**
 * Overview's Datasets (design/UNIFIED.md §4 Overview, "Datasets as the cards below"): the first four as the Datasets
 * page's cards (../datasets/DatasetCard) in their compact form, one row of up to four; the rest are a click away.
 */
import ArrowRight from '$icons/arrow-right';
import Images from '$icons/images';
import Plus from '$icons/plus';
import { css, For } from 'fw';
import type { ListedDataset } from '../datasets/problem';
import type { Resource } from '$lib/api/resource';
import { href } from '$lib/router';
import Button from '$lib/ui/Button';
import EmptyState from '$lib/ui/EmptyState';
import ErrorState from '$lib/ui/ErrorState';
import Panel from '$lib/ui/Panel';
import DatasetCard, { DatasetCardSkeleton } from '../datasets/DatasetCard';

/** Cards on Overview: one row. */
const SHOWN = 4;

export default function DatasetsPanel(props: { datasets: Resource<ListedDataset[]> }) {
  const datasets = props.datasets;
  const first = { get: () => (datasets.data ?? []).slice(0, SHOWN) };
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
        <ul class="cards" aria-busy="true" aria-label="Loading datasets">
          <For each={[0, 1, 2, 3]} key={(i) => i}>
            {() => <DatasetCardSkeleton compact />}
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
        <ul class="cards" aria-label="Datasets">
          <For each={first.get()} key={(d) => d.id}>
            {(d) => <DatasetCard dataset={d()} compact />}
          </For>
        </ul>
      )}
    </Panel>
  );
}

css`
  /* One row of four on a desktop (a short list keeps the columns' width); two on a tablet; one on a phone. */
  .cards {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: var(--space-4);
    padding: var(--space-4);
  }
  @media (max-width: 1023px) {
    .cards {
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
  }
  @media (max-width: 767px) {
    .cards {
      grid-template-columns: minmax(0, 1fr);
      gap: var(--space-2);
      padding: var(--space-3);
    }
  }
`;
