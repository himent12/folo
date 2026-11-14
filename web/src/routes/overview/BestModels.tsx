/**
 * Overview's Best models (design/UNIFIED.md §4 Overview, beside Recent runs): the top model of each comparable group
 * (lib/scoreSummary: a score ranks only against scores on the same dataset and split), its dataset named above it and
 * any split but validation tagged beside the score; the metric is named once, in the header, not under every value.
 */
import ArrowRight from '$icons/arrow-right';
import Box from '$icons/box';
import { css, For } from 'fw';
import type { Model } from '$lib/api/types';
import { bytes, count, metric } from '$lib/format';
import { prefs } from '$lib/prefs';
import { href } from '$lib/router';
import { scoreTag, type ScoreContext } from '$lib/scoreSummary';
import { balanced } from '$lib/ui/balancedGrid';
import Button from '$lib/ui/Button';
import EmptyState from '$lib/ui/EmptyState';
import ErrorState from '$lib/ui/ErrorState';
import Ident from '$lib/ui/Ident';
import Meta from '$lib/ui/Meta';
import Panel from '$lib/ui/Panel';
import Skeleton from '$lib/ui/Skeleton';
import type { Resource } from '$lib/api/resource';

export interface BestModelsProps {
  models: Resource<Model[]>;
  /** The ranking has what it needs (every model's run and dataset, or the reason it cannot). */
  ready: boolean;
  top: { m: Model; ctx: ScoreContext }[];
  /** What the list leaves out (./bestNote), or nothing. */
  note: string | null;
}

export default function BestModels(props: BestModelsProps) {
  const models = props.models;
  const topModels = { get: () => props.top };
  const rankReady = { get: () => props.ready };
  const note = { get: () => props.note };
  return (
    <Panel
      title="Best models"
      meta="mAP50-95"
      pad="none"
      id="best-models"
      actions={() => (
        <Button size="sm" variant="ghost" iconRight={ArrowRight} href={href('/models')}>
          All models
        </Button>
      )}
    >
      {models.error && !models.data ? (
        <ErrorState error={models.error} retry={() => models.refresh()} title="Could not load models" />
      ) : !rankReady.get() ? (
        // The loaded list's rows (dataset, name, meta; mAP on the right) at their height.
        <ul class="models" aria-busy="true" aria-label="Loading models" use:balanced={{ n: 4, min: 280 }}>
          <For each={[0, 1, 2, 3]} key={(i) => i}>
            {() => (
              <li aria-hidden="true">
                <span class="m-row">
                  <span class="m-text">
                    <span class="l-20">
                      <Skeleton w="64px" h="10px" />
                    </span>{' '}
                    <span class="l-20">
                      <Skeleton w="128px" h="12px" />
                    </span>{' '}
                    <span class="l-20">
                      <Skeleton w="152px" h="10px" />
                    </span>
                  </span>{' '}
                  <span class="m-map">
                    <span class="l-20 end">
                      <Skeleton w="40px" h="12px" />
                    </span>
                  </span>
                </span>
              </li>
            )}
          </For>
        </ul>
      ) : models.data?.length === 0 ? (
        <EmptyState icon={Box} title="No models yet" description="Finished runs save their best weights here." />
      ) : (
        <>
          {topModels.get().length ? (
            <ul class="models" use:balanced={{ n: topModels.get().length, min: 280 }}>
              <For each={topModels.get()} key={(t) => t.m.id}>
                {(t) => {
                  const tag = () => scoreTag(t().ctx);
                  return (
                    <li>
                      <a href={href(`/models/${t().m.id}`)}>
                        <span class="m-text">
                          <span class="m-ds truncate">{t().ctx.datasetName}</span> <span class="m-name truncate">{t().m.name}</span>{' '}
                          <span class="m-sub num">
                            <Meta line>
                              <Ident value={t().m.arch} />
                              <span>{count(t().m.classes.length, 'class')}</span>
                              <span>{bytes(t().m.bytes, prefs.units)}</span>
                            </Meta>
                          </span>
                        </span>{' '}
                        <span class="m-map">
                          <span class="num">
                            <span class="sr-only">mAP50-95 </span>
                            {metric(t().m.metrics?.map50_95)}
                          </span>{' '}
                          {tag() ? (
                            <span class="m-tag" data-fact>
                              {tag()}
                            </span>
                          ) : null}
                        </span>
                      </a>
                    </li>
                  );
                }}
              </For>
            </ul>
          ) : null}{' '}
          {note.get() ? (
            // What the list leaves out (./overview/bestNote): the datasets past the sixth (ranked, not shown),
            // weights trained elsewhere, models whose split is unknown.
            <p class="unranked" class:alone={!topModels.get().length}>
              {note.get()}
            </p>
          ) : null}
        </>
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
  /* Entries side by side wherever two or more fit, in balanced columns (ui/balancedGrid sets --cols): four models
     are 2 × 2, never 3 + 1 with one alone on a row. */
  .models {
    display: grid;
    grid-template-columns: repeat(var(--cols, 1), minmax(0, 1fr));
    column-gap: var(--space-4);
  }
  .models {
    padding: 0 var(--space-2) var(--space-2);
  }
  .models a,
  .m-row {
    display: flex;
    align-items: center;
    gap: var(--space-3);
    min-height: 64px;
    padding-top: var(--space-2) !important;
    padding-bottom: var(--space-2) !important;
    padding: 0 var(--space-2);
    border-radius: var(--radius-md);
    transition: background-color var(--dur-fast) var(--ease-standard);
  }
  .models a:hover {
    background: var(--bg-hover);
  }
  .m-text {
    flex: 1;
    min-width: 0;
    display: grid;
  }
  .m-name {
    font-weight: var(--weight-medium);
  }
  .m-ds {
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
  }
  .m-sub {
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
  }
  .m-map {
    display: grid;
    justify-items: end;
  }
  .m-map .num {
    font-weight: var(--weight-medium);
  }
  /* "Training split", "Split unknown": a label beside a score that is not on a validation split. It hangs under the
     score, so every score keeps the same place in its row with or without one. */
  .m-map {
    position: relative;
  }
  .m-tag {
    position: absolute;
    top: 100%;
    right: 0;
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-tertiary);
    white-space: nowrap;
  }
  /* The list's foot: on the panel's bottom edge (the body is a flex column, ui/Panel), so a panel stretched to Recent
     runs' height ends on its note, never on 70 px of nothing under it. */
  .unranked {
    margin-top: auto;
    padding: 0 var(--space-4) var(--space-4);
    font-size: var(--type-sm-size);
    color: var(--text-secondary);
  }
  .unranked.alone {
    padding-top: var(--space-1);
  }
`;
