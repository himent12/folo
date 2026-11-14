/**
 * Overview's Best models (design/UNIFIED.md §4 Overview, beside Recent runs): single-line rows as Desktop's, one per
 * comparable group (lib/scoreSummary: a score ranks only against scores on the same dataset and split): Dataset, Best
 * model, mAP50-95. A split other than validation is named after the dataset. The model's name is never cut while the
 * dataset has room: the dataset column takes what the name and the score leave. What the list leaves out is its foot.
 */
import ArrowRight from '$icons/arrow-right';
import Box from '$icons/box';
import { css, For } from 'fw';
import type { Model } from '$lib/api/types';
import { metric } from '$lib/format';
import { href, router } from '$lib/router';
import { scoreTag, type ScoreContext } from '$lib/scoreSummary';
import Button from '$lib/ui/Button';
import EmptyState from '$lib/ui/EmptyState';
import ErrorState from '$lib/ui/ErrorState';
import Panel from '$lib/ui/Panel';
import Skeleton from '$lib/ui/Skeleton';
import { tooltip } from '$lib/ui/tooltip';
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
  const head = () => (
    <thead>
      <tr>
        <th class="w-ds">Dataset</th>
        <th class="w-min">Best model</th>
        <th class="r">mAP50-95</th>
      </tr>
    </thead>
  );
  return (
    <Panel
      title="Best models"
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
      ) : !props.ready ? (
        // The loaded rows at their height.
        <div class="wrap">
          <table class="table" aria-busy="true" aria-label="Loading models">
            {head()}
            <tbody>
              <For each={[0, 1, 2, 3]} key={(i) => i}>
                {() => (
                  <tr aria-hidden="true">
                    <td>
                      <Skeleton w="min(72px, 100%)" h="12px" />
                    </td>
                    <td class="w-min">
                      <Skeleton w="112px" h="12px" />
                    </td>
                    <td class="r">
                      <span class="end">
                        <Skeleton w="40px" h="12px" />
                      </span>
                    </td>
                  </tr>
                )}
              </For>
            </tbody>
          </table>
        </div>
      ) : models.data?.length === 0 ? (
        <EmptyState icon={Box} title="No models yet" description="Finished runs save their best weights here." />
      ) : (
        <>
          {props.top.length ? (
            <div class="wrap">
              <table class="table" aria-label="Best models">
                {head()}
                <tbody>
                  <For each={props.top} key={(t) => t.m.id}>
                    {(t) => {
                      const tag = () => scoreTag(t().ctx);
                      const evalOn = () => (tag() ? `${t().ctx.datasetName} · ${tag()}` : (t().ctx.datasetName ?? ''));
                      return (
                        <tr class="link" onclick={() => router.go(`/models/${t().m.id}`)}>
                          <td class="ds">
                            {/* Only the name is ever cut (a free-text name, whole in the tooltip); the split tag is a fact. */}
                            <span class="ds-line" use:tooltip={evalOn()}>
                              <span class="ds-name">{t().ctx.datasetName}</span>
                              {tag() ? (
                                <span class="m-tag" data-fact>
                                  <span class="sep">{' · '}</span>
                                  {tag()}
                                </span>
                              ) : null}
                            </span>
                          </td>
                          <td class="w-min">
                            <a class="name cut" href={href(`/models/${t().m.id}`)} onclick={(e) => e.stopPropagation()} use:tooltip={t().m.name}>
                              {t().m.name}
                            </a>
                          </td>
                          <td class="r num">
                            <span class="sr-only">mAP50-95 </span>
                            {metric(t().m.metrics?.map50_95)}
                          </td>
                        </tr>
                      );
                    }}
                  </For>
                </tbody>
              </table>
            </div>
          ) : null}{' '}
          {props.note ? (
            // What the list leaves out (./overview/bestNote): the datasets past the sixth (ranked, not shown),
            // weights trained elsewhere, models whose split is unknown.
            <p class="unranked" class:alone={!props.top.length}>
              {props.note}
            </p>
          ) : null}
        </>
      )}
    </Panel>
  );
}

css`
  /* The name takes its own width (a 1 % column) and the dataset about a third, giving way first: it shrinks to what the
     name and the score leave; the name is cut only where the panel cannot hold it beside a 64 px dataset and the
     score. On a wide panel the score's column takes the rest. */
  .wrap {
    container-type: inline-size;
  }
  .w-ds {
    width: 36%;
  }
  .w-min {
    width: 1%;
    white-space: nowrap;
  }
  .cut {
    display: block;
    width: 0;
    min-width: 100%;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .ds-line {
    display: flex;
    width: 0;
    min-width: 100%;
    white-space: nowrap;
  }
  .ds-name {
    min-width: 0;
    max-width: 100%;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .name.cut {
    width: auto;
    min-width: 0;
    max-width: calc(100cqw - 176px);
    font-weight: var(--weight-medium);
    color: var(--text-primary);
  }
  .end {
    display: flex;
    justify-content: flex-end;
  }
  .r.num {
    font-weight: var(--weight-medium);
    color: var(--text-primary);
  }
  /* The dataset is the row's context, not its name: secondary and regular (the name is the model's). */
  .ds {
    color: var(--text-secondary);
    font-weight: var(--weight-regular);
  }
  .m-tag {
    flex: none;
    color: var(--text-tertiary);
  }
  /* A narrow panel (a 1280 px window's third, a phone): the split goes under the dataset's name, never cut after it;
     the name itself stays on one line, cut last with its whole name in the tooltip. */
  @container (max-width: 479.98px) {
    .ds-line {
      flex-direction: column;
    }
    .sep {
      display: none;
    }
  }
  /* The list's foot: on the panel's bottom edge (the body is a flex column, ui/Panel), so a panel stretched to Recent
     runs' height ends on its note, never on 70 px of nothing under it. */
  .unranked {
    margin-top: auto;
    padding: var(--space-3) var(--space-4);
    font-size: var(--type-sm-size);
    color: var(--text-secondary);
  }
`;
