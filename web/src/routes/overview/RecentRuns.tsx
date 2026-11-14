/**
 * Overview's Recent runs (design/UNIFIED.md §4 Overview): single-line rows, Run, Architecture · dataset, Status,
 * Progress, Best mAP50-95, Created. The table frame holds in every state: placeholder rows while loading, the message
 * inside it when there is nothing (without column heads over it, as on the list pages). The mAP trend is the Runs
 * list's column.
 */
import Activity from '$icons/activity';
import ArrowRight from '$icons/arrow-right';
import { css, For } from 'fw';
import { runProgress } from '$lib/api/progress';
import type { Run } from '$lib/api/types';
import { ago, metric } from '$lib/format';
import { failureCause } from '$lib/oom';
import { href, router } from '$lib/router';
import Button from '$lib/ui/Button';
import EmptyState from '$lib/ui/EmptyState';
import ErrorState from '$lib/ui/ErrorState';
import Ident from '$lib/ui/Ident';
import NoValue from '$lib/ui/NoValue';
import Panel from '$lib/ui/Panel';
import Progress from '$lib/ui/Progress';
import Skeleton from '$lib/ui/Skeleton';
import StatusBadge from '$lib/ui/StatusBadge';
import TableState from '$lib/ui/TableState';
import { tooltip } from '$lib/ui/tooltip';

export interface RecentRunsProps {
  /** Every run (undefined while loading). */
  all: Run[] | undefined;
  /** The rows: the latest runs but the one in the live panel. */
  recent: Run[];
  failed: boolean;
  error: unknown;
  retry: () => void;
  /** A run's dataset as people know it (lib/runDataset): undefined while it loads, null to leave it out. */
  dsLabel: (r: Run) => string | null | undefined;
}

export default function RecentRuns(props: RecentRunsProps) {
  return (
    <Panel
      title="Recent runs"
      meta={props.all ? `${props.recent.length} of ${props.all.length}` : undefined}
      pad="none"
      id="recent-runs"
      actions={() => (
        <Button size="sm" variant="ghost" iconRight={ArrowRight} href={href('/runs')}>
          All runs
        </Button>
      )}
    >
      <div class="scroll">
        <table class="table" aria-busy={!props.all} aria-label="Recent runs">
          {(props.all ? props.recent.length > 0 : !props.failed) ? (
            <thead>
              {/* Fixed widths on the shared header row: one geometry loading and loaded; the two text columns share the
                  rest, each cut with an ellipsis (whole in its tooltip). */}
              <tr>
                <th>Run</th>
                <th class="hide-sm">Architecture · dataset</th>
                <th style:width="112px">Status</th>
                <th class="hide-sm" style:width="120px">
                  Progress
                </th>
                <th class="r" style:width="128px">
                  Best mAP50-95
                </th>
                <th class="r hide-sm" style:width="88px">
                  Created
                </th>
              </tr>
            </thead>
          ) : null}
          <tbody>
            {props.failed ? (
              <TableState cols={6}>
                <ErrorState error={props.error} retry={props.retry} title="Could not load recent runs" />
              </TableState>
            ) : !props.all ? (
              <For each={[0, 1, 2, 3, 4, 5]} key={(i) => i}>
                {() => (
                  <tr aria-hidden="true">
                    <td>
                      <Skeleton w="min(112px, 100%)" h="12px" />
                    </td>
                    <td class="hide-sm">
                      <Skeleton w="min(128px, 100%)" h="12px" />
                    </td>
                    <td>
                      <Skeleton w="min(72px, 100%)" h="12px" />
                    </td>
                    <td class="hide-sm">
                      <div class="prog">
                        <Skeleton h="4px" />
                        <span>
                          <Skeleton w="min(40px, 100%)" h="10px" />
                        </span>
                      </div>
                    </td>
                    <td class="r">
                      <span class="end">
                        <Skeleton w="min(40px, 100%)" h="12px" />
                      </span>
                    </td>
                    <td class="r hide-sm">
                      <span class="end">
                        <Skeleton w="min(56px, 100%)" h="12px" />
                      </span>
                    </td>
                  </tr>
                )}
              </For>
            ) : props.recent.length === 0 ? (
              <TableState cols={6}>
                <EmptyState icon={Activity} title="No runs yet" description="Runs you start appear here with their progress and best mAP." />
              </TableState>
            ) : (
              <For each={props.recent} key={(r) => r.id}>
                {(r) => (
                  <tr class="link" onclick={() => router.go(`/runs/${r().id}`)}>
                    <td>
                      <a class="cut" href={href(`/runs/${r().id}`)} onclick={(e) => e.stopPropagation()} use:tooltip={r().name}>
                        {r().name}
                      </a>
                    </td>
                    <td class="hide-sm">
                      <span class="cut faint">
                        <Ident value={r().config.arch} />
                        {props.dsLabel(r()) === undefined ? (
                          <span class="ds">
                            {' · '}
                            <Skeleton inline w="72px" h="10px" />
                          </span>
                        ) : props.dsLabel(r()) ? (
                          <span class="ds" use:tooltip={props.dsLabel(r())}>
                            {' · '}
                            {props.dsLabel(r())}
                          </span>
                        ) : null}
                      </span>
                    </td>
                    <td use:tooltip={failureCause(r())}>
                      <StatusBadge run={r()} plain />
                    </td>
                    <td class="hide-sm">
                      <div class="prog">
                        <Progress value={runProgress(r())} size="xs" tone={r().status === 'running' ? 'accent' : 'neutral'} label={`${r().name} progress`} />
                        <span class="num faint">
                          {r().epoch} / {r().epochs}
                        </span>
                      </div>
                    </td>
                    <td class="r num">{r().best ? metric(r().best!.map50_95) : <NoValue />}</td>
                    <td class="r faint hide-sm">{ago(r().created)}</td>
                  </tr>
                )}
              </For>
            )}
          </tbody>
        </table>
      </div>
    </Panel>
  );
}

css`
  /* The table fits its panel at every width, never scrolling sideways: the two text columns share what the fixed ones
     leave, a long run name or dataset ending in an ellipsis rather than widen its column. */
  .scroll {
    overflow-x: auto;
  }
  .scroll :global(table) {
    table-layout: fixed;
  }
  /* A head longer than its column ("Architecture · dataset" beside Best models) ends in an ellipsis. */
  .scroll :global(th) {
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .cut {
    display: block;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .end {
    display: flex;
    justify-content: flex-end;
  }
  .prog {
    display: flex;
    align-items: center;
    gap: var(--space-2);
  }
  /* The count never shrinks under its text ("150 / 150"); the bar takes what is left. */
  .prog span {
    flex-shrink: 0;
    font-size: var(--type-sm-size);
    min-width: 56px;
    text-align: right;
  }
`;
