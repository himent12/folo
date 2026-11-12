/**
 * A queued run's side panels (the Run page's waiting layout): the queue, front first, with what trains now and this
 * run marked; and what the run will have once it trains, one quiet list instead of empty frames.
 */
import { css, For } from 'fw';
import { ordinal } from '$lib/api/queue';
import type { Run } from '$lib/api/types';
import { eta } from '$lib/format';
import { href } from '$lib/router';
import Panel from '$lib/ui/Panel';

export interface WaitingPanelsProps {
  /** The runs waiting, front first (this one among them); empty: no queue panel. */
  queue: Run[];
  /** The run training now. */
  ahead: Run | undefined;
  /** This run's id. */
  self: string;
  /** How many runs wait. */
  waiting: number;
  later: { name: string; what: string }[];
}

export default function WaitingPanels(props: WaitingPanelsProps) {
  return (
    <>
      {props.queue.length ? (
        <Panel title="Queue" meta={`${props.waiting} waiting`} pad="none" id="queue">
          <ol class="queue">
            {props.ahead ? (
              <li class="q-row">
                <span class="q-n">
                  <span class="live-dot" aria-hidden="true"></span>
                  <span class="sr-only">Training</span>
                </span>
                <a class="q-name" href={href(`/runs/${props.ahead.id}`)}>
                  {props.ahead.name}
                </a>
                <span class="q-meta num">{props.ahead.eta_s !== null ? `${eta(props.ahead.eta_s)} left` : 'training'}</span>
              </li>
            ) : null}
            <For each={props.queue} key={(q) => q.id}>
              {(q) => {
                const self = () => q().id === props.self;
                return (
                  <li class="q-row" class:self={self()} aria-current={self() ? 'true' : undefined}>
                    <span class="q-n num" aria-label={ordinal(q().queue_position ?? 0)}>
                      {q().queue_position}
                    </span>
                    {self() ? (
                      <>
                        <span class="q-name">{q().name}</span>
                        <span class="q-meta">this run</span>
                      </>
                    ) : (
                      <>
                        <a class="q-name" href={href(`/runs/${q().id}`)}>
                          {q().name}
                        </a>
                        <span class="q-meta"></span>
                      </>
                    )}
                  </li>
                );
              }}
            </For>
          </ol>
        </Panel>
      ) : null}
      <Panel title="When training starts" pad="none" id="later">
        <ul class="later">
          <For each={props.later} key={(l) => l.name}>
            {(l) => (
              <li>
                <span class="l-name">{l().name}</span>
                <span class="l-what">{l().what}</span>
              </li>
            )}
          </For>
        </ul>
      </Panel>
    </>
  );
}

css`
  .queue,
  .later {
    list-style: none;
    margin: 0;
    padding: 0 0 var(--space-2);
  }
  .q-row {
    display: grid;
    grid-template-columns: 24px minmax(0, 1fr) auto;
    align-items: center;
    gap: var(--space-2);
    height: 36px;
    padding: 0 var(--space-4);
    border-top: 1px solid var(--line-subtle);
  }
  .q-row:first-child {
    border-top: 0;
  }
  /* Each name is a link: a 44 px row on touch screens. */
  @media (pointer: coarse) {
    .q-row {
      height: 44px;
    }
  }
  .q-n {
    color: var(--text-tertiary);
    font-size: var(--type-sm-size);
  }
  .live-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--accent-line);
  }
  .q-name {
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    color: var(--text-primary);
  }
  a.q-name:hover {
    text-decoration: underline;
    text-decoration-color: var(--line-strong);
    text-underline-offset: 3px;
  }
  .q-row.self {
    background: var(--bg-active);
  }
  .q-row.self .q-name {
    font-weight: var(--weight-medium);
  }
  .q-meta {
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
    white-space: nowrap;
  }
  .later li {
    display: grid;
    gap: 2px;
    padding: var(--space-2) var(--space-4);
    border-top: 1px solid var(--line-subtle);
  }
  .later li:first-child {
    border-top: 0;
  }
  .l-name {
    font-weight: var(--weight-medium);
  }
  .l-what {
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
  }
`;
