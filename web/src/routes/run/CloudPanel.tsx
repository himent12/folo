/**
 * A cloud run's Cloud panel (docs/CLOUD.md): how far the hand-over is (the phases as done, current and to come, the
 * current one with its transfer), where it runs, what it has cost and the most it can, and when its checkpoint was
 * last copied home. It stays once the run has ended, with the final cost.
 */
import Check from '$icons/check';
import { computed, css, For } from 'fw';
import type { RunCloud } from '$lib/api/types';
import { offerLine, phaseIndex, phaseLabel, PHASES, PHASE_LABEL, transferShare, transferText } from '$lib/cloud';
import { ago, hoursText, timeOfDay, usd } from '$lib/format';
import { prefs } from '$lib/prefs';
import { href } from '$lib/router';
import Panel from '$lib/ui/Panel';
import Progress from '$lib/ui/Progress';

export interface CloudPanelProps {
  cloud: RunCloud;
  provider: string;
}

export default function CloudPanel(props: CloudPanelProps) {
  const at = computed(() => phaseIndex(props.cloud.phase));
  const released = computed(() => props.cloud.phase === 'released');
  const share = computed(() => transferShare(props.cloud.progress));
  const state = (i: number) => (released.get() || i < at.get() ? 'done' : i === at.get() ? 'now' : 'next');
  const cost = computed(() => {
    const c = props.cloud;
    return c.cost_usd === null ? 'not billed' : released.get() ? usd(c.cost_usd) : `${usd(c.cost_usd)} so far`;
  });
  const limit = computed(() => {
    const c = props.cloud;
    return !released.get() && c.deadline ? `${hoursText(c.max_hours)} · stops by ${timeOfDay(c.deadline)}` : hoursText(c.max_hours);
  });

  return (
    <Panel title="Cloud" meta={props.provider} id="cloud">
      <div class="body">
        <div class="wrap">
          <ol class="phases" aria-label="Hand-over">
            <For each={PHASES} key={(p) => p}>
              {(p, i) => {
                const s = () => state(i());
                return (
                  <li data-state={s()} aria-current={s() === 'now' ? 'step' : undefined}>
                    <span class="mark" aria-hidden="true">
                      {s() === 'done' ? <Check size={12} strokeWidth={2.25} /> : null}
                    </span>
                    <span class="name">{PHASE_LABEL[p()]}</span>
                    {s() === 'now' && props.cloud.progress ? (
                      <>
                        <span class="xfer num">{transferText(props.cloud.progress, prefs.units)}</span>
                        {share.get() !== null ? (
                          <span class="bar">
                            <Progress value={share.get()!} size="xs" label={`${PHASE_LABEL[p()]}: ${transferText(props.cloud.progress, prefs.units)}`} />
                          </span>
                        ) : null}
                      </>
                    ) : null}
                  </li>
                );
              }}
            </For>
            {at.get() < 0 ? (
              // A phase from a newer server: its own word, as the current one.
              <li data-state="now" aria-current="step">
                <span class="mark" aria-hidden="true"></span>
                <span class="name">{phaseLabel(props.cloud.phase)}</span>
              </li>
            ) : null}
          </ol>
          {/* One row per fact, each its own line: a long label never narrows another row's value. */}
          <dl class="facts">
            <div>
              <dt>Instance</dt>
              <dd>
                {props.cloud.instance_id ? (
                  <a class="link" href={href('/settings?section=cloud')}>
                    {offerLine(props.cloud.offer, prefs.units)}
                  </a>
                ) : (
                  offerLine(props.cloud.offer, prefs.units)
                )}
              </dd>
            </div>
            <div>
              <dt>Cost</dt>
              <dd class="num">{cost.get()}</dd>
            </div>
            <div>
              <dt>Limit</dt>
              <dd class="num">{limit.get()}</dd>
            </div>
            <div>
              <dt>Last checkpoint copy</dt>
              <dd class="num">{props.cloud.synced ? ago(props.cloud.synced) : 'none yet'}</dd>
            </div>
          </dl>
        </div>
      </div>
    </Panel>
  );
}

css`
  /* Where the panel is wide (a tablet's section, the full width), the phases and the facts sit side by side, so the
     panel uses its width instead of a narrow list with its right half empty. */
  .body {
    container-type: inline-size;
  }
  @container (min-width: 560px) {
    .wrap {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      column-gap: var(--space-8);
      align-items: start;
    }
    .wrap .phases {
      margin-bottom: 0;
    }
    .wrap .facts {
      border-top: 0;
    }
  }
  /* The phases, top to bottom: done ones with an ink check, the current one marked live (lime: it is happening now)
     with its transfer under it, those to come hollow. */
  .phases {
    display: grid;
    margin-bottom: var(--space-3);
  }
  .phases li {
    display: grid;
    grid-template-columns: 16px minmax(0, 1fr) auto;
    /* Every phase's own line is 24 px, also the current one's above its transfer bar, so the steps keep one rhythm. */
    grid-template-rows: 24px;
    align-items: center;
    column-gap: var(--space-2);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-tertiary);
  }
  .mark {
    display: grid;
    place-items: center;
    width: 14px;
    height: 14px;
    border-radius: 50%;
    border: 1px solid var(--line-strong);
  }
  [data-state='done'] {
    color: var(--text-secondary);
  }
  [data-state='done'] .mark {
    border-color: transparent;
    background: var(--bg-active);
    color: var(--text-primary);
  }
  [data-state='now'] {
    color: var(--text-primary);
  }
  [data-state='now'] .name {
    font-weight: var(--weight-medium);
  }
  [data-state='now'] .mark {
    border-color: var(--accent-line);
    box-shadow: inset 0 0 0 3px var(--bg-panel);
    background: var(--accent-line);
  }
  .xfer {
    color: var(--text-secondary);
    white-space: nowrap;
  }
  .bar {
    grid-column: 2 / -1;
    padding: 2px 0 var(--space-1);
  }
  .facts {
    display: grid;
    border-top: 1px solid var(--line-subtle);
  }
  .facts > div {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    flex-wrap: wrap;
    column-gap: var(--space-4);
    padding: 4px 0;
    border-bottom: 1px solid var(--line-subtle);
  }
  .facts > div:last-child {
    border-bottom: 0;
    padding-bottom: 0;
  }
  .facts dt {
    color: var(--text-tertiary);
  }
  .facts dd {
    margin-left: auto;
    text-align: right;
  }
  .link {
    color: var(--text-primary);
    text-decoration: underline;
    text-decoration-color: var(--line-strong);
    text-underline-offset: 3px;
  }
  .link:hover {
    text-decoration-color: var(--text-secondary);
  }
`;
