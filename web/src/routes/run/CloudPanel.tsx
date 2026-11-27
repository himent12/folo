/**
 * A cloud run's Cloud panel (docs/CLOUD.md): how far the hand-over is (the phases as done, current and to come, the
 * current one with its transfer; once released, where a failed run failed and what it never reached), where it runs,
 * what it has cost and the most it can, and when its checkpoint was last copied home. It stays once the run has ended.
 * A released run's cost is its instance's own, from the ledger (`GET /api/cloud/instances`), which keeps counting
 * after the run stopped waiting for the termination; while the provider has not confirmed it, releasing is still under
 * way and the instance may still be billed, and the instance is read again every 30 s (also while it cannot be read).
 */
import Check from '$icons/check';
import RotateCcw from '$icons/rotate-ccw';
import TriangleAlert from '$icons/triangle-alert';
import WifiOff from '$icons/wifi-off';
import X from '$icons/x';
import { computed, css, effect, For, signal, untrack } from 'fw';
import { hub } from '$lib/api/index';
import { Resource } from '$lib/api/resource';
import { pollTogether } from '$lib/api/share';
import type { RunCloud } from '$lib/api/types';
import { isAlive, offerLine, panelPhaseLabel, phaseIndex, phaseLabel, phaseStates, transferShare, transferText, type PhaseRun } from '$lib/cloud';
import { ago, hoursText, timeOfDay, usd } from '$lib/format';
import { prefs } from '$lib/prefs';
import { href } from '$lib/router';
import Badge from '$lib/ui/Badge';
import Panel from '$lib/ui/Panel';
import Progress from '$lib/ui/Progress';
import type { SpotRecovery } from './cloudRecovery';

export interface CloudPanelProps {
  cloud: RunCloud;
  provider: string;
  /** How the run ended and whether it trained: where a failed run failed, what it never reached. */
  run?: PhaseRun;
  /** The provider launches an instance per run (RunPod); a remote server does not (no provisioning, no release). */
  provisions?: boolean;
  /** Spot preemption auto-resume status derived from the run's logs and outcome (optional). */
  recovery?: SpotRecovery | null;
}

/** How often an instance whose termination is not confirmed is read again. */
const UNCONFIRMED_POLL_MS = 30_000;

export default function CloudPanel(props: CloudPanelProps) {
  const provisions = () => props.provisions ?? true;
  const released = computed(() => props.cloud.phase === 'released');
  // A released run's instance in the ledger: its cost, and whether the provider confirmed its termination (the run
  // stops waiting after a minute; until then it may still be billed). Only a provisioned instance, only once released.
  const instanceId = computed(() => (released.get() && provisions() ? props.cloud.instance_id : null));
  const instance = new Resource(async (s) => {
    const id = instanceId.get();
    if (!id) return null;
    return (await hub.api.cloudInstances(s)).find((i) => i.id === id) ?? null;
  });
  /** The instance of this run (the panel stays when the page moves to another run; an answer for the last one is not this one's). */
  const own = computed(() => (instance.data && instance.data.id === instanceId.get() ? instance.data : null));
  const unconfirmed = computed(() => (own.get() && isAlive(own.get()!.status) ? own.get() : null));
  /** The ledger could not be read: whether the instance is gone is not known. */
  const unknown = computed(() => !!instanceId.get() && !own.get() && instance.error !== undefined);
  effect(() => {
    if (!unconfirmed.get() && !unknown.get()) return;
    return pollTogether(UNCONFIRMED_POLL_MS, () => {
      if (document.visibilityState !== 'visible' || hub.mode === 'connecting') return;
      if (untrack(() => instance.loading || instance.refreshing)) return;
      void instance.refresh();
    });
  });
  const states = computed(() => phaseStates(props.cloud, props.run, provisions(), !!unconfirmed.get()));
  const share = computed(() => transferShare(props.cloud.progress));
  const cost = computed(() => {
    const c = props.cloud;
    const i = own.get();
    if (i) return i.cost_usd === null ? 'not billed' : isAlive(i.status) ? `${usd(i.cost_usd)} so far` : usd(i.cost_usd);
    return c.cost_usd === null ? 'not billed' : released.get() ? usd(c.cost_usd) : `${usd(c.cost_usd)} so far`;
  });
  // The facts that are times ago ("Last checkpoint copy: 2m ago") move on while nothing else on the run does (a
  // worker that does not answer sends nothing new).
  const now = signal(Date.now());
  effect(() => {
    if (released.get()) return;
    const t = setInterval(() => now.set(Date.now()), 30_000);
    return () => clearInterval(t);
  });
  const limit = computed(() => {
    const c = props.cloud;
    return !released.get() && c.deadline ? `${hoursText(c.max_hours)} · stops by ${timeOfDay(c.deadline)}` : hoursText(c.max_hours);
  });
  const recovery = () => props.recovery ?? null;
  const recoveryTone = computed(() => {
    const rec = recovery();
    if (!rec) return null;
    if (rec.kind === 'exhausted' || rec.kind === 'no_checkpoint') return 'danger' as const;
    if (rec.kind === 'recovering' || rec.kind === 'stopped') return 'warning' as const;
    return 'neutral' as const;
  });
  const recoveryFact = computed(() => {
    const rec = recovery();
    if (!rec || rec.attempts <= 0) return null;
    const count = rec.maxAttempts !== null ? `${rec.attempts} of ${rec.maxAttempts}` : `${rec.attempts}`;
    if (rec.kind === 'exhausted') return `${count} · exhausted`;
    if (rec.resumeEpoch !== null) return `${count} · epoch ${rec.resumeEpoch}`;
    return count;
  });

  return (
    <Panel
      title="Cloud"
      meta={props.provider}
      id="cloud"
      actions={
        recovery() ? (
          <span data-spot-recovery-badge={recovery()!.kind}>
            <Badge tone={recoveryTone.get() ?? 'neutral'}>
              {recovery()!.kind === 'exhausted' || recovery()!.kind === 'no_checkpoint' ? (
                <TriangleAlert size={12} aria-hidden="true" />
              ) : (
                <RotateCcw size={12} aria-hidden="true" />
              )}
              {recovery()!.badge}
            </Badge>
          </span>
        ) : undefined
      }
    >
      <div class="body">
        <div class="wrap">
          <ol class="phases" aria-label="Hand-over">
            <For each={states.get()} key={(p) => p.phase}>
              {(p) => {
                const s = () => p().state;
                const label = () => panelPhaseLabel(p().phase, provisions());
                return (
                  <li data-state={s()} aria-current={s() === 'now' ? 'step' : undefined}>
                    <span class="mark" aria-hidden="true">
                      {s() === 'done' ? <Check size={12} strokeWidth={2.25} /> : s() === 'failed' ? <X size={12} strokeWidth={2.25} /> : null}
                    </span>
                    <span class="name">{label()}</span>
                    {/* Out of the grid (absolute): the mark's meaning for a screen reader. */}
                    {s() === 'failed' ? <span class="sr-only">(failed)</span> : s() === 'skipped' ? <span class="sr-only">(not reached)</span> : null}
                    {s() === 'now' && props.cloud.progress ? (
                      <>
                        <span class="xfer num">{transferText(props.cloud.progress, prefs.units)}</span>
                        {share.get() !== null ? (
                          <span class="bar">
                            <Progress value={share.get()!} size="xs" label={`${label()}: ${transferText(props.cloud.progress, prefs.units)}`} />
                          </span>
                        ) : null}
                      </>
                    ) : null}
                  </li>
                );
              }}
            </For>
            {phaseIndex(props.cloud.phase) < 0 ? (
              // A phase from a newer server: its own word, as the current one.
              <li data-state="now" aria-current="step">
                <span class="mark" aria-hidden="true"></span>
                <span class="name">{phaseLabel(props.cloud.phase)}</span>
              </li>
            ) : null}
          </ol>
          {/* One row per fact, each its own line: a long label never narrows another row's value. */}
          <dl class="facts">
            {recovery() ? (
              <p
                class="recovery-line"
                class:danger={recovery()!.kind === 'exhausted' || recovery()!.kind === 'no_checkpoint'}
                class:recovering={recovery()!.kind === 'recovering'}
                role={recovery()!.kind === 'exhausted' || recovery()!.kind === 'no_checkpoint' ? 'alert' : 'status'}
                data-spot-recovery={recovery()!.kind}
              >
                {recovery()!.kind === 'exhausted' || recovery()!.kind === 'no_checkpoint' ? (
                  <TriangleAlert size={14} strokeWidth={2} aria-hidden="true" />
                ) : (
                  <RotateCcw size={14} strokeWidth={2} aria-hidden="true" />
                )}
                <span>{recovery()!.summary}</span>
              </p>
            ) : null}
            {props.cloud.unanswered_since ? (
              <p class="warn-line" role="status">
                <WifiOff size={14} strokeWidth={2} aria-hidden="true" />
                <span>
                  The worker has not answered since {timeOfDay(props.cloud.unanswered_since)}. FOLO keeps asking; the run
                  may still be training or waiting there.
                </span>
              </p>
            ) : unconfirmed.get() ? (
              <p class="warn-line" role="status">
                <TriangleAlert size={14} strokeWidth={2} aria-hidden="true" />
                <span>
                  Termination not confirmed: the instance may still be billed.{' '}
                  <a class="link" href={href('/settings?section=cloud')}>
                    Settings › Cloud
                  </a>
                </span>
              </p>
            ) : unknown.get() ? (
              <p class="warn-line" role="status">
                <TriangleAlert size={14} strokeWidth={2} aria-hidden="true" />
                <span>
                  Could not check that the instance is terminated.{' '}
                  <a class="link" href={href('/settings?section=cloud')}>
                    Settings › Cloud
                  </a>
                </span>
              </p>
            ) : null}
            <div>
              {/* A remote server is not an instance: the row says its GPU. */}
              <dt>{provisions() ? 'Instance' : 'GPU'}</dt>
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
              <dd class="num">{props.cloud.synced ? ago(props.cloud.synced, now.get()) : released.get() ? 'none' : 'none yet'}</dd>
            </div>
            {recoveryFact.get() ? (
              <div>
                <dt>Spot recoveries</dt>
                <dd class="num">{recoveryFact.get()}</dd>
              </div>
            ) : null}
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
  /* A failed phase: its mark in the danger hue, the word in ink (DESIGN_DIRECTION §2.6). Phases never reached: faint,
     their ring dashed. */
  [data-state='failed'] {
    color: var(--text-primary);
  }
  [data-state='failed'] .mark {
    border-color: transparent;
    color: var(--danger-solid);
  }
  [data-state='skipped'] .mark {
    border-style: dashed;
  }
  /* A worker that does not answer, a termination not confirmed, a ledger not read: the mark in its hue, the words
     secondary. */
  .warn-line,
  .recovery-line {
    display: flex;
    align-items: flex-start;
    gap: var(--space-2);
    padding: 4px 0;
    border-bottom: 1px solid var(--line-subtle);
    color: var(--text-secondary);
  }
  .warn-line :global(svg),
  .recovery-line :global(svg) {
    flex: none;
    margin-top: 2px;
    color: var(--warning-solid);
  }
  .recovery-line :global(svg) {
    color: var(--text-secondary);
  }
  .recovery-line.recovering :global(svg) {
    color: var(--warning-solid);
  }
  .recovery-line.danger {
    color: var(--text-primary);
  }
  .recovery-line.danger :global(svg) {
    color: var(--danger-solid);
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
