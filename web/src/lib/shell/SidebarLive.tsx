/**
 * The active training run, always one glance away in the sidebar. A polite live region says when one starts
 * training; its end is a toast (RunWatch), so it is never said twice.
 */
import { computed, css, effect, signal } from 'fw';
import { asOf, etaText, Now } from '$lib/api/freshness';
import { streamedRun } from '$lib/api/live';
import { ETA_PENDING, etaSettled, runProgress } from '$lib/api/progress';
import type { Run } from '$lib/api/types';
import { href } from '$lib/router';
import Progress from '$lib/ui/Progress';
import { tooltip } from '$lib/ui/tooltip';
import { watched } from './runList';

const training = (r: { status: string }) => r.status === 'running' || r.status === 'stopping';

export default function SidebarLive(props: { collapsed: boolean }) {
  const listed = computed(() => watched.runs?.find(training));
  // The page's own stream of this run, while it is up (Overview's hero, the run page): the card reads the same source
  // as the page, so the two never disagree and the card stays live when only the lists stopped refreshing. A stream
  // that reports the run finished hides the card before the list catches up.
  const streamed = computed(() => streamedRun(listed.get()?.id));
  const live = computed(() => {
    const s = streamed.get();
    return s ? (training(s) ? s : undefined) : listed.get();
  });
  // Runs waiting for this machine: a cloud run in its hand-over waits for no slot here.
  const queued = computed(() => watched.runs?.filter((r) => r.status === 'queued' && !r.cloud).length ?? 0);
  const progress = computed(() => {
    const l = live.get();
    return l ? runProgress(l) : 0;
  });
  // The list stopped refreshing and no stream covers the run: the card is its last known state (lib/api/freshness), so
  // it says so and stops pulsing. "Last known" only when both clocks are down.
  const since = computed(() => (streamed.get() ? null : watched.since));
  const now = new Now(() => since.get() !== null);

  // Said once, when a run starts training while Studio is open (not for the run already training at load).
  const said = signal('');
  let seen: string | null | undefined;
  effect(() => {
    const id = live.get()?.id ?? null;
    const name = live.get()?.name;
    // No list (yet) from the server in use: its first is where things stand there, not a run that started.
    if (!watched.runs) return void (seen = undefined);
    if (seen !== undefined && id && id !== seen) said.set(`${name} started training`);
    seen = id;
  });

  const run = () => live.get() as Run;
  // Text with values in it is one string (one text node, as Svelte renders it: split text shapes differently).
  const held = () => since.get() !== null;

  return (
    <>
      <p class="sr-only" aria-live="polite">
        {said.get()}
      </p>
      {live.get() ? (
        <>
          {props.collapsed ? (
            <a
              href={href(`/runs/${run().id}`)}
              class="mini"
              class:held={held()}
              aria-label={
                held()
                  ? `${run().name}, last known ${asOf(since.get() as number, now.current)}: epoch ${run().epoch} of ${run().epochs}`
                  : `Live run ${run().name}, epoch ${run().epoch} of ${run().epochs}`
              }
              use:tooltip={{ text: `${run().name} · epoch ${run().epoch} of ${run().epochs}`, side: 'right' }}
            >
              <svg viewBox="0 0 24 24" width="24" height="24" aria-hidden="true">
                <circle cx="12" cy="12" r="10" class="track" />
                <circle cx="12" cy="12" r="10" class="arc" stroke-dasharray={`${62.83 * progress.get()} 62.83`} transform="rotate(-90 12 12)" />
                <circle cx="12" cy="12" r="3" class="core" />
              </svg>
            </a>
          ) : (
            <a href={href(`/runs/${run().id}`)} class="card" class:held={held()}>
              <span class="top">
                <span class="pulse" aria-hidden="true"></span>
                <span class="label">{held() ? `Last known · ${asOf(since.get() as number, now.current)}` : 'Training'}</span>
                {/* As the run page says it: no time left until the first epoch is done (lib/api/progress etaSettled). */}
                <span class="eta num">
                  {etaSettled(run()) ? `${held() && run().eta_s !== null ? '~' : ''}${etaText(run().eta_s, null, now.current)}` : ETA_PENDING}
                </span>
              </span>
              <span class="name truncate">{run().name}</span>
              <Progress value={progress.get()} size="xs" label="Training progress" tone={held() ? 'neutral' : 'accent'} />
              <span class="meta num">
                <span>{`Epoch ${run().epoch} of ${run().epochs}`}</span>
                {run().best ? <span>{`mAP50-95 ${run().best?.map50_95.toFixed(3)}`}</span> : null}
              </span>
              {queued.get() > 0 ? <span class="queued">{`+${queued.get()} queued`}</span> : null}
            </a>
          )}
        </>
      ) : null}
    </>
  );
}

css`
  .card {
    display: grid;
    gap: 6px;
    margin: 0 var(--space-1);
    padding: 12px;
    border-radius: var(--radius-lg);
    background: var(--bg-panel);
    border: 1px solid var(--line-panel);
    transition:
      border-color var(--dur-fast) var(--ease-standard),
      background-color var(--dur-fast) var(--ease-standard);
  }
  .card:hover {
    border-color: var(--line-strong);
  }
  .top {
    display: flex;
    align-items: center;
    gap: 6px;
  }
  /* The pulsing dot carries "live"; the word stays in ink. */
  .label {
    font-size: var(--type-sm-size);
    font-weight: var(--weight-medium);
    color: var(--text-secondary);
  }
  .pulse {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--accent-solid);
    --pulse: var(--accent-line);
    animation: live-pulse 2.4s var(--ease-out) infinite;
  }
  :global([data-theme='light']) .pulse {
    background: var(--accent-line);
  }
  .held .arc {
    stroke: var(--text-tertiary);
  }
  .held .core {
    fill: var(--text-tertiary);
  }
  /* A last known state: the dot in ink and still (nothing here is current). */
  .held .pulse {
    background: var(--text-tertiary);
    animation: none;
  }
  .eta {
    margin-left: auto;
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
  }
  .name {
    font-weight: var(--weight-medium);
  }
  .meta {
    display: flex;
    justify-content: space-between;
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
  }
  .queued {
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
  }
  .mini {
    display: grid;
    place-items: center;
    width: 40px;
    height: 40px;
    margin: 0 auto;
    border-radius: var(--radius-md);
  }
  .mini:hover {
    background: var(--bg-hover);
  }
  .track {
    fill: none;
    stroke: var(--bg-active);
    stroke-width: 2.5;
  }
  .arc {
    fill: none;
    stroke: var(--accent-line);
    stroke-width: 2.5;
    stroke-linecap: round;
  }
  .core {
    fill: var(--accent-line);
  }
  /* The live heartbeat (\`live-pulse\`, base.css): one ring every 2.4 s; off under reduced motion. */
  @media (prefers-reduced-motion: reduce) {
    .pulse {
      animation: none;
    }
  }
`;
