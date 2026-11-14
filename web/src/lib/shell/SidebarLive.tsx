/**
 * The active training run, always one glance away in the sidebar: one 36 px row pinned above Settings (design/UNIFIED.md
 * §1), the run's name, its percent at the right and a 2 px progress rail under them; it opens the run. The epoch, the
 * time left and the score are the run's page's, so the row says them only in its tooltip and accessible name. A polite
 * live region says when one starts training; its end is a toast (RunWatch), so it is never said twice.
 */
import { computed, css, effect, signal } from 'fw';
import { asOf, etaText, Now } from '$lib/api/freshness';
import { streamedRun } from '$lib/api/live';
import { ETA_PENDING, etaSettled, runProgress } from '$lib/api/progress';
import type { Run } from '$lib/api/types';
import { href } from '$lib/router';
import Progress from '$lib/ui/Progress';
import { tooltip } from '$lib/ui/tooltip';
import { training, watched } from './runList';

export default function SidebarLive(props: { collapsed: boolean }) {
  const listed = computed(() => watched.runs?.find(training));
  // The page's own stream of this run, while it is up (Overview's hero, the run page): the row reads the same source
  // as the page, so the two never disagree and the row stays live when only the lists stopped refreshing. A stream
  // that reports the run finished hides the row before the list catches up.
  const streamed = computed(() => streamedRun(listed.get()?.id));
  const live = computed(() => {
    const s = streamed.get();
    return s ? (training(s) ? s : undefined) : listed.get();
  });
  const progress = computed(() => {
    const l = live.get();
    return l ? runProgress(l) : 0;
  });
  // The list stopped refreshing and no stream covers the run: the row is its last known state (lib/api/freshness), so
  // it says so and its rail turns neutral. "Last known" only when both clocks are down.
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
  const held = () => since.get() !== null;
  const percent = () => `${Math.floor(progress.get() * 100 + 1e-9)}%`;
  // As the run page says it: no time left until the first epoch is done (lib/api/progress etaSettled).
  const left = () => (etaSettled(run()) ? `${held() && run().eta_s !== null ? '~' : ''}${etaText(run().eta_s, null, now.current)}` : ETA_PENDING);
  /** Everything the row leaves to its page, in one line: its tooltip and (with "Live run") its accessible name. */
  const facts = () => `${run().name} · epoch ${run().epoch} of ${run().epochs} · ${left()}`;
  const name = () => (held() ? `${run().name}, last known ${asOf(since.get() as number, now.current)}: epoch ${run().epoch} of ${run().epochs}` : `Live run ${facts()}`);

  return (
    <>
      <p class="sr-only" aria-live="polite">
        {said.get()}
      </p>
      {live.get() ? (
        <a
          href={href(`/runs/${run().id}`)}
          class="live-run"
          class:rail={props.collapsed}
          class:held={held()}
          aria-label={name()}
          use:tooltip={{ text: held() ? `Last known · ${asOf(since.get() as number, now.current)} · ${facts()}` : facts(), side: props.collapsed ? 'right' : 'top' }}
        >
          {props.collapsed ? (
            // The icon rail: how far it is, over its rail (no room for the name).
            <span class="pct num">{percent()}</span>
          ) : (
            <span class="line">
              <span class="name truncate">{run().name}</span>
              <span class="pct num">{percent()}</span>
            </span>
          )}
          <Progress value={progress.get()} size="xs" tone={held() ? 'neutral' : 'accent'} />
        </a>
      ) : null}
    </>
  );
}

css`
  .live-run {
    display: flex;
    flex-direction: column;
    justify-content: center;
    gap: var(--space-1-5);
    height: var(--size-row);
    padding: 0 var(--space-2);
    border-radius: var(--radius-md);
    transition: background-color var(--dur-fast) var(--ease-standard);
  }
  .live-run:hover {
    background: var(--bg-hover);
  }
  .line {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
  }
  .name {
    flex: 1;
    min-width: 0;
    color: var(--text-primary);
  }
  .pct {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
  }
  /* The icon rail's form: the percent over a 24 px rail, in a 40 px tile with the panel's edge (as Desktop's). */
  .live-run.rail {
    align-items: center;
    width: var(--control-xl);
    height: var(--control-xl);
    margin: 0 auto;
    padding: 0;
    gap: var(--space-1);
    background: var(--bg-raised);
    border: 1px solid var(--line-subtle);
    border-radius: var(--radius-lg);
  }
  .live-run.rail:hover {
    border-color: var(--line-default);
  }
  .rail :global(.track) {
    width: var(--space-6);
  }
`;
