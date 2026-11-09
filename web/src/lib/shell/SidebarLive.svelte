<script lang="ts">
  /**
   * The active training run, always one glance away in the sidebar. A polite live region says when one starts
   * training; its end is a toast (RunWatch), so it is never said twice.
   */
  import { href } from '$lib/router.svelte';
  import { asOf, etaText, Now } from '$lib/api/freshness.svelte';
  import { runProgress } from '$lib/api/progress';
  import Progress from '$lib/ui/Progress.svelte';
  import { tooltip } from '$lib/ui/tooltip';
  import { watched } from './runList.svelte';

  let { collapsed }: { collapsed: boolean } = $props();

  const live = $derived(watched.runs?.find((r) => r.status === 'running' || r.status === 'stopping'));
  const queued = $derived(watched.runs?.filter((r) => r.status === 'queued').length ?? 0);
  const progress = $derived(live ? runProgress(live) : 0);
  // The list stopped refreshing: the card is its last known state (lib/api/freshness), so it says so and stops pulsing.
  const since = $derived(watched.since);
  const now = new Now(() => since !== null);

  // Said once, when a run starts training while Studio is open (not for the run already training at load).
  let said = $state('');
  let seen: string | null | undefined;
  $effect(() => {
    const id = live?.id ?? null;
    const name = live?.name;
    if (!watched.runs) return;
    if (seen !== undefined && id && id !== seen) said = `${name} started training`;
    seen = id;
  });
</script>

<p class="sr-only" aria-live="polite">{said}</p>

{#if live}
  {#if collapsed}
    <a
      href={href(`/runs/${live.id}`)}
      class="mini"
      class:held={since !== null}
      aria-label={since !== null ? `${live.name}, last known ${asOf(since, now.current)}: epoch ${live.epoch} of ${live.epochs}` : `Live run ${live.name}, epoch ${live.epoch} of ${live.epochs}`}
      use:tooltip={{ text: `${live.name} · epoch ${live.epoch}/${live.epochs}`, side: 'right' }}
    >
      <svg viewBox="0 0 24 24" width="24" height="24" aria-hidden="true">
        <circle cx="12" cy="12" r="10" class="track" />
        <circle cx="12" cy="12" r="10" class="arc" stroke-dasharray="{62.83 * progress} 62.83" transform="rotate(-90 12 12)" />
        <circle cx="12" cy="12" r="3" class="core" />
      </svg>
    </a>
  {:else}
    <a href={href(`/runs/${live.id}`)} class="card" class:held={since !== null}>
      <span class="top">
        <span class="pulse" aria-hidden="true"></span>
        <span class="label">{since !== null ? `Last known · ${asOf(since, now.current)}` : 'Training'}</span>
        <span class="eta num">{since !== null && live.eta_s !== null ? '~' : ''}{etaText(live.eta_s, null, now.current)}</span>
      </span>
      <span class="name truncate">{live.name}</span>
      <Progress value={progress} size="xs" label="Training progress" tone={since !== null ? 'neutral' : 'accent'} />
      <span class="meta num">
        <span>Epoch {live.epoch}/{live.epochs}</span>
        {#if live.best}<span>mAP {live.best.map50_95.toFixed(3)}</span>{/if}
      </span>
      {#if queued > 0}<span class="queued">+{queued} queued</span>{/if}
    </a>
  {/if}
{/if}

<style>
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
  /* The live heartbeat (`live-pulse`, base.css): one ring every 2.4 s; off under reduced motion. */
  @media (prefers-reduced-motion: reduce) {
    .pulse {
      animation: none;
    }
  }
</style>
