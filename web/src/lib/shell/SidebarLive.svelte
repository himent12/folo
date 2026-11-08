<script lang="ts">
  /** The active training run, always one glance away in the sidebar. */
  import { hub } from '$lib/api/index.svelte';
  import { Resource } from '$lib/api/resource.svelte';
  import { href } from '$lib/router.svelte';
  import { duration } from '$lib/format';
  import Progress from '$lib/ui/Progress.svelte';
  import { tooltip } from '$lib/ui/tooltip';

  let { collapsed }: { collapsed: boolean } = $props();

  const runs = new Resource((s) => hub.api.runs(s), { poll: 2000 });
  const live = $derived(runs.data?.find((r) => r.status === 'running' || r.status === 'stopping'));
  const queued = $derived(runs.data?.filter((r) => r.status === 'queued').length ?? 0);
  const progress = $derived(live ? (live.step / Math.max(1, live.steps_per_epoch * live.epochs)) : 0);
</script>

{#if live}
  {#if collapsed}
    <a
      href={href(`/runs/${live.id}`)}
      class="mini"
      aria-label="Live run {live.name}, epoch {live.epoch} of {live.epochs}"
      use:tooltip={{ text: `${live.name} · epoch ${live.epoch}/${live.epochs}`, side: 'right' }}
    >
      <svg viewBox="0 0 24 24" width="24" height="24" aria-hidden="true">
        <circle cx="12" cy="12" r="10" class="track" />
        <circle cx="12" cy="12" r="10" class="arc" stroke-dasharray="{62.83 * progress} 62.83" transform="rotate(-90 12 12)" />
        <circle cx="12" cy="12" r="3" class="core" />
      </svg>
    </a>
  {:else}
    <a href={href(`/runs/${live.id}`)} class="card">
      <span class="top">
        <span class="pulse" aria-hidden="true"></span>
        <span class="overline">Training</span>
        <span class="eta num">{duration(live.eta_s)} left</span>
      </span>
      <span class="name truncate">{live.name}</span>
      <Progress value={progress} size="xs" label="Training progress" />
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
    padding: 10px 12px 12px;
    border-radius: var(--radius-lg);
    background: var(--bg-panel);
    border: 1px solid var(--line-subtle);
    transition:
      border-color var(--dur-fast) var(--ease-standard),
      background-color var(--dur-fast) var(--ease-standard);
  }
  :global([data-theme='light']) .card {
    border-color: var(--line-default);
  }
  .card:hover {
    border-color: var(--line-strong);
  }
  .top {
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .overline {
    color: var(--accent-text);
  }
  .pulse {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--accent-solid);
    animation: pulse 1.6s var(--ease-out) infinite;
  }
  :global([data-theme='light']) .pulse {
    background: var(--accent-line);
  }
  .eta {
    margin-left: auto;
    font-size: var(--type-xs-size);
    color: var(--text-tertiary);
  }
  .name {
    font-weight: var(--weight-medium);
  }
  .meta {
    display: flex;
    justify-content: space-between;
    font-size: var(--type-xs-size);
    color: var(--text-tertiary);
  }
  .queued {
    font-size: var(--type-xs-size);
    color: var(--info-text);
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
  @keyframes pulse {
    0% {
      box-shadow: 0 0 0 0 color-mix(in srgb, var(--accent-solid) 60%, transparent);
    }
    100% {
      box-shadow: 0 0 0 5px transparent;
    }
  }
</style>
