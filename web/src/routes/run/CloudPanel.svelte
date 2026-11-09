<script lang="ts">
  /**
   * A cloud run's Cloud panel (docs/CLOUD.md): how far the hand-over is (the phases as done, current and to come, the
   * current one with its transfer), where it runs, what it has cost and the most it can, and when its checkpoint was
   * last copied home. It stays once the run has ended, with the final cost.
   */
  import Check from '@lucide/svelte/icons/check';
  import type { RunCloud } from '$lib/api/types';
  import { offerLine, phaseIndex, phaseLabel, PHASES, PHASE_LABEL, transferShare, transferText } from '$lib/cloud';
  import { ago, hoursText, timeOfDay, usd } from '$lib/format';
  import { prefs } from '$lib/prefs.svelte';
  import { href } from '$lib/router.svelte';
  import Panel from '$lib/ui/Panel.svelte';
  import Progress from '$lib/ui/Progress.svelte';

  let { cloud, provider }: { cloud: RunCloud; provider: string } = $props();

  const at = $derived(phaseIndex(cloud.phase));
  const released = $derived(cloud.phase === 'released');
  const share = $derived(transferShare(cloud.progress));
  const state = (i: number) => (released || i < at ? 'done' : i === at ? 'now' : 'next');
  const cost = $derived(cloud.cost_usd === null ? 'not billed' : released ? usd(cloud.cost_usd) : `${usd(cloud.cost_usd)} so far`);
  const limit = $derived(!released && cloud.deadline ? `${hoursText(cloud.max_hours)} · stops by ${timeOfDay(cloud.deadline)}` : hoursText(cloud.max_hours));
</script>

<Panel title="Cloud" meta={provider} id="cloud">
  <ol class="phases" aria-label="Hand-over">
    {#each PHASES as p, i (p)}
      {@const s = state(i)}
      <li data-state={s} aria-current={s === 'now' ? 'step' : undefined}>
        <span class="mark" aria-hidden="true">{#if s === 'done'}<Check size={12} strokeWidth={2.25} />{/if}</span>
        <span class="name">{PHASE_LABEL[p]}</span>
        {#if s === 'now' && cloud.progress}
          <span class="xfer num">{transferText(cloud.progress, prefs.units)}</span>
          {#if share !== null}<span class="bar"><Progress value={share} size="xs" label="{PHASE_LABEL[p]}: {transferText(cloud.progress, prefs.units)}" /></span>{/if}
        {/if}
      </li>
    {/each}
    {#if at < 0}
      <!-- A phase from a newer server: its own word, as the current one. -->
      <li data-state="now" aria-current="step"><span class="mark" aria-hidden="true"></span><span class="name">{phaseLabel(cloud.phase)}</span></li>
    {/if}
  </ol>
  <!-- One row per fact, each its own line: a long label never narrows another row's value. -->
  <dl class="facts">
    <div>
      <dt>Instance</dt>
      <dd>{#if cloud.instance_id}<a class="link" href={href('/settings?section=cloud')}>{offerLine(cloud.offer, prefs.units)}</a>{:else}{offerLine(cloud.offer, prefs.units)}{/if}</dd>
    </div>
    <div><dt>Cost</dt><dd class="num">{cost}</dd></div>
    <div><dt>Limit</dt><dd class="num">{limit}</dd></div>
    <div><dt>Last checkpoint copy</dt><dd class="num">{cloud.synced ? ago(cloud.synced) : 'none yet'}</dd></div>
  </dl>
</Panel>

<style>
  /* The phases, top to bottom: done ones with an ink check, the current one marked live (lime: it is happening now)
     with its transfer under it, those to come hollow. */
  .phases {
    display: grid;
    margin-bottom: var(--space-3);
  }
  .phases li {
    display: grid;
    grid-template-columns: 16px minmax(0, 1fr) auto;
    align-items: center;
    column-gap: var(--space-2);
    min-height: 24px;
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
</style>
