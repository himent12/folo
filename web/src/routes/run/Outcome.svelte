<script lang="ts">
  /**
   * A run's outcome under the header, every state in one pattern (./Callout): icon, title, cause, one action, and a
   * disclosure for the rest. Out of GPU memory says so plainly, with the smaller setting the memory forecast says fits
   * as the action, named with its forecast ("Retry with batch 8 · ~13.6 GiB", lib/oom `fitFixes`), and the other fixes
   * and the driver's report behind "More fixes". A queued run leads with what it waits for and when it
   * should start; on servers with queue control, with its place ("2nd in queue") and the way to the front. The trainer's warning that a run is too short to learn from (./warnings) is one line, "Too short to
   * learn: 125 optimizer steps", with the engine's sentences in plain words behind "Why?"; never a toast. Once the run
   * has an outcome, that line joins the outcome's callout as its second line ("Completed in 30s · best …", then "Too
   * short to learn: …"), so one run never stacks two banners.
   */
  import type { Snippet } from 'svelte';
  import CircleCheck from '@lucide/svelte/icons/circle-check';
  import Clock from '@lucide/svelte/icons/clock';
  import Play from '@lucide/svelte/icons/play';
  import RotateCcw from '@lucide/svelte/icons/rotate-ccw';
  import ScanSearch from '@lucide/svelte/icons/scan-search';
  import Square from '@lucide/svelte/icons/square';
  import TriangleAlert from '@lucide/svelte/icons/triangle-alert';
  import { hub } from '$lib/api/index.svelte';
  import type { Run, SystemInfo } from '$lib/api/types';
  import { bytes, duration, metric } from '$lib/format';
  import { fitFixes, isOom, oomCalibration, oomDetail, oomFixes, retryPath, vramForecaster, type OomFix } from '$lib/oom';
  import { prefs } from '$lib/prefs.svelte';
  import { href } from '$lib/router.svelte';
  import Button from '$lib/ui/Button.svelte';
  import Ident from '$lib/ui/Ident.svelte';
  import type { IconType } from '$lib/ui/types';
  import Callout from './Callout.svelte';
  import { plain, scheduleTitle, shortCause, suggestedEpochs } from './warnings';

  let {
    run,
    elapsed,
    resuming = false,
    onresume,
    waiting,
    waitingTitle = 'Waiting to start',
    waitingIcon = Clock,
    waitingActions,
    warnings = [],
    system,
  }: {
    run: Run;
    elapsed: number | null;
    resuming?: boolean;
    onresume: () => void;
    /** A queued run: what it waits for and when it should start. */
    waiting?: Snippet;
    /** A queued run's place, "2nd in queue", where the server reports it; a cloud run's phase. */
    waitingTitle?: string;
    /** The waiting state's mark: a clock in the queue, a cloud for a cloud run's hand-over. */
    waitingIcon?: IconType;
    /** A queued run's action: Start next. */
    waitingActions?: Snippet;
    /** The trainer's schedule warnings, as it wrote them (./warnings). */
    warnings?: string[];
    /**
     * `/api/system`: the GPU's memory and whether the server estimates runs, to forecast the fixes. Undefined while it
     * loads; null when it could not be read (the fixes are then offered unforecast).
     */
    system?: SystemInfo | null;
  } = $props();

  // Out of GPU memory: the same config fails again, so the ways out are smaller settings (a new run), most likely first.
  const oom = $derived(run.status === 'failed' && isOom(run.error));
  // Each fix is forecast as New run will forecast it (lib/oom `fitFixes`): the batch is the largest that fits the GPU
  // with 10% to spare, and the fixes that fit lead. Null while the forecast loads.
  let fixes = $state.raw<OomFix[] | null>(null);
  const gpuBytes = $derived.by(() => {
    const d = run.config.device;
    const g = system?.gpus.find((x) => (d.startsWith('gpu:') ? `gpu:${x.ordinal}` === d : true));
    return g && g.memory_bytes > 0 ? g.memory_bytes : null;
  });
  $effect(() => {
    if (!oom || !system) {
      fixes = !oom ? [] : system === null ? oomFixes(run.config) : null;
      return;
    }
    const ctrl = new AbortController();
    const c = run.config;
    fitFixes(c, gpuBytes, vramForecaster(hub.api, system, ctrl.signal), oomCalibration(c, gpuBytes)).then(
      (f) => !ctrl.signal.aborted && (fixes = f),
      () => {},
    );
    return () => ctrl.abort();
  });
  /** A fix with what it costs: "Retry with batch 8 · ~13.6 GiB". */
  const fixLabel = (f: OomFix) => (f.vram !== undefined ? `${f.label} · ~${bytes(f.vram, prefs.units)}` : f.label);
  const driver = $derived(oom ? oomDetail(run.error) : null);
  /** The epochs that clear the warning (⌈200 / steps per epoch⌉): its action opens New run with them filled in. */
  const suggested = $derived(suggestedEpochs(warnings));
  /**
   * The trainer's schedule warning as a callout line, "Too short to learn: 125 optimizer steps": the second line of the
   * run's outcome callout once there is one, else a callout of its own.
   */
  const short = $derived(warnings.length ? { icon: TriangleAlert, tone: 'warning' as const, title: scheduleTitle(warnings), msg: shortCause(warnings) } : undefined);
</script>

{#if run.status === 'failed' && oom}
  <!-- Out of memory blocks any run: its fix stays the action; a schedule warning joins as the second line, its fix and
       reasons behind More fixes. -->
  <Callout tone="danger" icon={TriangleAlert} title="Ran out of GPU memory" role="alert" stack moreLabel="More fixes" also={short}>
    Batch {run.config.batch} at {run.config.imgsz} px with <Ident value={run.config.arch} /> did not fit, at epoch {run.epoch}.
    {#snippet actions()}
      {#if fixes === null}
        <!-- The forecast is on its way: the action keeps its place and says it is choosing. -->
        <Button size="sm" variant="primary" icon={RotateCcw} loading>Retry with smaller settings</Button>
      {:else if fixes[0]}
        <Button size="sm" variant="primary" icon={RotateCcw} href={href(retryPath(run.id, fixes[0]))}>{fixLabel(fixes[0])}</Button>
      {:else}
        <Button size="sm" icon={RotateCcw} href={href(retryPath(run.id))}>New run like this</Button>
      {/if}
    {/snippet}
    {#snippet more()}
      {#if (fixes?.slice(1).some((f) => f.fits !== false) ?? false) || short}
        <div class="alts cta-row">
          {#each fixes?.slice(1).filter((f) => f.fits !== false) ?? [] as f (f.kind)}<Button size="sm" href={href(retryPath(run.id, f))}>{fixLabel(f)}</Button>{/each}
          {#if short}{@render scheduleFix()}{/if}
        </div>
      {/if}
      {#if driver}
        <p class="d-head">Driver report</p>
        <code class="mono">{driver}</code>
      {/if}
      {#if short}{@render why()}{/if}
    {/snippet}
  </Callout>
{:else if run.status === 'failed'}
  <Callout tone="danger" icon={TriangleAlert} title="Failed at epoch {run.epoch}" role="alert" moreLabel={short ? 'Why?' : 'More fixes'} also={short}>
    {run.error}
    {#snippet actions()}
      {#if short}{@render scheduleFix()}{/if}
      <!-- Anything but out of memory can continue from the last checkpoint. -->
      <Button size="sm" icon={Play} loading={resuming} onclick={onresume}>Resume</Button>
    {/snippet}
    {#snippet more()}
      {#if short}{@render why()}{:else}
        <div class="alts cta-row"><Button size="sm" icon={RotateCcw} href={href(retryPath(run.id))}>New run like this</Button></div>
      {/if}
    {/snippet}
  </Callout>
{:else if run.status === 'completed' && short}
  <!-- A run that completed but was too short to learn: the outcome and the warning in one callout, the fix its action
       (trying a model that learned nothing in the Playground is not the next step). -->
  <Callout tone="success" icon={CircleCheck} title="Completed in {duration(elapsed)}" moreLabel="Why?" also={short}>
    {#snippet facts()}<span>best mAP50-95 <span class="num">{metric(run.best?.map50_95)}</span> at epoch {run.best?.epoch ?? 0}</span>{/snippet}
    {#snippet actions()}{@render scheduleFix()}{/snippet}
    {#snippet more()}{@render why()}{/snippet}
  </Callout>
{:else if run.status === 'completed'}
  <Callout tone="success" icon={CircleCheck} title="Completed in {duration(elapsed)}" brief>
    {#snippet facts()}<span>best mAP50-95 <span class="num">{metric(run.best?.map50_95)}</span> at epoch {run.best?.epoch ?? 0}</span
      ><span>saved as a model</span>{/snippet}
    {#snippet actions()}
      {#if run.model_id}<Button size="sm" icon={ScanSearch} href={href(`/playground?model=${run.model_id}`)}>Try in Playground</Button>{/if}
    {/snippet}
  </Callout>
{:else if run.status === 'stopped'}
  <Callout tone="neutral" icon={Square} title="Stopped at epoch {run.epoch}" moreLabel="Why?" also={short} more={short ? why : undefined}>
    The last checkpoint was kept{run.model_id ? ' and saved as a model' : ''}.
    {#snippet actions()}
      {#if short}{@render scheduleFix()}{/if}
      <Button size="sm" icon={Play} loading={resuming} onclick={onresume}>Resume</Button>
    {/snippet}
  </Callout>
{:else if run.status === 'queued' && waiting}
  <!-- The dependency and the start estimate lead; the frames below stay compact until there is data. -->
  <Callout tone="neutral" icon={waitingIcon} title={waitingTitle} actions={waitingActions}>{@render waiting()}</Callout>
{:else if short}
  <!-- No outcome yet (a live run): the warning alone, one line: the cause in numbers and the fix; the engine's
       sentences, in plain words, behind "Why?". -->
  <Callout tone="warning" icon={TriangleAlert} title={short.title} role="note" label="Schedule warning" moreLabel="Why?" runIn more={why}>
    {short.msg}
    {#snippet actions()}{@render scheduleFix()}{/snippet}
  </Callout>
{/if}

{#snippet scheduleFix()}
  {#if suggested && suggested > run.config.epochs}
    <Button size="sm" icon={RotateCcw} href={href(`${retryPath(run.id)}&epochs=${suggested}`)}>Train {suggested} epochs</Button>
  {:else}
    <Button size="sm" icon={RotateCcw} href={href(retryPath(run.id))}>New run like this</Button>
  {/if}
{/snippet}

{#snippet why()}
  {#each warnings as w (w)}<p class="why">{plain(w)}</p>{/each}
{/snippet}

<style>
  .alts {
    display: flex;
    flex-wrap: wrap;
    gap: var(--space-2);
    padding-top: var(--space-1);
  }
  .d-head {
    margin-top: var(--space-2);
    color: var(--text-tertiary);
  }
  code {
    display: block;
    overflow-wrap: anywhere;
    color: var(--text-secondary);
  }
  /* The engine's sentences at a readable measure. */
  .why {
    max-inline-size: var(--measure);
  }
  .why + .why {
    margin-top: var(--space-2);
  }
  @media (max-width: 767px) {
    /* Phones: the other fixes as full-width secondary buttons, one per row (each names its setting and its forecast,
       "Retry at 640 px, batch 16 · ~12.1 GiB", too long to share a row), in the one phone button size (.cta-row) of the
       fix above them. */
    .alts {
      flex-direction: column;
      align-items: stretch;
    }
    .alts > :global(.btn.btn) {
      min-width: 0;
    }
  }
</style>
