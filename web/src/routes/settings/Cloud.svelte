<script lang="ts">
  /**
   * Settings › Cloud (docs/CLOUD.md): the cloud connections, each with its account, balance and the result of its last
   * check, and the instances FOLO launched with them, so nothing rented can hide. An instance whose termination the
   * provider has not confirmed is a problem, said above the list with what to look for in the provider's console.
   */
  import CircleAlert from '@lucide/svelte/icons/circle-alert';
  import CircleCheck from '@lucide/svelte/icons/circle-check';
  import Cloud from '@lucide/svelte/icons/cloud';
  import Plus from '@lucide/svelte/icons/plus';
  import Power from '@lucide/svelte/icons/power';
  import RefreshCw from '@lucide/svelte/icons/refresh-cw';
  import TriangleAlert from '@lucide/svelte/icons/triangle-alert';
  import Unplug from '@lucide/svelte/icons/unplug';
  import { hub } from '$lib/api/index.svelte';
  import { Resource } from '$lib/api/resource.svelte';
  import type { CloudConnection, CloudInstance } from '$lib/api/types';
  import { INSTANCE_LABEL, isAlive, offerParts, providerName } from '$lib/cloud';
  import { ago, timeOfDay, usd } from '$lib/format';
  import { prefs } from '$lib/prefs.svelte';
  import { href } from '$lib/router.svelte';
  import Button from '$lib/ui/Button.svelte';
  import Dialog from '$lib/ui/Dialog.svelte';
  import EmptyState from '$lib/ui/EmptyState.svelte';
  import ErrorState from '$lib/ui/ErrorState.svelte';
  import { DialogStatus } from '$lib/ui/feedback.svelte';
  import Ident from '$lib/ui/Ident.svelte';
  import Meta from '$lib/ui/Meta.svelte';
  import NoValue from '$lib/ui/NoValue.svelte';
  import Panel from '$lib/ui/Panel.svelte';
  import Skeleton from '$lib/ui/Skeleton.svelte';
  import { toast } from '$lib/ui/toast.svelte';
  import Callout from '../run/Callout.svelte';
  import ConnectDialog from './ConnectDialog.svelte';

  const providers = new Resource((s) => hub.api.cloudProviders(s));
  const connections = new Resource((s) => hub.api.cloudConnections(s));
  const instances = new Resource((s) => hub.api.cloudInstances(s));
  const runs = new Resource((s) => hub.api.runs(s));

  /** While an instance is alive (it costs money), its state and cost are read again every 5 s. */
  const anyAlive = $derived(!!instances.data?.some((i) => isAlive(i.status)));
  $effect(() => {
    if (!anyAlive) return;
    const t = setInterval(() => {
      if (document.visibilityState === 'visible') void instances.refresh();
    }, 5000);
    return () => clearInterval(t);
  });

  const runName = (id: string | null) => (id ? runs.data?.find((r) => r.id === id)?.name : undefined);
  const provName = (id: string) => providerName(id, providers.data);
  /** Instances shown at all: there are some, or a connection launches them (a remote server never does). */
  const provisioning = $derived(!!connections.data?.some((c) => providers.data?.find((p) => p.id === c.provider)?.provisions));
  const showInstances = $derived(!!instances.data?.length || provisioning || (!!instances.error && !!connections.data?.length));
  /** Terminations the provider has not confirmed: each one may still be billed. */
  const unconfirmed = $derived(instances.data?.filter((i) => i.status === 'terminating' && i.error) ?? []);

  // ---- connect ----
  let connecting = $state(false);
  function connected(c: CloudConnection) {
    connections.set([...(connections.data ?? []), c]);
    void instances.refresh();
  }

  // ---- check ----
  let checking = $state<string | null>(null);
  async function check(c: CloudConnection) {
    checking = c.id;
    try {
      const next = await hub.api.checkCloudConnection(c.id);
      connections.set((connections.data ?? []).map((x) => (x.id === next.id ? next : x)));
    } catch (e) {
      // page-level: Check is a row action, no dialog is open.
      toast.error(e, `Could not check ${c.name}`);
    } finally {
      checking = null;
    }
  }

  // ---- disconnect ----
  let leaving = $state<CloudConnection | null>(null);
  let leaveOpen = $state(false);
  let leaveBusy = $state(false);
  const leaveStatus = new DialogStatus();
  /** What disconnecting forgets: the provider's secret fields by name ("the API key", "the token"). */
  const secrets = $derived.by(() => {
    const p = leaving ? providers.data?.find((x) => x.id === leaving?.provider) : undefined;
    const names = p?.fields.filter((f) => f.secret).map((f) => (/^[A-Z]{2}/.test(f.label) ? f.label : f.label.toLowerCase())) ?? [];
    return names.length ? `the ${names.join(' and the ')}` : 'its credentials';
  });
  function askDisconnect(c: CloudConnection) {
    leaving = c;
    leaveOpen = true;
  }
  async function disconnect() {
    const c = leaving;
    if (!c) return;
    leaveBusy = true;
    try {
      await hub.api.disconnectCloud(c.id);
      leaveOpen = false;
      toast.success(`Disconnected ${c.name}`);
      connections.set((connections.data ?? []).filter((x) => x.id !== c.id));
    } catch (e) {
      // `409 connection_in_use` says which instance or run holds it, in the dialog.
      leaveStatus.error(e, `Could not disconnect ${c.name}`);
    } finally {
      leaveBusy = false;
    }
  }

  // ---- terminate ----
  let ending = $state<CloudInstance | null>(null);
  let endOpen = $state(false);
  let endBusy = $state(false);
  const endStatus = new DialogStatus();
  const endText = $derived.by(() => {
    const i = ending;
    if (!i) return '';
    const run = runName(i.run_id);
    const own = `The ${i.offer.gpu} at ${provName(i.provider)} is shut down now.`;
    return run ? `${own} Its run ${run} fails; the checkpoints already copied to this machine stay.` : own;
  });
  function askTerminate(i: CloudInstance) {
    ending = i;
    endOpen = true;
    // The safe choice takes focus: a terminated instance cannot be had back.
    requestAnimationFrame(() => document.querySelector<HTMLElement>('dialog[open] [data-safe]')?.focus());
  }
  async function terminate() {
    const i = ending;
    if (!i) return;
    endBusy = true;
    try {
      const next = await hub.api.terminateCloudInstance(i.id);
      endOpen = false;
      toast.info(next.status === 'terminated' ? `Terminated the ${i.offer.gpu}` : `Terminating the ${i.offer.gpu}`, runName(i.run_id) ? `${runName(i.run_id)} fails; its copied checkpoints stay.` : undefined);
      instances.set((instances.data ?? []).map((x) => (x.id === next.id ? next : x)));
      void runs.refresh();
    } catch (e) {
      endStatus.error(e, 'Could not terminate the instance');
    } finally {
      endBusy = false;
    }
  }
</script>

<Panel title="Cloud" id="cloud" divided pad="none">
  {#snippet actions()}
    {#if connections.data?.length}<Button size="sm" icon={Plus} disabled={!providers.data} onclick={() => (connecting = true)}>Connect</Button>{/if}
  {/snippet}
  {#if connections.error && !connections.data}
    <div class="state"><ErrorState error={connections.error} title="Could not load the connections" retry={() => connections.refresh()} compact /></div>
  {:else if !connections.data}
    <ul class="cx" aria-hidden="true">
      {#each [0, 1] as i (i)}
        <li class="cx-row">
          <span class="cx-main"><Skeleton w="96px" h="14px" /><Skeleton w="220px" h="12px" /></span>
          <span class="cx-end"><Skeleton w="72px" h="12px" /><Skeleton w="128px" h="24px" r="var(--radius-md)" /></span>
        </li>
      {/each}
    </ul>
  {:else if !connections.data.length}
    <!-- The empty state holds the one way to connect; the header shows it once there is a list. -->
    <EmptyState icon={Cloud} title="No cloud connected" description="Connect RunPod to train on a rented GPU, or another FOLO server of yours.">
      {#snippet actions()}<Button icon={Plus} disabled={!providers.data} onclick={() => (connecting = true)}>Connect</Button>{/snippet}
    </EmptyState>
  {:else}
    <ul class="cx" aria-label="Cloud connections">
      {#each connections.data as c (c.id)}
        {@const bad = c.status === 'error'}
        <li class="cx-row">
          <div class="cx-main">
            <p class="cx-name">{c.name}</p>
            <p class="cx-meta num">
              <Meta>
                {#if c.name !== provName(c.provider)}<span>{provName(c.provider)}</span>{/if}
                {#if c.account}<span>{c.account}</span>{/if}
                {#if c.balance_usd !== null}<span>Balance {usd(c.balance_usd)}</span>{/if}
                {#if c.checked}<span>checked {ago(c.checked)}</span>{/if}
              </Meta>
            </p>
            {#if bad}
              <p class="cx-error"><CircleAlert size={14} strokeWidth={1.75} aria-hidden="true" /><span>{c.error ?? 'The last check failed.'}</span></p>
            {:else if c.note}
              <p class="cx-note"><TriangleAlert size={14} strokeWidth={1.75} aria-hidden="true" /><span>{c.note}</span></p>
            {/if}
          </div>
          <div class="cx-end">
            <span class="cx-state" data-state={c.status}>
              {#if bad}<CircleAlert size={14} strokeWidth={1.75} aria-hidden="true" />Check failed{:else}<CircleCheck size={14} strokeWidth={1.75} aria-hidden="true" />Connected{/if}
            </span>
            <span class="cx-acts">
              <Button size="sm" icon={RefreshCw} loading={checking === c.id} onclick={() => check(c)}>{bad ? 'Check again' : 'Check'}</Button>
              <Button size="sm" variant="danger" icon={Unplug} onclick={() => askDisconnect(c)}>Disconnect</Button>
            </span>
          </div>
        </li>
      {/each}
    </ul>
  {/if}
</Panel>

{#if showInstances}
  <Panel title="Instances" id="cloud-instances" meta="last 7 days" divided pad="none">
    {#if unconfirmed.length}
      <div class="warn">
        {#each unconfirmed as i (i.id)}
          <Callout tone="warning" icon={TriangleAlert} title="Termination not confirmed" role="alert">
            FOLO could not confirm that <Ident value={i.name} /> is terminated: {i.error}. It keeps trying. Check the provider’s console{#if i.provider_ref}{' '}for <Ident value={i.provider_ref} />{/if}; the instance may still be billed.
          </Callout>
        {/each}
      </div>
    {/if}
    {#if instances.error && !instances.data}
      <div class="state"><ErrorState error={instances.error} title="Could not load the instances" retry={() => instances.refresh()} compact /></div>
    {:else if instances.data && !instances.data.length}
      <p class="none">No instances. FOLO starts one per cloud run and terminates it when the run ends.</p>
    {:else}
      <table class="table tall inst" aria-label="Cloud instances" aria-busy={!instances.data}>
        <thead>
          <tr>
            <th scope="col">Instance</th>
            <th scope="col">Run</th>
            <th scope="col">Status</th>
            <th scope="col" class="r" style:width="80px">Cost so far</th>
            <th scope="col" class="act"><span class="sr-only">Actions</span></th>
          </tr>
        </thead>
        <tbody>
          {#if !instances.data}
            {#each [0, 1] as i (i)}
              <tr aria-hidden="true">
                <td><Skeleton w="200px" h="12px" /><Skeleton w="140px" h="10px" /></td>
                <td><Skeleton w="88px" h="12px" /></td>
                <td><Skeleton w="72px" h="12px" /></td>
                <td class="r"><Skeleton w="40px" h="12px" /></td>
                <td></td>
              </tr>
            {/each}
          {:else}
            {#each instances.data as i (i.id)}
              {@const alive = isAlive(i.status)}
              {@const run = runName(i.run_id)}
              <tr class:ended={!alive}>
                <td>
                  <span class="gpu"><Meta>{#each offerParts(i.offer, prefs.units) as p (p)}<span>{p}</span>{/each}</Meta></span>
                  <span class="sub num"><Meta><span><Ident value={i.name} /></span><span>started {ago(i.created)}</span></Meta></span>
                </td>
                <td class="run">{#if i.run_id}<a href={href(`/runs/${i.run_id}`)}>{run ?? i.run_id}</a>{:else}<NoValue label="no run" />{/if}</td>
                <td>
                  <span class="st" data-status={i.status}><span class="dot" aria-hidden="true"></span>{INSTANCE_LABEL[i.status]}</span>
                  {#if alive && i.status !== 'terminating' && i.deadline}<span class="sub num">stops by {timeOfDay(i.deadline)}</span>{/if}
                </td>
                <td class="r num">{i.cost_usd === null ? 'not billed' : usd(i.cost_usd)}</td>
                <td class="act">
                  {#if alive}<Button size="sm" variant="danger" icon={Power} onclick={() => askTerminate(i)}>Terminate</Button>{/if}
                </td>
              </tr>
            {/each}
          {/if}
        </tbody>
      </table>
      <!-- Phones: a row each, the GPU over its state, cost and run, Terminate on the right. -->
      {#if instances.data}
        <ul class="m-rows" aria-label="Cloud instances">
          {#each instances.data as i (i.id)}
            {@const alive = isAlive(i.status)}
            <li class="m-row" class:ended={!alive}>
              <div class="m-main">
                <span class="gpu"><Meta>{#each offerParts(i.offer, prefs.units) as p (p)}<span>{p}</span>{/each}</Meta></span>
                <span class="sub num">
                  <Meta>
                    <span class="st" data-status={i.status}><span class="dot" aria-hidden="true"></span>{INSTANCE_LABEL[i.status]}</span>
                    <span>{i.cost_usd === null ? 'not billed' : usd(i.cost_usd)}</span>
                    {#if alive && i.status !== 'terminating' && i.deadline}<span>stops by {timeOfDay(i.deadline)}</span>{/if}
                    {#if i.run_id}<span><a href={href(`/runs/${i.run_id}`)}>{runName(i.run_id) ?? i.run_id}</a></span>{/if}
                  </Meta>
                </span>
              </div>
              {#if alive}<Button size="sm" variant="danger" icon={Power} onclick={() => askTerminate(i)}>Terminate</Button>{/if}
            </li>
          {/each}
        </ul>
      {/if}
    {/if}
  </Panel>
{/if}

{#if providers.data}
  <ConnectDialog bind:open={connecting} providers={providers.data} onconnected={connected} />
{/if}

<Dialog
  bind:open={leaveOpen}
  title="Disconnect {leaving?.name ?? ''}?"
  description="FOLO forgets {secrets}. Runs and models stay."
  size="sm"
  status={leaveStatus}
>
  {#snippet footer()}
    <Button onclick={() => (leaveOpen = false)}>Cancel</Button>
    <Button variant="destructive" icon={Unplug} loading={leaveBusy} onclick={disconnect}>Disconnect</Button>
  {/snippet}
</Dialog>

<Dialog bind:open={endOpen} title="Terminate this instance?" description={endText} size="sm" status={endStatus}>
  {#snippet footer()}
    <Button data-safe onclick={() => (endOpen = false)}>Keep it running</Button>
    <Button variant="destructive" icon={Power} loading={endBusy} onclick={terminate}>Terminate</Button>
  {/snippet}
</Dialog>

<style>
  .state {
    padding: 0 var(--space-4);
  }
  /* A connection: its name over its account line (and a failed check's reason), its state and actions on the right. */
  .cx {
    display: grid;
  }
  .cx-row {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: var(--space-4);
    padding: var(--space-3) var(--space-4);
  }
  .cx-row + .cx-row {
    border-top: 1px solid var(--line-subtle);
  }
  .cx-main {
    display: grid;
    gap: 2px;
    min-width: 0;
  }
  .cx-name {
    font-weight: var(--weight-medium);
  }
  .cx-meta {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
  }
  /* The failed check's reason in the danger ink with its mark (never red text alone), at the measure. */
  .cx-error,
  .cx-note {
    display: flex;
    align-items: flex-start;
    gap: var(--space-1-5);
    max-inline-size: var(--measure);
    margin-top: var(--space-1);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    text-wrap: pretty;
  }
  .cx-error {
    color: var(--danger-text);
  }
  /* What a working connection still has against it (an unencrypted URL, a worker that cannot be installed): the
     warning mark, the words in secondary ink (it is not a failure, and never coloured text alone). */
  .cx-note {
    color: var(--text-secondary);
    overflow-wrap: anywhere;
  }
  .cx-note :global(svg) {
    color: var(--warning-text);
  }
  .cx-error :global(svg),
  .cx-note :global(svg) {
    flex-shrink: 0;
    margin-top: 1px;
  }
  .cx-end {
    flex-shrink: 0;
    display: flex;
    align-items: center;
    gap: var(--space-4);
  }
  .cx-acts {
    display: flex;
    gap: var(--space-2);
  }
  /* The state as its mark and a word, the mark in its hue (as Settings › System's capabilities). */
  .cx-state {
    display: inline-flex;
    align-items: center;
    gap: var(--space-1-5);
    font-size: var(--type-sm-size);
    color: var(--text-secondary);
    white-space: nowrap;
  }
  .cx-state[data-state='ok'] :global(svg) {
    color: var(--success-text);
  }
  .cx-state[data-state='error'] :global(svg) {
    color: var(--danger-text);
  }
  .warn {
    display: grid;
    gap: var(--space-2);
    padding: var(--space-3) var(--space-4) 0;
  }
  .none {
    padding: var(--space-3) var(--space-4) var(--space-4);
    max-inline-size: calc(var(--measure) + 2 * var(--space-4));
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-tertiary);
  }
  .inst .gpu {
    display: block;
  }
  /* The instance is the flexible column: where the form is narrow (laptops) its GPU line wraps between words, and its
     meta line between items, instead of pushing Terminate out of the panel. */
  .inst td:first-child {
    white-space: normal;
  }
  .sub {
    display: block;
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    font-weight: var(--weight-regular);
    color: var(--text-tertiary);
  }
  .inst td {
    vertical-align: top;
    padding-top: var(--space-2);
    padding-bottom: var(--space-2);
  }
  .inst td.act {
    vertical-align: middle;
    text-align: right;
  }
  .run a {
    color: var(--text-primary);
  }
  /* Ended instances are quiet: the record stays, nothing in it asks for attention. */
  .ended,
  .ended .gpu,
  .ended .run a {
    color: var(--text-tertiary);
  }
  .ended .gpu {
    font-weight: var(--weight-regular);
  }
  /* Status: a dot and its word; a hue only where the state means something (live, terminating, lost). */
  .st {
    display: inline-flex;
    align-items: center;
    gap: var(--space-1-5);
    white-space: nowrap;
  }
  .st .dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--text-tertiary);
  }
  .st[data-status='running'] .dot {
    background: var(--accent-line);
  }
  .st[data-status='terminating'] .dot {
    background: var(--warning-solid);
  }
  .st[data-status='lost'] .dot {
    background: var(--danger-solid);
  }
  .m-rows {
    display: none;
  }
  @media (max-width: 767px) {
    .cx-row {
      flex-direction: column;
      align-items: stretch;
      gap: var(--space-3);
    }
    /* The state, then the actions; at the narrowest phones the actions take a line of their own. */
    .cx-end {
      flex-wrap: wrap;
      justify-content: space-between;
      row-gap: var(--space-2);
    }
    .inst {
      display: none;
    }
    .m-rows {
      display: block;
    }
    .m-row {
      display: flex;
      align-items: center;
      gap: var(--space-3);
      min-height: 56px;
      padding: var(--space-2) var(--space-4);
    }
    .m-row + .m-row {
      border-top: 1px solid var(--line-subtle);
    }
    .m-main {
      flex: 1;
      min-width: 0;
      display: grid;
    }
    .m-row .gpu {
      font-weight: var(--weight-medium);
    }
    .m-row.ended .gpu {
      font-weight: var(--weight-regular);
    }
    .m-row a {
      color: inherit;
      text-decoration: underline;
      text-decoration-color: var(--line-strong);
      text-underline-offset: 3px;
    }
  }
</style>
