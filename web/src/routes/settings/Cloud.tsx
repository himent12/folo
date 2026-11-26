/**
 * Settings › Cloud (docs/CLOUD.md): the cloud connections, each with its account, balance and the result of its last
 * check, and the instances FOLO launched with them, so nothing rented can hide. An instance whose termination the
 * provider has not confirmed is a problem, said above the list with what to look for in the provider's console.
 */
import CircleAlert from '$icons/circle-alert';
import CircleCheck from '$icons/circle-check';
import CloudIcon from '$icons/cloud';
import Plus from '$icons/plus';
import Power from '$icons/power';
import RefreshCw from '$icons/refresh-cw';
import TriangleAlert from '$icons/triangle-alert';
import Unplug from '$icons/unplug';
import { computed, css, effect, For, Index, onCleanup, ReactiveSet, signal, untrack } from 'fw';
import { hub } from '$lib/api/index';
import { Resource } from '$lib/api/resource';
import type { CloudConnection, CloudInstance } from '$lib/api/types';
import { INSTANCE_LABEL, isAlive, offerParts, providerName } from '$lib/cloud';
import { ago, timeOfDay, usd } from '$lib/format';
import { prefs } from '$lib/prefs';
import { href } from '$lib/router';
import Button from '$lib/ui/Button';
import Dialog from '$lib/ui/Dialog';
import EmptyState from '$lib/ui/EmptyState';
import ErrorState from '$lib/ui/ErrorState';
import { DialogStatus } from '$lib/ui/feedback';
import Ident from '$lib/ui/Ident';
import Meta from '$lib/ui/Meta';
import NoValue from '$lib/ui/NoValue';
import Panel from '$lib/ui/Panel';
import Skeleton from '$lib/ui/Skeleton';
import { toast } from '$lib/ui/toast';
import Callout from '../run/Callout';
import ConnectDialog from './ConnectDialog';

/** Two placeholder rows while a list loads. */
const TWO = [0, 1];

export default function Cloud() {
  const providers = new Resource((s) => hub.api.cloudProviders(s));
  const connections = new Resource((s) => hub.api.cloudConnections(s));
  const instances = new Resource((s) => hub.api.cloudInstances(s));
  const runs = new Resource((s) => hub.api.runs(s));

  /** While an instance is alive (it costs money), its state and cost are read again every 5 s. */
  const anyAlive = computed(() => !!instances.data?.some((i) => isAlive(i.status)));
  effect(() => {
    if (!anyAlive.get()) return;
    const t = setInterval(() => {
      if (document.visibilityState === 'visible') void instances.refresh();
    }, 5000);
    return () => clearInterval(t);
  });

  const runName = (id: string | null) => (id ? runs.data?.find((r) => r.id === id)?.name : undefined);
  const provName = (id: string) => providerName(id, providers.data);
  /** Instances shown at all: there are some, or a connection launches them (a remote server never does). */
  const provisioning = computed(() => !!connections.data?.some((c) => providers.data?.find((p) => p.id === c.provider)?.provisions));
  const showInstances = computed(() => !!instances.data?.length || provisioning.get() || (!!instances.error && !!connections.data?.length));
  /** Terminations the provider has not confirmed: each one may still be billed. */
  const unconfirmed = computed(() => instances.data?.filter((i) => i.status === 'terminating' && i.error) ?? []);

  let alive = true;
  /** Bumps whenever a new connection attempt starts (`hub.mode === 'connecting'`) or `hub.epoch` changes. */
  let serverGen = 0;
  onCleanup(() => {
    alive = false;
    serverGen++;
  });
  const reconnecting = computed(() => hub.mode === 'connecting');
  /** Live check (avoids TS narrowing `hub.mode` across `await`). */
  const isConnecting = () => hub.mode === 'connecting';

  // ---- connect ----
  const connecting = signal(false);
  function connected(c: CloudConnection) {
    const cur = connections.data ?? [];
    if (!cur.some((x) => x.id === c.id)) connections.set([...cur, c]);
    void instances.refresh();
  }

  // ---- check ----
  const checking = new ReactiveSet<string>();
  async function check(c: CloudConnection) {
    if (checking.has(c.id) || isConnecting()) return;
    const gen = serverGen;
    const epoch = hub.epoch;
    checking.add(c.id);
    try {
      const next = await hub.api.checkCloudConnection(c.id);
      if (!alive || gen !== serverGen || isConnecting() || hub.epoch !== epoch) return;
      connections.set((connections.data ?? []).map((x) => (x.id === next.id ? next : x)));
    } catch (e) {
      if (!alive || gen !== serverGen || isConnecting() || hub.epoch !== epoch) return;
      // page-level: Check is a row action, no dialog is open.
      toast.error(e, `Could not check ${c.name}`);
    } finally {
      if (alive && gen === serverGen && hub.epoch === epoch) checking.delete(c.id);
    }
  }

  // ---- disconnect ----
  const leaving = signal<CloudConnection | null>(null);
  const leaveOpen = signal(false);
  const leaveBusy = signal(false);
  const leaveStatus = new DialogStatus();
  let leaveSeq = 0;
  effect(() => {
    if (!leaveOpen.get()) {
      leaveSeq++;
      leaveBusy.set(false);
      leaveStatus.clear();
    }
  });
  /** What disconnecting forgets: the provider's secret fields by name ("the API key", "the token"). */
  const secrets = computed(() => {
    const l = leaving.get();
    const p = l ? providers.data?.find((x) => x.id === l.provider) : undefined;
    const names = p?.fields.filter((f) => f.secret).map((f) => (/^[A-Z]{2}/.test(f.label) ? f.label : f.label.toLowerCase())) ?? [];
    return names.length ? `the ${names.join(' and the ')}` : 'its credentials';
  });
  function askDisconnect(c: CloudConnection) {
    if (hub.mode === 'connecting') return;
    leaveSeq++;
    leaveBusy.set(false);
    leaveStatus.clear();
    leaving.set(c);
    leaveOpen.set(true);
  }
  async function disconnect() {
    const c = leaving.peek();
    if (!c || leaveBusy.peek() || isConnecting()) return;
    const seq = ++leaveSeq;
    const gen = serverGen;
    const epoch = hub.epoch;
    leaveBusy.set(true);
    try {
      await hub.api.disconnectCloud(c.id);
      if (!alive || gen !== serverGen || isConnecting() || hub.epoch !== epoch) return;
      if (seq === leaveSeq || leaving.peek()?.id === c.id) leaveOpen.set(false);
      toast.success(`Disconnected ${c.name}`);
      connections.set((connections.data ?? []).filter((x) => x.id !== c.id));
    } catch (e) {
      if (!alive || seq !== leaveSeq || gen !== serverGen || isConnecting() || hub.epoch !== epoch) return;
      // `409 connection_in_use` says which instance or run holds it, in the dialog.
      leaveStatus.error(e, `Could not disconnect ${c.name}`);
    } finally {
      if (alive && seq === leaveSeq) leaveBusy.set(false);
    }
  }

  // ---- terminate ----
  const ending = signal<CloudInstance | null>(null);
  const endOpen = signal(false);
  const endBusy = signal(false);
  const endStatus = new DialogStatus();
  let endSeq = 0;
  effect(() => {
    if (!endOpen.get()) {
      endSeq++;
      endBusy.set(false);
      endStatus.clear();
    }
  });
  let prevEpoch = untrack(() => hub.epoch);
  let prevConnecting = untrack(() => hub.mode === 'connecting');
  effect(() => {
    const ep = hub.epoch;
    const isConnecting = hub.mode === 'connecting';
    if (ep === prevEpoch && isConnecting === prevConnecting) return;
    prevEpoch = ep;
    prevConnecting = isConnecting;
    serverGen++;
    leaveSeq++;
    endSeq++;
    checking.clear();
    connecting.set(false);
    leaveBusy.set(false);
    endBusy.set(false);
    leaveOpen.set(false);
    endOpen.set(false);
    leaveStatus.clear();
    endStatus.clear();
  });
  const endText = computed(() => {
    const i = ending.get();
    if (!i) return '';
    const run = runName(i.run_id);
    const own = `The ${i.offer.gpu} at ${provName(i.provider)} is shut down now.`;
    return run ? `${own} Its run ${run} fails; the checkpoints already copied to this machine stay.` : own;
  });
  function askTerminate(i: CloudInstance) {
    if (hub.mode === 'connecting') return;
    endSeq++;
    endBusy.set(false);
    endStatus.clear();
    ending.set(i);
    endOpen.set(true);
    // The safe choice takes focus: a terminated instance cannot be had back.
    requestAnimationFrame(() => document.querySelector<HTMLElement>('dialog[open] [data-safe]')?.focus());
  }
  async function terminate() {
    const i = ending.peek();
    if (!i || endBusy.peek() || isConnecting()) return;
    const seq = ++endSeq;
    const gen = serverGen;
    const epoch = hub.epoch;
    endBusy.set(true);
    try {
      const next = await hub.api.terminateCloudInstance(i.id);
      if (!alive || gen !== serverGen || isConnecting() || hub.epoch !== epoch) return;
      if (seq === endSeq) endOpen.set(false);
      toast.info(next.status === 'terminated' ? `Terminated the ${i.offer.gpu}` : `Terminating the ${i.offer.gpu}`, runName(i.run_id) ? `${runName(i.run_id)} fails; its copied checkpoints stay.` : undefined);
      instances.set((instances.data ?? []).map((x) => (x.id === next.id ? next : x)));
      void runs.refresh();
    } catch (e) {
      if (!alive || seq !== endSeq || gen !== serverGen || isConnecting() || hub.epoch !== epoch) return;
      endStatus.error(e, 'Could not terminate the instance');
    } finally {
      if (alive && seq === endSeq) endBusy.set(false);
    }
  }

  const status = (i: () => CloudInstance) => (
    <span class="st" data-status={i().status}>
      <span class="dot" aria-hidden="true"></span>
      {INSTANCE_LABEL[i().status]}
    </span>
  );
  const gpu = (i: () => CloudInstance) => (
    <span class="gpu">
      <Meta>
        <For each={offerParts(i().offer, prefs.units)} key={(p) => p}>
          {(p) => <span>{p()}</span>}
        </For>
      </Meta>
    </span>
  );
  const cost = (i: CloudInstance) => (i.cost_usd === null ? 'not billed' : usd(i.cost_usd));
  const stopsBy = (i: CloudInstance) => isAlive(i.status) && i.status !== 'terminating' && !!i.deadline;

  return (
    <>
      <Panel
        title="Cloud"
        id="cloud"
        divided
        pad="none"
        actions={() => (
          <>
            {connections.data?.length ? (
              <Button size="sm" icon={Plus} disabled={!providers.data || reconnecting.get()} onclick={() => connecting.set(true)}>
                Connect
              </Button>
            ) : null}
          </>
        )}
      >
        {/* Connect waits for the providers: when they fail, say so (with Retry) rather than leave it disabled unexplained. */}
        {providers.error && !providers.data ? (
          <div class="state">
            <ErrorState error={providers.error} title="Could not load the cloud providers" retry={() => providers.refresh()} compact />
          </div>
        ) : null}
        {connections.error && !connections.data ? (
          <div class="state">
            <ErrorState error={connections.error} title="Could not load the connections" retry={() => connections.refresh()} compact />
          </div>
        ) : !connections.data ? (
          <ul class="cx" aria-hidden="true">
            <For each={TWO} key={(i) => i}>
              {() => (
                <li class="cx-row">
                  <span class="cx-main">
                    <Skeleton w="96px" h="14px" />
                    <Skeleton w="220px" h="12px" />
                  </span>
                  <span class="cx-end">
                    <Skeleton w="72px" h="12px" />
                    <Skeleton w="128px" h="24px" r="var(--radius-md)" />
                  </span>
                </li>
              )}
            </For>
          </ul>
        ) : !connections.data.length ? (
          /* The empty state holds the one way to connect; the header shows it once there is a list. */
          <EmptyState
            icon={CloudIcon}
            title="No cloud connected"
            description="Connect RunPod to train on a rented GPU, or another FOLO server of yours."
            actions={() => (
              <Button icon={Plus} disabled={!providers.data || reconnecting.get()} onclick={() => connecting.set(true)}>
                Connect
              </Button>
            )}
          />
        ) : (
          <ul class="cx" aria-label="Cloud connections">
            <For each={connections.data ?? []} key={(c) => c.id}>
              {(c) => {
                const bad = () => c().status === 'error';
                return (
                  <li class="cx-row">
                    <div class="cx-main">
                      <p class="cx-name">{c().name}</p>
                      <p class="cx-meta num">
                        <Meta>
                          {c().name !== provName(c().provider) ? <span>{provName(c().provider)}</span> : null}
                          {c().account ? <span>{c().account}</span> : null}
                          {c().balance_usd !== null ? <span>Balance {usd(c().balance_usd!)}</span> : null}
                          {c().checked ? <span>checked {ago(c().checked!)}</span> : null}
                        </Meta>
                      </p>
                      {bad() ? (
                        <p class="cx-error">
                          <CircleAlert size={14} strokeWidth={1.75} aria-hidden="true" />
                          <span>{c().error ?? 'The last check failed.'}</span>
                        </p>
                      ) : c().note ? (
                        <p class="cx-note">
                          <TriangleAlert size={14} strokeWidth={1.75} aria-hidden="true" />
                          <span>{c().note}</span>
                        </p>
                      ) : null}
                    </div>
                    <div class="cx-end">
                      <span class="cx-state" data-state={c().status}>
                        {bad() ? (
                          <>
                            <CircleAlert size={14} strokeWidth={1.75} aria-hidden="true" />
                            Check failed
                          </>
                        ) : (
                          <>
                            <CircleCheck size={14} strokeWidth={1.75} aria-hidden="true" />
                            Connected
                          </>
                        )}
                      </span>
                      <span class="cx-acts">
                        <Button size="sm" icon={RefreshCw} loading={checking.has(c().id)} disabled={reconnecting.get()} onclick={() => check(c())}>
                          {bad() ? 'Check again' : 'Check'}
                        </Button>
                        <Button size="sm" variant="danger" icon={Unplug} disabled={reconnecting.get()} onclick={() => askDisconnect(c())}>
                          Disconnect
                        </Button>
                      </span>
                    </div>
                  </li>
                );
              }}
            </For>
          </ul>
        )}
      </Panel>

      {showInstances.get() ? (
        <Panel title="Instances" id="cloud-instances" meta="last 7 days" divided pad="none">
          {unconfirmed.get().length ? (
            <div class="warn">
              <For each={unconfirmed.get()} key={(i) => i.id}>
                {(i) => (
                  <Callout tone="warning" icon={TriangleAlert} title="Termination not confirmed" role="alert">
                    FOLO could not confirm that <Ident value={i().name} /> is terminated: {i().error}. It keeps trying. Check the provider’s console
                    {i().provider_ref ? (
                      <>
                        {' '}for <Ident value={i().provider_ref!} />
                      </>
                    ) : null}
                    ; the instance may still be billed.
                  </Callout>
                )}
              </For>
            </div>
          ) : null}
          {instances.error && !instances.data ? (
            <div class="state">
              <ErrorState error={instances.error} title="Could not load the instances" retry={() => instances.refresh()} compact />
            </div>
          ) : instances.data && !instances.data.length ? (
            <p class="none">No instances. FOLO starts one per cloud run and terminates it when the run ends.</p>
          ) : (
            <>
              <table class="table tall inst" aria-label="Cloud instances" aria-busy={!instances.data && !instances.error}>
                <thead>
                  <tr>
                    <th scope="col">Instance</th>
                    <th scope="col">Run</th>
                    <th scope="col">Status</th>
                    <th scope="col" class="r" style:width="80px">
                      Cost so far
                    </th>
                    <th scope="col" class="act">
                      <span class="sr-only">Actions</span>
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {!instances.data ? (
                    <For each={TWO} key={(i) => i}>
                      {() => (
                        <tr aria-hidden="true">
                          <td>
                            <Skeleton w="200px" h="12px" />
                            <Skeleton w="140px" h="10px" />
                          </td>
                          <td>
                            <Skeleton w="88px" h="12px" />
                          </td>
                          <td>
                            <Skeleton w="72px" h="12px" />
                          </td>
                          <td class="r">
                            <Skeleton w="40px" h="12px" />
                          </td>
                          <td></td>
                        </tr>
                      )}
                    </For>
                  ) : (
                    <For each={instances.data ?? []} key={(i) => i.id}>
                      {(i) => {
                        const alive = () => isAlive(i().status);
                        const run = () => runName(i().run_id);
                        return (
                          <tr class:ended={!alive()}>
                            <td>
                              {gpu(i)}
                              <span class="sub num">
                                <Meta>
                                  <span>
                                    <Ident value={i().name} />
                                  </span>
                                  <span>started {ago(i().created)}</span>
                                </Meta>
                              </span>
                            </td>
                            <td class="run">
                              {i().run_id ? <a href={href(`/runs/${i().run_id}`)}>{run() ?? i().run_id}</a> : <NoValue label="no run" />}
                            </td>
                            <td>
                              {status(i)}
                              {stopsBy(i()) ? <span class="sub num">stops by {timeOfDay(i().deadline!)}</span> : null}
                            </td>
                            <td class="r num">{cost(i())}</td>
                            <td class="act">
                              {alive() ? (
                                <Button size="sm" variant="danger" icon={Power} disabled={reconnecting.get()} onclick={() => askTerminate(i())}>
                                  Terminate
                                </Button>
                              ) : null}
                            </td>
                          </tr>
                        );
                      }}
                    </For>
                  )}
                </tbody>
              </table>
              {/* Phones: a row each, the GPU over its state, cost and run, Terminate on the right. */}
              {instances.data ? (
                <ul class="m-rows" aria-label="Cloud instances">
                  <For each={instances.data ?? []} key={(i) => i.id}>
                    {(i) => {
                      const alive = () => isAlive(i().status);
                      return (
                        <li class="m-row" class:ended={!alive()}>
                          <div class="m-main">
                            {gpu(i)}
                            <span class="sub num">
                              <Meta>
                                {status(i)}
                                <span>{cost(i())}</span>
                                {stopsBy(i()) ? <span>stops by {timeOfDay(i().deadline!)}</span> : null}
                                {i().run_id ? (
                                  <span>
                                    <a href={href(`/runs/${i().run_id}`)}>{runName(i().run_id) ?? i().run_id}</a>
                                  </span>
                                ) : null}
                              </Meta>
                            </span>
                          </div>
                          {alive() ? (
                            <Button size="sm" variant="danger" icon={Power} disabled={reconnecting.get()} onclick={() => askTerminate(i())}>
                              Terminate
                            </Button>
                          ) : null}
                        </li>
                      );
                    }}
                  </For>
                </ul>
              ) : null}
            </>
          )}
        </Panel>
      ) : null}

      {providers.data ? <ConnectDialog bind:open={connecting} providers={providers.data ?? []} onconnected={connected} /> : null}

      <Dialog
        bind:open={leaveOpen}
        title={`Disconnect ${leaving.get()?.name ?? ''}?`}
        description={`FOLO forgets ${secrets.get()}. Runs and models stay.`}
        size="sm"
        status={leaveStatus}
        footer={() => (
          <>
            <Button onclick={() => leaveOpen.set(false)}>Cancel</Button>
            <Button variant="destructive" icon={Unplug} loading={leaveBusy.get()} disabled={reconnecting.get()} onclick={disconnect}>
              Disconnect
            </Button>
          </>
        )}
      />

      <Dialog
        bind:open={endOpen}
        title="Terminate this instance?"
        description={endText.get()}
        size="sm"
        status={endStatus}
        footer={() => (
          <>
            <Button data-safe onclick={() => endOpen.set(false)}>
              Keep it running
            </Button>
            <Button variant="destructive" icon={Power} loading={endBusy.get()} disabled={reconnecting.get()} onclick={terminate}>
              Terminate
            </Button>
          </>
        )}
      />
    </>
  );
}

css`
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
`;
