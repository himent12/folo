// A queued run's place in the training queue, as its page reads it (servers that list `queue_control`, docs/API.md):
// what trains ahead of it, the runs waiting in front, how long that should take (servers with `run_estimate`), whether
// Start next is a move, and the move itself (the new order at once, the old one back on a refusal).
import { computed, onCleanup, untrack } from 'fw';
import { hub } from '$lib/api/index';
import type { LiveRun } from '$lib/api/live';
import { settledEta } from '$lib/api/progress';
import { hasFeature, withQueueMove } from '$lib/api/queue';
import { Resource } from '$lib/api/resource';
import type { QueueRequest, Run, SystemInfo } from '$lib/api/types';
import { withoutInexactSeed } from '$lib/seed';
import { timeRange } from '$lib/throughput';
import { moveQueued, queueWait, type Length } from './queueActions';

export interface QueueSource {
  run: () => Run | undefined;
  live: LiveRun;
  /** Every run (polled): the queue and what trains. */
  others: Resource<Run[]>;
  system: Resource<SystemInfo>;
}

export function queueState(src: QueueSource) {
  const { run, live, others, system } = src;
  const status = () => run()?.status;
  const cloud = () => run()?.cloud;
  const ahead = computed(() => others.data?.find((r) => r.status === 'running' || r.status === 'stopping'));
  const queueCtl = computed(() => hasFeature(system.data, 'queue_control'));
  const place = computed(() => {
    const r = run();
    return r?.status === 'queued' && queueCtl.get() ? (r.queue_position ?? null) : null;
  });
  /** The runs waiting, front first (this one among them). */
  const queue = computed(() =>
    (others.data ?? []).filter((r) => r.status === 'queued' && r.queue_position != null).sort((a, b) => (a.queue_position ?? 0) - (b.queue_position ?? 0)),
  );
  const waitingCount = computed(() => Math.max(live.waiting ?? 0, queue.get().length, place.get() ?? 0));
  /** The waiting run just in front of this one. */
  const before = computed(() => {
    const p = place.get();
    return p && p > 1 ? queue.get().find((r) => r.id !== run()?.id && r.queue_position === p - 1) : undefined;
  });
  /** The waiting runs in front of this one, front first; keyed by their ids, so a poll that changes nothing asks nothing. */
  const inFront = computed(() => {
    const p = place.get();
    return p && p > 1 ? queue.get().filter((r) => r.id !== run()?.id && (r.queue_position ?? 0) < p) : [];
  });
  const frontKey = computed(() => inFront.get().map((r) => r.id).join(' '));
  // Their lengths, from the server's estimate of each one's settings (servers with `run_estimate`): what makes "starts
  // in about …" an estimate rather than a guess. One that cannot be estimated stays unknown, and the page says so. A
  // seed over 2^53 − 1 arrived rounded and is left out (lib/seed, rule B): it does not change a run's length.
  const frontLengths = new Resource(async (s) => {
    const key = frontKey.get();
    const rs = untrack(() => inFront.get());
    // No estimates on this server: every run in front stays unknown (one null each), never "nothing in front".
    if (!key || !hasFeature(system.data, 'run_estimate')) return { key, lengths: rs.map((): Length => null) };
    const lengths = await Promise.all(
      rs.map((r) =>
        hub.api.estimateRun(withoutInexactSeed({ ...r.config, name: r.name }), s).then(
          (e): Length => ({ lo: e.est_total_s_low, hi: e.est_total_s_high }),
          (): Length => null,
        ),
      ),
    );
    return { key, lengths };
  });
  /** Each run in front's length, in queue order (null: unknown, or not asked yet). */
  const lengths = computed<Length[]>(() => (frontLengths.data?.key === frontKey.get() ? frontLengths.data.lengths : inFront.get().map(() => null)));
  /** The wait: what the training run has left plus the lengths in front (`queueWait`); a floor when any is unknown. */
  const wait = computed(() => {
    const a = ahead.get();
    return queueWait(a ? settledEta(a) : undefined, lengths.get());
  });
  /** The queued runs in front, together: one run's estimate, or their sum when every one is known. */
  const frontTime = computed(() => {
    const all = queueWait(undefined, lengths.get());
    return all?.known ? timeRange(all.lo, all.hi) : inFront.get().length > 1 ? 'lengths not known yet' : 'length not known yet';
  });
  /**
   * Nothing trains and the run is at the front of the queue (or the worker has taken it off): the worker is reading
   * its dataset and building its model, until the server reports it `running`.
   */
  const starting = computed(() => !cloud() && status() === 'queued' && !!others.data && !ahead.get() && (place.get() ?? 1) === 1);
  /** Start next is a move only when something is in front (or nothing trains, and it starts at once). */
  const canStart = computed(() => place.get() !== null && (place.get()! > 1 || !ahead.get()) && !starting.get());

  let mounted = true;
  onCleanup(() => (mounted = false));

  /** Moves this run in the queue: the new order at once (here and in the queue list), the old one back on a refusal. */
  async function move(req: QueueRequest) {
    const cur = untrack(run);
    if (!cur) return;
    const id = cur.id;
    const epoch = hub.epoch;
    const refresh = () => {
      if (!mounted || hub.epoch !== epoch) return;
      void others.refresh();
      hub.api.run(id).then(
        (r) => {
          if (!mounted || hub.epoch !== epoch) return;
          // Asked for while the trainer was being built: a stream that has since said the run left the queue is newer,
          // and a run that left it never goes back (no status event would correct it until the run ends).
          if (r.status === 'queued' && live.run?.status !== 'queued') return;
          if (live.run?.id === id) live.run = { ...live.run, status: r.status, started: r.started, queue_position: r.queue_position ?? null, resolved: r.resolved ?? live.run.resolved };
        },
        () => {},
      );
    };
    const r = await moveQueued(
      cur,
      req,
      () => {
        const list = others.data;
        const prev = live.run?.queue_position ?? null;
        const optimisticList = list ? withQueueMove(list, id, req) : null;
        if (optimisticList) others.set(optimisticList);
        const next = optimisticList?.find((x) => x.id === id)?.queue_position ?? null;
        if (live.run?.id === id && next != null) live.run = { ...live.run, queue_position: next };
        return () => {
          if (!mounted || hub.epoch !== epoch) return;
          if (list && others.data === optimisticList) others.set(list);
          else void others.refresh();
          if (live.run?.id === id && next != null) {
            if (live.run.queue_position === next) live.run = { ...live.run, queue_position: prev };
            else refresh();
          }
        };
      },
      refresh,
      untrack(() => ahead.get()),
    );
    if (!r || !mounted || hub.epoch !== epoch) return;
    if (live.run?.id === id) live.run = { ...live.run, queue_position: r.queue_position ?? null };
    void others.refresh();
  }

  return { ahead, place, queue, waitingCount, before, inFront, wait, frontTime, starting, canStart, move };
}
