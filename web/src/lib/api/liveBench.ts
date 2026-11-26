// A benchmark kept live (docs/API.md, Vision benchmarks → Events): follows its SSE stream and loads the benchmark, its
// entries and its loops. As for a labeling job (LiveLabelJob), the stream replays nothing but the benchmark, so we
// subscribe *before* loading and merge the two: an entry or loop event that arrived while the lists were loading is
// applied over the answer, and a `bench` event over the benchmark. A `bench` event also loads the lists again (coalesced
// with the load the stream's opening asks for): it is what the server sends a subscriber that fell behind, and it comes
// when the entries change in a way the lists should show (one added, the leader changed). An action's answer (a stop, a
// new entry, a rename) shows at once, except where the stream has told of that item since the action began. The
// entries are kept in the leaderboard's order ($lib/bench/rules `rankEntries`), the loops newest first.

import { computed, effect, signal, untrack } from 'fw';
import { BENCH_FEATURE, rankEntries } from '../bench/rules';
import { ApiError } from './client';
import { hub } from './index';
import { STREAM_GRACE_MS } from './live';
import type { Bench, BenchEntry, BenchLoop } from './types';

/** How long a `bench` event or the stream's opening waits for another before the lists load again. */
export const RESYNC_MS = 150;

/** `items` with the changes applied (null: removed), changed ones in place, new ones at the end. */
export function applyById<T extends { id: string }>(items: readonly T[], changes: Iterable<readonly [string, T | null]>): T[] {
  const left = new Map(changes);
  if (!left.size) return [...items];
  const out: T[] = [];
  for (const x of items) {
    if (!left.has(x.id)) out.push(x);
    else {
      const v = left.get(x.id);
      left.delete(x.id);
      if (v) out.push(v);
    }
  }
  for (const v of left.values()) if (v) out.push(v);
  return out;
}

const byCreatedDesc = (a: BenchLoop, b: BenchLoop) => b.created.localeCompare(a.created) || b.id.localeCompare(a.id);

export class LiveBench {
  #bench = signal<Bench | undefined>(undefined);
  #entries = signal<BenchEntry[]>([]);
  #loops = signal<BenchLoop[]>([]);
  #error = signal<unknown>(undefined);
  #resyncError = signal<unknown>(undefined);
  #loading = signal<boolean>(true);
  #listed = signal<boolean>(false);
  #connection = signal<'idle' | 'open' | 'reconnecting'>('idle');
  #lost = signal<number | null>(null);

  get bench(): Bench | undefined {
    return this.#bench.get();
  }
  set bench(v: Bench | undefined) {
    this.#bench.set(v);
  }
  /** The leaderboard: finished entries by score, then the others, newest first. */
  get entries(): BenchEntry[] {
    return this.#entries.get();
  }
  set entries(v: BenchEntry[]) {
    this.#entries.set(v);
  }
  /** Newest first. */
  get loops(): BenchLoop[] {
    return this.#loops.get();
  }
  set loops(v: BenchLoop[]) {
    this.#loops.set(v);
  }
  get error(): unknown {
    return this.#error.get();
  }
  set error(v: unknown) {
    this.#error.set(v);
  }
  /** Error from a post-load background resync (kept separate from `error` so the loaded board stays visible). */
  get resyncError(): unknown {
    return this.#resyncError.get();
  }
  set resyncError(v: unknown) {
    this.#resyncError.set(v);
  }
  get loading(): boolean {
    return this.#loading.get();
  }
  set loading(v: boolean) {
    this.#loading.set(v);
  }
  /** The lists have loaded at least once: until then a failed load is the page's error, never "no entries yet". */
  get listed(): boolean {
    return this.#listed.get();
  }
  set listed(v: boolean) {
    this.#listed.set(v);
  }
  get connection(): 'idle' | 'open' | 'reconnecting' {
    return this.#connection.get();
  }
  set connection(v: 'idle' | 'open' | 'reconnecting') {
    this.#connection.set(v);
  }
  /** When the stream was last heard from, set once it has been down for `STREAM_GRACE_MS`; null while it is up. */
  get lost(): number | null {
    return this.#lost.get();
  }
  set lost(v: number | null) {
    this.#lost.set(v);
  }
  #retry = signal(0);
  /** Every `bench` event bumps it: a snapshot or an action's answer older than the latest event is not applied. */
  #benchSeq = 0;
  /** Bumps whenever the bound benchmark view starts or tears down so late `act()` answers never cross switches. */
  #viewSeq = 0;
  /** The clock items are ordered by: every entry, loop or removal event and every load sent bumps it. */
  #seq = 0;
  /** Per entry or loop id, the clock at its latest event; `#loaded` the clock at which the lists shown were asked for. */
  #heard = new Map<string, number>();
  #loaded = 0;
  #inflight = new Set<{ sent: number; entries: Map<string, BenchEntry | null>; loops: Map<string, BenchLoop | null> }>();

  constructor(id: () => string | null) {
    // Memoized boolean so `hub.features` flipping `['vision_bench'] -> null` during an outage keeps the last snapshot.
    const lacksBench = computed(() => hub.lacks(BENCH_FEATURE));
    effect(() => {
      const benchId = id();
      const resetView = () => {
        this.#viewSeq++;
        this.#heard.clear();
        this.#loaded = ++this.#seq;
        untrack(() => {
          this.bench = undefined;
          this.entries = [];
          this.loops = [];
          this.error = undefined;
          this.resyncError = undefined;
          this.loading = false;
          this.listed = false;
          this.lost = null;
          this.connection = 'idle';
        });
      };
      if (!benchId) {
        resetView();
        return;
      }
      if (hub.mode === 'connecting') return;
      if (lacksBench.get()) {
        resetView();
        return;
      }
      void hub.epoch;
      void this.#retry.get();
      this.#viewSeq++;
      this.#heard.clear();
      this.#loaded = ++this.#seq;
      const api = hub.api;
      let closed = false;
      const view = new AbortController();
      let heard = Date.now();
      let lostTimer: ReturnType<typeof setTimeout> | undefined;
      let syncTimer: ReturnType<typeof setTimeout> | undefined;
      let resync: ReturnType<typeof setTimeout> | undefined;
      untrack(() => {
        this.bench = undefined;
        this.entries = [];
        this.loops = [];
        this.error = undefined;
        this.resyncError = undefined;
        this.loading = true;
        this.listed = false;
        this.lost = null;
        this.connection = 'idle';
      });

      // Answers can arrive out of order (a load per reconnect): one older than the last applied is dropped.
      let issued = 0;
      let applied = 0;
      const load = async () => {
        if (closed) return;
        const my = ++issued;
        const seq = this.#benchSeq;
        const flight = { sent: ++this.#seq, entries: new Map<string, BenchEntry | null>(), loops: new Map<string, BenchLoop | null>() };
        this.#inflight.add(flight);
        try {
          const [bench, entries, loops] = await Promise.all([
            api.bench(benchId, view.signal),
            api.benchEntries(benchId, view.signal),
            api.benchLoops(benchId, view.signal),
          ]);
          if (closed || my < applied) return;
          applied = my;
          if (this.#benchSeq === seq || !this.bench) this.bench = bench;
          this.entries = rankEntries(applyById(entries, flight.entries));
          this.loops = applyById(loops, flight.loops).sort(byCreatedDesc);
          this.#loaded = flight.sent;
          this.loading = false;
          this.listed = true;
          this.error = undefined;
          this.resyncError = undefined;
          heard = Date.now();
          if (this.lost !== null) this.lost = this.connection === 'reconnecting' ? heard : null;
        } catch (e) {
          if (closed || my < applied) return;
          throw e;
        } finally {
          this.#inflight.delete(flight);
        }
      };
      // The benchmark was deleted (404) or this server does not implement benchmarks (501): stop retrying and close SSE.
      let unsub = (): void => {};
      const gone = (e: unknown): boolean => {
        const terminal = e instanceof ApiError && (e.status === 404 || (e.status === 501 && e.code === 'not_implemented'));
        if (closed || !terminal) return false;
        closed = true;
        this.#viewSeq++;
        this.#heard.clear();
        this.#loaded = ++this.#seq;
        view.abort();
        clearTimeout(lostTimer);
        lostTimer = undefined;
        clearTimeout(syncTimer);
        syncTimer = undefined;
        clearTimeout(resync);
        resync = undefined;
        unsub();
        this.error = e;
        this.resyncError = undefined;
        this.bench = undefined;
        this.entries = [];
        this.loops = [];
        this.lost = null;
        this.connection = 'idle';
        this.loading = false;
        this.listed = false;
        return true;
      };
      // Re-synchronising (on open, on a `bench` event, on reconnect) never gives up silently: a gap would stay.
      const sync = (attempt = 0) => {
        if (closed) return;
        clearTimeout(syncTimer);
        load().catch((e) => {
          if (closed || gone(e)) return;
          // Never listed yet: the failure is the page's error (with its retry), not an empty board.
          if (!this.listed) {
            this.error = e;
            this.loading = false;
          } else {
            this.resyncError = e;
          }
          clearTimeout(syncTimer);
          syncTimer = setTimeout(() => sync(attempt + 1), Math.min(15_000, 1000 * 2 ** attempt));
        });
      };
      const soon = () => {
        if (resync !== undefined || closed) return;
        resync = setTimeout(() => {
          resync = undefined;
          if (!closed) sync();
        }, RESYNC_MS);
      };
      const alive = () => {
        heard = Date.now();
        this.lost = null;
        this.resyncError = undefined;
      };
      const change = <T extends { id: string }>(kind: 'entries' | 'loops', item: T | null, itemId: string) => {
        this.#heard.set(itemId, ++this.#seq);
        for (const f of this.#inflight) (f[kind] as Map<string, T | null>).set(itemId, item);
      };

      unsub = api.benchEvents(benchId, {
        connection: (s) => {
          if (closed) return;
          // Events between the first load and the stream opening would be lost: load again once it is open.
          if (s === 'open' && this.connection === 'idle') soon();
          this.connection = s;
          // A stream that dropped may mean the benchmark is gone: one load says so (a 404) or catches up.
          if (s === 'reconnecting') soon();
          if (s === 'open') {
            clearTimeout(lostTimer);
            lostTimer = undefined;
          } else if (this.lost === null && lostTimer === undefined) {
            const since = heard;
            lostTimer = setTimeout(() => {
              lostTimer = undefined;
              if (!closed && this.connection === 'reconnecting') this.lost = since;
            }, STREAM_GRACE_MS);
          }
        },
        bench: (b) => {
          if (closed) return;
          alive();
          this.#benchSeq++;
          // While the first load's failure is shown, the benchmark alone does not make an (empty) board: the load does.
          if (this.listed || this.error === undefined) this.bench = b;
          soon();
        },
        entry: (e) => {
          if (closed) return;
          alive();
          change('entries', e, e.id);
          this.entries = rankEntries(applyById(this.entries, [[e.id, e]]));
        },
        loop: (l) => {
          if (closed) return;
          alive();
          change('loops', l, l.id);
          this.loops = applyById(this.loops, [[l.id, l]]).sort(byCreatedDesc);
        },
        removed: ({ id: gone }) => {
          if (closed) return;
          alive();
          // An entry or a loop: whichever list has it.
          change('entries', null, gone);
          for (const f of this.#inflight) f.loops.set(gone, null);
          this.entries = this.entries.filter((e) => e.id !== gone);
          this.loops = this.loops.filter((l) => l.id !== gone);
        },
        reconnect: () => {
          if (!closed) sync();
        },
      });

      load().catch((e) => {
        if (closed || gone(e)) return;
        this.error = e;
        this.loading = false;
        this.bench = undefined;
      });

      return () => {
        closed = true;
        this.#viewSeq++;
        view.abort();
        clearTimeout(lostTimer);
        clearTimeout(syncTimer);
        clearTimeout(resync);
        unsub();
      };
    });
  }

  /** Loads again from scratch (after a load error). */
  retry() {
    this.#retry.set((n) => n + 1);
  }

  /** Where the stream is now: an action takes a mark before it starts, and `putEntry` / `putLoop` give its answer with it. */
  mark(): number {
    return this.#seq;
  }

  /** Runs an action that answers with the benchmark (rename, a new plan) and shows its answer, unless an event or server/benchmark switch was newer. */
  async act(call: () => Promise<Bench>): Promise<Bench> {
    const epoch = hub.epoch;
    const view = this.#viewSeq;
    const seq = this.#benchSeq;
    const b = await call();
    if (hub.epoch === epoch && hub.mode !== 'connecting' && this.#viewSeq === view && this.#benchSeq === seq && this.bench?.id === b.id) this.bench = b;
    return b;
  }

  #fresh(id: string, since: number) {
    return (this.#heard.get(id) ?? 0) <= since && this.#loaded <= since;
  }

  /** Shows an entry as an action's answer has it (started, stopped), unless the stream told of it since `since`. */
  putEntry(e: BenchEntry, since: number) {
    for (const f of this.#inflight) if (f.sent <= since && !f.entries.has(e.id)) f.entries.set(e.id, e);
    if (this.#fresh(e.id, since)) this.entries = rankEntries(applyById(this.entries, [[e.id, e]]));
  }

  /** Drops a deleted entry (also from a load in flight, which may still have it). */
  dropEntry(id: string) {
    for (const f of this.#inflight) f.entries.set(id, null);
    this.entries = this.entries.filter((e) => e.id !== id);
  }

  /** Shows a loop as an action's answer has it, unless the stream told of it since `since`. */
  putLoop(l: BenchLoop, since: number) {
    for (const f of this.#inflight) if (f.sent <= since && !f.loops.has(l.id)) f.loops.set(l.id, l);
    if (this.#fresh(l.id, since)) this.loops = applyById(this.loops, [[l.id, l]]).sort(byCreatedDesc);
  }

  /** Drops a deleted loop. */
  dropLoop(id: string) {
    for (const f of this.#inflight) f.loops.set(id, null);
    this.loops = this.loops.filter((l) => l.id !== id);
  }
}
