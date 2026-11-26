// Server-Sent Events with reconnect, read with fetch rather than EventSource: EventSource hides the server's
// `: keep-alive` comments (docs/API.md: one every 15 s), so a stream that went silent without closing (a stuck proxy, a
// sleeping laptop's link) could not be told from a quiet one and stayed "live" for good (stress d6-connection). Here
// every byte counts as a sign of life; a stream silent for `STREAM_SILENCE_MS` (three keep-alives missed, as Desktop's
// STREAM_SILENCE) is aborted and reopened like a dropped one, and the reopening asks for a resync (`reconnect`). A
// stream the server ended, an error status and a failed request are all reopened with exponential backoff.

import type { AccessWatch } from './access';
import type { Reachability } from './reach';
import type { BenchEventMap, LabelEventMap, RunEventMap } from './types';

/** Handlers for a stream whose payloads are `M`, keyed by the SSE `event:` name, plus the connection's own. */
export type EventHandlers<M> = {
  [K in keyof M]?: (data: M[K]) => void;
} & {
  /** Called after the stream reconnects; callers should refetch history to fill the gap. */
  reconnect?: () => void;
  /** Connection state changes, for "reconnecting…" indicators. */
  connection?: (state: 'open' | 'reconnecting') => void;
};

/**
 * The events of a run's stream (`GET /api/runs/{id}/events`); `queue` comes from servers with `queue_control`,
 * `cloud` from servers with `cloud` (a run on a cloud connection).
 */
export const RUN_EVENTS: readonly (keyof RunEventMap)[] = ['status', 'step', 'epoch', 'log', 'queue', 'cloud'];
/** The events of a labeling job's stream (`GET /api/label/jobs/{id}/events`). */
export const LABEL_EVENTS: readonly (keyof LabelEventMap)[] = ['job', 'image', 'removed', 'reload', 'log'];
/** The events of a benchmark's stream (`GET /api/bench/benchmarks/{id}/events`). */
export const BENCH_EVENTS: readonly (keyof BenchEventMap)[] = ['bench', 'entry', 'loop', 'removed'];

/**
 * How long a stream (or its handshake) may pass no byte at all before it is taken for dead: the server sends a
 * keep-alive comment every 15 s, so this is three of them missed.
 */
export const STREAM_SILENCE_MS = 45_000;

export interface Backoff {
  /** Delay before the given (0-based) retry attempt, in ms. */
  delay(attempt: number): number;
}

export const defaultBackoff: Backoff = {
  delay: (attempt) => Math.min(15_000, 1000 * 2 ** attempt),
};

/**
 * Splits an SSE text stream into events: the returned function takes each chunk and calls `emit` with every complete
 * event's name (`message` when it has none) and data. Comments and other fields are skipped. Exported for tests.
 */
export function sseParser(emit: (name: string, data: string) => void): (chunk: string) => void {
  let buf = '';
  let name = '';
  let data: string[] = [];
  const line = (l: string) => {
    if (l === '') {
      if (data.length) emit(name || 'message', data.join('\n'));
      name = '';
      data = [];
      return;
    }
    if (l.startsWith(':')) return;
    const colon = l.indexOf(':');
    const field = colon < 0 ? l : l.slice(0, colon);
    let value = colon < 0 ? '' : l.slice(colon + 1);
    if (value.startsWith(' ')) value = value.slice(1);
    if (field === 'event') name = value;
    else if (field === 'data') data.push(value);
  };
  return (chunk) => {
    buf += chunk;
    // A `\r` at the end may be the first half of a `\r\n`: it waits for the next chunk.
    const end = buf.endsWith('\r') ? buf.length - 1 : buf.length;
    const lines = buf.slice(0, end).split(/\r\n|\r|\n/);
    buf = (lines.pop() ?? '') + buf.slice(end);
    for (const l of lines) line(l);
  };
}

export interface StreamWatch {
  reach?: Reachability;
  access?: AccessWatch;
}

/**
 * Opens `url` and hands the JSON payload of each event named in `events` to its handler. Other event names are ignored
 * (servers may add some). Returns the function that closes it.
 */
export function openEventStream<M>(
  url: string,
  events: readonly (keyof M & string)[],
  handlers: EventHandlers<M>,
  backoff: Backoff = defaultBackoff,
  silence: number = STREAM_SILENCE_MS,
  watch?: StreamWatch,
): () => void {
  const wanted = new Set<string>(events);
  let closed = false;
  let attempt = 0;
  let hadError = false;
  let timer: ReturnType<typeof setTimeout> | undefined;
  let watchdog: ReturnType<typeof setTimeout> | undefined;
  /** The connection in use (null between one and the next): a reader whose controller is no longer it stops. */
  let ctrl: AbortController | null = null;
  /** When the connection in use last passed a byte (or was opened). */
  let heard = 0;

  const emit = (name: string, data: string) => {
    if (!wanted.has(name)) return;
    const handler = handlers[name as keyof M] as ((d: unknown) => void) | undefined;
    if (!handler) return;
    try {
      handler(JSON.parse(data));
    } catch (e) {
      console.warn(`bad SSE '${name}' payload`, e);
    }
  };

  // The connection is gone (ended, failed, refused or silent): say so and open another after the backoff.
  const drop = (notifyReachFailed = true) => {
    if (closed) return;
    const old = ctrl;
    ctrl = null;
    old?.abort();
    clearTimeout(watchdog);
    hadError = true;
    if (notifyReachFailed) watch?.reach?.failed();
    handlers.connection?.('reconnecting');
    clearTimeout(timer);
    timer = setTimeout(() => void connect(), backoff.delay(attempt++));
  };
  const checkSilence = () => {
    const idle = Date.now() - heard;
    if (idle >= silence) drop();
    else watchdog = setTimeout(checkSilence, silence - idle);
  };

  const connect = async () => {
    if (closed) return;
    const my = new AbortController();
    ctrl = my;
    heard = Date.now();
    clearTimeout(watchdog);
    watchdog = setTimeout(checkSilence, silence);
    const seen = watch?.access?.checks ?? 0;
    let res: Response;
    try {
      res = await fetch(url, { headers: { accept: 'text/event-stream' }, cache: 'no-store', signal: my.signal });
    } catch {
      if (ctrl === my) drop();
      return;
    }
    if (ctrl !== my) return void res.body?.cancel().catch(() => {});
    // A 401 or 403 reached the server (or its auth proxy) and was refused: let Access confirm whether Studio is still
    // let in instead of probing health as an outage (health stays open under `folo serve --token`).
    if (res.status === 401 || res.status === 403) {
      res.body?.cancel().catch(() => {});
      watch?.reach?.ok();
      watch?.access?.refused(seen);
      return drop(false);
    }
    // An error status, or an answer that is not an event stream (a proxy's page): as EventSource, a failed connection.
    if (!res.ok || !res.body || !(res.headers.get('content-type') ?? '').includes('text/event-stream')) {
      res.body?.cancel().catch(() => {});
      return drop();
    }
    heard = Date.now();
    attempt = 0;
    watch?.reach?.ok();
    handlers.connection?.('open');
    if (hadError) {
      hadError = false;
      handlers.reconnect?.();
    }
    const feed = sseParser(emit);
    const decoder = new TextDecoder();
    const reader = res.body.getReader();
    try {
      for (;;) {
        const { done, value } = await reader.read();
        if (ctrl !== my) return;
        if (done) break;
        heard = Date.now();
        feed(decoder.decode(value, { stream: true }));
      }
    } catch {
      // Aborted (closed, or silent) or the connection broke: whether it is still the one in use decides.
    }
    if (ctrl === my) drop();
  };

  void connect();
  return () => {
    closed = true;
    clearTimeout(timer);
    clearTimeout(watchdog);
    const old = ctrl;
    ctrl = null;
    old?.abort();
  };
}
