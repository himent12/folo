// Server-Sent Events with reconnect. EventSource retries by itself after a dropped connection, but gives up for
// good when the server answers with an error status; then we reopen it ourselves with exponential backoff.

import type { LabelEventMap, RunEventMap } from './types';

/** Handlers for a stream whose payloads are `M`, keyed by the SSE `event:` name, plus the connection's own. */
export type EventHandlers<M> = {
  [K in keyof M]?: (data: M[K]) => void;
} & {
  /** Called after the stream reconnects; callers should refetch history to fill the gap. */
  reconnect?: () => void;
  /** Connection state changes, for "reconnecting…" indicators. */
  connection?: (state: 'open' | 'reconnecting') => void;
};

/** The events of a run's stream (`GET /api/runs/{id}/events`); `queue` comes from servers with `queue_control`. */
export const RUN_EVENTS: readonly (keyof RunEventMap)[] = ['status', 'step', 'epoch', 'log', 'queue'];
/** The events of a labeling job's stream (`GET /api/label/jobs/{id}/events`). */
export const LABEL_EVENTS: readonly (keyof LabelEventMap)[] = ['job', 'image', 'removed', 'reload', 'log'];

export interface Backoff {
  /** Delay before the given (0-based) retry attempt, in ms. */
  delay(attempt: number): number;
}

export const defaultBackoff: Backoff = {
  delay: (attempt) => Math.min(15_000, 1000 * 2 ** attempt),
};

/**
 * Opens `url` and hands the JSON payload of each event named in `events` to its handler. Other event names are ignored
 * (servers may add some). Returns the function that closes it.
 */
export function openEventStream<M>(
  url: string,
  events: readonly (keyof M & string)[],
  handlers: EventHandlers<M>,
  backoff: Backoff = defaultBackoff,
): () => void {
  let es: EventSource | null = null;
  let closed = false;
  let attempt = 0;
  let hadError = false;
  let timer: ReturnType<typeof setTimeout> | undefined;

  const connect = () => {
    if (closed) return;
    es = new EventSource(url);
    es.onopen = () => {
      attempt = 0;
      handlers.connection?.('open');
      if (hadError) {
        hadError = false;
        handlers.reconnect?.();
      }
    };
    es.onerror = () => {
      hadError = true;
      handlers.connection?.('reconnecting');
      if (es && es.readyState === EventSource.CLOSED) {
        es = null;
        timer = setTimeout(connect, backoff.delay(attempt++));
      }
    };
    for (const name of events) {
      es.addEventListener(name, (ev) => {
        const handler = handlers[name] as ((d: unknown) => void) | undefined;
        if (!handler) return;
        try {
          handler(JSON.parse((ev as MessageEvent<string>).data));
        } catch (e) {
          console.warn(`bad SSE '${name}' payload`, e);
        }
      });
    }
  };

  connect();
  return () => {
    closed = true;
    clearTimeout(timer);
    es?.close();
    es = null;
  };
}
