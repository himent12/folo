// Server-Sent Events with reconnect. EventSource retries by itself after a dropped connection, but gives up for
// good when the server answers with an error status; then we reopen it ourselves with exponential backoff.

import type { RunEventHandlers } from './client';
import type { RunEventMap } from './types';

/** Every event name a run's stream carries (docs/API.md, `GET /api/runs/{id}/events`); others are ignored. */
export const EVENTS: (keyof RunEventMap)[] = ['status', 'step', 'epoch', 'log', 'queue', 'cloud'];

export interface Backoff {
  /** Delay before the given (0-based) retry attempt, in ms. */
  delay(attempt: number): number;
}

export const defaultBackoff: Backoff = {
  delay: (attempt) => Math.min(15_000, 1000 * 2 ** attempt),
};

export function openEventStream(url: string, handlers: RunEventHandlers, backoff: Backoff = defaultBackoff): () => void {
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
    for (const name of EVENTS) {
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
