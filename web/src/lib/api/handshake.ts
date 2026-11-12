// The first handshake with a server: its `GET /api/health` must answer with FOLO's health payload (docs/API.md,
// Health) before Studio uses it. An address that answers with something else is not "no answer": what came back is
// said in plain words (the footer, the offline notice, Settings › Connection) while Studio keeps asking.

import { ApiError } from './client';
import type { Health } from './types';

/** Whether a health answer is FOLO's: `status: "ok"`, a version, whether it is the mock (and the API, when sent). */
export function isHealth(h: unknown): h is Health {
  if (typeof h !== 'object' || h === null) return false;
  const o = h as Record<string, unknown>;
  return (
    o.status === 'ok' &&
    typeof o.version === 'string' &&
    typeof o.mock === 'boolean' &&
    (o.api === undefined || typeof o.api === 'string')
  );
}

export const NOT_FOLO = 'This address answered, but not as a FOLO server.';

/**
 * What a failed health check got back from an address that answered but cannot be used, in words for the user; null
 * when nothing answered: no connection, a timeout, or a proxy with nothing behind it (Vite's dev proxy answers 500
 * with an empty body, a gateway 502 to 504).
 */
export function refusal(e: unknown): string | null {
  if (!(e instanceof ApiError) || e.status === 0) return null;
  if ((e.status === 500 && e.code === 'http_500') || (e.status >= 502 && e.status <= 504)) return null;
  if (e.status === 401) return 'This server asks for sign-in (401).';
  if (e.status === 403) return 'This server refuses access (403).';
  return NOT_FOLO;
}
