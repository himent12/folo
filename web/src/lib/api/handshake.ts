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

/** A FOLO server started with `--token`: Studio has no token to send (docs/CLOUD.md, Securing a server). */
export const TOKEN_WALL = "This server needs a token (folo serve --token), which Studio can't send.";

/**
 * Whether the FOLO server at `base` refuses Studio's requests for want of a token. `GET /api/health` is the one path
 * `folo serve --token` leaves open, so its answer cannot tell; a `HEAD` of the runs list (what Studio polls anyway, without
 * the body) can: `401` is the token wall. No answer is not a wall: the requests that follow will find the server gone.
 */
export async function tokenWall(base: string, signal: AbortSignal): Promise<boolean> {
  try {
    return (await fetch(`${base}/api/runs`, { method: 'HEAD', signal })).status === 401;
  } catch (e) {
    if (signal.aborted) throw e;
    return false;
  }
}

/**
 * What a failed health check got back from an address that answered but cannot be used, in words for the user; null
 * when nothing answered: no connection, a timeout, or a proxy with nothing behind it (Vite's dev proxy answers 500
 * with an empty body, a gateway 502 to 504).
 */
export function refusal(e: unknown): string | null {
  if (!(e instanceof ApiError) || e.status === 0) return null;
  if ((e.status === 500 && e.code === 'http_500') || (e.status >= 502 && e.status <= 504)) return null;
  if (e.status === 401 && e.code === 'token_wall') return TOKEN_WALL;
  if (e.status === 401) return 'This server asks for sign-in (401).';
  if (e.status === 403) return 'This server refuses access (403).';
  return NOT_FOLO;
}
