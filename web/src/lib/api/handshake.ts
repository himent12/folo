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

/** What `HEAD /api/runs` answers on a FOLO server that wants no token (axum answers HEAD as GET, without the body). */
export const NO_WALL = 200;

/** A server (or what is in front of it) that answers 403 to Studio's requests. */
export const REFUSES_ACCESS = 'This server refuses access (403).';

/**
 * What the token check (`HEAD /api/runs`, as `tokenWall`) says about the server in use letting Studio in: `'in'` for a
 * 2xx, the status for a 401 or 403 (refused), null when it cannot tell (no answer, a timeout, any other status).
 */
export type AccessAnswer = 'in' | 401 | 403 | null;

export async function accessCheck(base: string, signal: AbortSignal): Promise<AccessAnswer> {
  try {
    const { status } = await fetch(`${base}/api/runs`, { method: 'HEAD', signal });
    return status >= 200 && status < 300 ? 'in' : status === 401 || status === 403 ? status : null;
  } catch {
    return null;
  }
}

/** A refusal `accessCheck` found, in the words a connect uses: a 401 is the token wall. */
export function refusedWords(status: 401 | 403): string {
  return status === 401 ? TOKEN_WALL : REFUSES_ACCESS;
}

/**
 * The token check sent together with the health check (see `Hub`'s connect for when its answer is used). `answer`:
 * true for `401` (the token wall), false for `NO_WALL`, null for anything else: any other status, no HTTP answer, or
 * cancelled. `cancel` aborts the request while it is still waiting for its answer, and does nothing once it has one
 * (or failed): aborting a request that has answered only shows it as cancelled in the browser's network log.
 */
export function earlyTokenWall(base: string, signal: AbortSignal): { answer: Promise<boolean | null>; cancel: () => void } {
  const mine = new AbortController();
  let out = true;
  const answer = fetch(`${base}/api/runs`, { method: 'HEAD', signal: AbortSignal.any([signal, mine.signal]) }).then(
    (r) => {
      out = false;
      return r.status === 401 ? true : r.status === NO_WALL ? false : null;
    },
    () => {
      out = false;
      return null;
    },
  );
  return {
    answer,
    cancel: () => {
      if (out) mine.abort();
    },
  };
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
  if (e.status === 403) return REFUSES_ACCESS;
  return NOT_FOLO;
}
