/**
 * A seed as Studio takes, shows and sends it: every seed field and every reader of a run's seed goes through here.
 *
 * The server reads a seed as any 64-bit whole number (docs/API.md), but a browser's number is exact only up to
 * 2^53 − 1 (`Number.MAX_SAFE_INTEGER`, `MAX_SEED`): a larger one would be sent as another seed, and one read from the
 * server arrives already rounded by JSON parsing. Two rules follow.
 *
 * A. A seed field (New run, Generate synthetic) is a text field, and its seed is the text as typed, checked before any
 *    conversion (`parseSeed`): digits only, leading zeros allowed (`007` is 7) and spaces around them ignored; no sign,
 *    no separators, no decimal point, no exponent (`5.0`, `1e3`, `9007199254740990.5` and `1,000` are refused, never
 *    rounded to a seed), and at most `MAX_SEED`.
 * B. A run's seed over `MAX_SEED` (`exactSeed` false) is never put into anything that leaves Studio or that a person
 *    would run: it is left out where it does not matter to the answer or is said to be left out
 *    (`withoutInexactSeed`: the wait estimate, Copy JSON), and New run, copying such a run, leaves its Seed empty with
 *    `COPIED_SEED_ERROR` (kept in the draft) and holds Start and the CLI command until one is typed. It is shown as
 *    `OVER_SEED`, never as the rounded number.
 */
import { num } from './format';

/** The largest seed Studio holds exactly: 9,007,199,254,740,991. */
export const MAX_SEED = Number.MAX_SAFE_INTEGER;

/** A seed field's text as a seed (rule A), or null when it is not one. */
export function parseSeed(text: string): number | null {
  const t = text.trim();
  if (!/^\d+$/.test(t)) return null;
  // Past 2^53 − 1 a digit string reads as 2^53 or more, never as a safe integer: the bound needs no BigInt.
  const v = Number(t);
  return Number.isSafeInteger(v) ? v : null;
}

/** A seed field's error, or null when its text is a seed. */
export const seedError = (text: string): string | null => (parseSeed(text) === null ? `A whole number from 0 to ${num(MAX_SEED)}.` : null);

/** Whether a seed read from the server is the seed itself, not a rounded stand-in (one over `MAX_SEED`). */
export const exactSeed = (v: number) => Number.isSafeInteger(v) && v >= 0;

/** A request or a copy made from a run's config (rule B): its seed left out when Studio does not hold it exactly. */
export function withoutInexactSeed<T extends { seed?: number }>(c: T): T {
  if (c.seed === undefined || exactSeed(c.seed)) return c;
  const { seed: _, ...rest } = c;
  return rest as T;
}

/** What a seed over `MAX_SEED` is shown as: the bound it is over, never the rounded number. */
export const OVER_SEED = `Over ${num(MAX_SEED)}`;

/** New run's Seed field, copied from a run whose seed is over `MAX_SEED`: left empty with this, until one is typed. */
export const COPIED_SEED_ERROR = `The copied run's seed is over ${num(MAX_SEED)}, more than Studio holds exactly. Enter a seed.`;

/** Copy JSON's note for a run whose seed is over `MAX_SEED`, left out of the copy. */
export const SEED_LEFT_OUT = `Its seed is left out: it is over ${num(MAX_SEED)}, more than Studio can copy exactly.`;
