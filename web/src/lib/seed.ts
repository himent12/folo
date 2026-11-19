/**
 * A seed as Studio takes and shows it. The server reads a seed as any 64-bit whole number (docs/API.md), but a browser's
 * number is exact only up to 2^53 − 1 (`Number.MAX_SAFE_INTEGER`): a larger one typed here would be sent as another
 * seed, and one read from the server arrives already rounded by JSON parsing. So the forms take 0 to 2^53 − 1 (the
 * limit the server itself puts on a benchmark's seed), and a run's seed above it is never shown or copied as if exact.
 */
import { num } from './format';

/** The largest seed Studio holds exactly: 9,007,199,254,740,991. */
export const MAX_SEED = Number.MAX_SAFE_INTEGER;

/** A seed field's error (number inputs bind `null` when cleared), or null when it holds a seed. */
export const seedError = (v: unknown): string | null =>
  typeof v === 'number' && Number.isSafeInteger(v) && v >= 0 ? null : `A whole number from 0 to ${num(MAX_SEED)}.`;

/** Whether a seed read from the server is the seed itself, not a rounded stand-in (one over `MAX_SEED`). */
export const exactSeed = (v: number) => Number.isSafeInteger(v);

/** What a seed over `MAX_SEED` is shown as: the bound it is over, never the rounded number. */
export const OVER_SEED = `Over ${num(MAX_SEED)}`;

/** New run's Seed field, copied from a run whose seed is over `MAX_SEED`: left empty with this, until one is typed. */
export const COPIED_SEED_ERROR = `The copied run's seed is over ${num(MAX_SEED)}, more than Studio holds exactly. Enter a seed.`;
