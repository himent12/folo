// Measured throughput and run estimates (docs/API.md, Throughput and estimates): which numbers come from this server
// and which are the reference, in words every screen shares. Every figure shown from here names its source: "measured
// here" only for a row of this very adapter or CPU; a row from another adapter under the same ordinal says where it
// was measured; everything else is the reference (docs/BENCHMARKS.md), named to people as `referenceSource` words it.
// The rules every screen follows: a measured figure is the primary one and the reference stays secondary beside it
// (or under it), each labelled; a run's time is a range (`timeRange`), never one precise figure; a rough estimate
// (low confidence) reads one step quieter; and a server without these features keeps the reference figures, said to
// be the reference, so nothing breaks.

import type { Api } from '$lib/api/client';
import type { Arch, InferThroughput, RunEstimate, SystemInfo, ThroughputTable, TrainThroughput } from '$lib/api/types';
import { ago, num } from '$lib/format';

export type ThroughputFeature = 'throughput' | 'run_estimate' | 'benchmark';

/** Whether the server lists `f` in `/api/system`'s `features` (older servers list none of these). */
export const hasFeature = (sys: SystemInfo | null | undefined, f: ThroughputFeature) => !!sys?.features?.includes(f);

/**
 * The throughput table, or null where the server has none (or it cannot be read): callers then keep the reference
 * figures, so a missing table never breaks a page.
 */
export async function loadThroughput(api: Api, signal?: AbortSignal): Promise<ThroughputTable | null> {
  try {
    const sys = await api.system(signal);
    return hasFeature(sys, 'throughput') ? await api.throughput(signal) : null;
  } catch (e) {
    if (e instanceof DOMException && e.name === 'AbortError') throw e;
    return null;
  }
}

/** A measured row of this server's own adapter or CPU (not one measured elsewhere under the same ordinal). */
export function measuredHere(t: ThroughputTable, r: TrainThroughput | InferThroughput): boolean {
  if (r.source !== 'measured') return false;
  if (r.device === t.device) return r.measured_on === t.device_name;
  // CPU rows: the table names the default device's adapter only; a CPU row is this machine's unless it says otherwise.
  return r.device === 'cpu';
}

/** The measured rows of this server, training then inference, newest first within each. */
export function measuredRows(t: ThroughputTable) {
  return { training: t.training.filter((r) => r.source === 'measured'), inference: t.inference.filter((r) => r.source === 'measured') };
}

/**
 * This server's measured end-to-end latency for `arch` at `imgsz` on its default device: the row with the most
 * samples (any precision; the row says which). Undefined when only the reference has one.
 */
export function measuredLatency(t: ThroughputTable | null | undefined, arch: Arch, imgsz = 640): InferThroughput | undefined {
  if (!t) return undefined;
  return t.inference
    .filter((r) => r.arch === arch && r.imgsz === imgsz && r.device === t.device && measuredHere(t, r))
    .sort((a, b) => b.samples - a.samples)[0];
}

/** This server's measured training row for `arch` at `imgsz`, batch 16, f32 on its default device, if any. */
export function measuredTraining(t: ThroughputTable | null | undefined, arch: Arch, imgsz = 640): TrainThroughput | undefined {
  if (!t) return undefined;
  return t.training
    .filter((r) => r.arch === arch && r.imgsz === imgsz && !r.amp && r.device === t.device && measuredHere(t, r))
    .sort((a, b) => Math.abs(a.batch - 16) - Math.abs(b.batch - 16) || b.samples - a.samples)[0];
}

/** "5.90 ms" / "12.3 ms": two decimals under 10 ms, as every latency in Studio. */
export const latencyMs = (x: number) => `${num(x, x < 10 ? 2 : 1)} ms`;
/** "5.50–6.40 ms": a range in one unit. */
export const latencyRange = (lo: number, hi: number) => `${num(lo, lo < 10 ? 2 : 1)}–${num(hi, hi < 10 ? 2 : 1)} ms`;

const minutes = (s: number, up: boolean) => Math.max(1, up ? Math.ceil(s / 60) : Math.floor(s / 60));
const hours = (s: number) => (s / 3600).toFixed(s < 36_000 ? 1 : 0);

/**
 * A run's time as the range it is ("18–24 min", "1.2–1.6 h", "50 min – 1.2 h"), never one precise figure; "~18 min"
 * only when both ends round to the same value.
 */
export function timeRange(lo: number, hi: number): string {
  const a = Math.max(0, Math.min(lo, hi));
  const b = Math.max(lo, hi);
  if (b < 60) return 'under 1 min';
  if (b < 3600) {
    const x = minutes(a, false);
    const y = minutes(b, true);
    return x === y ? `~${x} min` : `${x}–${y} min`;
  }
  if (a >= 3600) {
    const x = hours(a);
    const y = hours(b);
    return x === y ? `~${x} h` : `${x}–${y} h`;
  }
  return `${minutes(a, false)} min – ${hours(b)} h`;
}

/** One rough time ("~18 min", "~1.2 h"), for figures that have no range. */
export function roughTime(s: number): string {
  return s < 3600 ? `~${Math.max(1, Math.round(s / 60))} min` : `~${hours(s)} h`;
}

/** "this GPU" / "this CPU": the device an estimate is for, in a sentence. */
export const thisDevice = (device: string) => (device === 'cpu' ? 'this CPU' : 'this GPU');

/**
 * The basis of a reference estimate, saying why its range is wide: "Reference benchmarks · not run here yet"
 * on the reference machine's GPU or CPU model (-30% / +60%), "Wide range: not run on this GPU yet" elsewhere
 * (-60% / +200%).
 */
export const referenceBasis = (sameModel: boolean, cpu: boolean) =>
  sameModel ? 'Reference benchmarks · not run here yet' : `Wide range: not run on this ${cpu ? 'CPU' : 'GPU'} yet`;

/**
 * What an estimate stands on, in plain words, short enough for one line under the figure:
 * "Measured on this GPU · 226 samples", "Scaled from folo-n at 640 px on this GPU" ("Wide range: scaled from folo-n at
 * 640 px" when that is far from these settings), or a reference basis (`referenceBasis`): a wide range says why.
 */
export function basisLine(e: RunEstimate): string {
  const b = e.basis;
  if (b.source === 'measured' && b.exact) return `Measured on ${thisDevice(e.device)} · ${num(b.samples)} ${b.samples === 1 ? 'sample' : 'samples'}`;
  // Carried far from what was measured (low confidence), the range is wide, and the line says so and why.
  if (b.source === 'measured')
    return e.confidence === 'low' ? `Wide range: scaled from ${b.arch} at ${b.imgsz} px` : `Scaled from ${b.arch} at ${b.imgsz} px on ${thisDevice(e.device)}`;
  return referenceBasis(b.measured_on === e.device_name, e.device === 'cpu');
}

/** What the range means, for the disclosure under the estimate. */
export function confidenceLine(e: RunEstimate): string {
  if (e.confidence === 'high') return 'Measured here with these very settings, so the range is narrow.';
  if (e.confidence === 'medium') return 'Measured here, but with few samples or nearby settings: a wider range.';
  return 'Nothing close to these settings was measured here: a wide range.';
}

/** "2h ago", or "reference" for a row that was never measured. */
export const updatedWords = (r: { updated: string | null }, now = Date.now()) => (r.updated ? ago(r.updated, now) : 'reference');
