// How a run's readings are said: the hatched end of its progress rail (mosaic off), a learning rate's power of ten, the
// epoch a per-step reading comes from, and the loss terms in plain words. Shared by the KPI strip and the charts.
import { lr as fmtLr } from '$lib/format';

const PLAIN: Record<string, string> = { '⁻': '−', '⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4', '⁵': '5', '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9' };

/**
 * What a progress rail's hatched end means (the last `n` of `epochs` train without mosaic, the trainer's capped count),
 * said for where the run is (`epoch` done, `live`), so a phase still ahead never reads as the current state (round
 * 27's "mosaic off" at epoch 38 of 100): before it, "Mosaic off after epoch 90"; while a live run is in it, "Mosaic off
 * · final 10 epochs"; once past, "Last 10 epochs: mosaic off". A run the trainer stopped early (`early`, lib/api/progress
 * `stoppedEarly`) never gets there, or not to its end: "Stopped early, before mosaic off" / "Mosaic off from epoch 91".
 * `cap` is the caption under the hatch, `full` its tooltip and the rail's accessible description.
 */
export function mosaicOffText(n: number, epochs: number, epoch: number, live: boolean, early = false): { cap: string; full: string } {
  const from = epochs - n;
  const last = `${n} ${n === 1 ? 'epoch' : 'epochs'}`;
  if (early && epoch <= from) return { cap: 'Stopped early, before mosaic off', full: `Stopped early at epoch ${epoch}: mosaic was to be off after epoch ${from}` };
  if (early) return { cap: `Mosaic off from epoch ${from + 1}`, full: `Mosaic off from epoch ${from + 1} until the early stop at epoch ${epoch}` };
  if (epoch < from) return { cap: `Mosaic off after epoch ${from}`, full: `Mosaic off after epoch ${from}: the last ${last} train without mosaic` };
  if (live) {
    const final = n === 1 ? 'final epoch' : `final ${n} epochs`;
    return { cap: `Mosaic off · ${final}`, full: `Mosaic off now: the ${final} train without mosaic` };
  }
  return { cap: `Last ${last}: mosaic off`, full: `Last ${last}: mosaic off` };
}

/**
 * "3.42" and "× 10⁻⁴" from 3.42e-4: the value set large, its power of ten as the unit, so a tile never cuts it off.
 * `exp` ("−4") is the exponent in plain digits, for a real superscript at a legible size (Unicode superscripts render
 * at about 8 px in a 12 px unit).
 */
export function lrParts(v: number | null | undefined): { value: string; unit?: string; exp?: string } {
  const [m, e] = fmtLr(v).split(' × ');
  return e ? { value: m!, unit: `× ${e}`, exp: [...e.slice(2)].map((c) => PLAIN[c] ?? c).join('') } : { value: m! };
}

/**
 * The epoch a per-step reading (loss, LR, throughput) comes from: "ep 24". A run that ended inside an epoch (failed or
 * stopped part-way) has steps past its last finished epoch, so their epoch says so, "ep 24 (partial)", and never seems
 * to contradict the run's 23 finished epochs. `observed` is the progress in epochs (23.4).
 */
export function stepEpoch(status: string, observed: number, finished: number): string {
  const n = Math.max(1, Math.ceil(observed - 1e-9));
  return `ep ${n}${status !== 'running' && status !== 'queued' && n > finished ? ' (partial)' : ''}`;
}

/**
 * The epochs a run has finished, the `epoch` [`mosaicOffText`] takes. `Run.epoch` is the epoch in progress while a run
 * is queued, running or stopping, and the one it failed in for a failed run (docs/API.md `Run.epoch`); only a
 * completed or stopped run carries its last finished epoch. Read as finished, epoch 2 of 3 in progress said "Mosaic
 * off · final epoch" a whole epoch before the trainer closes mosaic, and a run that failed in epoch 2 said "Last 1
 * epoch: mosaic off" for an epoch it never trained (stress s3-live-1).
 */
export function epochsDone(status: string, epoch: number): number {
  return status === 'completed' || status === 'stopped' ? epoch : Math.max(0, epoch - 1);
}

/** The loss terms in plain words (DESIGN_DIRECTION §2.3: no jargon without its definition). */
export const LOSS_TERMS: [term: string, means: string][] = [
  ['total', 'box, cls and dfl weighted and added: what training lowers.'],
  ['box', 'box regression: how far predicted boxes are from the true ones.'],
  ['cls', 'classification: how wrong the class scores are.'],
  ['dfl', 'distribution focal: how sharply each box edge is placed.'],
  ['EMA', "smoothing over steps (0.98: about the last 50); a higher value reacts more slowly. The chart's header gives the last raw step beside the smoothed total."],
];

/**
 * The change of a score since the last epoch as Per class writes it, "▲ 0.012" / "▼ 0.004" (direction by
 * mark, in ink), or "no change" when it rounds to nothing.
 */
export function scoreDelta(now: number, prev: number): string {
  const d = Math.round((now - prev) * 1000) / 1000;
  return d === 0 ? 'no change' : `${d > 0 ? '▲' : '▼'} ${Math.abs(d).toFixed(3)}`;
}
