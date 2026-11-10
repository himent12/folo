// Cloud compute in words and rules (docs/CLOUD.md): the phases of a cloud run as people read them, an offer in one
// line, the offer New run preselects, the default time limit, the states of an instance. One module, so Settings, New
// run and the run page say the same thing with the same numbers.

import type { CloudAvailability, CloudInstanceStatus, CloudOffer, CloudPhase, CloudProvider, Run, RunCloud, SystemInfo, TransferProgress } from './api/types';
import { bytes, hoursText, pct, pricePerHour, usd } from './format';

/**
 * Whether the server has cloud compute (`/api/system`'s `cloud` feature): without it no cloud control renders and no
 * request carries `compute`.
 */
export const hasCloud = (sys: SystemInfo | null | undefined): boolean => !!sys?.features?.includes('cloud');

/** Every phase of a cloud run, in order (`CloudPhase`). */
export const PHASES: readonly CloudPhase[] = ['provisioning', 'booting', 'syncing', 'training', 'collecting', 'releasing', 'released'];

/** What a phase is called on screen. */
export const PHASE_LABEL: Record<CloudPhase, string> = {
  provisioning: 'Provisioning',
  booting: 'Booting',
  syncing: 'Uploading data',
  training: 'Training',
  collecting: 'Collecting weights',
  releasing: 'Releasing',
  released: 'Released',
};

const known = (phase: string): phase is CloudPhase => (PHASES as readonly string[]).includes(phase);

/** A phase's words; a phase from a newer server that this Studio does not know is its raw word. */
export const phaseLabel = (phase: string): string => (known(phase) ? PHASE_LABEL[phase] : phase);

/** A phase's place in `PHASES`, or -1 for one this Studio does not know. */
export const phaseIndex = (phase: string): number => (known(phase) ? PHASES.indexOf(phase) : -1);

/** Before `training`: nothing trains yet, so a queued run is the hand-over, not a wait in the training queue. */
export const beforeTraining = (phase: string): boolean => {
  const i = phaseIndex(phase);
  return i >= 0 && i < PHASES.indexOf('training');
};

/**
 * A queued cloud run whose hand-over is still before `training` (provisioning, booting, uploading its data): it waits in
 * no queue, so it is shown by its phase, not as "Queued", and has no place in the queue.
 */
export const inHandover = <R extends Pick<Run, 'status' | 'cloud'>>(run: R): run is R & { cloud: RunCloud } =>
  run.status === 'queued' && !!run.cloud && beforeTraining(run.cloud.phase);

/** The share of a transfer done (0..1), or null when its size is not known. */
export function transferShare(p: TransferProgress | null | undefined): number | null {
  return p && p.total_bytes > 0 ? Math.min(1, p.done_bytes / p.total_bytes) : null;
}

/** "118.0 of 460.0 MiB" (the unit once when both ends share it), "118.0 MiB" while the total is unknown. */
export function transferText(p: TransferProgress, units: 'binary' | 'metric' = 'binary'): string {
  const done = bytes(p.done_bytes, units);
  if (p.total_bytes <= 0) return done;
  const total = bytes(p.total_bytes, units);
  const unit = (s: string) => s.slice(s.lastIndexOf(' ') + 1);
  return unit(done) === unit(total) ? `${done.slice(0, done.lastIndexOf(' '))} of ${total}` : `${done} of ${total}`;
}

/** "a" or "an" before a GPU's name as it is read aloud: "an RTX 4090", "an A100", "an H100", "a GeForce". */
export function article(name: string): 'a' | 'an' {
  // An acronym or model number is read letter by letter: the letters whose names start with a vowel sound take "an".
  if (/^[A-Z][A-Z0-9]/.test(name)) return /^[AEFHILMNORSX]/.test(name) ? 'an' : 'a';
  return /^[aeiou]/i.test(name) ? 'an' : 'a';
}

/** A provider's display name ("RunPod") from its id, the id itself until the list is in. */
export const providerName = (id: string, providers: readonly CloudProvider[] | undefined): string => providers?.find((p) => p.id === id)?.name ?? id;

/**
 * The waiting state's title while a cloud run is queued: the phase it is in ("Provisioning an RTX 4090 at RunPod",
 * "Booting the instance", "Uploading data · 25%"), or, once the worker has the run, that it starts there.
 */
export function phaseTitle(cloud: RunCloud, provider: string): string {
  switch (cloud.phase) {
    case 'provisioning':
      return `Provisioning ${article(cloud.offer.gpu)} ${cloud.offer.gpu} at ${provider}`;
    case 'booting':
      // A remote server launches nothing: the wait is for it to answer.
      return cloud.instance_id ? 'Booting the instance' : `Connecting to ${provider}`;
    case 'syncing': {
      const share = transferShare(cloud.progress);
      return share === null ? PHASE_LABEL.syncing : `${PHASE_LABEL.syncing} · ${pct(Math.floor(share * 100) / 100, 0)}`;
    }
    case 'training':
      return `Starting on ${cloud.instance_id ? 'the instance' : provider}`;
    default:
      return phaseLabel(cloud.phase);
  }
}

/** What the hand-over is doing, in one sentence under the waiting state's title (`phaseTitle`). */
export function phaseWhat(cloud: RunCloud, provider: string, units: 'binary' | 'metric' = 'binary'): string {
  const gpu = cloud.offer.gpu;
  switch (cloud.phase) {
    case 'provisioning':
      return `${provider} is asked for ${article(gpu)} ${gpu}${cloud.offer.price_per_hour === null ? '' : ` at ${pricePerHour(cloud.offer.price_per_hour)}`}. Once it has booted, the dataset is uploaded and training starts.`;
    case 'booting':
      return cloud.instance_id ? 'The instance is starting. FOLO waits for its worker to answer, then checks it.' : `FOLO waits for ${provider} to answer, then checks it.`;
    case 'syncing':
      return cloud.progress ? `${transferText(cloud.progress, units)} sent. Training starts once the dataset is there.` : 'Training starts once the dataset is there.';
    case 'training':
      return 'The worker has the run and starts it as soon as its GPU is free.';
    default:
      return '';
  }
}

/** An offer's own words after its GPU: "low stock", "none available now"; null when there is nothing to say. */
export function availabilityNote(a: CloudAvailability): string | null {
  return a === 'low' ? 'low stock' : a === 'none' ? 'none available now' : null;
}

/** An offer in one line, without its price: "RTX 4090 · 24.0 GiB · Secure Cloud" (memory in the app's one size format). */
export function offerLine(o: CloudOffer, units: 'binary' | 'metric' = 'binary'): string {
  return offerParts(o, units).join(' · ');
}

/** `offerLine`'s items, for markup that sets them as a meta line (`ui/Meta`), so a wrapped line never ends on a "·". */
export function offerParts(o: CloudOffer, units: 'binary' | 'metric' = 'binary'): string[] {
  return [o.gpu_count > 1 ? `${o.gpu_count} × ${o.gpu}` : o.gpu, o.vram_bytes > 0 ? bytes(o.vram_bytes, units) : null, o.detail].filter((p): p is string => !!p);
}

/**
 * The offer New run preselects: the cheapest available one with at least `needBytes` of GPU memory, or, without an
 * estimate, the cheapest available. When none has the memory, the one with the most (the cheapest of those), so the
 * "won't fit" warning stands against the nearest miss. Null when nothing is available.
 */
export function pickOffer(offers: readonly CloudOffer[], needBytes: number | null): string | null {
  const open = offers.filter((o) => o.availability !== 'none');
  if (!open.length) return null;
  const cheapest = (list: CloudOffer[]) => list.reduce((a, b) => ((b.price_per_hour ?? 0) < (a.price_per_hour ?? 0) ? b : a));
  if (needBytes === null) return cheapest(open).id;
  const fit = open.filter((o) => o.vram_bytes >= needBytes);
  if (fit.length) return cheapest(fit).id;
  const most = Math.max(...open.map((o) => o.vram_bytes));
  return cheapest(open.filter((o) => o.vram_bytes === most)).id;
}

/** `Compute.max_hours`' limits and default (folo_api::cloud). */
export const MIN_HOURS = 0.25;
export const MAX_HOURS = 720;
export const DEFAULT_HOURS = 12;

/**
 * New run's "Stop after" until it is set by hand: the high end of the time estimate × 1.5, plus half an hour, rounded up
 * to a whole hour, at least 1 (12 until there is an estimate).
 */
export function defaultMaxHours(highSeconds: number | null | undefined): number {
  if (highSeconds === null || highSeconds === undefined || !Number.isFinite(highSeconds)) return DEFAULT_HOURS;
  return Math.min(MAX_HOURS, Math.max(1, Math.ceil((highSeconds * 1.5) / 3600 + 0.5)));
}

/** The most a run on `price` can cost within `hours` (`CostEstimate.max_usd`); null without a price. */
export const maxCost = (price: number | null, hours: number): number | null => (price === null ? null : price * hours);

/** "$8.28 at 12 h": the limit in money and time, or the time alone for compute that is not billed. */
export const limitText = (price: number | null, hours: number): string => {
  const most = maxCost(price, hours);
  return most === null ? hoursText(hours) : `${usd(most)} at ${hoursText(hours)}`;
};

/** An instance's state in one word. */
export const INSTANCE_LABEL: Record<CloudInstanceStatus, string> = {
  launching: 'Launching',
  starting: 'Starting',
  running: 'Running',
  terminating: 'Terminating',
  terminated: 'Terminated',
  lost: 'Lost',
};

/** True until it is `terminated` or `lost`: it may still cost money. */
export const isAlive = (s: CloudInstanceStatus): boolean => s !== 'terminated' && s !== 'lost';
