// The mock's cloud compute (docs/API.md, Cloud compute and Mock mode → Cloud): the real providers and their forms, a
// RunPod connection with a fixed list of offers, and a cloud run's hand-over as a pure function of its timeline and the
// clock (as the mock's runs are), so the run page, its stream and the instances list agree at any instant.

import { bytes } from '../../format';
import type { CloudConnection, CloudInstance, CloudInstanceStatus, CloudOffer, CloudPhase, CloudProvider, LogLevel, LogLine, RunCloud } from '../types';

const GiB = 2 ** 30;
const MIN = 60_000;
const HOUR = 60 * MIN;

/** The providers as folo-cloud lists them (`info()` of crates/folo-cloud's runpod/provider.rs and remote.rs, word for word). */
export const PROVIDERS: CloudProvider[] = [
  {
    id: 'runpod',
    name: 'RunPod',
    about: 'Rents an on-demand GPU pod on RunPod for each run and terminates it when the run ends.',
    docs_url: 'https://www.runpod.io/console/user/settings',
    fields: [
      {
        key: 'api_key',
        label: 'API key',
        secret: true,
        required: true,
        placeholder: 'rpa_…',
        help: 'RunPod console → Settings → API Keys; the key needs read and write access to pods.',
      },
      {
        key: 'worker_base_url',
        label: 'Worker download URL',
        secret: false,
        required: false,
        placeholder: 'https://example.com/folo',
        help: 'Where instances download FOLO from (install.sh and the release archive) when not the public release of this version: a mirror, or your own build.',
      },
    ],
    provisions: true,
    self_terminates: true,
  },
  {
    id: 'remote',
    name: 'Remote FOLO server',
    about: 'A FOLO server you run yourself (`folo serve --host 0.0.0.0 --token …`), on a workstation or a VM at any cloud.',
    docs_url: null,
    fields: [
      {
        key: 'url',
        label: 'Server URL',
        secret: false,
        required: true,
        placeholder: 'https://gpu-box.example.com:7878',
        help: 'The address of its `folo serve`; use https:// (a TLS proxy or a tunnel) across networks you do not control.',
      },
      { key: 'token', label: 'Token', secret: true, required: false, placeholder: null, help: 'Its `--token` (or FOLO_TOKEN); empty when it runs without one.' },
    ],
    provisions: false,
    self_terminates: false,
  },
];

const offer = (id: string, gpu: string, vram: number, price: number, detail: string, availability: CloudOffer['availability']): CloudOffer => ({
  id,
  gpu,
  gpu_count: 1,
  vram_bytes: vram * GiB,
  price_per_hour: price,
  region: null,
  detail,
  availability,
});

/** RunPod's offers, cheapest first: both clouds, one with nothing available now and one low in stock. */
export const RUNPOD_OFFERS: CloudOffer[] = [
  offer('NVIDIA RTX A4000@community', 'RTX A4000', 16, 0.17, 'Community Cloud', 'high'),
  offer('NVIDIA GeForce RTX 3090@community', 'RTX 3090', 24, 0.22, 'Community Cloud', 'medium'),
  offer('NVIDIA GeForce RTX 4090@community', 'RTX 4090', 24, 0.34, 'Community Cloud', 'low'),
  offer('NVIDIA GeForce RTX 4090@secure', 'RTX 4090', 24, 0.69, 'Secure Cloud', 'high'),
  offer('NVIDIA L40S@secure', 'L40S', 48, 0.86, 'Secure Cloud', 'medium'),
  offer('NVIDIA A100 80GB PCIe@secure', 'A100 PCIe', 80, 1.64, 'Secure Cloud', 'none'),
  offer('NVIDIA H100 80GB HBM3@secure', 'H100 SXM', 80, 2.99, 'Secure Cloud', 'high'),
];

/** A remote server's one offer: the GPU it trains on, not billed. */
export const REMOTE_OFFER: CloudOffer = {
  id: 'gpu:0',
  gpu: 'RTX 3090',
  gpu_count: 1,
  vram_bytes: 24 * GiB,
  price_per_hour: null,
  region: null,
  detail: null,
  availability: 'unknown',
};

/** The secure-cloud RTX 4090 the seeded cloud runs rented. */
export const RTX_4090 = RUNPOD_OFFERS[3]!;

/** The demo's connection (`cc_4d2a`, as on folo-server's mock); `error` is the `cloud-trouble` scenario's revoked key. */
export function seedConnection(t0: number, error: boolean): CloudConnection {
  return {
    id: 'cc_4d2a',
    provider: 'runpod',
    name: 'RunPod',
    created: new Date(t0 - 12 * 24 * HOUR).toISOString(),
    status: error ? 'error' : 'ok',
    account: 'you@example.com',
    balance_usd: error ? null : 42.17,
    error: error ? 'RunPod rejected this API key' : null,
    note: null,
    checked: new Date(t0 - 3 * MIN).toISOString(),
    values: { api_key: '••••a1b2' },
  };
}

/** A secret as a connection shows it: `••••` and its last 4 characters. */
export const secretHint = (v: string) => `••••${v.slice(-4)}`;

/** A cloud run's hand-over around its training (times in ms since the epoch). */
export interface CloudTimeline {
  connectionId: string;
  provider: string;
  offer: CloudOffer;
  maxHours: number;
  /** Its instance (`null` for a provider that launches none) and how the provider knows it. */
  instanceId: string | null;
  instanceName: string;
  providerRef: string;
  /** The launch request. */
  launchedAt: number;
  /** How long each phase before training takes. */
  provisionMs: number;
  bootMs: number;
  syncMs: number;
  /** Bytes the upload sends and the collection copies back. */
  uploadBytes: number;
  weightsBytes: number;
  /** When the worker's run ended before its last epoch (a stop) or the instance went away (`lost`). */
  endAt?: number;
  /** Terminated by hand (`DELETE /api/cloud/instances/{id}`): no weights to collect, the run fails. */
  lost?: boolean;
  /** A termination the provider has not confirmed: the instance stays `terminating` with this error. */
  stuck?: string;
}

/** Collecting the weights and releasing the instance, after training (the run's status waits for both). */
export const COLLECT_MS = 2000;
export const RELEASE_MS = 2000;
/** How long a termination by hand takes to be confirmed. */
const CONFIRM_MS = 1500;
/** The ledger keeps ended instances for 7 days. */
const KEEP_MS = 7 * 24 * HOUR;

/** When the worker starts training: after provisioning, booting and the upload. */
export const trainAt = (c: CloudTimeline) => c.launchedAt + c.provisionMs + c.bootMs + c.syncMs;

/** When training stopped on the worker: its natural end (`trainEnd`), or earlier when it was stopped or lost. */
const stopAt = (c: CloudTimeline, trainEnd: number) => Math.min(trainEnd, c.endAt ?? Infinity);

/** When nothing of the run is left at the provider (`released`). */
export function releasedAt(c: CloudTimeline, trainEnd: number): number {
  const end = stopAt(c, trainEnd);
  if (c.lost) return end;
  return end + (end > trainAt(c) ? COLLECT_MS : 0) + RELEASE_MS;
}

/** The phase at `now`, and how far its transfer is. */
function phaseAt(c: CloudTimeline, trainEnd: number, now: number): { phase: CloudPhase; progress: RunCloud['progress'] } {
  const end = stopAt(c, trainEnd);
  const share = (from: number, ms: number) => Math.max(0, Math.min(1, (now - from) / ms));
  if (now < end) {
    const booted = c.launchedAt + c.provisionMs;
    const synced = booted + c.bootMs;
    if (now < booted) return { phase: 'provisioning', progress: null };
    if (now < synced) return { phase: 'booting', progress: null };
    if (now < synced + c.syncMs)
      return { phase: 'syncing', progress: { done_bytes: Math.round(c.uploadBytes * share(synced, c.syncMs)), total_bytes: c.uploadBytes } };
    return { phase: 'training', progress: null };
  }
  const released = releasedAt(c, trainEnd);
  if (now >= released) return { phase: 'released', progress: null };
  if (now < released - RELEASE_MS) return { phase: 'collecting', progress: { done_bytes: Math.round(c.weightsBytes * share(end, COLLECT_MS)), total_bytes: c.weightsBytes } };
  return { phase: 'releasing', progress: null };
}

/**
 * When `weights/last.folo` was last copied home: after the first epoch, then after the first epoch at least 10 minutes
 * past the copy before, and once more when the weights are collected (docs/CLOUD.md, Checkpoints).
 */
function syncedAt(c: CloudTimeline, trainEnd: number, epochMs: number, now: number): number | null {
  const start = trainAt(c);
  const end = stopAt(c, trainEnd);
  if (!c.lost && now >= releasedAt(c, trainEnd) - RELEASE_MS && end > start) return releasedAt(c, trainEnd) - RELEASE_MS;
  let last: number | null = null;
  for (let t = start + epochMs; t <= Math.min(now, end); t += epochMs) if (last === null || t - last >= 10 * MIN) last = t;
  return last;
}

/** The instance's own end: its release, or the confirmation of a termination by hand; never while stuck. */
const instanceEnd = (c: CloudTimeline, trainEnd: number) => (c.stuck ? null : releasedAt(c, trainEnd) + (c.lost ? CONFIRM_MS : 0));

/** Its cost so far: the offer's price over its lifetime (docs/CLOUD.md: an estimate; the provider's bill counts). */
function costAt(c: CloudTimeline, trainEnd: number, now: number): number | null {
  if (c.offer.price_per_hour === null) return null;
  const end = instanceEnd(c, trainEnd);
  const until = end === null ? now : Math.min(now, end);
  return Math.round(c.offer.price_per_hour * (Math.max(0, until - c.launchedAt) / HOUR) * 100) / 100;
}

/** The phase the hand-over was in when the run ended: the collection after training, else the one it stopped or was lost in. */
function endedIn(c: CloudTimeline, trainEnd: number): CloudPhase {
  const end = stopAt(c, trainEnd);
  return !c.lost && end > trainAt(c) ? 'collecting' : phaseAt(c, trainEnd, end - 1).phase;
}

const iso = (t: number | null) => (t === null ? null : new Date(t).toISOString());

/** `Run.cloud` at `now`; `trainEnd` is when training ends on the worker if nothing stops it, `epochMs` an epoch's length. */
export function runCloudAt(c: CloudTimeline, trainEnd: number, epochMs: number, now: number): RunCloud {
  const { phase, progress } = phaseAt(c, trainEnd, now);
  const exists = now >= c.launchedAt + c.provisionMs;
  return {
    connection_id: c.connectionId,
    provider: c.provider,
    offer: c.offer,
    instance_id: c.instanceId && exists ? c.instanceId : null,
    phase,
    progress,
    max_hours: c.maxHours,
    deadline: exists ? iso(c.launchedAt + c.maxHours * HOUR) : null,
    cost_usd: costAt(c, trainEnd, now),
    synced: iso(syncedAt(c, trainEnd, epochMs, now)),
    ended_in: phase === 'releasing' || phase === 'released' ? endedIn(c, trainEnd) : null,
  };
}

/** The run's instance at `now` (`GET /api/cloud/instances`), or null for a provider that launches none or one long gone. */
export function instanceAt(c: CloudTimeline, runId: string, trainEnd: number, now: number): CloudInstance | null {
  if (!c.instanceId) return null;
  const end = instanceEnd(c, trainEnd);
  if (end !== null && now - end > KEEP_MS) return null;
  const { phase } = phaseAt(c, trainEnd, now);
  const status: CloudInstanceStatus = c.stuck
    ? phase === 'released'
      ? 'terminating'
      : 'running'
    : end !== null && now >= end
      ? 'terminated'
      : c.lost || phase === 'releasing' || phase === 'released'
        ? 'terminating'
        : phase === 'provisioning'
          ? 'launching'
          : phase === 'booting'
            ? 'starting'
            : 'running';
  return {
    id: c.instanceId,
    connection_id: c.connectionId,
    provider: c.provider,
    provider_ref: status === 'launching' ? null : c.providerRef,
    name: c.instanceName,
    run_id: runId,
    offer: c.offer,
    status,
    created: new Date(c.launchedAt).toISOString(),
    ended: status === 'terminated' && end !== null ? new Date(end).toISOString() : null,
    deadline: status === 'launching' ? null : new Date(c.launchedAt + c.maxHours * HOUR).toISOString(),
    cost_usd: costAt(c, trainEnd, now),
    error: c.stuck && status === 'terminating' ? c.stuck : null,
  };
}

/**
 * The hand-over's lines in the run's log as far as `now`, before and after the trainer's own. After training, the
 * real control plane's words (folo-server's `cloud/executor`): "the run ended on the worker; copying its weights",
 * then "terminating the instance". A run lost with its instance has nothing to copy or terminate, one stopped before
 * it trained nothing to copy, and one on a remote FOLO server (no instance) no instance to terminate.
 */
export function cloudLog(c: CloudTimeline, trainEnd: number, now: number, dataset: string): LogLine[] {
  const booted = c.launchedAt + c.provisionMs;
  const start = trainAt(c);
  const end = stopAt(c, trainEnd);
  const released = releasedAt(c, trainEnd);
  const price = c.offer.price_per_hour === null ? 'not billed' : `$${c.offer.price_per_hour.toFixed(2)}/h`;
  const lines: [number, LogLevel, string][] = [];
  if (c.instanceId) {
    lines.push([c.launchedAt, 'info', `launching ${c.instanceName}: ${c.offer.gpu}, ${c.offer.detail ?? c.offer.id}, ${price}; terminated after ${c.maxHours} h at the latest`]);
    lines.push([booted, 'info', `instance ${c.providerRef} is up; waiting for its worker`]);
  }
  lines.push([booted + c.bootMs, 'info', `worker checked: folo 0.1.0 on ${c.offer.gpu} (vulkan); uploading ${dataset} (${bytes(c.uploadBytes)})`]);
  if (end > start) lines.push([start, 'info', 'upload done; the worker trains']);
  if (c.lost) lines.push([end, 'error', `instance ${c.instanceName} was terminated before the run ended`]);
  else {
    if (end > start) {
      // 1 ms after `end`, where the trainer's final epoch line is stamped: the log sorts by time, and this follows it.
      lines.push([end + 1, 'info', 'the run ended on the worker; copying its weights']);
      lines.push([end + COLLECT_MS, 'info', `copied best.folo and last.folo (${bytes(c.weightsBytes)})`]);
    }
    if (c.instanceId) {
      lines.push([released - RELEASE_MS, 'info', 'terminating the instance']);
      lines.push([released, c.stuck ? 'warn' : 'info', c.stuck ? `could not confirm that ${c.instanceName} is terminated: ${c.stuck}; retrying every 30 s` : `terminated ${c.instanceName}`]);
    }
  }
  return lines.filter(([t]) => t <= now).map(([t, level, message]) => ({ t: new Date(t).toISOString(), level, message }));
}
