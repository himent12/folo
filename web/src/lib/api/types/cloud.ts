// Cloud compute: providers, connections, offers, instances and cloud runs (docs/API.md, Cloud compute).

import type { Time } from './time';

// ---------- Cloud compute (docs/API.md, Cloud compute; docs/CLOUD.md); servers list `cloud` in `features` ----------

/** One field of a connection form (`CloudProvider.fields`, `ConnectCloud.values`). */
export interface CloudField {
  /** Key in `ConnectCloud.values`, e.g. `api_key`. */
  key: string;
  label: string;
  /** A credential: entered masked, never returned (connections show only a hint of it). */
  secret: boolean;
  required: boolean;
  placeholder: string | null;
  help: string | null;
}

/** `GET /api/cloud/providers`: a kind of compute, with the fields a connection to it takes. */
export interface CloudProvider {
  /** `runpod`, `remote`. */
  id: string;
  name: string;
  /** One sentence on what it is. */
  about: string;
  /** Where its credentials are made, if anywhere. */
  docs_url: string | null;
  /** In form order. */
  fields: CloudField[];
  /** It launches an instance per run and terminates it afterwards (`false`: a machine that already runs). */
  provisions: boolean;
  /** Its instances terminate themselves after a run's deadline, also while this server is off. */
  self_terminates: boolean;
}

/** `POST /api/cloud/connections` body. `values` holds credentials: never log, store or show them. */
export interface ConnectCloud {
  provider: string;
  /** Empty means the provider's name (with ` 2`, ` 3`, ... when taken). */
  name?: string;
  /** A value per `CloudField.key`; empty values count as missing. */
  values: Record<string, string>;
}

export type CloudConnectionStatus = 'ok' | 'error';

/** A provider with credentials (`GET /api/cloud/connections`). */
export interface CloudConnection {
  /** e.g. `cc_1f2e`. */
  id: string;
  provider: string;
  name: string;
  created: Time;
  /** Result of the last check. */
  status: CloudConnectionStatus;
  /** Who the credentials belong to (an email, a host); `null` when unknown. */
  account: string | null;
  /** In US dollars, where the provider reports one. */
  balance_usd: number | null;
  /** Why the last check failed (`status: error`). */
  error: string | null;
  /** Something to know about a working connection (an unencrypted `http://` URL). */
  note: string | null;
  /** Time of the last check. */
  checked: Time | null;
  /** Fields that are not secret as entered; secrets as a hint (`••••` and their last 4 characters). */
  values: Record<string, string>;
}

/** How likely an offer is to start now (`none`: launching would be refused). */
export type CloudAvailability = 'high' | 'medium' | 'low' | 'none' | 'unknown';

/** Something a connection can run on: a GPU type with its price (`GET /api/cloud/connections/{id}/offers`). */
export interface CloudOffer {
  /** Opaque, unique within the connection (`Compute.offer_id`). */
  id: string;
  /** e.g. `RTX 4090`. */
  gpu: string;
  /** GPUs of the instance (a run trains on one). */
  gpu_count: number;
  /** Memory of one GPU in bytes (0 when unknown). */
  vram_bytes: number;
  /** US dollars per hour; `null` for compute that is not billed (your own machine). */
  price_per_hour: number | null;
  region: string | null;
  /** A qualifier of the offer, e.g. `Secure Cloud`. */
  detail: string | null;
  availability: CloudAvailability;
}

export interface CloudOfferList {
  /** Cheapest first (offers without a price first). */
  offers: CloudOffer[];
  /** When the provider was asked (answers are cached for a minute). */
  fetched: Time;
}

/** Where an instance is in its life; alive until `terminated` or `lost`. */
export type CloudInstanceStatus = 'launching' | 'starting' | 'running' | 'terminating' | 'terminated' | 'lost';

/** A machine FOLO launched at a provider (`GET /api/cloud/instances`). */
export interface CloudInstance {
  /** e.g. `ci_8a2f`. */
  id: string;
  connection_id: string;
  provider: string;
  /** The provider's own id (to find it in its console); `null` while launching. */
  provider_ref: string | null;
  /** Its name at the provider, `folo-<run id>-<nonce>`. */
  name: string;
  run_id: string | null;
  offer: CloudOffer;
  status: CloudInstanceStatus;
  /** Time of the launch request. */
  created: Time;
  /** When it was confirmed gone. */
  ended: Time | null;
  /** When it is terminated at the latest (the run's `max_hours`). */
  deadline: Time | null;
  /** Estimated cost so far in US dollars; `null` without a price. */
  cost_usd: number | null;
  /** The last failed provider request about it (a termination being retried). */
  error: string | null;
}

/** `TrainConfig.compute`: train on a cloud connection instead of this machine. */
export interface Compute {
  connection_id: string;
  offer_id: string;
  /** The run is stopped and its instance terminated after this many hours at the latest (0.25 to 720; default 12). */
  max_hours?: number;
  /** Disk of the instance in GB; missing sizes it for the dataset (twice its size plus 20 GB). */
  disk_gb?: number;
}

/**
 * Where a cloud run is in its lifecycle (docs/CLOUD.md), in order. A newer server may send a phase this client does not
 * know: read the field as a string and show an unknown one as its raw word.
 */
export type CloudPhase = 'provisioning' | 'booting' | 'syncing' | 'training' | 'collecting' | 'releasing' | 'released';

/** Bytes moved of a transfer (`RunCloud.progress`). */
export interface TransferProgress {
  done_bytes: number;
  /** 0 when unknown. */
  total_bytes: number;
}

/** `Run.cloud` and the SSE `cloud` event: where a cloud run executes and how far it is. */
export interface RunCloud {
  connection_id: string;
  provider: string;
  /** The offer as it was when the run started. */
  offer: CloudOffer;
  /** `null` before the launch and for providers that launch none. */
  instance_id: string | null;
  /** A `CloudPhase`, or a newer server's phase this client does not know. */
  phase: CloudPhase | (string & {});
  /** Progress of the phase's transfer (`syncing`, `collecting`), else `null`. */
  progress: TransferProgress | null;
  max_hours: number;
  /** When the time limit ends; `null` until the instance exists. */
  deadline: Time | null;
  /** Estimated cost so far in US dollars (final once `released`); `null` without a price. */
  cost_usd: number | null;
  /** When `weights/last.folo` was last copied to this machine. */
  synced: Time | null;
  /**
   * The phase it was in when the run ended (set at `releasing`): where a failed run failed, and the phases after it it
   * never reached. `null` before; absent from servers before it.
   */
  ended_in?: CloudPhase | (string & {}) | null;
}

/** `RunEstimate.cost`: what a cloud run would cost. */
export interface CostEstimate {
  price_per_hour: number;
  /** `price_per_hour` × `est_total_s` (training only: booting and the upload come on top). */
  est_usd: number;
  est_usd_low: number;
  est_usd_high: number;
  /** The most the run can cost: `price_per_hour` × `compute.max_hours`. */
  max_usd: number;
}
