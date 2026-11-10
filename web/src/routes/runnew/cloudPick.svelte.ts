// New run's "Run on" (docs/CLOUD.md): this machine or a cloud connection, the connection's offers with one preselected,
// and the run's time limit, as the config's `compute`. Only on a server with the `cloud` feature: without it nothing
// here asks the server anything and no config carries `compute`.

import { ApiError, hub } from '$lib/api/index.svelte';
import { Resource } from '$lib/api/resource.svelte';
import type { CloudConnection, CloudProvider, Compute, SystemInfo } from '$lib/api/types';
import { DEFAULT_HOURS, defaultMaxHours, hasCloud, MAX_HOURS, MIN_HOURS, pickOffer, providerName } from '$lib/cloud';
import { sentence } from '$lib/format';

/** "Run on" when the run trains here. */
export const LOCAL = 'local';

/** A control of Run on that Start's answer can be about. */
export type CloudField = 'runon' | 'offer' | 'hours';

export class CloudPick {
  /** `LOCAL` or a connection's id. */
  target = $state<string>(LOCAL);
  offerId = $state<string | null>(null);
  /** Picked by hand: the preselection no longer moves it. */
  offerChosen = $state(false);
  hours = $state<number | null>(DEFAULT_HOURS);
  /** Set by hand: the default no longer follows the estimate. */
  hoursEdited = $state(false);
  /** Start's refusal at the control it is about, while that control still holds the value it refused. */
  refused = $state.raw<{ field: CloudField; value: string; message: string } | null>(null);
  /** Why a copied cloud run's connection or GPU could not be preselected (it trains here instead). */
  lost = $state<string | null>(null);

  readonly connections: Resource<CloudConnection[] | null>;
  readonly providers: Resource<CloudProvider[] | null>;
  readonly offers;
  /** The next offers request asks the provider again instead of the server's cached answer. */
  #fresh = false;
  /** A copied run's choice, held until the lists can say whether it still exists. */
  #wanted = $state.raw<{ compute: Compute; gpu: string } | null>(null);

  constructor(system: () => SystemInfo | undefined) {
    const on = () => hasCloud(system());
    this.connections = new Resource((s) => (on() ? hub.api.cloudConnections(s) : Promise.resolve(null)));
    this.providers = new Resource((s) => (on() ? hub.api.cloudProviders(s) : Promise.resolve(null)));
    this.offers = new Resource(async (s) => {
      const id = this.target;
      if (!on() || id === LOCAL) return null;
      const fresh = this.#fresh;
      this.#fresh = false;
      return { id, ...(await hub.api.cloudOffers(id, s, fresh)) };
    });

    // A copied run's connection and GPU, once the lists say they still exist; else this machine, and a note says why.
    $effect(() => {
      const want = this.#wanted;
      const list = this.connections.data;
      if (!want || !list) return;
      const conn = list.find((c) => c.id === want.compute.connection_id);
      if (!conn || conn.status === 'error') {
        this.#drop(`The run’s cloud connection is ${conn ? 'not working' : 'gone'}, so this one trains on this machine.`);
        return;
      }
      const got = this.offers.data;
      if (!got || got.id !== conn.id) return;
      if (got.offers.some((o) => o.id === want.compute.offer_id)) {
        this.offerId = want.compute.offer_id;
        this.offerChosen = true;
        this.#wanted = null;
      } else this.#drop(`${conn.name} no longer offers the run’s ${want.gpu}, so this one trains on this machine.`);
    });
  }

  /** The chosen connection, while one is chosen. */
  get connection(): CloudConnection | undefined {
    return this.target === LOCAL ? undefined : this.connections.data?.find((c) => c.id === this.target);
  }
  /** Its offers (null while they load, or for this machine). */
  get offerList() {
    const o = this.offers.data;
    return o && o.id === this.target ? o.offers : null;
  }
  get offer() {
    return this.offerList?.find((o) => o.id === this.offerId);
  }
  /** "RunPod": the chosen connection's provider by name. */
  get provider(): string {
    const c = this.connection;
    return c ? providerName(c.provider, this.providers.data ?? undefined) : '';
  }
  /** Whether the provider launches an instance per run (`remote` uses a machine that already runs). */
  get provisions(): boolean {
    const c = this.connection;
    return !!c && !!this.providers.data?.find((p) => p.id === c.provider)?.provisions;
  }
  /** A connection is chosen: the run trains elsewhere. */
  get cloud(): boolean {
    return this.target !== LOCAL;
  }
  /** The config's `compute`, once a connection and one of its offers are chosen. */
  get compute(): Compute | null {
    const o = this.offer;
    return this.cloud && o && this.hours !== null ? { connection_id: this.target, offer_id: o.id, max_hours: this.hours } : null;
  }

  /** What is wrong with the choice, per control (the form's errors). */
  get errors(): Record<CloudField, string | null> {
    const r = this.refused;
    const at = (f: CloudField, value: string) => (r && r.field === f && r.value === value ? r.message : null);
    const h = this.hours;
    return {
      runon: at('runon', this.target),
      offer: this.cloud && this.offerList && !this.offer ? 'Pick a GPU.' : at('offer', this.offerId ?? ''),
      hours: !this.cloud ? null : h === null || !(h >= MIN_HOURS && h <= MAX_HOURS) ? `From ${MIN_HOURS} to ${MAX_HOURS} hours.` : at('hours', String(h)),
    };
  }

  /** Picks a connection (or `LOCAL`): its offers load and the cheapest that fits is preselected again. */
  choose(target: string) {
    this.target = target;
    this.offerId = null;
    this.offerChosen = false;
    this.lost = null;
    this.#wanted = null;
  }

  /** The preselection: the cheapest available offer with `needBytes` of memory, until one is picked by hand. */
  preselect(needBytes: number | null) {
    const list = this.offerList;
    // A copied run's GPU is settled by the lists (above), never by the preselection.
    if (!list || this.#wanted || (this.offerChosen && this.offer)) return;
    this.offerId = pickOffer(list, needBytes);
  }

  /** The default time limit from the estimate's high end, until it is set by hand. */
  follow(highSeconds: number | null) {
    if (!this.hoursEdited) this.hours = defaultMaxHours(highSeconds);
  }

  /** "New run from this config" of a cloud run: its connection, GPU and limit, if they still exist. */
  prefill(compute: Compute | null | undefined, gpu: string) {
    if (!compute) return;
    this.choose(compute.connection_id);
    this.#wanted = { compute, gpu };
    if (compute.max_hours !== undefined) {
      this.hours = compute.max_hours;
      this.hoursEdited = true;
    }
  }

  #drop(why: string) {
    this.choose(LOCAL);
    this.lost = why;
  }

  /**
   * Start's refusal, said at the control it is about (docs/API.md, Cloud runs): an offer the provider no longer has or
   * will not start is the GPU's (and the offers are read again), a connection that is gone or rejected Run on's, a time
   * limit out of range Stop after's. Returns that control, or null when the error is about none of them.
   */
  refuse(err: unknown): CloudField | null {
    if (!(err instanceof ApiError) || !this.cloud) return null;
    const message = sentence(err.message, true);
    let field: CloudField | null = null;
    if (err.code === 'offer_not_found' || err.code === 'cloud_refused') field = 'offer';
    else if (err.code === 'connection_not_found' || err.code === 'cloud_auth_failed') field = 'runon';
    else if (err.code === 'invalid_config' && /max_hours/.test(err.message)) field = 'hours';
    if (!field) return null;
    const value = field === 'offer' ? (this.offerId ?? '') : field === 'runon' ? this.target : String(this.hours);
    this.refused = { field, value, message };
    if (field === 'offer') {
      this.#fresh = true;
      void this.offers.refresh();
    }
    if (field === 'runon') void this.connections.refresh();
    return field;
  }

  /** Reads the offers again (the GPU's Retry). */
  retryOffers() {
    this.#fresh = true;
    void this.offers.refresh();
  }
}
