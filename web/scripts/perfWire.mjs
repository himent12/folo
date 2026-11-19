// What a page sent and received over the network, for scripts/perf.mjs: from CDP's Network events, each counted at
// its own time (CDP's monotonic clock, in seconds), so a count cut at a moment of the page's (its content showing) holds
// what happened by then and nothing after, however late CDP delivered the events.

/** An event stream (a live run's, a labeling job's, a benchmark's): it never finishes, so waits leave it out. */
export const isStream = (/** @type {string} */ u) => {
  try {
    return /\/events$/.test(new URL(u).pathname);
  } catch {
    return false;
  }
};

/** @typedef {{ id: string, hop: number, ts: number, url: string, kind: 'sent' | 'bytes', n: number }} WireEvent */
/**
 * @typedef {object} Wire
 * @property {Set<string>} cached the hops the cache answered (never on the network), as `<request id>#<hop>`: a redirect's
 *   hops share the request's id, and one can come from the network and the next from the cache
 * @property {WireEvent[]} events every request sent and every byte received, in order
 * @property {Map<string, string>} inflight requests not finished (id → URL)
 * @property {Set<string>} streams requests that are event streams
 * @property {number} last when (`now()`) a request that is not a stream last sent or received anything
 * @property {(t: number, from?: number) => { bytes: number, requests: number, urls: string[] }} upTo what went over the
 *   network by `t` (and from `from`: an event of the document before, delivered late, is not this one's)
 * @property {() => void} reset forgets what came before (a new document)
 */

/**
 * Follows `cdp`'s Network events (Network.enable sent). `upTo(t)`: the HTTP requests sent by `t` (a redirect's every hop
 * a request of its own, one still waiting for its first byte too; none the cache answered, revalidations counted), with
 * their URLs, and the bytes received by `t` as Chrome counts them on the wire (encoded: every response's headers and
 * body, a redirect hop's too; a response still arriving with what had arrived). Chrome reports no chunks of a document's
 * body: it counts when the document finishes.
 * @param {{ on(event: string, f: (e: any) => void): unknown }} cdp
 * @param {() => number} now the clock of `last` (tests)
 * @returns {Wire}
 */
export function wireOf(cdp, now = Date.now) {
  /** @type {Wire} */
  const wire = {
    cached: new Set(),
    events: [],
    inflight: new Map(),
    streams: new Set(),
    last: now(),
    upTo(t, from = -Infinity) {
      let bytes = 0;
      /** @type {string[]} */
      const urls = [];
      for (const e of this.events) {
        if (e.ts > t || e.ts < from || this.cached.has(`${e.id}#${e.hop}`)) continue;
        if (e.kind === 'sent') urls.push(e.url);
        else bytes += e.n;
      }
      return { bytes, requests: urls.length, urls };
    },
    reset() {
      this.cached.clear();
      this.events.length = 0;
    },
  };
  /** @type {Map<string, string>} */
  const urlOf = new Map();
  /** The hop each request is on (0, then one more for each redirect). */
  /** @type {Map<string, number>} */
  const hopOf = new Map();
  const hopKey = (/** @type {string} */ id) => `${id}#${hopOf.get(id) ?? 0}`;
  const touch = (/** @type {string} */ id) => {
    const u = urlOf.get(id);
    if (u === undefined || !(wire.streams.has(id) || isStream(u))) wire.last = now();
  };
  const push = (/** @type {string} */ id, /** @type {number} */ ts, /** @type {'sent' | 'bytes'} */ kind, /** @type {number} */ n) =>
    wire.events.push({ id, hop: hopOf.get(id) ?? 0, ts, url: urlOf.get(id) ?? '', kind, n });
  cdp.on('Network.requestWillBeSent', (e) => {
    if (!/^https?:/.test(e.request.url)) return;
    // A redirect: the hop's response (headers and body) came in, and the next hop goes out under the same id.
    if (e.redirectResponse) {
      const r = e.redirectResponse;
      if (r.fromDiskCache || r.fromPrefetchCache || r.fromServiceWorker) wire.cached.add(hopKey(e.requestId));
      push(e.requestId, e.timestamp, 'bytes', r.encodedDataLength ?? 0);
      hopOf.set(e.requestId, (hopOf.get(e.requestId) ?? 0) + 1);
    }
    urlOf.set(e.requestId, e.request.url);
    wire.inflight.set(e.requestId, e.request.url);
    touch(e.requestId);
    push(e.requestId, e.timestamp, 'sent', 1);
  });
  cdp.on('Network.requestServedFromCache', (e) => void wire.cached.add(hopKey(e.requestId)));
  // The response's headers: what had been received of the request by then.
  cdp.on('Network.responseReceived', (e) => {
    if (!urlOf.has(e.requestId)) return;
    if (/event-stream/.test(e.response.mimeType)) wire.streams.add(e.requestId);
    if (e.response.fromDiskCache || e.response.fromPrefetchCache || e.response.fromServiceWorker) wire.cached.add(hopKey(e.requestId));
    touch(e.requestId);
    push(e.requestId, e.timestamp, 'bytes', e.response.encodedDataLength ?? 0);
  });
  cdp.on('Network.dataReceived', (e) => {
    if (!urlOf.has(e.requestId)) return;
    touch(e.requestId);
    push(e.requestId, e.timestamp, 'bytes', e.encodedDataLength);
  });
  // Finished: what loadingFinished counts beyond the headers and the chunks reported so far: nothing for a subresource
  // (checked on Chrome 157: the two agree to the byte), the body of the document itself.
  cdp.on('Network.loadingFinished', (e) => {
    if (!urlOf.has(e.requestId)) return;
    touch(e.requestId);
    wire.inflight.delete(e.requestId);
    let had = 0;
    let hop = 0;
    for (const x of wire.events) {
      if (x.id !== e.requestId) continue;
      // loadingFinished's total is the last hop's: the redirect hops' bytes are not in it.
      if (x.kind === 'sent') hop = had;
      else had += x.n;
    }
    const more = e.encodedDataLength - (had - hop);
    if (more > 0) push(e.requestId, e.timestamp, 'bytes', more);
  });
  cdp.on('Network.loadingFailed', (e) => {
    if (!urlOf.has(e.requestId)) return;
    touch(e.requestId);
    wire.inflight.delete(e.requestId);
  });
  return wire;
}

/**
 * Waits until no request but the event streams (a live run's, which never finish) is out and none has sent or received
 * anything for `quiet` ms, at most `max` ms; throws, naming what is still out, when it never goes quiet. The activity
 * comes from the events themselves (`wire.last`), so a request that starts and ends between two looks still counts.
 * @param {Pick<Wire, 'inflight' | 'streams' | 'last'>} wire
 * @param {(ms: number) => Promise<unknown>} sleep
 * @param {() => number} now the clock of `wire.last`
 */
export async function settleRequests(wire, sleep, quiet = 500, max = 15_000, now = Date.now) {
  const out = () => [...wire.inflight].filter(([id, u]) => !wire.streams.has(id) && !isStream(u));
  const end = now() + max;
  while (now() < end) {
    if (!out().length && now() - wire.last >= quiet) return;
    await sleep(25);
  }
  throw new Error(`the requests never went quiet within ${max / 1000} s: ${out().slice(0, 5).map(([, u]) => new URL(u).pathname).join(', ')}`);
}
