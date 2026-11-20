// perf.mjs's network accounting (./perfWire.mjs) on made-up CDP events: counts cut at a moment, and the warm-up's wait.
import { EventEmitter } from 'node:events';
import { describe, expect, it } from 'vitest';
import { settleRequests, wireOf } from './perfWire.mjs';

/** A CDP session that is an event emitter, and a clock that moves only when told. */
function fake() {
  const cdp = new EventEmitter();
  let t = 0;
  return { cdp, now: () => t, advance: (/** @type {number} */ ms) => void (t += ms) };
}
const sent = (/** @type {string} */ requestId, /** @type {number} */ timestamp, url = `http://x/${requestId}`, /** @type {{ encodedDataLength: number } | undefined} */ redirectResponse = undefined) => ({
  requestId,
  timestamp,
  request: { url },
  ...(redirectResponse ? { redirectResponse } : {}),
});
const response = (/** @type {string} */ requestId, /** @type {number} */ timestamp, /** @type {number} */ encodedDataLength, mimeType = 'text/javascript') => ({
  requestId,
  timestamp,
  response: { mimeType, encodedDataLength },
});

describe('what went over the network by a moment', () => {
  it('counts headers, partial bodies, redirect hops and unanswered requests by then, nothing after', () => {
    const { cdp } = fake();
    const w = wireOf(cdp);
    const emit = (/** @type {string} */ name, /** @type {any} */ e) => cdp.emit(`Network.${name}`, e);
    // The document: Chrome reports no chunks of its body, only its total when it finishes.
    emit('requestWillBeSent', sent('doc', 0.9));
    emit('responseReceived', response('doc', 1.0, 336, 'text/html'));
    emit('loadingFinished', { requestId: 'doc', timestamp: 1.1, encodedDataLength: 2965 });
    // A script whose body is still arriving at the mark (2.0): its headers and what had arrived.
    emit('requestWillBeSent', sent('a', 1.0));
    emit('responseReceived', response('a', 1.1, 300));
    emit('dataReceived', { requestId: 'a', timestamp: 1.2, encodedDataLength: 1000 });
    emit('dataReceived', { requestId: 'a', timestamp: 2.5, encodedDataLength: 500 });
    // Headers in, body pending.
    emit('requestWillBeSent', sent('h', 1.0));
    emit('responseReceived', response('h', 1.9, 250));
    // Waiting for its first byte.
    emit('requestWillBeSent', sent('b', 1.5));
    // A redirect: two hops, each a request, the first hop's response counted too.
    emit('requestWillBeSent', sent('r', 1.0));
    emit('requestWillBeSent', sent('r', 1.2, 'http://x/r2', { encodedDataLength: 200 }));
    emit('responseReceived', response('r', 1.3, 100));
    emit('dataReceived', { requestId: 'r', timestamp: 1.4, encodedDataLength: 50 });
    emit('loadingFinished', { requestId: 'r', timestamp: 1.5, encodedDataLength: 150 });
    // After the mark, though CDP may say so before the count is taken.
    emit('requestWillBeSent', sent('late', 2.1));
    emit('responseReceived', response('late', 2.2, 99));
    // Answered by the cache, and not over HTTP: never on the wire.
    emit('requestWillBeSent', sent('c', 1.0));
    emit('requestServedFromCache', { requestId: 'c' });
    emit('responseReceived', response('c', 1.0, 999));
    emit('requestWillBeSent', sent('d', 1.0, 'data:image/png;base64,AA'));
    const got = w.upTo(2.0);
    expect(got.urls).toEqual(['http://x/doc', 'http://x/a', 'http://x/h', 'http://x/b', 'http://x/r', 'http://x/r2']);
    expect(got.bytes).toBe(2965 + 300 + 1000 + 250 + 200 + 100 + 50);
    expect(w.upTo(3).bytes).toBe(got.bytes + 500 + 99);
  });

  it('counts a redirect hop that went over the network when the cache answers the next one', () => {
    const { cdp } = fake();
    const w = wireOf(cdp);
    const emit = (/** @type {string} */ name, /** @type {any} */ e) => cdp.emit(`Network.${name}`, e);
    // A 302 from the network, then the final response from the disk cache, under the same request id.
    emit('requestWillBeSent', sent('r', 1.0, 'http://x/a'));
    emit('requestWillBeSent', sent('r', 1.1, 'http://x/b', { encodedDataLength: 180 }));
    emit('requestServedFromCache', { requestId: 'r' });
    emit('responseReceived', { requestId: 'r', timestamp: 1.2, response: { mimeType: 'text/javascript', encodedDataLength: 0, fromDiskCache: true } });
    emit('loadingFinished', { requestId: 'r', timestamp: 1.3, encodedDataLength: 0 });
    expect(w.upTo(2)).toEqual({ bytes: 180, requests: 1, urls: ['http://x/a'] });
    // And the other way round: a cached redirect, then the final hop from the network.
    emit('requestWillBeSent', sent('s', 1.0, 'http://x/c'));
    emit('requestWillBeSent', sent('s', 1.1, 'http://x/d', /** @type {any} */ ({ encodedDataLength: 0, fromDiskCache: true })));
    emit('responseReceived', response('s', 1.2, 120));
    emit('loadingFinished', { requestId: 's', timestamp: 1.3, encodedDataLength: 120 });
    expect(w.upTo(2)).toEqual({ bytes: 180 + 120, requests: 2, urls: ['http://x/a', 'http://x/d'] });
  });

  it('leaves out what came before a moment (a document before, its events delivered late)', () => {
    const { cdp } = fake();
    const w = wireOf(cdp);
    const emit = (/** @type {string} */ name, /** @type {any} */ e) => cdp.emit(`Network.${name}`, e);
    w.reset();
    // Delivered after the reset, but of the load before (navigation of this one at 5.0).
    emit('requestWillBeSent', sent('old', 4.0));
    emit('responseReceived', response('old', 4.1, 700));
    emit('requestWillBeSent', sent('doc', 5.1));
    emit('responseReceived', response('doc', 5.2, 300, 'text/html'));
    expect(w.upTo(6, 5.0)).toEqual({ bytes: 300, requests: 1, urls: ['http://x/doc'] });
  });
});

describe('a document before, by its loader', () => {
  it("leaves out an event stream of the document before that receives after this navigation started (issue #309)", () => {
    const { cdp } = fake();
    const w = wireOf(cdp);
    const emit = (/** @type {string} */ name, /** @type {any} */ e) => cdp.emit(`Network.${name}`, e);
    emit('requestWillBeSent', { ...sent('L1', 1.0), loaderId: 'L1' });
    emit('requestWillBeSent', { ...sent('s', 1.5, 'http://x/api/runs/run_1/events'), loaderId: 'L1' });
    emit('responseReceived', response('s', 1.6, 200, 'text/event-stream'));
    w.reset();
    // This navigation starts at 5.0 and commits at 5.3; the old document's stream still receives at 5.2.
    emit('requestWillBeSent', { ...sent('L2', 5.1), loaderId: 'L2' });
    emit('dataReceived', { requestId: 's', timestamp: 5.2, encodedDataLength: 400 });
    emit('responseReceived', response('L2', 5.3, 300, 'text/html'));
    // A request without a loader (a worker's, say) is this document's.
    emit('requestWillBeSent', sent('w', 5.4));
    expect(w.upTo(6, 5.0)).toEqual({ bytes: 300, requests: 2, urls: ['http://x/L2', 'http://x/w'] });
  });
});

describe('the warm-up waits for the network to go quiet', () => {
  it('waits the whole quiet window after a request that started and ended between two looks', async () => {
    const f = fake();
    const w = wireOf(f.cdp, f.now);
    // At 490 ms a request goes out and finishes within 25 ms, never seen in flight.
    let fired = false;
    const sleep = async (/** @type {number} */ ms) => {
      f.advance(ms);
      if (!fired && f.now() >= 490) {
        fired = true;
        f.cdp.emit('Network.requestWillBeSent', sent('x', 1));
        f.cdp.emit('Network.loadingFinished', { requestId: 'x', timestamp: 1.01, encodedDataLength: 10 });
      }
    };
    await settleRequests(w, sleep, 500, 15_000, f.now);
    expect(fired).toBe(true);
    expect(f.now()).toBeGreaterThanOrEqual(490 + 500);
  });

  it('waits for a request still out, and not for an event stream', async () => {
    const f = fake();
    const w = wireOf(f.cdp, f.now);
    f.cdp.emit('Network.requestWillBeSent', sent('s', 0, 'http://x/api/runs/run_1/events'));
    f.cdp.emit('Network.requestWillBeSent', sent('img', 0, 'http://x/api/img.jpg'));
    const sleep = async (/** @type {number} */ ms) => {
      f.advance(ms);
      // The stream keeps sending; the image finishes at 1 s.
      f.cdp.emit('Network.dataReceived', { requestId: 's', timestamp: f.now() / 1000, encodedDataLength: 5 });
      if (f.now() === 1000) f.cdp.emit('Network.loadingFinished', { requestId: 'img', timestamp: 1, encodedDataLength: 10 });
    };
    await settleRequests(w, sleep, 500, 15_000, f.now);
    expect(f.now()).toBe(1500);
  });

  it('says what is still out when it never goes quiet', async () => {
    const f = fake();
    const w = wireOf(f.cdp, f.now);
    f.cdp.emit('Network.requestWillBeSent', sent('slow', 0, 'http://x/api/slow'));
    await expect(settleRequests(w, async (ms) => f.advance(ms), 500, 2000, f.now)).rejects.toThrow('/api/slow');
  });
});
