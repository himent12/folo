// @vitest-environment happy-dom
// The offline state (shell F1, F6): a failed request or a dropped stream is a suspicion one health probe confirms;
// while offline the probe repeats, and the first answer brings Studio back, with what failed meanwhile refetched.
import { createRoot, flush } from 'fw';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import ConnectionStatus from '$lib/shell/ConnectionStatus';
import OfflineNotice from '$lib/shell/OfflineNotice';
import StatusBadge from '$lib/ui/StatusBadge';
import { render } from '../../../framework/testing';
import { hub } from './index';
import { PROBE_DELAYS, Reach } from './reach';
import { Resource } from './resource';
import type { Dataset } from './types';

describe('Reach', () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it('goes offline only when the probe fails too, and back when one answers', async () => {
    let up = true;
    let probes = 0;
    const reach = new Reach();
    reach.reset(async () => {
      probes++;
      if (!up) throw new Error('down');
    });
    reach.failed();
    await vi.advanceTimersByTimeAsync(0);
    // A blip: the probe answered, nothing to say.
    expect([reach.offline, probes, reach.back]).toEqual([false, 1, 0]);

    up = false;
    reach.failed();
    reach.failed(); // a second failure while the probe is out adds nothing
    await vi.advanceTimersByTimeAsync(0);
    expect([reach.offline, probes]).toEqual([true, 2]);
    // Offline: further failures wait for the probes, which repeat on their own.
    reach.failed();
    await vi.advanceTimersByTimeAsync(PROBE_DELAYS[0]!);
    expect(probes).toBe(3);
    await vi.advanceTimersByTimeAsync(PROBE_DELAYS[1]!);
    expect(probes).toBe(4);

    up = true;
    await vi.advanceTimersByTimeAsync(PROBE_DELAYS[2]!);
    expect([reach.offline, probes, reach.back]).toEqual([false, 5, 1]);
    // Answering, it does not poll.
    await vi.advanceTimersByTimeAsync(60_000);
    expect(probes).toBe(5);
  });

  it('comes back at once when any request answers, and stops probing', async () => {
    let probes = 0;
    const reach = new Reach();
    reach.reset(async () => {
      probes++;
      throw new Error('down');
    });
    reach.failed();
    await vi.advanceTimersByTimeAsync(0);
    expect(reach.offline).toBe(true);
    reach.ok();
    expect([reach.offline, reach.back]).toEqual([false, 1]);
    await vi.advanceTimersByTimeAsync(60_000);
    expect(probes).toBe(1);
  });

  it('drops a suspicion when anything answers while its probe is out (review R1)', async () => {
    let reject!: (e: Error) => void;
    const reach = new Reach();
    reach.reset(() => new Promise((_, no) => (reject = no)));
    reach.failed();
    // A newer request (or the reopened stream) answered: the probe that ends in a timeout afterwards proves nothing.
    reach.ok();
    reject(new Error('timed out'));
    await vi.advanceTimersByTimeAsync(0);
    expect([reach.offline, reach.back]).toEqual([false, 0]);
    // The next failure is a suspicion of its own, with a probe of its own.
    reach.failed();
    reject(new Error('down'));
    await vi.advanceTimersByTimeAsync(0);
    expect(reach.offline).toBe(true);
  });

  it('has nothing to probe without a server (the mock), and a new server forgets the outage', async () => {
    const reach = new Reach();
    reach.failed();
    await vi.advanceTimersByTimeAsync(0);
    expect(reach.offline).toBe(false);
    reach.reset(() => Promise.reject(new Error('down')));
    reach.failed();
    await vi.advanceTimersByTimeAsync(0);
    expect(reach.offline).toBe(true);
    reach.reset(null);
    expect([reach.offline, reach.back]).toEqual([false, 0]);
  });
});

describe('hub.offline with the server', () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.useRealTimers();
  });

  it('stays online when the health probe gets an HTTP error: the server answered (review R2)', async () => {
    let health = 0;
    vi.stubGlobal('fetch', async (url: string) => {
      if (!url.endsWith('/api/health')) throw new TypeError('Failed to fetch');
      // The connect's check answers; the probe's gets a 500, then a 200 that is not JSON.
      if (health++ === 0) return Response.json({ status: 'ok', version: '0.1.0', api: 'v1', mock: false });
      return health === 2 ? Response.json({ error: { code: 'internal', message: 'boom' } }, { status: 500 }) : new Response('<html>');
    });
    await hub.connect();
    await expect(hub.api.datasets()).rejects.toThrow();
    await vi.advanceTimersByTimeAsync(0);
    expect([health, hub.offline]).toEqual([2, false]);
    await expect(hub.api.datasets()).rejects.toThrow();
    await vi.advanceTimersByTimeAsync(0);
    expect([health, hub.offline]).toEqual([3, false]);
  });

  it('turns on when requests stop reaching it, the shell says so, and a failed list refetches by itself when it is back', async () => {
    let up = true;
    let lists = 0;
    vi.stubGlobal('fetch', async (url: string) => {
      if (!up) throw new TypeError('Failed to fetch');
      if (url.endsWith('/api/health')) return Response.json({ status: 'ok', version: '0.1.0', api: 'v1', mock: false });
      lists++;
      return Response.json({ datasets: [] });
    });
    await hub.connect();
    expect([hub.mode, hub.offline]).toEqual(['live', false]);
    const shell = document.createElement('div');
    const view = render(
      () => (
        <>
          <ConnectionStatus collapsed={false} />
          <OfflineNotice />
          <StatusBadge run={{ status: 'running' }} plain />
          <StatusBadge run={{ status: 'completed' }} plain />
        </>
      ),
      shell,
    );
    const said = () => [shell.querySelector('.conn .label')!.textContent!.trim(), shell.querySelector('[role=status]')!.textContent!.trim()];
    const held = () => [...shell.querySelectorAll('.badge')].map((b) => b.classList.contains('held'));
    expect(said()).toEqual(['Connected', '']);
    expect(held()).toEqual([false, false]);

    up = false;
    let r!: Resource<Dataset[]>;
    const stop = createRoot((dispose) => {
      r = new Resource((s) => hub.api.datasets(s));
      return dispose;
    });
    flush();
    await vi.advanceTimersByTimeAsync(0);
    expect(r.error).toBeDefined();
    expect(hub.offline).toBe(true);
    flush();
    expect(said()).toEqual(['Server offline', "Can't reach folo serve. Retrying…"]);
    // A run's live status is only last known now: neutral and still; a finished one stays as it is.
    expect(held()).toEqual([true, false]);

    up = true;
    await vi.advanceTimersByTimeAsync(PROBE_DELAYS[0]!);
    flush();
    await vi.advanceTimersByTimeAsync(0);
    expect(hub.offline).toBe(false);
    expect(r.error).toBeUndefined();
    expect(r.data).toEqual([]);
    expect(lists).toBe(1);
    expect(said()).toEqual(['Connected', '']);
    expect(held()).toEqual([false, false]);
    stop();
    view.unmount();
  });
});
