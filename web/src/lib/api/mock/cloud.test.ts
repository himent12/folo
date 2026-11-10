// The mock's cloud compute (docs/API.md, Mock mode → Cloud): the connection, its offers, the documented errors, and a
// cloud run through every phase on the clock, with its instance and its cost.
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { RunCloud, TrainConfig } from '../types';
import { MockApi } from './index';
import { RTX_4090, RUNPOD_OFFERS } from './cloud';

const NOW = Date.parse('2026-11-08T14:30:00Z');

beforeEach(() => {
  vi.useFakeTimers();
  vi.setSystemTime(NOW);
});
afterEach(() => vi.useRealTimers());

async function cloudConfig(api: MockApi, extra: Partial<TrainConfig['compute']> = {}): Promise<TrainConfig> {
  const base = (await api.run('run_31aa')).config;
  return { ...base, name: 'shapes-cloud', epochs: 10, device: 'auto', compute: { connection_id: 'cc_4d2a', offer_id: RTX_4090.id, max_hours: 2, ...extra } };
}

describe('mock cloud: connections and offers', () => {
  it('has a RunPod connection whose offers include ones short of stock, cheapest first', async () => {
    const api = new MockApi('normal', 0);
    expect((await api.system()).features).toContain('cloud');
    const [conn] = await api.cloudConnections();
    expect(conn).toMatchObject({ id: 'cc_4d2a', status: 'ok' });
    const { offers } = await api.cloudOffers('cc_4d2a');
    expect(offers.map((o) => o.availability)).toContain('none');
    expect(offers.map((o) => o.availability)).toContain('low');
    const prices = offers.map((o) => o.price_per_hour ?? 0);
    expect(prices).toEqual([...prices].sort((a, b) => a - b));
  });

  it('has no connection in cloud-none, no cloud at all in cloud-off', async () => {
    expect(await new MockApi('cloud-none', 0).cloudConnections()).toEqual([]);
    const off = new MockApi('cloud-off', 0);
    expect((await off.system()).features).not.toContain('cloud');
    await expect(off.cloudConnections()).rejects.toMatchObject({ status: 404 });
    expect((await off.runs()).some((r) => r.cloud)).toBe(false);
  });

  it('refuses a secret with "bad" in it, never repeating the secret', async () => {
    const api = new MockApi('cloud-none', 0);
    const err = await api.connectCloud({ provider: 'runpod', values: { api_key: 'rpa_badkey123' } }).catch((e) => e);
    expect(err).toMatchObject({ status: 422, code: 'cloud_auth_failed' });
    expect(`${err.message} ${err.hint}`).not.toContain('rpa_badkey123');
    await expect(api.connectCloud({ provider: 'runpod', values: { api_key: ' ' } })).rejects.toMatchObject({ code: 'invalid_body', message: expect.stringContaining('`api_key`') });
    await expect(api.connectCloud({ provider: 'vast', values: {} })).rejects.toMatchObject({ status: 404, code: 'provider_not_found' });
  });

  it('keeps a secret as its hint only, and names a second connection of a provider "RunPod 2"', async () => {
    const api = new MockApi('normal', 0);
    const c = await api.connectCloud({ provider: 'runpod', values: { api_key: 'rpa_SECRETxyz9', worker_base_url: '' } });
    expect(c).toMatchObject({ name: 'RunPod 2', status: 'ok', values: { api_key: '••••xyz9' } });
    expect(JSON.stringify(await api.cloudConnections())).not.toContain('SECRET');
    const remote = await api.connectCloud({ provider: 'remote', name: 'Lab', values: { url: 'http://gpu-box.example.com:7878', token: 'tok_1234' } });
    expect(remote).toMatchObject({ name: 'Lab', account: 'gpu-box.example.com:7878', note: expect.stringContaining('unencrypted') });
    expect((await api.cloudOffers(remote.id)).offers).toHaveLength(1);
  });

  it('checks a connection again; a revoked key stays in error and its offers cannot be read', async () => {
    const api = new MockApi('cloud-trouble', 0);
    vi.setSystemTime(NOW + 60_000);
    const c = await api.checkCloudConnection('cc_4d2a');
    expect(c).toMatchObject({ status: 'error', error: expect.stringContaining('rejected this API key'), checked: new Date(NOW + 60_000).toISOString() });
    await expect(api.cloudOffers('cc_4d2a')).rejects.toMatchObject({ status: 422, code: 'cloud_auth_failed' });
  });

  it('keeps a connection with a live instance (409 connection_in_use), forgets one without', async () => {
    const busy = new MockApi('cloud-training', 0);
    await expect(busy.disconnectCloud('cc_4d2a')).rejects.toMatchObject({ status: 409, code: 'connection_in_use', message: expect.stringContaining('folo-run_c7b1-9d03') });
    const idle = new MockApi('normal', 0);
    await idle.disconnectCloud('cc_4d2a');
    expect(await idle.cloudConnections()).toEqual([]);
    await expect(idle.disconnectCloud('cc_4d2a')).rejects.toMatchObject({ code: 'connection_not_found' });
  });
});

describe('mock cloud: runs and instances', () => {
  it('seeds a finished cloud run, released with its final cost, and its terminated instance', async () => {
    const api = new MockApi('normal', 0);
    const run = await api.run('run_c3d5');
    expect(run).toMatchObject({ status: 'completed', device: 'NVIDIA GeForce RTX 4090 (vulkan)', model_id: 'm_shapes_m' });
    expect(run.cloud).toMatchObject({ phase: 'released', provider: 'runpod', instance_id: 'ci_4c1a', progress: null });
    expect(run.cloud!.cost_usd).toBeGreaterThan(0.1);
    const [inst] = await api.cloudInstances();
    expect(inst).toMatchObject({ id: 'ci_4c1a', status: 'terminated', run_id: 'run_c3d5', cost_usd: run.cloud!.cost_usd });
    expect(inst!.ended).not.toBeNull();
  });

  it('lists a termination that is not confirmed as terminating, with its error', async () => {
    const [inst] = await new MockApi('cloud-trouble', 0).cloudInstances();
    expect(inst).toMatchObject({ status: 'terminating', error: expect.stringContaining('rejected this API key'), ended: null });
  });

  it('simulates a cloud run through every phase, out of the training queue, its cost growing', async () => {
    const api = new MockApi('normal', 0);
    const created = await api.createRun(await cloudConfig(api));
    expect(created).toMatchObject({ status: 'queued', queue_position: null, device: '' });
    const at = async (ms: number) => {
      vi.setSystemTime(NOW + ms);
      return api.run(created.id);
    };
    const seen: [string, string, string | undefined][] = [];
    for (let ms = 0; ms <= 610_000 + 5000; ms += 500) {
      const r = await at(ms);
      const inst = (await api.cloudInstances()).find((i) => i.run_id === created.id);
      const key: [string, string, string | undefined] = [r.cloud!.phase, r.status, inst?.status];
      if (JSON.stringify(seen.at(-1)) !== JSON.stringify(key)) seen.push(key);
    }
    expect(seen).toEqual([
      ['provisioning', 'queued', 'launching'],
      ['booting', 'queued', 'starting'],
      ['syncing', 'queued', 'running'],
      ['training', 'running', 'running'],
      ['collecting', 'running', 'running'],
      ['releasing', 'running', 'terminating'],
      ['released', 'completed', 'terminated'],
    ]);
    const half = (await at(5000)).cloud!;
    expect(half.progress!.done_bytes / half.progress!.total_bytes).toBeCloseTo(0.5, 2);
    expect(half.deadline).toBe(new Date(NOW + 2 * 3_600_000).toISOString());
    const training = await at(300_000);
    expect(training.device).toBe('NVIDIA GeForce RTX 4090 (vulkan)');
    expect(training.cloud!.cost_usd).toBeCloseTo((0.69 * 300) / 3600, 2);
    expect(training.cloud!.synced).not.toBeNull();
    const done = await at(700_000);
    expect(done.cloud!.cost_usd).toBeCloseTo((0.69 * 610) / 3600, 2);
    expect((await at(800_000)).cloud!.cost_usd).toBe(done.cloud!.cost_usd);
  });

  it('refuses what the server refuses: no capacity, an unknown connection or offer, a time limit out of range, a device', async () => {
    const api = new MockApi('normal', 0);
    const none = RUNPOD_OFFERS.find((o) => o.availability === 'none')!;
    await expect(api.createRun(await cloudConfig(api, { offer_id: none.id }))).rejects.toMatchObject({ status: 409, code: 'cloud_refused' });
    await expect(api.createRun(await cloudConfig(api, { offer_id: 'nope' }))).rejects.toMatchObject({ status: 404, code: 'offer_not_found' });
    await expect(api.createRun(await cloudConfig(api, { connection_id: 'cc_x' }))).rejects.toMatchObject({ status: 404, code: 'connection_not_found' });
    await expect(api.createRun(await cloudConfig(api, { max_hours: 0.1 }))).rejects.toMatchObject({ code: 'invalid_config', message: expect.stringContaining('max_hours') });
    await expect(api.createRun({ ...(await cloudConfig(api)), device: 'gpu:0' })).rejects.toMatchObject({ code: 'invalid_config', message: expect.stringContaining('device') });
  });

  it('estimates a cloud config on the reference, with what it would cost', async () => {
    const api = new MockApi('normal', 0);
    const e = await api.estimateRun(await cloudConfig(api));
    expect(e).toMatchObject({ device: 'cloud', device_name: 'RTX 4090', source: 'reference', confidence: 'low' });
    expect(e.cost!.price_per_hour).toBe(0.69);
    expect(e.cost!.max_usd).toBe(1.38);
    expect(e.cost!.est_usd_low).toBeLessThanOrEqual(e.cost!.est_usd);
    expect(e.cost!.est_usd_high).toBeGreaterThanOrEqual(e.cost!.est_usd);
    // A time limit is no reason to refuse an estimate's offer: nothing launches.
    const none = RUNPOD_OFFERS.find((o) => o.availability === 'none')!;
    expect((await api.estimateRun(await cloudConfig(api, { offer_id: none.id }))).cost).toBeDefined();
  });

  it('fails the run with cloud_instance_lost when its instance is terminated by hand', async () => {
    const api = new MockApi('cloud-training', 0);
    const inst = await api.terminateCloudInstance('ci_9d03');
    expect(inst.status).toBe('terminating');
    const run = await api.run('run_c7b1');
    expect(run).toMatchObject({ status: 'failed', error_code: 'cloud_instance_lost' });
    expect(run.cloud!.phase).toBe('released');
    vi.setSystemTime(NOW + 2000);
    expect((await api.cloudInstances()).find((i) => i.id === 'ci_9d03')).toMatchObject({ status: 'terminated' });
    await expect(api.terminateCloudInstance('ci_x')).rejects.toMatchObject({ status: 404, code: 'instance_not_found' });
  });

  it('stops a cloud run before training by releasing its instance', async () => {
    const api = new MockApi('normal', 0);
    const created = await api.createRun(await cloudConfig(api));
    vi.setSystemTime(NOW + 3000);
    expect((await api.stopRun(created.id)).status).toBe('stopping');
    expect((await api.run(created.id)).cloud!.phase).toBe('releasing');
    vi.setSystemTime(NOW + 6000);
    const r = await api.run(created.id);
    expect(r).toMatchObject({ status: 'stopped', started: null });
    expect(r.cloud!.phase).toBe('released');
  });

  it('streams the cloud state as `cloud` events, a phase at once and progress about once a second', async () => {
    const api = new MockApi('normal', 0);
    const created = await api.createRun(await cloudConfig(api));
    const events: RunCloud[] = [];
    const stop = api.runEvents(created.id, { cloud: (c) => events.push(c) });
    await vi.advanceTimersByTimeAsync(7000);
    stop();
    expect(events.map((e) => e.phase).filter((p, i, a) => a[i - 1] !== p)).toEqual(['booting', 'syncing', 'training']);
    const uploads = events.filter((e) => e.phase === 'syncing');
    expect(uploads.length).toBeGreaterThanOrEqual(2);
    expect(uploads.length).toBeLessThanOrEqual(3);
  });

  it('never holds the local training slot', async () => {
    const api = new MockApi('normal', 0);
    await api.createRun(await cloudConfig(api));
    const local = await api.createRun({ ...(await api.run('run_31aa')).config, name: 'local' });
    expect(local.queue_position).toBe(3);
  });
});
