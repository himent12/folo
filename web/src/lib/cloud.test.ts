import { describe, expect, it } from 'vitest';
import type { CloudOffer, RunCloud, RunStatus } from './api/types';
import { article, beforeTraining, cloudGpu, defaultMaxHours, inHandover, isAlive, limitText, menuOrder, panelPhaseLabel, phaseIndex, phaseLabel, phaseStates, phaseTitle, pickOffer, providerName, transferShare, transferText, type PhaseRun } from './cloud';

const GiB = 2 ** 30;
const offer = (id: string, gpu: string, vram: number, price: number | null, availability: CloudOffer['availability'] = 'high'): CloudOffer => ({
  id,
  gpu,
  gpu_count: 1,
  vram_bytes: vram * GiB,
  price_per_hour: price,
  region: null,
  detail: 'Secure Cloud',
  availability,
});

describe('cloud phases', () => {
  it('takes a phase it does not know (a newer server) as its raw word, never as before training', () => {
    expect(phaseLabel('migrating')).toBe('migrating');
    expect(phaseIndex('migrating')).toBe(-1);
    expect(beforeTraining('syncing')).toBe(true);
    expect(beforeTraining('training')).toBe(false);
    expect(beforeTraining('migrating')).toBe(false);
  });

  it('counts a queued cloud run as in its hand-over only until it trains', () => {
    const run = (status: RunStatus, phase: string | null) => ({ status, cloud: phase === null ? undefined : ({ phase } as RunCloud) });
    expect(inHandover(run('queued', 'provisioning'))).toBe(true);
    expect(inHandover(run('queued', 'syncing'))).toBe(true);
    // The worker has it and waits for its GPU, a newer server's phase, a run of this machine, one already running.
    expect(inHandover(run('queued', 'training'))).toBe(false);
    expect(inHandover(run('queued', 'migrating'))).toBe(false);
    expect(inHandover(run('queued', null))).toBe(false);
    expect(inHandover(run('stopped', 'booting'))).toBe(false);
  });

  it('titles the hand-over by its phase: the instance it waits for, or the upload with its share', () => {
    const c = (phase: string, extra: Partial<RunCloud> = {}): RunCloud => ({
      connection_id: 'cc_1',
      provider: 'runpod',
      offer: offer('o', 'RTX 4090', 24, 0.69),
      instance_id: 'ci_1',
      phase,
      progress: null,
      max_hours: 12,
      deadline: null,
      cost_usd: null,
      synced: null,
      ...extra,
    });
    expect(phaseTitle(c('provisioning'), 'RunPod')).toBe('Provisioning an RTX 4090 at RunPod');
    expect(phaseTitle(c('booting', { instance_id: null }), 'Lab server')).toBe('Connecting to Lab server');
    expect(phaseTitle(c('syncing', { progress: { done_bytes: 1.2 * GiB, total_bytes: 4.8 * GiB } }), 'RunPod')).toBe('Uploading data · 25%');
    expect(phaseTitle(c('syncing', { progress: { done_bytes: 5, total_bytes: 0 } }), 'RunPod')).toBe('Uploading data');
  });

  it('says how far a transfer is, the unit once', () => {
    expect(transferText({ done_bytes: 1.2 * GiB, total_bytes: 4.8 * GiB })).toBe('1.2 of 4.8 GiB');
    expect(transferText({ done_bytes: 300 * 2 ** 20, total_bytes: 4.8 * GiB })).toBe('300.0 MiB of 4.8 GiB');
    expect(transferText({ done_bytes: 300 * 2 ** 20, total_bytes: 0 })).toBe('300.0 MiB');
    expect(transferShare({ done_bytes: 1, total_bytes: 4 })).toBe(0.25);
    expect(transferShare({ done_bytes: 1, total_bytes: 0 })).toBeNull();
    expect(transferShare(null)).toBeNull();
  });

  it('reads a GPU name aloud for its article', () => {
    expect(['RTX 4090', 'A100 PCIe', 'H100 SXM', 'L40S', 'GeForce GTX', 'Tesla T4'].map(article)).toEqual(['an', 'an', 'an', 'an', 'a', 'a']);
  });
});

describe('the Cloud panel’s phases', () => {
  const cloud = (phase: string, extra: Partial<RunCloud> = {}): RunCloud => ({
    connection_id: 'cc_1',
    provider: 'runpod',
    offer: offer('o', 'RTX 4090', 24, 0.69),
    instance_id: 'ci_1',
    phase,
    progress: null,
    max_hours: 12,
    deadline: null,
    cost_usd: null,
    synced: null,
    ...extra,
  });
  const run = (status: RunStatus, error_code: string | undefined, step: number, epoch = 0): PhaseRun => ({ status, error_code, step, epoch });
  const marks = (list: { phase: string; state: string }[]) => list.map((p) => `${p.state} ${p.phase}`);

  it('marks the phases before the current one done while the run is alive', () => {
    expect(marks(phaseStates(cloud('syncing'), run('queued', undefined, 0), true))).toEqual([
      'done provisioning',
      'done booting',
      'now syncing',
      'next training',
      'next collecting',
      'next releasing',
      'next released',
    ]);
  });

  it('marks where a released run failed, and never a phase it did not reach as done', () => {
    // The upload failed (the remote server stopped at 1%): no training, no collection.
    expect(marks(phaseStates(cloud('released'), run('failed', 'cloud_sync_failed', 0), true))).toEqual([
      'done provisioning',
      'done booting',
      'failed syncing',
      'skipped training',
      'skipped collecting',
      'done releasing',
      'done released',
    ]);
    // The same code after training is the collection of the weights.
    expect(marks(phaseStates(cloud('released'), run('failed', 'cloud_sync_failed', 40, 3), true)).filter((m) => !m.startsWith('done'))).toEqual(['failed collecting']);
    expect(marks(phaseStates(cloud('released'), run('failed', 'cloud_boot_failed', 0), true)).slice(0, 3)).toEqual(['done provisioning', 'failed booting', 'skipped syncing']);
    expect(marks(phaseStates(cloud('released'), run('failed', 'cloud_instance_lost', 40, 3), true))[3]).toBe('failed training');
    // Stopped before it trained: the hand-over before training happened, training did not.
    expect(marks(phaseStates(cloud('released'), run('stopped', undefined, 0), true)).filter((m) => !m.startsWith('done'))).toEqual(['skipped training', 'skipped collecting']);
    expect(phaseStates(cloud('released'), run('completed', undefined, 400, 10), true).every((p) => p.state === 'done')).toBe(true);
    // Without the run (an older caller), as before: all done.
    expect(phaseStates(cloud('released'), undefined, true).every((p) => p.state === 'done')).toBe(true);
  });

  it('lists no provisioning or releasing for a remote server, and words its steps as connecting and done', () => {
    const remote = cloud('released', { provider: 'remote', instance_id: null });
    expect(marks(phaseStates(remote, run('failed', 'cloud_unavailable', 0), false))).toEqual(['failed booting', 'skipped syncing', 'skipped training', 'skipped collecting', 'done released']);
    expect(['booting', 'released', 'syncing'].map((p) => panelPhaseLabel(p as never, false))).toEqual(['Connecting', 'Done', 'Uploading data']);
    expect(panelPhaseLabel('booting', true)).toBe('Booting');
  });

  it('keeps releasing under way while the termination is not confirmed', () => {
    expect(marks(phaseStates(cloud('released'), run('completed', undefined, 400, 10), true, true)).slice(-2)).toEqual(['now releasing', 'next released']);
  });
});

describe('a cloud run’s names', () => {
  it('names the GPU once: the offer’s until the worker reports its device, never "<gpu> at <provider>"', () => {
    const c = { offer: offer('o', 'NVIDIA GeForce RTX 4080 SUPER', 16, null) } as RunCloud;
    expect(cloudGpu({ device: 'NVIDIA GeForce RTX 4080 SUPER at Remote FOLO server', cloud: c })).toBe('RTX 4080 SUPER');
    expect(cloudGpu({ device: '', cloud: c })).toBe('RTX 4080 SUPER');
    expect(cloudGpu({ device: 'NVIDIA GeForce RTX 4080 SUPER (vulkan)', cloud: c })).toBe('RTX 4080 SUPER');
  });

  it('names a known provider before the server’s list is in, never by its id', () => {
    expect(providerName('remote', undefined)).toBe('Remote FOLO server');
    expect(providerName('runpod', undefined)).toBe('RunPod');
    expect(providerName('lab', undefined)).toBe('lab');
    expect(providerName('remote', [{ id: 'remote', name: 'Your server' } as never])).toBe('Your server');
  });

});

describe('offers', () => {
  const offers = [
    offer('a4000', 'RTX A4000', 16, 0.17),
    offer('3090', 'RTX 3090', 24, 0.22, 'low'),
    offer('4090', 'RTX 4090', 24, 0.69),
    offer('a100', 'A100 PCIe', 80, 1.64, 'none'),
    offer('h100', 'H100 SXM', 80, 2.99),
  ];

  it('preselects the cheapest available offer with the memory the run needs', () => {
    expect(pickOffer(offers, null)).toBe('a4000');
    expect(pickOffer(offers, 6 * GiB)).toBe('a4000');
    expect(pickOffer(offers, 20 * GiB)).toBe('3090');
    // The A100 has the memory but none are available: the H100 is.
    expect(pickOffer(offers, 40 * GiB)).toBe('h100');
    // Nothing has 100 GiB: the most memory, cheapest first.
    expect(pickOffer(offers.slice(0, 3), 100 * GiB)).toBe('3090');
    expect(pickOffer([offers[3]!], null)).toBeNull();
    // Compute that is not billed is the cheapest.
    expect(pickOffer([offers[2]!, offer('own', 'RTX 3090', 24, null)], 8 * GiB)).toBe('own');
  });

  it('lists what can start before what has none in stock, each as the server ordered it', () => {
    const catalogue = [offer('a2000', 'RTX A2000', 6, 0.12, 'none'), ...offers];
    expect(menuOrder(catalogue).map((o) => o.id)).toEqual(['a4000', '3090', '4090', 'h100', 'a2000', 'a100']);
    expect(menuOrder(offers.slice(0, 3))).toEqual(offers.slice(0, 3));
    expect(menuOrder([])).toEqual([]);
  });
});

describe('the time limit', () => {
  it('defaults to the high estimate × 1.5 plus half an hour, rounded up to a whole hour, at least 1', () => {
    expect(defaultMaxHours(null)).toBe(12);
    expect(defaultMaxHours(10 * 60)).toBe(1);
    expect(defaultMaxHours(3600)).toBe(2);
    expect(defaultMaxHours(5 * 3600)).toBe(8);
    expect(defaultMaxHours(1000 * 3600)).toBe(720);
  });

  it('says the limit in money and time', () => {
    expect(limitText(0.69, 12)).toBe('$8.28 at 12 h');
    expect(limitText(null, 12)).toBe('12 h');
  });

  it('counts an instance alive until it is gone', () => {
    expect(['launching', 'starting', 'running', 'terminating'].every((s) => isAlive(s as never))).toBe(true);
    expect(isAlive('terminated') || isAlive('lost')).toBe(false);
  });
});
