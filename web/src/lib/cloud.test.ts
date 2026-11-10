import { describe, expect, it } from 'vitest';
import type { CloudOffer, RunCloud, RunStatus } from './api/types';
import {
  article,
  availabilityNote,
  beforeTraining,
  defaultMaxHours,
  inHandover,
  isAlive,
  limitText,
  menuOrder,
  offerLine,
  offerParts,
  phaseIndex,
  phaseLabel,
  phaseTitle,
  phaseWhat,
  pickOffer,
  providerName,
  transferShare,
  transferText,
} from './cloud';

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
  it('names the phases in the words people read, an unknown one as its raw word', () => {
    expect(['provisioning', 'booting', 'syncing', 'training', 'collecting', 'releasing', 'released'].map(phaseLabel)).toEqual([
      'Provisioning',
      'Booting',
      'Uploading data',
      'Training',
      'Collecting weights',
      'Releasing',
      'Released',
    ]);
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

  it('titles the hand-over by its phase, never as a wait in the queue', () => {
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
    expect(phaseTitle(c('booting'), 'RunPod')).toBe('Booting the instance');
    expect(phaseTitle(c('booting', { instance_id: null }), 'Lab server')).toBe('Connecting to Lab server');
    expect(phaseTitle(c('syncing', { progress: { done_bytes: 1.2 * GiB, total_bytes: 4.8 * GiB } }), 'RunPod')).toBe('Uploading data · 25%');
    expect(phaseTitle(c('syncing', { progress: { done_bytes: 5, total_bytes: 0 } }), 'RunPod')).toBe('Uploading data');
    expect(phaseTitle(c('training'), 'RunPod')).toBe('Starting on the instance');
    expect(phaseWhat(c('provisioning'), 'RunPod')).toBe('RunPod is asked for an RTX 4090 at $0.69/h. Once it has booted, the dataset is uploaded and training starts.');
    expect(phaseWhat(c('syncing', { progress: { done_bytes: 1.2 * GiB, total_bytes: 4.8 * GiB } }), 'RunPod')).toBe('1.2 of 4.8 GiB sent. Training starts once the dataset is there.');
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

  it('names a provider by its list, by its id until the list is in', () => {
    expect(providerName('runpod', [{ id: 'runpod', name: 'RunPod' } as never])).toBe('RunPod');
    expect(providerName('runpod', undefined)).toBe('runpod');
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

  it('writes an offer in one line and says what its stock means', () => {
    expect(offerLine(offers[2]!)).toBe('RTX 4090 · 24.0 GiB · Secure Cloud');
    expect(offerLine({ ...offers[2]!, gpu_count: 2, vram_bytes: 0, detail: null })).toBe('2 × RTX 4090');
    expect(offerParts(offers[2]!, 'metric')).toEqual(['RTX 4090', '25.8 GB', 'Secure Cloud']);
    expect(availabilityNote('low')).toBe('low stock');
    expect(availabilityNote('none')).toBe('none available now');
    expect(availabilityNote('high')).toBeNull();
  });

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
