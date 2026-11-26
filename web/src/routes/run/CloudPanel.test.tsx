// @vitest-environment happy-dom
// A cloud run's page (docs/CLOUD.md): the hand-over is the waiting state's title, never a place in the queue; the
// Cloud panel shows the phases, the transfer, the instance, the cost and the limit; a `cloud` event replaces the state.
import { flush as flushSync, signal } from 'fw';
import { render, type Rendered } from '../../../framework/testing';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { RunEventHandlers } from '$lib/api/client';
import { hub } from '$lib/api/index';
import { MockApi } from '$lib/api/mock';
import type { CloudInstance, RunCloud } from '$lib/api/types';
import { parse, router } from '$lib/router';
import Run from '../Run';
import CloudPanel, { type CloudPanelProps } from './CloudPanel';

/** Ports of `mount(X, { target, props })`; `unmount(c)` is `c.unmount()`. */
function mount(C: typeof CloudPanel | typeof Run, o: { target: HTMLElement; props?: CloudPanelProps }): Rendered {
  return C === Run ? render(() => <Run />, o.target) : render(() => <CloudPanel {...o.props!} />, o.target);
}
const unmount = (c: Rendered) => c.unmount();

let target: HTMLElement;
let c: Rendered | undefined;

async function until<T>(get: () => T | null | undefined | false, what = 'the condition'): Promise<T> {
  let v: T | null | undefined | false = undefined;
  await vi.waitFor(
    () => {
      flushSync();
      v = get();
      if (!v) throw new Error(`still waiting for ${what}`);
    },
    { timeout: 4000, interval: 10 },
  );
  return v as T;
}

const text = (el: Element | null | undefined) => el?.textContent?.replace(/\s+/g, ' ').trim() ?? '';
const phases = () => [...target.querySelectorAll('#cloud .phases li')].map((li) => `${li.getAttribute('data-state')} ${text(li.querySelector('.name'))}`);
const fact = (label: string) => text([...target.querySelectorAll('#cloud .facts > div')].find((d) => text(d.querySelector('dt')) === label)?.querySelector('dd'));

beforeEach(() => {
  hub.mode = 'mock';
  // A released run's panel asks for its instance: the mock answers, never a network.
  hub.api = new MockApi('normal', 0);
  target = document.createElement('div');
  document.body.appendChild(target);
});

afterEach(() => {
  if (c) unmount(c);
  c = undefined;
  target.remove();
  vi.restoreAllMocks();
  vi.useRealTimers();
});

const GiB = 2 ** 30;
const state = (extra: Partial<RunCloud> = {}): RunCloud => ({
  connection_id: 'cc_1',
  provider: 'runpod',
  offer: { id: 'o', gpu: 'RTX 4090', gpu_count: 1, vram_bytes: 24 * GiB, price_per_hour: 0.69, region: null, detail: 'Secure Cloud', availability: 'high' },
  instance_id: 'ci_1',
  phase: 'syncing',
  progress: { done_bytes: 1.2 * GiB, total_bytes: 4.8 * GiB },
  max_hours: 12,
  deadline: new Date(Date.now() + 3_600_000).toISOString(),
  cost_usd: 0.42,
  synced: null,
  ...extra,
});

describe('CloudPanel', () => {
  // Noon in the machine's own time zone: the deadline an hour later is today, so the limit says a clock time alone.
  beforeEach(() => {
    vi.useFakeTimers({ toFake: ['Date'] });
    vi.setSystemTime(new Date(2026, 10, 9, 12, 0));
  });
  afterEach(() => vi.useRealTimers());

  it('shows the phases done, current and to come, the transfer, and the facts while the run is alive', () => {
    // At noon, so the deadline an hour on is today (from 23:00 it is tomorrow, and the limit names the date too).
    vi.useFakeTimers({ toFake: ['Date'] });
    vi.setSystemTime(new Date(2026, 10, 9, 12, 0));
    c = mount(CloudPanel, { target, props: { cloud: state(), provider: 'RunPod' } });
    flushSync();
    expect(phases()).toEqual([
      'done Provisioning',
      'done Booting',
      'now Uploading data',
      'next Training',
      'next Collecting weights',
      'next Releasing',
      'next Released',
    ]);
    expect(text(target.querySelector('#cloud .xfer'))).toBe('1.2 of 4.8 GiB');
    expect(target.querySelector('#cloud [role="progressbar"]')?.getAttribute('aria-valuenow')).toBe('25');
    expect(fact('Instance')).toBe('RTX 4090 · 24.0 GiB · Secure Cloud');
    expect(target.querySelector('#cloud .facts a')?.getAttribute('href')).toBe('#/settings?section=cloud');
    expect(fact('Cost')).toBe('$0.42 so far');
    expect(fact('Limit')).toBe('12 h · stops by 13:00');
    expect(fact('Last checkpoint copy')).toBe('none yet');
  });

  it('keeps the final cost once released, "not billed" without a price, and a newer server’s phase as its word', () => {
    c = mount(CloudPanel, { target, props: { cloud: state({ phase: 'released', progress: null, cost_usd: 1.84 }), provider: 'RunPod' } });
    flushSync();
    expect(phases().every((p) => p.startsWith('done'))).toBe(true);
    expect(fact('Cost')).toBe('$1.84');
    expect(fact('Limit')).toBe('12 h');
    unmount(c);
    c = mount(CloudPanel, { target, props: { cloud: state({ phase: 'migrating', progress: null, cost_usd: null, instance_id: null }), provider: 'Lab' } });
    flushSync();
    expect(phases().at(-1)).toBe('now migrating');
    expect(fact('Cost')).toBe('not billed');
    expect(target.querySelector('#cloud .facts a')).toBeNull();
  });

  it('marks where a released run failed and what it never reached, and a remote server’s steps as its own', () => {
    // The worker stopped answering during the upload: no training happened, so none is ticked.
    c = mount(CloudPanel, { target, props: { cloud: state({ phase: 'released', progress: null, ended_in: 'syncing' }), provider: 'RunPod', run: { status: 'failed' } } });
    flushSync();
    expect(phases()).toEqual(['done Provisioning', 'done Booting', 'failed Uploading data', 'skipped Training', 'skipped Collecting weights', 'done Releasing', 'done Released']);
    expect(fact('Last checkpoint copy')).toBe('none');
    unmount(c);
    // A remote server: nothing provisioned or released, connected to; its GPU is not an instance.
    const remote = state({ provider: 'remote', instance_id: null, phase: 'training', progress: null, cost_usd: null });
    c = mount(CloudPanel, { target, props: { cloud: remote, provider: 'Remote FOLO server', provisions: false } });
    flushSync();
    expect(phases()).toEqual(['done Connecting', 'done Uploading data', 'now Training', 'next Collecting weights', 'next Done']);
    expect(fact('GPU')).toBe('RTX 4090 · 24.0 GiB · Secure Cloud');
    expect(fact('Instance')).toBe('');
  });

  it('takes a released run’s cost from its instance, and never another run’s', async () => {
    vi.useRealTimers();
    const inst = (id: string, status: CloudInstance['status'], cost: number): CloudInstance =>
      ({ id, status, cost_usd: cost, offer: state().offer, error: null }) as CloudInstance;
    const answer = vi.spyOn(hub.api, 'cloudInstances').mockResolvedValue([inst('ci_2', 'terminating', 9.99), inst('ci_1', 'terminated', 1.91)]);
    // The run stopped waiting for the termination at $1.84; the reaper confirmed it later, at $1.91.
    const first = state({ phase: 'released', progress: null, cost_usd: 1.84, ended_in: 'collecting' });
    const cloud = signal(first);
    c = render(() => <CloudPanel cloud={cloud.get()} provider="RunPod" run={{ status: 'completed' }} />, target);
    await until(() => fact('Cost') === '$1.91', 'the ledger’s cost');
    expect(target.querySelector('#cloud .warn-line')).toBeNull();
    // The same panel, another run's page (whose instance is ci_3), while its lookup is still on the way: nothing of
    // ci_1 is shown for it, and ci_2's warning never is.
    let answerLater: (v: CloudInstance[]) => void = () => {};
    answer.mockReturnValueOnce(new Promise((r) => (answerLater = r)));
    cloud.set(state({ phase: 'released', progress: null, cost_usd: 0.5, instance_id: 'ci_3', ended_in: 'collecting' }));
    flushSync();
    expect(fact('Cost')).toBe('$0.50');
    expect(target.querySelector('#cloud .warn-line')).toBeNull();
    answerLater([inst('ci_2', 'terminating', 9.99)]);
    await until(() => answer.mock.calls.length >= 2, 'the second lookup');
    flushSync();
    expect(fact('Cost')).toBe('$0.50');
    expect(target.querySelector('#cloud .warn-line')).toBeNull();
  });

  it('says since when the worker is silent, and keeps its times ago moving while nothing else does', () => {
    vi.useFakeTimers({ toFake: ['Date', 'setInterval', 'clearInterval'] });
    vi.setSystemTime(new Date(2026, 10, 9, 12, 0));
    const synced = new Date(2026, 10, 9, 11, 59, 50).toISOString();
    const silent = new Date(2026, 10, 9, 11, 59, 55).toISOString();
    c = mount(CloudPanel, { target, props: { cloud: state({ phase: 'training', progress: null, synced, unanswered_since: silent }), provider: 'RunPod' } });
    flushSync();
    expect(text(target.querySelector('#cloud .warn-line'))).toBe('The worker has not answered since 11:59. FOLO keeps asking; the run may still be training or waiting there.');
    expect(fact('Last checkpoint copy')).toBe('just now');
    // Ninety seconds on, with no new state from the server: the copy is no longer "just now".
    vi.advanceTimersByTime(90_000);
    flushSync();
    expect(fact('Last checkpoint copy')).toBe('2m ago');
  });

  it('says when the ledger cannot be read, and reads it again until it can', async () => {
    vi.useFakeTimers({ toFake: ['setInterval', 'clearInterval', 'setTimeout', 'clearTimeout'] });
    const answer = vi.spyOn(hub.api, 'cloudInstances').mockRejectedValue(new Error('down'));
    c = mount(CloudPanel, { target, props: { cloud: state({ phase: 'released', progress: null, ended_in: 'collecting' }), provider: 'RunPod', run: { status: 'completed' } } });
    const note = await until(() => target.querySelector('#cloud .warn-line'), 'the note');
    expect(text(note)).toBe('Could not check that the instance is terminated. Settings › Cloud');
    answer.mockResolvedValue([{ id: 'ci_1', status: 'terminated', cost_usd: 1.2, offer: state().offer, error: null } as CloudInstance]);
    vi.advanceTimersByTime(30_000);
    await until(() => !target.querySelector('#cloud .warn-line') && fact('Cost') === '$1.20', 'the ledger read again');
  });

  it('skips unconfirmed-instance polls while connecting or while a previous poll is still in flight (#309)', async () => {
    vi.useFakeTimers({ toFake: ['setInterval', 'clearInterval', 'setTimeout', 'clearTimeout'] });
    const prevMode = hub.mode;
    try {
      let calls = 0;
      let releaseSecond!: () => void;
      const answer = vi.spyOn(hub.api, 'cloudInstances').mockImplementation(async () => {
        calls++;
        if (calls === 2) {
          await new Promise<void>((r) => (releaseSecond = r));
        }
        return [{ id: 'ci_1', status: 'terminating', cost_usd: 1.2, offer: state().offer, error: 'timeout' } as CloudInstance];
      });
      c = mount(CloudPanel, {
        target,
        props: { cloud: state({ phase: 'released', progress: null, ended_in: 'collecting' }), provider: 'RunPod', run: { status: 'completed' } },
      });
      await until(() => target.querySelector('#cloud .warn-line'), 'unconfirmed warning');
      expect(answer).toHaveBeenCalledTimes(1);

      hub.mode = 'connecting';
      flushSync();
      vi.advanceTimersByTime(30_000);
      expect(answer).toHaveBeenCalledTimes(1);

      hub.mode = 'mock';
      flushSync();
      vi.advanceTimersByTime(30_000);
      expect(answer).toHaveBeenCalledTimes(2);
      // Still in flight: next 30s tick must not overlap/abort it.
      vi.advanceTimersByTime(30_000);
      expect(answer).toHaveBeenCalledTimes(2);
      releaseSecond();
      await Promise.resolve();
    } finally {
      hub.mode = prevMode;
      vi.useRealTimers();
    }
  });
});

describe('Run: a cloud run', () => {
  async function open(api: MockApi, id: string) {
    hub.api = api;
    router.route = parse(`#/runs/${id}`);
    c = mount(Run, { target });
    await until(() => target.querySelector('#cloud .phases'), 'the Cloud panel');
  }

  it('titles the waiting state by its phase, with no queue controls, and says where it runs', async () => {
    await open(new MockApi('cloud-upload', 0), 'run_c9e4');
    const callout = await until(() => target.querySelector('.callout'), 'the waiting state');
    expect(text(callout.querySelector('.c-title'))).toMatch(/^Uploading data · \d+%$/);
    expect(text(callout.querySelector('.c-msg'))).toMatch(/ of 460\.0 MiB sent\. Training starts once the dataset is there\.$/);
    expect(target.querySelector('#queue')).toBeNull();
    expect(text(callout)).not.toContain('queue');
    expect(text(target.querySelector('.dev.cloud'))).toBe('RunPod · RTX 4090');
    expect(phases().find((p) => p.startsWith('now'))).toBe('now Uploading data');
  });

  it('follows the `cloud` event: the state it carries replaces the run’s', async () => {
    const api = new MockApi('cloud-upload', 0);
    let handlers: RunEventHandlers = {};
    vi.spyOn(api, 'runEvents').mockImplementation((_id, h) => {
      handlers = h;
      return () => {};
    });
    await open(api, 'run_c9e4');
    const run = await api.run('run_c9e4');
    handlers.cloud?.({ ...run.cloud!, phase: 'provisioning', progress: null, instance_id: null });
    await until(() => text(target.querySelector('.callout .c-title')) === 'Provisioning an RTX 4090 at RunPod', 'the new title');
    expect(phases().find((p) => p.startsWith('now'))).toBe('now Provisioning');
    handlers.cloud?.({ ...run.cloud!, phase: 'booting', progress: null });
    await until(() => text(target.querySelector('.callout .c-title')) === 'Booting the instance', 'the next title');
  });

  it('says in the header that the worker does not answer, and stops saying it when it does', async () => {
    const api = new MockApi('cloud-training', 0);
    let handlers: RunEventHandlers = {};
    vi.spyOn(api, 'runEvents').mockImplementation((_id, h) => {
      handlers = h;
      return () => {};
    });
    const id = (await api.runs()).find((r) => r.cloud && r.status === 'running')!.id;
    await open(api, id);
    const run = await api.run(id);
    const badges = () => [...target.querySelectorAll('.badge')].map(text);
    expect(badges()).not.toContain('Worker not answering');
    handlers.cloud?.({ ...run.cloud!, unanswered_since: new Date().toISOString() });
    await until(() => badges().includes('Worker not answering'), 'the badge');
    handlers.cloud?.({ ...run.cloud!, unanswered_since: null });
    await until(() => !badges().includes('Worker not answering'), 'the badge gone');
  });

  it('keeps the Cloud panel on a finished run, released, with its final cost', async () => {
    await open(new MockApi('normal', 0), 'run_c3d5');
    expect(phases().every((p) => p.startsWith('done'))).toBe(true);
    expect(fact('Cost')).toMatch(/^\$\d+\.\d\d$/);
    expect(target.querySelector('#cloud .warn-line')).toBeNull();
  });

  it('keeps releasing under way on a finished run whose instance’s termination is not confirmed', async () => {
    // cloud-trouble: the key was revoked, so RunPod never confirmed the termination of shapes-m's instance.
    await open(new MockApi('cloud-trouble', 0), 'run_c3d5');
    const note = await until(() => target.querySelector('#cloud .warn-line'), 'the unconfirmed termination');
    expect(text(note)).toBe('Termination not confirmed: the instance may still be billed. Settings › Cloud');
    expect(phases().slice(-2)).toEqual(['now Releasing', 'next Released']);
    expect(fact('Cost')).toMatch(/^\$\d+\.\d\d so far$/);
  });
});
