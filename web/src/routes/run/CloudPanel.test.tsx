// @vitest-environment happy-dom
// A cloud run's page (docs/CLOUD.md): the hand-over is the waiting state's title, never a place in the queue; the
// Cloud panel shows the phases, the transfer, the instance, the cost and the limit; a `cloud` event replaces the state.
import { flush as flushSync } from 'fw';
import { render, type Rendered } from '../../../framework/testing';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { RunEventHandlers } from '$lib/api/client';
import { hub } from '$lib/api/index';
import { MockApi } from '$lib/api/mock';
import type { RunCloud } from '$lib/api/types';
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

  it('keeps the Cloud panel on a finished run, released, with its final cost', async () => {
    await open(new MockApi('normal', 0), 'run_c3d5');
    expect(phases().every((p) => p.startsWith('done'))).toBe(true);
    expect(fact('Cost')).toMatch(/^\$\d+\.\d\d$/);
  });
});
