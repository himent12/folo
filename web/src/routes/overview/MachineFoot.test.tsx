// @vitest-environment happy-dom
// The live panel's foot: only the run's own GPU, never another adapter's load beside it, and none for a cloud run,
// whose machine this server has no readings of (review round 1, findings 1).
import { flush } from 'fw';
import { render } from '../../../framework/testing';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { hub } from '$lib/api/index';
import { MockApi } from '$lib/api/mock';
import type { Run, SystemInfo } from '$lib/api/types';
import MachineFoot from './MachineFoot';

let target: HTMLElement;
let c: { unmount(): void } | undefined;

afterEach(() => {
  c?.unmount();
  c = undefined;
  target?.remove();
});

async function mount(edit: (r: Run) => Run): Promise<{ sys: SystemInfo }> {
  const api = new MockApi('normal', 0);
  hub.api = api;
  hub.mode = 'mock';
  const run = edit((await api.runs()).find((r) => r.status === 'running')!);
  const sys = await api.system();
  target = document.createElement('div');
  document.body.appendChild(target);
  c = render(() => <MachineFoot run={run} />, target);
  flush();
  return { sys };
}

const foot = () =>
  vi.waitFor(() => {
    flush();
    const el = target.querySelector('.foot');
    if (!el) throw new Error('still waiting for the foot');
    return el;
  });

describe('MachineFoot', () => {
  it("shows the GPU the run's device names", async () => {
    const { sys } = await mount((r) => ({ ...r, config: { ...r.config, device: 'gpu:0' } }));
    const el = await foot();
    expect(el.querySelectorAll('.meter').length).toBe(3);
    expect(el.querySelector('.meter .m-label')?.textContent).toContain(sys.gpus.find((g) => g.ordinal === 0)!.name.replace(/^NVIDIA (GeForce )?/, ''));
  });

  it('shows no GPU for a run that auto resolved to the CPU', async () => {
    await mount((r) => ({ ...r, device: 'cpu', config: { ...r.config, device: 'auto' } }));
    const el = await foot();
    expect(el.querySelectorAll('.meter').length).toBe(2);
    expect(el.querySelector('.meter .m-label')?.textContent).toMatch(/^CPU · \d+ threads$/);
  });

  it('shows no GPU when the run names one this machine does not have', async () => {
    await mount((r) => ({ ...r, config: { ...r.config, device: 'gpu:99' } }));
    const el = await foot();
    expect(el.querySelectorAll('.meter').length).toBe(2);
  });

  it("has no foot for a cloud run: this server has no readings of the rented machine", async () => {
    await mount((r) => ({ ...r, cloud: { provider: 'runpod' } as unknown as Run['cloud'] }));
    await new Promise((ok) => setTimeout(ok, 50));
    flush();
    expect(target.querySelector('.foot')).toBeNull();
  });
});
