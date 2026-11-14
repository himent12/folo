// @vitest-environment happy-dom
import { flush as flushSync } from 'fw';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { hub } from '$lib/api/index';
import { MockApi } from '$lib/api/mock';
import type { TrainConfig } from '$lib/api/types';
import { parse, router } from '$lib/router';
import { render } from '../../../framework/testing';
import Settings from '../Settings';
import Throughput from './Throughput';

/** Port of `mount(Component, { target, props })`: renders into `target`; `unmount(c)` is `c.unmount()`. */
const mount = (C: typeof Settings | typeof Throughput, o: { target: HTMLElement; props?: { benchmark?: boolean } }) =>
  render(() => (C === Throughput ? <Throughput {...(o.props ?? {})} /> : <Settings />), o.target);
const unmount = (c: { unmount(): void }) => c.unmount();

let target: HTMLElement;
let c: ReturnType<typeof mount> | undefined;

async function until<T>(get: () => T | null | undefined | false, what = 'the condition'): Promise<T> {
  let v: T | null | undefined | false = undefined;
  await vi.waitFor(
    () => {
      flushSync();
      v = get();
      if (!v) throw new Error(`still waiting for ${what}`);
    },
    { timeout: 3000, interval: 10 },
  );
  return v as T;
}

beforeEach(() => {
  target = document.createElement('div');
  document.body.appendChild(target);
  router.route = parse('#/settings');
  hub.mode = 'mock';
});

afterEach(() => {
  if (c) unmount(c);
  c = undefined;
  target.remove();
});

const rows = () => [...target.querySelectorAll('.tp-row')].map((r) => r.textContent!.replace(/\s+/g, ' ').trim());
/** A table row's cells, by its column's heading. */
const cells = () =>
  [...target.querySelectorAll('.tp-row')].map((r) => {
    const heads = [...target.querySelectorAll('#throughput thead th')].map((th) => th.textContent!.trim());
    return Object.fromEntries([...r.querySelectorAll('td')].map((td, i) => [heads[i], td.textContent!.replace(/\s+/g, ' ').trim()]));
  });
const tab = (name: string) => [...target.querySelectorAll<HTMLElement>('#throughput [role="radio"]')].find((b) => b.textContent?.startsWith(name))!;
const measureBtn = () => [...target.querySelectorAll('button')].find((b) => /Measure now|Measuring/.test(b.textContent ?? ''));
const setWidth = (width: number) => (window as unknown as { happyDOM: { setViewport(v: { width: number }): void } }).happyDOM.setViewport({ width });

describe('Settings › Throughput', () => {
  // design/UNIFIED.md §4: a table (Device, Architecture, Image size, Batch, Precision, the figure, Source, Updated) with
  // Training / Inference in the panel's header.
  it('lists the measured rows only as a table, Training or Inference, with their spread, samples and age, and says which came from another adapter', async () => {
    hub.api = new MockApi('normal', 0);
    c = mount(Throughput, { target, props: { benchmark: true } });
    await until(() => target.querySelector('.tp-row'), 'the rows');
    expect([...target.querySelectorAll('#throughput thead th')].map((th) => th.textContent?.trim())).toEqual([
      'Device',
      'Architecture',
      'Image size',
      'Batch',
      'Precision',
      'img/s',
      'Range',
      'Source',
      'Updated',
    ]);
    // Three training rows and two prediction rows (the switch's counts); none of the reference's 40.
    expect([tab('Training').textContent, tab('Inference').textContent]).toEqual(['Training3', 'Inference2']);
    expect(rows()).toHaveLength(3);
    const first = cells()[0]!;
    expect(first).toMatchObject({ Device: 'GPU 0', Architecture: 'folo-n', 'Image size': '640 px', Batch: '16', Precision: 'f32', 'img/s': '212', Range: '205–220', Source: 'Measured · 99 samples' });
    expect(first.Updated).toMatch(/^\d+m ago$/);
    tab('Inference').click();
    flushSync();
    expect(rows()).toHaveLength(2);
    expect([...target.querySelectorAll('#throughput thead th')].map((th) => th.textContent?.trim())).toContain('Latency');
    const other = [...target.querySelectorAll<HTMLElement>('.tp-row')].find((r) => r.textContent?.includes('RTX 3060'))!;
    expect(other.classList.contains('other')).toBe(true);
    // Its samples are said too (as on phones): a row from another adapter loses no fact.
    expect(other.textContent).toMatch(/Measured on RTX 3060 · \d+ samples?( \(probes\))? · not used here/);
    expect(other.querySelector('[title]')?.getAttribute('title')).toContain('Measured on another GPU (RTX 3060), so not used for this one');
  });

  it('keeps a two-line list on phones, the switch choosing which', async () => {
    const width = window.innerWidth;
    setWidth(390);
    try {
      hub.api = new MockApi('normal', 0);
      c = mount(Throughput, { target, props: { benchmark: true } });
      await until(() => target.querySelector('.tp-row'), 'the rows');
      expect(target.querySelector('#throughput table')).toBeNull();
      expect(rows()).toHaveLength(3);
      expect(rows()[0]).toMatch(/^folo-n 640 px · batch 16 · f32 99 samples · \d+m ago 212 img\/s 205–220$/);
      tab('Inference').click();
      flushSync();
      expect(rows()).toHaveLength(2);
      expect(rows().find((r) => r.includes('RTX 3060'))).toContain('measured on another GPU');
    } finally {
      setWidth(width);
    }
  });

  it('measures on demand: busy while it runs, then the probe row is new', async () => {
    const api = new MockApi('normal', 0);
    hub.api = api;
    let release!: () => void;
    const gate = new Promise<void>((r) => (release = r));
    const real = api.benchmark.bind(api);
    vi.spyOn(api, 'benchmark').mockImplementation(async (b) => {
      await gate;
      return real(b);
    });
    c = mount(Throughput, { target, props: { benchmark: true } });
    await until(() => target.querySelector('.tp-row'), 'the rows');
    measureBtn()!.click();
    flushSync();
    expect(measureBtn()!.getAttribute('aria-busy')).toBe('true');
    expect(target.querySelector('.tp-status[role="status"]')).not.toBeNull();
    release();
    // The default probe (folo-n, 320 px, batch 16, f32) adds a training row and a prediction row, each first as the newest.
    await until(() => rows().length === 4, 'the probe');
    expect(target.querySelector('.tp-status')?.textContent ?? null).toBeNull();
    expect(cells()[0]).toMatchObject({ Architecture: 'folo-n', 'Image size': '320 px', Batch: '16', Precision: 'f32', Source: 'Measured · 1 sample (a probe)' });
    expect(tab('Inference').textContent).toBe('Inference3');
  });

  it('says why it cannot measure while a run is training', async () => {
    const api = new MockApi('normal', 0);
    hub.api = api;
    await api.createRun({ ...((await api.runs())[0]!.config as TrainConfig), name: 'busy' });
    c = mount(Throughput, { target, props: { benchmark: true } });
    await until(() => target.querySelector('.tp-row'), 'the rows');
    measureBtn()!.click();
    const err = await until(() => target.querySelector('.tp-status.err'), 'the error');
    expect(err.getAttribute('role')).toBe('alert');
    expect(err.textContent).toContain('busy');
  });

  it('has no Measure now without the benchmark feature', async () => {
    hub.api = new MockApi('normal', 0);
    c = mount(Throughput, { target, props: { benchmark: false } });
    await until(() => target.querySelector('.tp-row'), 'the rows');
    expect(measureBtn()).toBeUndefined();
  });

  it('is absent from Settings on a server without the throughput feature', async () => {
    hub.api = new MockApi('no-throughput', 0);
    c = mount(Settings, { target });
    await until(() => target.querySelector('#system .cap'), 'the System panel');
    expect(target.querySelector('#throughput')).toBeNull();
    unmount(c);
    hub.api = new MockApi('normal', 0);
    c = mount(Settings, { target });
    await until(() => target.querySelector('#throughput .tp-row'), 'the Throughput panel');
  });
});
