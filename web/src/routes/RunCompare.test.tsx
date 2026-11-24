// @vitest-environment happy-dom
import { flush as flushSync } from 'fw';
import { render, type Rendered } from '../../framework/testing';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ApiError, hub } from '$lib/api/index';
import { metric } from '$lib/format';
import { MockApi } from '$lib/api/mock';
import { parse, router } from '$lib/router';
import RunCompare, { COMPARE_POLL_MS } from './RunCompare';
import { emaSetting } from './run/resolved';

/** Port of `mount(RunCompare, { target })`; `unmount(c)` is `c.unmount()`. */
const mount = (_: typeof RunCompare, o: { target: HTMLElement }) => render(() => <RunCompare />, o.target);
const unmount = (c: Rendered) => c.unmount();

let target: HTMLElement;
let c: Rendered | undefined;
const width = window.innerWidth;
const setWidth = (w: number) => (window as unknown as { happyDOM: { setViewport(v: { width: number }): void } }).happyDOM.setViewport({ width: w });

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
  hub.api = new MockApi('normal', 0);
  hub.mode = 'mock';
});

afterEach(() => {
  if (c) unmount(c);
  c = undefined;
  target.remove();
  setWidth(width);
});

describe('Compare runs on a phone: two at a time', () => {
  it('keeps every row on the baseline and the run paged to; the baseline is its own choice, never the line in front', async () => {
    const runs = (await hub.api.runs()).filter((r) => r.status !== 'queued').slice(0, 4);
    expect(runs).toHaveLength(4);
    // Each run's best epoch tells its precision · recall cell apart from the others'.
    expect(new Set(runs.map((r) => r.best?.epoch)).size).toBe(4);
    const names = runs.map((r) => r.name);
    setWidth(390);
    router.route = parse(`#/runs/compare?ids=${runs.map((r) => r.id).join(',')}`);
    c = mount(RunCompare, { target });
    const table = await until(() => target.querySelector('#cmp-table table.two'), 'the paired table');
    // The precision · recall row needs the runs' metrics: wait for a value in it.
    await until(() => [...table.querySelectorAll('tbody tr')].some((tr) => tr.querySelector('td.k')?.textContent === 'Precision · recall' && tr.textContent?.includes('ep ')), 'precision · recall');

    /** Every body row has the setting cell and one cell per shown run; header names say which runs they are. */
    function check(baseline: string, other: string) {
      const heads = [...table.querySelectorAll('thead th')].slice(1).map((th) => th.querySelector('.th-name')?.textContent);
      expect(heads).toEqual([baseline, other]);
      // No colgroup whose keyed columns could outlive a page of runs.
      expect(table.querySelector('colgroup')).toBeNull();
      expect(table.querySelectorAll('thead th')[1]!.querySelector('.base-tag')).not.toBeNull();
      const rows = [...table.querySelectorAll('tbody tr:not(.fold)')];
      expect(rows.length).toBeGreaterThan(3);
      for (const tr of rows) expect(tr.querySelectorAll('td'), tr.querySelector('td.k')?.textContent ?? '').toHaveLength(3);
      // Each run's own epoch shows in its precision · recall cell: the values sit under their run's name.
      const pr = rows.find((tr) => tr.querySelector('td.k')?.textContent === 'Precision · recall')!;
      const epochOf = (name: string) => runs.find((r) => r.name === name)?.best?.epoch;
      const cells = [...pr.querySelectorAll('td')].slice(1).map((td) => td.textContent ?? '');
      [baseline, other].forEach((n, i) => {
        const ep = epochOf(n);
        expect(cells[i]).toContain(ep != null ? `ep ${ep}` : '—');
      });
    }

    check(names[0]!, names[1]!);
    // Page to the third run: the baseline stays, the second column is run C in every row.
    target.querySelector<HTMLButtonElement>('button[aria-label="Next run"]')!.click();
    flushSync();
    check(names[0]!, names[2]!);
    // Keep run D's line in front: emphasis is not the baseline, so the table still reads against run A beside run C.
    const chip = [...target.querySelectorAll<HTMLButtonElement>('.chip button.key')].find((b) => b.textContent === names[3])!;
    chip.click();
    flushSync();
    expect(chip.getAttribute('aria-pressed')).toBe('true');
    check(names[0]!, names[2]!);
    // The explicit control makes run C the baseline; run A, which it replaces, takes the other column.
    target.querySelector<HTMLButtonElement>(`#cmp-table button[aria-label="Make ${names[2]} the baseline"]`)!.click();
    flushSync();
    check(names[2]!, names[0]!);
    // It holds while another line is brought forward and the pager steps through the rest.
    [...target.querySelectorAll<HTMLButtonElement>('.chip button.key')].find((b) => b.textContent === names[1])!.click();
    flushSync();
    check(names[2]!, names[0]!);
    target.querySelector<HTMLButtonElement>('button[aria-label="Next run"]')!.click();
    flushSync();
    check(names[2]!, target.querySelectorAll('#cmp-table thead th .th-name')[1]?.textContent ?? '');
    expect(target.querySelectorAll('#cmp-table thead th .th-name')[1]?.textContent).not.toBe(names[0]);
  });

  it('forgets a baseline removed from the comparison: adding that run back does not make it the baseline again', async () => {
    const runs = (await hub.api.runs()).filter((r) => r.status !== 'queued').slice(0, 4);
    const names = runs.map((r) => r.name);
    const all = runs.map((r) => r.id);
    setWidth(390);
    router.route = parse(`#/runs/compare?ids=${all.join(',')}`);
    c = mount(RunCompare, { target });
    await until(() => target.querySelector('#cmp-table table.two'), 'the paired table');
    const heads = () => [...target.querySelectorAll('#cmp-table thead th .th-name')].map((th) => th.textContent);

    // Run C becomes the baseline.
    target.querySelector<HTMLButtonElement>('button[aria-label="Next run"]')!.click();
    flushSync();
    target.querySelector<HTMLButtonElement>(`#cmp-table button[aria-label="Make ${names[2]} the baseline"]`)!.click();
    flushSync();
    expect(heads()[0]).toBe(names[2]);

    // Remove run C (its chip's Remove, in the phone's Edit sheet): run A is the baseline.
    target.querySelector<HTMLButtonElement>('button.manage')!.click();
    flushSync();
    const rm = await until(() => document.querySelector<HTMLButtonElement>(`button[aria-label="Remove ${names[2]}"]`), 'the Remove button');
    rm.click();
    flushSync();
    expect(router.route.query.get('ids')).toBe([all[0], all[1], all[3]].join(','));
    await until(() => target.querySelector('#cmp-table table.two'), 'the paired table again');
    expect(heads()[0]).toBe(names[0]);

    // Add run C back: run A stays the baseline; C is just another run to page to.
    router.go(`/runs/compare?ids=${all.join(',')}`, { replace: true });
    flushSync();
    await until(() => target.querySelectorAll('.chip').length === 4, 'four runs again');
    expect(heads()[0]).toBe(names[0]);
    expect(target.querySelector('#cmp-table thead th .base-tag')?.closest('th')?.querySelector('.th-name')?.textContent).toBe(names[0]);
  });
});

describe('Compare: the side by side reads against its baseline', () => {
  it("marks each dataset's best, sets values equal to the baseline's in tertiary, and keeps the keys to the chips", async () => {
    const runs = (await hub.api.runs()).filter((r) => r.status !== 'queued').slice(0, 4);
    setWidth(1440);
    router.route = parse(`#/runs/compare?ids=${runs.map((r) => r.id).join(',')}`);
    c = mount(RunCompare, { target });
    const table = await until(() => target.querySelector('#cmp-table table'), 'the table');
    expect(table.querySelector('thead .base-tag')?.closest('th')?.querySelector('.th-name')?.textContent).toBe(runs[0]!.name);
    // The best row says it is any epoch's best, and each dataset with two runs here has its best marked.
    const best = [...table.querySelectorAll('tbody tr')].find((tr) => tr.querySelector('td.k')?.textContent?.startsWith('Best mAP50-95'))!;
    const datasets = new Set(runs.map((r) => r.config.dataset_id));
    const shared = [...datasets].filter((d) => runs.filter((r) => r.config.dataset_id === d).length > 1);
    expect(best.querySelectorAll('.tag')).toHaveLength(shared.length);
    if (datasets.size > 1) for (const t of best.querySelectorAll('.tag')) expect(t.textContent).toMatch(/^(joint )?best on /);
    // Every other column's value is either marked as differing from the baseline's or tertiary as equal to it.
    const rows = [...table.querySelectorAll('tbody tr')].filter((tr) => tr.querySelector('td.cfg'));
    expect(rows.length).toBeGreaterThan(0);
    for (const tr of rows) {
      const cells = [...tr.querySelectorAll('td.cfg')];
      expect(cells[0]!.classList.contains('base')).toBe(true);
      for (const td of cells.slice(1)) expect(td.classList.contains('diff') !== td.classList.contains('same')).toBe(true);
    }
    expect(table.querySelector('td.cfg.same')).toBeTruthy();
  });
});

describe('Compare: how many settings differ', () => {
  it('says one setting in the singular (#382, the twin of "AP for each of 1 classes")', async () => {
    // A copy of a run that trained at another image size: that is the one setting the two differ in.
    const api = hub.api as MockApi;
    const a = (await api.runs()).find((r) => r.status === 'completed' && !r.cloud)!;
    const twin = { ...a, id: 'run_twin', name: 'twin', config: { ...a.config, imgsz: a.config.imgsz + 32 } };
    const runs = api.runs.bind(api);
    api.runs = async (...rest) => [...(await runs(...rest)), twin];
    const metrics = api.runMetrics.bind(api);
    api.runMetrics = (id, ...rest) => metrics(id === twin.id ? a.id : id, ...rest);
    setWidth(1440);
    router.route = parse(`#/runs/compare?ids=${a.id},${twin.id}`);
    c = mount(RunCompare, { target });
    const meta = await until(() => target.querySelector('#cmp-table .meta'), 'the side by side');
    expect(meta.textContent).toBe('1 setting differs');
    // The EMA is a resolved setting like the warmup: its row is there, the same for both (PR #399).
    target.querySelector<HTMLButtonElement>('#cmp-table tr.fold button')!.click();
    const ema = await until(() => [...target.querySelectorAll('#cmp-table tbody tr')].find((tr) => tr.querySelector('td.k')?.textContent === 'EMA'), 'the EMA row');
    expect([...ema!.querySelectorAll('td.cfg')].map((td) => td.textContent)).toEqual([emaSetting(a.resolved!)!.value, emaSetting(a.resolved!)!.value]);
  });
});

describe('Compare: the keyboard', () => {
  it('B makes the next run the baseline, wrapping, as Swap does one column at a time', async () => {
    const runs = (await hub.api.runs()).filter((r) => r.status !== 'queued').slice(0, 3);
    setWidth(1440);
    router.route = parse(`#/runs/compare?ids=${runs.map((r) => r.id).join(',')}`);
    c = mount(RunCompare, { target });
    const table = await until(() => target.querySelector('#cmp-table table'), 'the table');
    const baseline = () => table.querySelector('thead .base-tag')?.closest('th')?.querySelector('.th-name')?.textContent;
    expect(baseline()).toBe(runs[0]!.name);
    const press = () => {
      window.dispatchEvent(new KeyboardEvent('keydown', { key: 'b', bubbles: true, cancelable: true }));
      flushSync();
    };
    press();
    expect(baseline()).toBe(runs[1]!.name);
    press();
    press();
    expect(baseline()).toBe(runs[0]!.name);
  });
});

describe('Compare runs from a deep link', () => {
  it('takes a run named twice in ?ids= once', async () => {
    // `?ids=a,a,b` (a shared or edited link) put run a in every keyed row twice: Svelte refuses duplicate keys, and
    // the page never rendered.
    const [a, b] = (await hub.api.runs()).filter((r) => r.status !== 'queued');
    router.route = parse(`#/runs/compare?ids=${a!.id},${a!.id},${b!.id}`);
    c = mount(RunCompare, { target });
    const table = await until(() => target.querySelector('#cmp-table table'), 'the table');
    const heads = [...table.querySelectorAll('thead th .th-name')].map((n) => n.textContent);
    expect(heads).toEqual([a!.name, b!.name]);
  });
});

describe('Compare runs from a link that names a deleted run', () => {
  it('draws the runs that exist (a run deleted since the link was made fails no one else)', async () => {
    // `?ids=a,run_gone,b`: the deleted run has no chip to remove, and its `404 run_not_found` failed the metrics of
    // every run, so the chart was an error whose Retry could never succeed.
    const [a, b] = (await hub.api.runs()).filter((r) => r.status !== 'queued');
    router.route = parse(`#/runs/compare?ids=${a!.id},run_gone,${b!.id}`);
    c = mount(RunCompare, { target });
    await until(() => target.querySelector('#cmp-table table'), 'the table');
    await until(() => target.querySelector('.swap') || target.textContent?.includes("Could not load these runs' metrics"), 'the chart or its error');
    expect(target.textContent).not.toContain("Could not load these runs' metrics");
  });
});

describe('Compare: a run that trains while compared (stress s3-live-5)', () => {
  afterEach(() => vi.useRealTimers());
  /** `ms` of fake time on, then the answers and what they set off settled. */
  const tick = async (ms = 0) => {
    await vi.advanceTimersByTimeAsync(ms);
    for (let i = 0; i < 6; i++) {
      await vi.advanceTimersByTimeAsync(0);
      flushSync();
    }
  };
  /**
   * Run `done` compared while it trains: "running" in its second epoch in the list, one finished epoch in its metrics,
   * each as the server holds them when asked. `state.training = false` completes it; `state.gate` holds its metrics
   * answer (asked while it trained) until released; `state.fail` makes its next metrics reads fail.
   */
  async function trainingCompare(gate: Promise<void> | null = null, training = true) {
    const api = hub.api as MockApi;
    const [done, other] = (await api.runs()).filter((r) => r.status === 'completed').slice(0, 2);
    const full = (await api.runMetrics(done!.id)).epochs;
    const listRuns = api.runs.bind(api);
    const readMetrics = api.runMetrics.bind(api);
    const state = { training, gate, fail: 0, epoch: 2, deleted: false };
    vi.spyOn(api, 'runs').mockImplementation(async (...a) => {
      const training = state.training;
      const list = (await listRuns(...a)).filter((r) => !(state.deleted && r.id === other!.id));
      return list.map((r) => (training && r.id === done!.id ? { ...r, status: 'running', epoch: state.epoch, best: null } : r));
    });
    vi.spyOn(api, 'runMetrics').mockImplementation(async (id, ...rest) => {
      const training = state.training;
      const epoch = state.epoch;
      const held = id === done!.id ? state.gate : null;
      if (id === done!.id && state.fail > 0) {
        state.fail--;
        throw new ApiError(500, 'internal', 'the metrics could not be read');
      }
      if (id === other!.id && state.deleted) throw new ApiError(404, 'run_not_found', 'no such run');
      const m = await readMetrics(id, ...rest);
      if (held) await held;
      return training && id === done!.id ? { ...m, epochs: m.epochs.slice(0, epoch - 1) } : m;
    });
    vi.useFakeTimers();
    router.route = parse(`#/runs/compare?ids=${done!.id},${other!.id}`);
    c = mount(RunCompare, { target });
    const chip = () => [...target.querySelectorAll('.chip')].find((el) => el.textContent?.includes(done!.name))?.textContent ?? '';
    const last = metric(full.at(-1)!.val.map50_95);
    expect(metric(full[0]!.val.map50_95)).not.toBe(last);
    return { state, chip, first: metric(full[0]!.val.map50_95), second: metric(full[1]!.val.map50_95), last, other: other! };
  }

  it('follows it: its line and value grow as it trains, and its end mark says how it ended', async () => {
    const { state, chip, first, last } = await trainingCompare();
    await tick();
    expect(chip()).toContain('live');
    expect(chip()).toContain(first);
    // It completes while the page is open: within a poll and the metrics read it sets off.
    state.training = false;
    await tick(COMPARE_POLL_MS);
    expect(chip()).not.toContain('live');
    expect(chip()).toContain(last);
  });

  it('never loses a completion seen while the metrics read is still out (review round 1)', async () => {
    let release!: () => void;
    const { state, chip, last } = await trainingCompare(new Promise<void>((r) => (release = r)));
    await tick();
    expect(chip()).toContain('live');
    // The run completes and the poll sees it while the first metrics answer (asked while it trained) is still out.
    state.training = false;
    state.gate = null;
    await tick(COMPARE_POLL_MS);
    expect(chip()).not.toContain('live');
    release();
    await tick();
    expect(chip()).toContain(last);
  });

  it('says a failed refresh may leave the curves behind, and tries again until it gets them (review round 1)', async () => {
    const { state, chip, first, last } = await trainingCompare();
    await tick();
    expect(chip()).toContain(first);
    state.training = false;
    state.fail = 1;
    await tick(COMPARE_POLL_MS);
    expect(chip()).not.toContain('live');
    expect(target.querySelector('.stale-note')?.textContent).toContain("Could not refresh these runs' metrics");
    // Tried again a poll later, with no click.
    await tick(COMPARE_POLL_MS);
    expect(target.querySelector('.stale-note')).toBeNull();
    expect(chip()).toContain(last);
  });

  it('keeps following it after the other compared run is deleted elsewhere (review round 2)', async () => {
    const { state, chip, second, last, other } = await trainingCompare();
    await tick();
    // An epoch later its metrics are read again, with both runs in the list.
    state.epoch = 3;
    await tick(COMPARE_POLL_MS);
    expect(chip()).toContain(second);
    // The other run is deleted in another tab: its chip goes and the link drops it.
    state.deleted = true;
    await tick(COMPARE_POLL_MS);
    expect([...target.querySelectorAll('.chip')].some((el) => el.textContent?.includes(other.name))).toBe(false);
    // Then this one completes: its line and value still reach the end (they froze at epoch 2).
    state.training = false;
    await tick(COMPARE_POLL_MS);
    expect(chip()).not.toContain('live');
    expect(chip()).toContain(last);
  });

  it('reads again a failed run whose metrics came before the runs list (review round 3)', async () => {
    // The metrics answer is asked while epoch 1's checkpoint is still being written (no epochs); the runs list comes
    // after epoch 2 failed before its first step: `failed, epoch: 1`, epoch 1 finished (docs/API.md `Run.epoch`).
    const api = hub.api as MockApi;
    const [run, other] = (await api.runs()).filter((r) => r.status === 'completed').slice(0, 2);
    const full = (await api.runMetrics(run!.id)).epochs;
    const listRuns = api.runs.bind(api);
    const readMetrics = api.runMetrics.bind(api);
    let release!: () => void;
    const listed = new Promise<void>((r) => (release = r));
    let reads = 0;
    vi.spyOn(api, 'runs').mockImplementation(async (...a) => {
      const list = await listRuns(...a);
      await listed;
      return list.map((r) => (r.id === run!.id ? { ...r, status: 'failed', epoch: 1, best: null } : r));
    });
    vi.spyOn(api, 'runMetrics').mockImplementation(async (id, ...rest) => {
      const m = await readMetrics(id, ...rest);
      if (id !== run!.id) return m;
      return { ...m, epochs: m.epochs.slice(0, reads++ === 0 ? 0 : 1) };
    });
    vi.useFakeTimers();
    router.route = parse(`#/runs/compare?ids=${run!.id},${other!.id}`);
    c = mount(RunCompare, { target });
    await tick();
    expect(reads).toBe(1);
    release();
    await tick();
    await tick(COMPARE_POLL_MS);
    const chip = [...target.querySelectorAll('.chip')].find((el) => el.textContent?.includes(run!.name))?.textContent ?? '';
    expect(chip).toContain(metric(full[0]!.val.map50_95));
    // One read more, not one per poll: the second answer was asked with the list in hand.
    await tick(COMPARE_POLL_MS);
    expect(reads).toBe(2);
  });

  it('stops polling once all compared runs are terminal and synced, and checks again on tab return', async () => {
    // Completed runs when the page opens: once metrics are synced, no 3s poll fires while idle.
    const { state, chip } = await trainingCompare(null, false);
    const runsSpy = hub.api.runs as unknown as ReturnType<typeof vi.fn>;
    await tick();
    expect(chip()).not.toContain('live');
    const callsAfterSync = runsSpy.mock.calls.length;
    await tick(COMPARE_POLL_MS * 4);
    expect(runsSpy.mock.calls.length).toBe(callsAfterSync);
    // Resumed in another tab: switching away and returning to this tab refreshes once and restarts live polling.
    state.training = true;
    const vis = vi.spyOn(document, 'visibilityState', 'get');
    try {
      vis.mockReturnValue('hidden');
      document.dispatchEvent(new Event('visibilitychange'));
      await tick();
      expect(chip()).not.toContain('live');
      vis.mockReturnValue('visible');
      document.dispatchEvent(new Event('visibilitychange'));
      await tick();
      expect(chip()).toContain('live');
    } finally {
      vis.mockRestore();
    }
    // Completes again: once final metrics catch up, polling stops again.
    state.training = false;
    await tick(COMPARE_POLL_MS);
    expect(chip()).not.toContain('live');
    const callsAfterComplete = runsSpy.mock.calls.length;
    await tick(COMPARE_POLL_MS * 4);
    expect(runsSpy.mock.calls.length).toBe(callsAfterComplete);
  });

  it('stops polling for a run that failed mid-epoch once synced', async () => {
    const api = hub.api as MockApi;
    const [run, other] = (await api.runs()).filter((r) => r.status === 'completed').slice(0, 2);
    const listRuns = api.runs.bind(api);
    const readMetrics = api.runMetrics.bind(api);
    const runsSpy = vi.spyOn(api, 'runs').mockImplementation(async (...a) => {
      const list = await listRuns(...a);
      // Failed during epoch 2: only epoch 1 has validated metrics.
      return list.map((r) => (r.id === run!.id ? { ...r, status: 'failed', epoch: 2, best: null } : r));
    });
    vi.spyOn(api, 'runMetrics').mockImplementation(async (id, ...rest) => {
      const m = await readMetrics(id, ...rest);
      return id === run!.id ? { ...m, epochs: m.epochs.slice(0, 1) } : m;
    });
    vi.useFakeTimers();
    router.route = parse(`#/runs/compare?ids=${run!.id},${other!.id}`);
    c = mount(RunCompare, { target });
    await tick();
    const callsAfterSync = runsSpy.mock.calls.length;
    await tick(COMPARE_POLL_MS * 4);
    expect(runsSpy.mock.calls.length).toBe(callsAfterSync);
  });

  it('keeps advancing a healthy live run while another run fails persistently, and recovers cleanly', async () => {
    const api = hub.api as MockApi;
    const [live, broken] = (await api.runs()).filter((r) => r.status === 'completed').slice(0, 2);
    const fullLive = (await api.runMetrics(live!.id)).epochs;
    const fullBroken = (await api.runMetrics(broken!.id)).epochs;
    const listRuns = api.runs.bind(api);
    const readMetrics = api.runMetrics.bind(api);
    const st = { training: true, epoch: 2, failBroken: true };
    vi.spyOn(api, 'runs').mockImplementation(async (...a) => {
      const list = await listRuns(...a);
      return list.map((r) => (st.training && r.id === live!.id ? { ...r, status: 'running', epoch: st.epoch, best: null } : r));
    });
    vi.spyOn(api, 'runMetrics').mockImplementation(async (id, ...rest) => {
      if (id === broken!.id && st.failBroken) throw new ApiError(500, 'internal', 'broken');
      const m = await readMetrics(id, ...rest);
      return st.training && id === live!.id ? { ...m, epochs: m.epochs.slice(0, st.epoch - 1) } : m;
    });
    vi.useFakeTimers();
    router.route = parse(`#/runs/compare?ids=${live!.id},${broken!.id}`);
    c = mount(RunCompare, { target });
    const chip = (name: string) => [...target.querySelectorAll('.chip')].find((el) => el.textContent?.includes(name))?.textContent ?? '';
    await tick();
    expect(chip(live!.name)).toContain(metric(fullLive[0]!.val.map50_95));
    expect(chip(broken!.name)).toContain('metrics failed');

    // Live run advances and completes while broken still fails 500.
    st.training = false;
    await tick(COMPARE_POLL_MS);
    expect(chip(live!.name)).not.toContain('live');
    expect(chip(live!.name)).toContain(metric(fullLive.at(-1)!.val.map50_95));
    expect(chip(broken!.name)).toContain('metrics failed');

    // Broken recovers on the next retry; live keeps its final value.
    st.failBroken = false;
    await tick(COMPARE_POLL_MS);
    expect(target.querySelector('.stale-note')).toBeNull();
    expect(chip(broken!.name)).toContain(metric(fullBroken.at(-1)!.val.map50_95));
    expect(chip(live!.name)).toContain(metric(fullLive.at(-1)!.val.map50_95));
  });
});

describe('Compare: deleted run slot reclamation, partial metrics error, and Precision', () => {
  it('reconciles 4 missing IDs out of ?ids=, announces them, and restores the picker', async () => {
    setWidth(1440);
    router.route = parse('#/runs/compare?ids=missing_1,missing_2,missing_3,missing_4');
    c = mount(RunCompare, { target });
    const notice = await until(() => target.querySelector('[data-missing-runs]'), 'the missing runs notice');
    expect(notice.textContent).toContain('Removed 4 missing runs');
    expect(notice.textContent).toContain('missing_1');
    expect(notice.textContent).toContain('missing_4');
    expect(router.route.query.get('ids')).toBeNull();
    expect(target.textContent).toContain('Pick runs to compare');
    // Picker is restored (not trapped in an empty dead end).
    expect(target.querySelector('.empty select[aria-label="Add a run to the comparison"]')).not.toBeNull();
    // Dismiss clears the notice.
    notice.querySelector<HTMLButtonElement>('button')!.click();
    flushSync();
    expect(target.querySelector('[data-missing-runs]')).toBeNull();
  });

  it('reclaims slots for deleted IDs alongside valid runs so Add run stays available', async () => {
    setWidth(1440);
    const [a] = (await hub.api.runs()).filter((r) => r.status !== 'queued');
    router.route = parse(`#/runs/compare?ids=${a!.id},gone_1,gone_2,gone_3`);
    c = mount(RunCompare, { target });
    await until(() => target.querySelector('[data-missing-runs]'), 'the missing runs notice');
    expect(router.route.query.get('ids')).toBe(a!.id);
    expect(target.querySelectorAll('.chip')).toHaveLength(1);
    // Add run control is available again because 3 dead slots were reclaimed.
    expect(target.querySelector('select[aria-label="Add a run to the comparison"]')).not.toBeNull();
  });

  it('preserves healthy curves when one run fails to load metrics and recovers via per-run Retry', async () => {
    setWidth(1440);
    const api = hub.api as MockApi;
    const [a, b] = (await api.runs()).filter((r) => r.status === 'completed').slice(0, 2);
    const real = api.runMetrics.bind(api);
    const fullA = (await real(a!.id)).epochs;
    const fullB = (await real(b!.id)).epochs;
    let failB = true;
    const metricsSpy = vi.spyOn(api, 'runMetrics').mockImplementation(async (id, ...rest) => {
      if (id === b!.id && failB) throw new ApiError(500, 'internal', 'metrics unavailable');
      return real(id, ...rest);
    });
    vi.useFakeTimers();
    try {
      const settle = async () => {
        for (let i = 0; i < 6; i++) {
          await vi.advanceTimersByTimeAsync(0);
          flushSync();
        }
      };
      router.route = parse(`#/runs/compare?ids=${a!.id},${b!.id}`);
      c = mount(RunCompare, { target });
      await settle();

      // Healthy run A's curve renders and its chip shows its value; run B reports metrics failure with a per-run Retry and keeps Open run link.
      const chipText = (name: string) => [...target.querySelectorAll('.chip')].find((el) => el.textContent?.includes(name))?.textContent ?? '';
      expect(target.querySelector('.swap')).not.toBeNull();
      expect(chipText(a!.name)).toContain(metric(fullA.at(-1)!.val.map50_95));
      expect(chipText(b!.name)).toContain('metrics failed');
      expect(target.querySelector(`a[aria-label="Open ${b!.name}"]`)).not.toBeNull();
      expect(target.querySelector('.stale-note')?.textContent).toContain(`Could not load metrics for ${b!.name}`);

      // Clicking per-run Retry fetches only run B and restores its value without wiping run A.
      metricsSpy.mockClear();
      failB = false;
      const retryBtn = target.querySelector<HTMLButtonElement>(`button[aria-label="Retry metrics for ${b!.name}"]`)!;
      retryBtn.click();
      await settle();
      expect(target.querySelector('.stale-note')).toBeNull();
      expect(metricsSpy).toHaveBeenCalledTimes(1);
      expect(metricsSpy).toHaveBeenCalledWith(b!.id, expect.any(AbortSignal));
      expect(chipText(a!.name)).toContain(metric(fullA.at(-1)!.val.map50_95));
      expect(chipText(b!.name)).toContain(metric(fullB.at(-1)!.val.map50_95));
    } finally {
      vi.useRealTimers();
    }
  });

  it('recovers cleanly from a total initial metrics failure via ErrorState Retry', async () => {
    setWidth(1440);
    const api = hub.api as MockApi;
    const [a, b] = (await api.runs()).filter((r) => r.status === 'completed').slice(0, 2);
    const real = api.runMetrics.bind(api);
    let failAll = true;
    vi.spyOn(api, 'runMetrics').mockImplementation(async (id, ...rest) => {
      if (failAll) throw new ApiError(500, 'internal', 'all metrics down');
      return real(id, ...rest);
    });
    router.route = parse(`#/runs/compare?ids=${a!.id},${b!.id}`);
    c = mount(RunCompare, { target });
    await until(() => target.textContent?.includes("Could not load these runs' metrics"), 'full error state');
    // Total initial failure is owned by ErrorState, not duplicated in chip/stale notices.
    expect(target.querySelector('.stale-note')).toBeNull();

    failAll = false;
    const retry = [...target.querySelectorAll<HTMLButtonElement>('button')].find((btn) => btn.textContent?.trim() === 'Retry')!;
    retry.click();
    await until(() => target.querySelector('.swap') && !target.textContent?.includes("Could not load these runs' metrics"), 'chart recovered');
    expect(target.querySelector('.swap')).not.toBeNull();
  });

  it('supports Precision in the metric selector and annotates its best-F1 operating point', async () => {
    setWidth(1440);
    const [a, b] = (await hub.api.runs()).filter((r) => r.status === 'completed').slice(0, 2);
    const mA = await hub.api.runMetrics(a!.id);
    router.route = parse(`#/runs/compare?ids=${a!.id},${b!.id}`);
    c = mount(RunCompare, { target });
    await until(() => target.querySelector('.swap'), 'the chart');

    const precRadio = [...target.querySelectorAll<HTMLButtonElement>('[role="radiogroup"][aria-label="Metric"] [role="radio"]')].find((el) => el.textContent?.trim() === 'Precision')!;
    expect(precRadio).toBeDefined();
    precRadio.click();
    flushSync();

    expect(precRadio.getAttribute('aria-checked')).toBe('true');
    const chipA = [...target.querySelectorAll('.chip')].find((el) => el.textContent?.includes(a!.name))?.textContent ?? '';
    expect(chipA).toContain(metric(mA.epochs.at(-1)!.val.precision));
    expect(target.querySelector('.chips-head .at')?.textContent).toContain('at best-F1 conf');
  });
});

describe('Compare: a run that detected nothing', () => {
  it('says "no detections" in its precision · recall cell, never "conf 1.00 · ep 1" (studio-train F1)', async () => {
    const api = hub.api as MockApi;
    const [blank, other] = (await api.runs()).filter((r) => r.status === 'completed').slice(0, 2);
    const metrics = api.runMetrics.bind(api);
    vi.spyOn(api, 'runMetrics').mockImplementation(async (id, ...rest) => {
      const m = await metrics(id, ...rest);
      if (id !== blank!.id) return m;
      const val = { map50: 0, map50_95: 0, precision: 0, recall: 0, per_class: [], best_f1_conf: 1 };
      return { ...m, epochs: m.epochs.map((e) => ({ ...e, val })) };
    });
    router.route = parse(`#/runs/compare?ids=${blank!.id},${other!.id}`);
    c = mount(RunCompare, { target });
    const pr = await until(
      () => [...target.querySelectorAll('#cmp-table tbody tr')].find((tr) => tr.querySelector('td.k')?.textContent === 'Precision · recall' && tr.textContent?.includes('ep ')),
      'precision · recall',
    );
    const cells = [...pr.querySelectorAll('td')].slice(1).map((td) => td.textContent ?? '');
    expect(cells[0]).toContain('no detections');
    expect(cells[0]).not.toContain('conf 1.00');
    expect(cells[1]).toContain(`ep ${other!.best?.epoch}`);
  });
});
