// @vitest-environment happy-dom
// FOLO v2 (`folo2-n..x`) end-to-end parity tests: arch helpers, ModelSection family toggle, estimation,
// OOM step-down within family, cloud guard, draft & query parsing, MockApi, and HttpApi `x-folo-arch` header.
import { flush as flushSync, signal } from 'fw';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { render } from '../../../framework/testing';
import { HttpApi } from '$lib/api/client';
import { MockApi } from '$lib/api/mock';
import type { Resource } from '$lib/api/resource';
import type { Arch, Model, Run, ThroughputTable, TrainConfig } from '$lib/api/types';
import { archFamily, archFromCli, archNameSuffix, archWithFamily, FOLO2_FEATURE, isArch, isFolo2Arch, smallerArchInFamily, V2_ARCHES } from '$lib/arch';
import { oomFixes } from '$lib/oom';
import { trainCommand } from '$lib/trainCli';
import type { CloudPick } from './cloudPick';
import { isDraft } from './draft';
import { arches, archesForFamily, estimate } from './estimate';
import { formErrors, type FormErrors } from './jump';
import ModelSection from './ModelSection';
import { DEFAULT_AUGMENT, parseInitialQuery } from './newPrefill';

const cleanErrors = (): FormErrors => ({
  name: null,
  epochs: null,
  lr0: null,
  seed: null,
  close: null,
  dataset: null,
  weights: null,
  device: null,
  amp: null,
  runon: null,
  offer: null,
  hours: null,
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe('FOLO v2 architecture helpers, OOM fixes, estimation, and CLI', () => {
  it('maps families, scales, run-name suffixes, and stays inside folo2 on OOM step-down', () => {
    for (const a of V2_ARCHES) {
      expect(isArch(a)).toBe(true);
      expect(isFolo2Arch(a)).toBe(true);
      expect(archFamily(a)).toBe('folo2');
      expect(archNameSuffix(a)).toBe(a);
    }
    expect(archNameSuffix('folo-s')).toBe('s');
    expect(archWithFamily('folo-s', 'folo2')).toBe('folo2-s');
    expect(archWithFamily('folo2-l', 'folo')).toBe('folo-l');
    expect(smallerArchInFamily('folo2-m')).toBe('folo2-s');
    expect(smallerArchInFamily('folo2-n')).toBeUndefined();

    const fixes = oomFixes({ arch: 'folo2-n', batch: 4, imgsz: 320, device: 'gpu:0' });
    expect(fixes.some((f) => f.kind === 'arch')).toBe(false);
    const medFixes = oomFixes({ arch: 'folo2-m', batch: 16, imgsz: 640, device: 'gpu:0' });
    expect(medFixes.find((f) => f.kind === 'arch')?.query.arch).toBe('folo2-s');
  });

  it('estimates training time/VRAM and formats CLI commands for all folo2-* scales', () => {
    for (const a of V2_ARCHES) {
      const e = estimate(a, 1000, 10, 16, 640, false, false);
      expect(e.imgsPerS).toBeGreaterThan(0);
      expect(e.totalSeconds).toBeGreaterThan(0);
      expect(e.vramBytes).toBeGreaterThan(0);
    }
    const cfg: TrainConfig = {
      name: 'shapes-folo2-s',
      dataset_id: 'ds_shapes',
      arch: 'folo2-s',
      epochs: 20,
      batch: 16,
      imgsz: 640,
      device: 'auto',
      optimizer: 'auto',
      lr0: 0.01,
      seed: 0,
      weights: null,
      augment: DEFAULT_AUGMENT,
    };
    const cli = trainCommand(cfg, 'D:/data/shapes');
    expect(cli.line).toContain('--model folo2-s');
    expect(archesForFamily('folo2')).toHaveLength(5);
  });

  it('parses ?arch=folo2-* queries, persists folo2 in drafts, and blocks cloud runs for folo2', () => {
    const q = new URLSearchParams('arch=folo2-m&dataset=ds_shapes');
    expect(parseInitialQuery(q, null, null).arch).toBe('folo2-m');

    expect(
      isDraft({
        server: '',
        query: '',
        datasetId: 'ds_shapes',
        arch: 'folo2-l',
        epochs: 50,
        batch: '16',
        imgsz: '640',
        imgszChosen: false,
        device: 'auto',
        optimizer: 'auto',
        amp: false,
        e2e: false,
        lr0: 0.01,
        seed: '0',
        seedOver: false,
        weights: 'none',
        name: 'shapes-folo2-l',
        nameEdited: false,
        augment: DEFAULT_AUGMENT,
        closeEdited: false,
        advancedOpen: false,
        augOpen: false,
        cloud: { target: 'local', offerId: null, offerChosen: false, hours: 2, hoursEdited: false },
      }),
    ).toBe(true);

    const fakeCloud = { errors: { runon: null, offer: null, hours: null } } as unknown as CloudPick;
    const errs = formErrors(
      'shapes-folo2-s',
      50,
      0.01,
      '0',
      DEFAULT_AUGMENT,
      { id: 'ds_shapes', name: 'shapes', splits: { train: 100, val: 20, test: 0 }, classes: ['circle'] } as never,
      null,
      'none',
      undefined,
      'folo2-s',
      null,
      true,
      'auto',
      null,
      false,
      true,
      fakeCloud,
    );
    expect(errs.runon).toContain('FOLO v2 trains locally only');
  });
});

describe('ModelSection FOLO v1 / FOLO v2 family selector', () => {
  const emptyTp = { data: null, loading: false, error: undefined } as unknown as Resource<ThroughputTable | null>;

  it('switches cleanly between FOLO v1 and FOLO v2 while preserving scale and disables v2 when unsupported', () => {
    const host = document.body.appendChild(document.createElement('div'));
    const arch = signal<Arch>('folo-s');
    const weights = signal('none');
    const folo2Supported = signal(true);
    const view = render(
      () => (
        <ModelSection
          bind:arch={arch}
          bind:weights={weights}
          folo2Supported={folo2Supported.get()}
          tp={emptyTp}
          oomRun={null}
          was={() => undefined}
          pickedPublished={null}
          selArch={arches.find((a) => a.arch === arch.get())}
          startHint="Fine-tuning from a trained model converges much faster."
          pickedAbout={null}
          errors={cleanErrors()}
          weightChoices={[]}
          otherChoices={[]}
          publishedOptions={[]}
          scoreCtx={() => ({ split: null, dataset: null }) as never}
          pickWeights={() => {}}
        />
      ),
      host,
    );
    flushSync();

    // Initially FOLO v1 with folo-s selected.
    const familyRadios = () => [...host.querySelectorAll<HTMLButtonElement>('#arch-family [role="radio"]')];
    expect(familyRadios().map((b) => b.textContent?.trim())).toEqual(['FOLO v1', 'FOLO v2']);
    expect(host.querySelector<HTMLInputElement>('input[name="arch"]:checked')?.value).toBe('folo-s');

    // Clicking FOLO v2 preserves scale -> folo2-s and swaps table rows to folo2-n..x.
    familyRadios()[1]!.click();
    flushSync();
    expect(arch.get()).toBe('folo2-s');
    expect([...host.querySelectorAll<HTMLInputElement>('input[name="arch"]')].map((i) => i.value)).toEqual([...V2_ARCHES]);

    // Switching back to FOLO v1 after picking folo2-m maps cleanly to folo-m.
    arch.set('folo2-m');
    flushSync();
    familyRadios()[0]!.click();
    flushSync();
    expect(arch.get()).toBe('folo-m');

    // Server without folo2_arch disables the FOLO v2 option with an explanatory note.
    folo2Supported.set(false);
    flushSync();
    expect(familyRadios()[1]!.disabled).toBe(true);
    expect(host.querySelector('#arch-family-note')?.textContent).toContain('folo2_arch');

    view.unmount();
    host.remove();
  });
});

describe('MockApi & HttpApi folo2 contracts', () => {
  it('MockApi advertises folo2_arch, creates/estimates local folo2 runs, and refuses cloud folo2 runs', async () => {
    const api = new MockApi('normal', 0);
    const sys = await api.system();
    expect(sys.features).toContain(FOLO2_FEATURE);

    const cfg: TrainConfig = {
      name: 'shapes-folo2-n',
      dataset_id: 'ds_shapes',
      arch: 'folo2-n',
      epochs: 10,
      batch: 16,
      imgsz: 640,
      device: 'auto',
      optimizer: 'auto',
      lr0: 0.01,
      seed: 0,
      weights: null,
      augment: DEFAULT_AUGMENT,
    };
    const est = await api.estimateRun(cfg);
    expect(est.cli).toContain('--model folo2-n');

    const run = await api.createRun(cfg);
    expect(run.config.arch).toBe('folo2-n');
    const cli = await api.runCli(run.id);
    expect(cli.cli).toContain('--model folo2-n');

    const conns = await api.cloudConnections();
    const offers = await api.cloudOffers(conns[0]!.id);
    await expect(
      api.createRun({
        ...cfg,
        compute: { connection_id: conns[0]!.id, offer_id: offers.offers[0]!.id, max_hours: 1 },
      }),
    ).rejects.toMatchObject({ status: 400, code: 'invalid_config' });
  });

  it('HttpApi.model() promotes wire folo-* arch to canonical folo2-* from x-folo-arch header', async () => {
    const wireModel: Model = {
      id: 'm_v2',
      name: 'shapes-folo2-s-best',
      arch: 'folo-s',
      imgsz: 640,
      classes: ['circle'],
      params: 9_850_000,
      flops_g: 22.4,
      bytes: 20_000_000,
      created: '2026-11-27T00:00:00Z',
      run_id: null,
      metrics: null,
    };
    vi.stubGlobal(
      'fetch',
      vi.fn(
        async () =>
          new Response(JSON.stringify(wireModel), {
            status: 200,
            headers: { 'content-type': 'application/json', 'x-folo-arch': 'folo2-s' },
          }),
      ),
    );
    const api = new HttpApi('http://127.0.0.1:7878');
    const loaded = await api.model('m_v2');
    expect(loaded.arch).toBe('folo2-s');
  });

  it('archFromCli extracts folo2-* from argv and HttpApi preserves canonical folo2-* across runs(), run(), models(), and runCli()', async () => {
    expect(archFromCli({ argv: ['folo', 'train', '--model', 'folo2-m'] })).toBe('folo2-m');
    expect(archFromCli({ argv: ['folo', 'train', '--model=folo2-l'] })).toBe('folo2-l');
    expect(archFromCli({ argv: ['folo', 'train', '--model', 'folo-s'] })).toBe('folo-s');

    const wireRun: Run = {
      id: 'run_v2',
      name: 'shapes-folo2-m',
      status: 'running',
      config: {
        name: 'shapes-folo2-m',
        dataset_id: 'ds_shapes',
        arch: 'folo-m',
        epochs: 20,
        batch: 16,
        imgsz: 640,
        device: 'auto',
        optimizer: 'auto',
        lr0: 0.01,
        seed: 0,
        weights: null,
        augment: DEFAULT_AUGMENT,
      },
      device: 'gpu:0',
      epoch: 1,
      epochs: 20,
      step: 10,
      steps_per_epoch: 100,
      eta_s: 60,
      model_id: null,
      created: '2026-11-27T00:00:00Z',
      started: '2026-11-27T00:00:01Z',
      finished: null,
      error: null,
      best: null,
    };
    const wireModel: Model = {
      id: 'm_v2_run',
      name: 'shapes-folo2-m-best',
      arch: 'folo-m',
      imgsz: 640,
      classes: ['circle'],
      params: 21_000_000,
      flops_g: 50.0,
      bytes: 42_000_000,
      created: '2026-11-27T00:00:00Z',
      run_id: 'run_v2',
      metrics: null,
    };
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: string | URL) => {
        const url = String(input);
        if (url.endsWith('/api/runs/run_v2/cli')) {
          return new Response(JSON.stringify({ argv: ['folo', 'train', '--model', 'folo2-m'], cli: 'folo train --model folo2-m' }), {
            status: 200,
            headers: { 'content-type': 'application/json' },
          });
        }
        if (url.endsWith('/api/runs')) {
          return new Response(JSON.stringify({ runs: [wireRun] }), { status: 200, headers: { 'content-type': 'application/json' } });
        }
        if (url.endsWith('/api/runs/run_v2')) {
          return new Response(JSON.stringify(wireRun), { status: 200, headers: { 'content-type': 'application/json' } });
        }
        if (url.endsWith('/api/models')) {
          return new Response(JSON.stringify({ models: [wireModel] }), { status: 200, headers: { 'content-type': 'application/json' } });
        }
        return new Response(JSON.stringify(wireRun), { status: 200, headers: { 'content-type': 'application/json' } });
      }),
    );

    const api = new HttpApi('http://127.0.0.1:7878');
    // Reading runCli caches the canonical folo2-m arch for run_v2.
    await api.runCli('run_v2');
    expect((await api.run('run_v2')).config.arch).toBe('folo2-m');
    expect((await api.runs())[0]!.config.arch).toBe('folo2-m');
    // models() also promotes any checkpoint linked via run_id to run_v2!
    expect((await api.models())[0]!.arch).toBe('folo2-m');
  });

  it('HttpApi cold-load runs() and models() hydrate canonical folo2-* once per id and skip probes when folo2_arch is absent', async () => {
    const wireRun: Run = {
      id: 'run_cold',
      name: 'shapes-folo2-n',
      status: 'completed',
      config: {
        name: 'shapes-folo2-n',
        dataset_id: 'ds_shapes',
        arch: 'folo-n',
        epochs: 1,
        batch: 4,
        imgsz: 128,
        device: 'cpu',
        optimizer: 'auto',
        lr0: 0.01,
        seed: 0,
        weights: null,
        augment: DEFAULT_AUGMENT,
      },
      device: 'cpu',
      epoch: 1,
      epochs: 1,
      step: 3,
      steps_per_epoch: 3,
      eta_s: 0,
      model_id: 'm_cold',
      created: '2026-11-27T00:00:00Z',
      started: '2026-11-27T00:00:01Z',
      finished: '2026-11-27T00:00:02Z',
      error: null,
      best: null,
    };
    const wireModel: Model = {
      id: 'm_cold',
      name: 'shapes-folo2-n-best',
      arch: 'folo-n',
      imgsz: 128,
      classes: ['circle'],
      params: 3_000_000,
      flops_g: 8.0,
      bytes: 6_000_000,
      created: '2026-11-27T00:00:02Z',
      run_id: 'run_cold',
      metrics: null,
    };
    const urls: string[] = [];
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: string | URL) => {
        const url = String(input);
        urls.push(url);
        if (url.endsWith('/api/system')) {
          return new Response(JSON.stringify({ features: [] }), { status: 200, headers: { 'content-type': 'application/json' } });
        }
        if (url.endsWith('/api/runs')) {
          return new Response(JSON.stringify({ runs: [wireRun] }), { status: 200, headers: { 'content-type': 'application/json' } });
        }
        if (url.endsWith('/api/models')) {
          return new Response(JSON.stringify({ models: [wireModel] }), { status: 200, headers: { 'content-type': 'application/json' } });
        }
        if (url.endsWith('/api/models/m_cold')) {
          return new Response(JSON.stringify(wireModel), {
            status: 200,
            headers: { 'content-type': 'application/json', 'x-folo-arch': 'folo2-n' },
          });
        }
        return new Response(JSON.stringify({ argv: ['folo', 'train', '--model', 'folo2-n'], cli: 'folo train --model folo2-n' }), {
          status: 200,
          headers: { 'content-type': 'application/json' },
        });
      }),
    );

    // Cold session: runs() hydrates via model(m_cold) x-folo-arch and promotes BOTH runs() and models().
    const api = new HttpApi('http://127.0.0.1:7895');
    const runs1 = await api.runs();
    expect(runs1[0]!.config.arch).toBe('folo2-n');
    const models1 = await api.models();
    expect(models1[0]!.arch).toBe('folo2-n');
    const probesAfterFirst = urls.filter((u) => u.includes('/models/m_cold') || u.includes('/cli')).length;
    expect(probesAfterFirst).toBe(1);

    // Second poll makes ZERO extra per-item probe requests.
    await api.runs();
    await api.models();
    expect(urls.filter((u) => u.includes('/models/m_cold') || u.includes('/cli')).length).toBe(probesAfterFirst);

    // Older server without folo2_arch makes ZERO probe requests on list calls.
    urls.length = 0;
    const oldApi = new HttpApi('http://127.0.0.1:7895');
    await oldApi.system();
    const oldRuns = await oldApi.runs();
    const oldModels = await oldApi.models();
    expect(oldRuns[0]!.config.arch).toBe('folo-n');
    expect(oldModels[0]!.arch).toBe('folo-n');
    expect(urls.filter((u) => u.includes('/models/m_cold') || u.includes('/cli'))).toEqual([]);
  });
});
