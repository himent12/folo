import { readFileSync, writeFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import type { Model, TrainConfig } from '$lib/api/types';
import { cliServerUrl, quote, serverCommand, trainCommand, weightsRef } from './trainCli';

const base: TrainConfig = {
  name: 'fasteners-s',
  dataset_id: 'ds_fasteners',
  arch: 'folo-s',
  epochs: 100,
  batch: 16,
  imgsz: 640,
  device: 'auto',
  optimizer: 'auto',
  lr0: 0.01,
  weights: null,
  augment: { mosaic: 1, mixup: 0, hsv: [0.015, 0.7, 0.4], fliplr: 0.5, scale: 0.5, translate: 0.1, close_mosaic: 10 },
  seed: 0,
};
const model = (extra: Partial<Model> = {}) => ({ id: 'm_d849e303', name: 'fasteners-n-best', arch: 'folo-n', ...extra }) as Model;
const ft = (extra: Partial<TrainConfig> = {}): TrainConfig => ({ ...base, name: 'fasteners-n-2', arch: 'folo-n', epochs: 15, weights: 'm_d849e303', ...extra });

/**
 * PowerShell's word splitting for the commands this module writes: bare words and single-quoted ones, where a doubled
 * quote is one (`'O''Brien'`), the typographic single quotes counting as `'` (as crates/folo-cli/tests/studio_cli.rs).
 */
function split(line: string): string[] {
  const out: string[] = [];
  const re = /['‘’‚‛]((?:[^'‘’‚‛]|['‘’‚‛]{2})*)['‘’‚‛]|(\S+)/g;
  for (let m = re.exec(line); m; m = re.exec(line)) out.push(m[1] !== undefined ? m[1].replace(/(['‘’‚‛])['‘’‚‛]/g, '$1') : m[2]!);
  return out;
}

describe('trainCommand (DOGFOOD-3 S1)', () => {
  it('fine-tunes from the weights file without --model, which conflicts with --weights', () => {
    const c = trainCommand(ft({ amp: true }), 'D:/folo-dogfood3/fasteners', model({ path: 'D:/folo/models/fasteners-n-best.folo' }));
    expect(c.line).toBe('folo train --data D:/folo-dogfood3/fasteners --weights D:/folo/models/fasteners-n-best.folo --epochs 15 --batch 16 --imgsz 640 --amp --head o2m --name fasteners-n-2');
    expect(c.argv).not.toContain('--model');
    expect(c.placeholder).toBeNull();
  });

  it('names the file by placeholder where the server does not report its path, never the model id', () => {
    const c = trainCommand(ft(), 'D:/fasteners', model());
    expect(c.line).toContain('--weights <path to fasteners-n-best.folo>');
    expect(c.line).not.toContain('m_d849e303');
    expect(c.placeholder).toBe('fasteners-n-best.folo');
    // An unknown model: its id is all there is to name the file by.
    expect(weightsRef('m_x', undefined)).toEqual({ kind: 'placeholder', value: '<path to m_x.folo>', file: 'm_x.folo' });
  });

  it('names published weights by their published name, as the server does', () => {
    const coco = model({ id: 'folo-n-coco', name: 'folo-n-coco', source: 'published', about: 'folo-n trained on COCO' });
    const c = trainCommand(ft({ weights: 'folo-n-coco' }), 'D:/fasteners', coco);
    expect(c.line).toBe('folo train --data D:/fasteners --weights folo-n-coco --epochs 15 --batch 16 --imgsz 640 --head o2m --name fasteners-n-2');
    expect(c.placeholder).toBeNull();
    // Even where the server also reports its cached file: the name is what the CLI looks up.
    expect(weightsRef('folo-n-coco', { ...coco, path: 'C:/Users/me/.folo/weights/folo-n-coco.folo' })).toEqual({ kind: 'path', value: 'folo-n-coco' });
  });

  it('trains from scratch with --model, leaving out what the CLI defaults to', () => {
    expect(trainCommand(base, 'D:/fasteners').line).toBe('folo train --data D:/fasteners --model folo-s --epochs 100 --batch 16 --imgsz 640 --name fasteners-s');
  });

  it('carries every TrainConfig setting that differs from the CLI default', () => {
    const c = trainCommand(
      { ...base, device: 'gpu:1', optimizer: 'adamw', lr0: 0.002, head: 'o2o', seed: 7, augment: { mosaic: 0.5, mixup: 0.1, hsv: [0, 0.5, 0.3], fliplr: 0, scale: 0.25, translate: 0.2, close_mosaic: 5 } },
      'D:/fasteners',
    );
    expect(c.parts.slice(6)).toEqual([
      '--device gpu:1',
      '--optimizer adamw',
      '--lr0 0.002',
      '--head o2o',
      '--mosaic 0.5',
      '--mixup 0.1',
      '--hsv 0 0.5 0.3',
      '--fliplr 0',
      '--scale 0.25',
      '--translate 0.2',
      '--close-mosaic 5',
      '--seed 7',
      '--name fasteners-s',
    ]);
  });

  it('quotes words as PowerShell reads them: an apostrophe is doubled inside single quotes', () => {
    expect(quote('D:/data/fasteners')).toBe('D:/data/fasteners');
    expect(quote('D:\\data\\fasteners')).toBe('D:\\data\\fasteners');
    expect(quote('D:/my data/fasteners')).toBe("'D:/my data/fasteners'");
    expect(quote("D:/O'Brien/data")).toBe("'D:/O''Brien/data'");
    // PowerShell ends a single-quoted string at a typographic quote too: doubled the same way.
    expect(quote('D:/O’Brien’s/data')).toBe("'D:/O’’Brien’’s/data'");
    // An array, splatting, a variable, a comment, an escape, a subexpression, a separator, the stop-parsing token.
    for (const w of ['a,b', '@x', '$x', '#x', 'a`b', '(x)', 'a;b', '--%']) expect(quote(w)).toBe(`'${w}'`);
    const c = trainCommand(base, 'C:/Users/me/My Datasets/fasteners');
    expect(split(c.line).slice(1)).toEqual(c.argv);
  });

  it("quotes apostrophes in the dataset and weights paths, and in the run's name", () => {
    const c = trainCommand(ft({ name: "Bob's fasteners" }), "D:/O'Brien/data", model({ path: "D:/models/O'Brien's best.folo" }));
    expect(c.line).toBe("folo train --data 'D:/O''Brien/data' --weights 'D:/models/O''Brien''s best.folo' --epochs 15 --batch 16 --imgsz 640 --head o2m --name 'Bob''s fasteners'");
    expect(split(c.line).slice(1)).toEqual(c.argv);
    expect(c.argv).toContain("D:/O'Brien/data");
  });

  it('attaches a value that starts with a dash, so the CLI never reads it as a flag (as the server does)', () => {
    const c = trainCommand({ ...base, name: '--amp' }, 'D:/fasteners');
    expect(c.argv).toContain('--name=--amp');
    expect(c.argv).not.toContain('--amp');
    expect(c.line).toBe('folo train --data D:/fasteners --model folo-s --epochs 100 --batch 16 --imgsz 640 --name=--amp');
    const spaced = trainCommand({ ...base, name: '-my run' }, 'D:/fasteners');
    expect(spaced.argv.at(-1)).toBe('--name=-my run');
    expect(split(spaced.line).slice(1)).toEqual(spaced.argv);
  });

  it("prefers the server's own command where it sends one", () => {
    expect(serverCommand({ cli: 'folo train --data x' })).toBe('folo train --data x');
    expect(serverCommand({})).toBeNull();
    expect(serverCommand(null)).toBeNull();
  });

  it('appends --set auto_batch=true when autoBatch option is enabled without altering TrainConfig', () => {
    const c = trainCommand(base, 'D:/fasteners', undefined, { autoBatch: true });
    expect(c.argv.slice(-2)).toEqual(['--set', 'auto_batch=true']);
    expect(c.parts.at(-1)).toBe('--set auto_batch=true');
    expect(split(c.line).slice(1)).toEqual(c.argv);
  });

  it('emits cloud flags and --max-preempt-retries while omitting conflicting --device and --set auto_batch=true', () => {
    const cloudCfg: TrainConfig = {
      ...base,
      device: 'gpu:0',
      compute: { connection_id: 'cc_4d2a', offer_id: 'NVIDIA GeForce RTX 4090@secure', max_hours: 8, disk_gb: 60 },
    };
    const custom = trainCommand(cloudCfg, 'D:/fasteners', undefined, { autoBatch: true, maxPreemptRetries: 5 });
    expect(custom.argv).not.toContain('--device');
    expect(custom.argv).not.toContain('--set');
    expect(custom.line).toBe(
      "folo train --data D:/fasteners --model folo-s --epochs 100 --batch 16 --imgsz 640 --name fasteners-s --cloud cc_4d2a --offer 'NVIDIA GeForce RTX 4090@secure' --max-hours 8 --disk-gb 60 --max-preempt-retries 5",
    );
    expect(split(custom.line).slice(1)).toEqual(custom.argv);

    // Default max_hours (12) and default max_preempt_retries (3) are omitted; 0 is emitted to disable auto-resume.
    const defaultCfg: TrainConfig = {
      ...base,
      compute: { connection_id: 'cc_4d2a', offer_id: 'gpu:0', max_hours: 12 },
    };
    const def = trainCommand(defaultCfg, 'D:/fasteners', undefined, { maxPreemptRetries: 3 });
    expect(def.line).toBe('folo train --data D:/fasteners --model folo-s --epochs 100 --batch 16 --imgsz 640 --name fasteners-s --cloud cc_4d2a --offer gpu:0');
    const disabled = trainCommand(defaultCfg, 'D:/fasteners', undefined, { maxPreemptRetries: 0 });
    expect(disabled.line).toContain('--max-preempt-retries 0');
    expect(split(disabled.line).slice(1)).toEqual(disabled.argv);

    // Out-of-range retries (>20) and empty connection_id are ignored safely.
    const invalidRetry = trainCommand(defaultCfg, 'D:/fasteners', undefined, { maxPreemptRetries: 21 });
    expect(invalidRetry.argv).not.toContain('--max-preempt-retries');
    const emptyConn = trainCommand({ ...base, compute: { connection_id: '', offer_id: 'gpu:0' } }, 'D:/fasteners');
    expect(emptyConn.argv).not.toContain('--cloud');

    // Cloud fine-tuning passes the registered store model ID (never a local file path or placeholder),
    // and includes `--server` when connected to a FOLO server.
    const cloudFt = trainCommand(
      { ...ft({ weights: 'm_saved42' }), compute: { connection_id: 'cc_4d2a', offer_id: 'gpu:0' } },
      'D:/fasteners',
      model({ id: 'm_saved42', path: 'D:/folo/models/saved.folo', source: 'run' }),
      { serverUrl: 'http://gpu-box:7878' },
    );
    expect(cloudFt.placeholder).toBeNull();
    expect(cloudFt.argv).toContain('--weights');
    expect(cloudFt.argv[cloudFt.argv.indexOf('--weights') + 1]).toBe('m_saved42');
    expect(cloudFt.line).not.toContain('saved.folo');
    expect(cloudFt.line).toContain('--server http://gpu-box:7878');
    expect(split(cloudFt.line).slice(1)).toEqual(cloudFt.argv);

    // Even when Model metadata has no path, cloud fine-tune never emits a placeholder.
    const cloudFtNoPath = trainCommand(
      { ...ft({ weights: 'm_imported' }), compute: { connection_id: 'cc_4d2a', offer_id: 'gpu:0' } },
      'D:/fasteners',
      model({ id: 'm_imported', path: undefined }),
    );
    expect(cloudFtNoPath.placeholder).toBeNull();
    expect(cloudFtNoPath.argv[cloudFtNoPath.argv.indexOf('--weights') + 1]).toBe('m_imported');
  });

  it('resolves connected server URL for live and explicit server targets while staying null in bare mock mode', () => {
    expect(cliServerUrl('http://gpu-box:7878/', 'mock', 'http://localhost:3000')).toBe('http://gpu-box:7878');
    expect(cliServerUrl('', 'live', 'https://studio.example.com/')).toBe('https://studio.example.com');
    expect(cliServerUrl('', 'mock', 'http://localhost:3000')).toBeNull();
  });
});

// The commands crates/folo-cli/tests/studio_cli.rs parses with the real clap parser (the CLI's `Cli`): representative
// configs, each with the values the parsed arguments must hold. Placeholder commands are not runnable and stay out.
const FIXTURE = new URL('./trainCli.fixtures.json', import.meta.url);
const withPath = model({ path: 'D:/folo/data/models/fasteners-n-best.folo' });
const cases: { name: string; config: TrainConfig; data: string; model?: Model }[] = [
  { name: 'from scratch', config: base, data: 'D:/folo-dogfood3/fasteners' },
  { name: 'fine-tune from a path', config: ft(), data: 'D:/folo-dogfood3/fasteners', model: withPath },
  { name: 'fine-tune, mixed precision, end to end', config: ft({ amp: true, head: 'o2o', device: 'gpu:0' }), data: 'D:/folo-dogfood3/fasteners', model: withPath },
  { name: 'mixed precision', config: { ...base, amp: true }, data: 'D:/fasteners' },
  { name: 'end to end', config: { ...base, head: 'o2o' }, data: 'D:/fasteners' },
  { name: 'device auto', config: { ...base, device: 'auto' }, data: 'D:/fasteners' },
  { name: 'device cpu', config: { ...base, device: 'cpu' }, data: 'D:/fasteners' },
  { name: 'device gpu:0', config: { ...base, device: 'gpu:0' }, data: 'D:/fasteners' },
  { name: 'device gpu:1', config: { ...base, device: 'gpu:1' }, data: 'D:/fasteners' },
  { name: 'adamw at a custom lr0', config: { ...base, optimizer: 'adamw', lr0: 0.002 }, data: 'D:/fasteners' },
  { name: 'sgd at a custom lr0', config: { ...base, optimizer: 'sgd', lr0: 0.05 }, data: 'D:/fasteners' },
  {
    name: 'augmentation and seed',
    config: { ...base, arch: 'folo-m', batch: 8, imgsz: 960, epochs: 300, seed: 42, augment: { mosaic: 0.5, mixup: 0.1, hsv: [0, 0.5, 0.3], fliplr: 0, scale: 0.25, translate: 0.2, close_mosaic: 0 } },
    data: 'D:/fasteners',
  },
  { name: 'a dataset path with spaces', config: { ...base, arch: 'folo-n' }, data: 'C:/Users/me/My Datasets/fasteners' },
  { name: 'a Windows path', config: { ...base, arch: 'folo-l' }, data: 'D:\\data\\shapes' },
  { name: 'an apostrophe in the dataset path and the name', config: { ...base, name: "Bob's fasteners" }, data: "D:/O'Brien/data" },
  { name: 'a name that starts with a dash', config: { ...base, name: '--amp' }, data: 'D:/fasteners' },
  { name: 'an apostrophe in the weights path', config: ft(), data: 'D:/fasteners', model: model({ path: "C:/Users/O'Brien/FOLO models/fasteners-n-best.folo" }) },
  {
    name: 'fine-tune from published weights',
    config: ft({ weights: 'folo-n-coco' }),
    data: 'D:/folo-dogfood3/fasteners',
    model: model({ id: 'folo-n-coco', name: 'folo-n-coco', source: 'published' }),
  },
];

describe('the CLI fixture (crates/folo-cli/tests/studio_cli.rs)', () => {
  it('is up to date (rewritten when it is not: commit it)', () => {
    const fixture = cases.map(({ name, config, data, model: m }) => {
      const c = trainCommand(config, data, m);
      expect(c.placeholder).toBeNull();
      expect(split(c.line).slice(1)).toEqual(c.argv);
      return {
        name,
        command: c.line,
        argv: c.argv,
        expect: {
          data,
          model: config.weights ? null : config.arch,
          weights: config.weights ? (weightsRef(config.weights, m).value ?? null) : null,
          epochs: config.epochs,
          batch: config.batch,
          imgsz: config.imgsz,
          device: config.device,
          optimizer: config.optimizer,
          lr0: config.lr0,
          amp: !!config.amp,
          head: config.head === 'o2o' ? 'o2o' : config.weights ? 'o2m' : null,
          mosaic: config.augment.mosaic,
          mixup: config.augment.mixup,
          hsv: config.augment.hsv,
          fliplr: config.augment.fliplr,
          scale: config.augment.scale,
          translate: config.augment.translate,
          close_mosaic: config.augment.close_mosaic,
          seed: config.seed,
          name: config.name,
        },
      };
    });
    const text = `${JSON.stringify(fixture, null, 2)}\n`;
    let committed = '';
    try {
      committed = readFileSync(FIXTURE, 'utf8');
    } catch {
      // First run: written below.
    }
    if (committed !== text) writeFileSync(FIXTURE, text);
    expect(committed, 'trainCli.fixtures.json was out of date and has been rewritten: commit it').toBe(text);
  });
});
