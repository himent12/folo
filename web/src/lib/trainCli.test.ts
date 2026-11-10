import { readFileSync, writeFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import type { Model, TrainConfig } from '$lib/api/types';
import { quote, serverCommand, trainCommand, weightsRef } from './trainCli';

const base: TrainConfig = {
  name: 'r6-s',
  dataset_id: 'ds_r6',
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
const ft = (extra: Partial<TrainConfig> = {}): TrainConfig => ({ ...base, name: 'r6-n-2', arch: 'folo-n', epochs: 15, weights: 'm_d849e303', ...extra });

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
    const c = trainCommand(ft({ amp: true }), 'D:/folo-dogfood3/r6', model({ path: 'D:/folo/models/fasteners-n-best.folo' }));
    expect(c.line).toBe('folo train --data D:/folo-dogfood3/r6 --weights D:/folo/models/fasteners-n-best.folo --epochs 15 --batch 16 --imgsz 640 --amp --head o2m --name r6-n-2');
    expect(c.argv).not.toContain('--model');
    expect(c.placeholder).toBeNull();
  });

  it('names the file by placeholder where the server does not report its path, never the model id', () => {
    const c = trainCommand(ft(), 'D:/r6', model());
    expect(c.line).toContain('--weights <path to fasteners-n-best.folo>');
    expect(c.line).not.toContain('m_d849e303');
    expect(c.placeholder).toBe('fasteners-n-best.folo');
    // An unknown model: its id is all there is to name the file by.
    expect(weightsRef('m_x', undefined)).toEqual({ kind: 'placeholder', value: '<path to m_x.folo>', file: 'm_x.folo' });
  });

  it('names published weights by their published name, as the server does', () => {
    const coco = model({ id: 'folo-n-coco', name: 'folo-n-coco', source: 'published', about: 'folo-n trained on COCO' });
    const c = trainCommand(ft({ weights: 'folo-n-coco' }), 'D:/r6', coco);
    expect(c.line).toBe('folo train --data D:/r6 --weights folo-n-coco --epochs 15 --batch 16 --imgsz 640 --head o2m --name r6-n-2');
    expect(c.placeholder).toBeNull();
    // Even where the server also reports its cached file: the name is what the CLI looks up.
    expect(weightsRef('folo-n-coco', { ...coco, path: 'C:/Users/me/.folo/weights/folo-n-coco.folo' })).toEqual({ kind: 'path', value: 'folo-n-coco' });
  });

  it('trains from scratch with --model, leaving out what the CLI defaults to', () => {
    expect(trainCommand(base, 'D:/r6').line).toBe('folo train --data D:/r6 --model folo-s --epochs 100 --batch 16 --imgsz 640 --name r6-s');
  });

  it('carries every TrainConfig setting that differs from the CLI default', () => {
    const c = trainCommand(
      { ...base, device: 'gpu:1', optimizer: 'adamw', lr0: 0.002, head: 'o2o', seed: 7, augment: { mosaic: 0.5, mixup: 0.1, hsv: [0, 0.5, 0.3], fliplr: 0, scale: 0.25, translate: 0.2, close_mosaic: 5 } },
      'D:/r6',
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
      '--name r6-s',
    ]);
  });

  it('quotes words as PowerShell reads them: an apostrophe is doubled inside single quotes', () => {
    expect(quote('D:/data/r6')).toBe('D:/data/r6');
    expect(quote('D:\\data\\r6')).toBe('D:\\data\\r6');
    expect(quote('D:/my data/r6')).toBe("'D:/my data/r6'");
    expect(quote("D:/O'Brien/data")).toBe("'D:/O''Brien/data'");
    // PowerShell ends a single-quoted string at a typographic quote too: doubled the same way.
    expect(quote('D:/O’Brien’s/data')).toBe("'D:/O’’Brien’’s/data'");
    // An array, splatting, a variable, a comment, an escape, a subexpression, a separator, the stop-parsing token.
    for (const w of ['a,b', '@x', '$x', '#x', 'a`b', '(x)', 'a;b', '--%']) expect(quote(w)).toBe(`'${w}'`);
    const c = trainCommand(base, 'C:/Users/me/My Datasets/r6');
    expect(split(c.line).slice(1)).toEqual(c.argv);
  });

  it("quotes apostrophes in the dataset and weights paths, and in the run's name", () => {
    const c = trainCommand(ft({ name: "Bob's r6" }), "D:/O'Brien/data", model({ path: "D:/models/O'Brien's best.folo" }));
    expect(c.line).toBe("folo train --data 'D:/O''Brien/data' --weights 'D:/models/O''Brien''s best.folo' --epochs 15 --batch 16 --imgsz 640 --head o2m --name 'Bob''s r6'");
    expect(split(c.line).slice(1)).toEqual(c.argv);
    expect(c.argv).toContain("D:/O'Brien/data");
  });

  it('attaches a value that starts with a dash, so the CLI never reads it as a flag (as the server does)', () => {
    const c = trainCommand({ ...base, name: '--amp' }, 'D:/r6');
    expect(c.argv).toContain('--name=--amp');
    expect(c.argv).not.toContain('--amp');
    expect(c.line).toBe('folo train --data D:/r6 --model folo-s --epochs 100 --batch 16 --imgsz 640 --name=--amp');
    const spaced = trainCommand({ ...base, name: '-my run' }, 'D:/r6');
    expect(spaced.argv.at(-1)).toBe('--name=-my run');
    expect(split(spaced.line).slice(1)).toEqual(spaced.argv);
  });

  it("prefers the server's own command where it sends one", () => {
    expect(serverCommand({ cli: 'folo train --data x' })).toBe('folo train --data x');
    expect(serverCommand({})).toBeNull();
    expect(serverCommand(null)).toBeNull();
  });
});

// The commands crates/folo-cli/tests/studio_cli.rs parses with the real clap parser (the CLI's `Cli`): representative
// configs, each with the values the parsed arguments must hold. Placeholder commands are not runnable and stay out.
const FIXTURE = new URL('./trainCli.fixtures.json', import.meta.url);
const withPath = model({ path: 'D:/folo/data/models/fasteners-n-best.folo' });
const cases: { name: string; config: TrainConfig; data: string; model?: Model }[] = [
  { name: 'from scratch', config: base, data: 'D:/folo-dogfood3/r6' },
  { name: 'fine-tune from a path', config: ft(), data: 'D:/folo-dogfood3/r6', model: withPath },
  { name: 'fine-tune, mixed precision, end to end', config: ft({ amp: true, head: 'o2o', device: 'gpu:0' }), data: 'D:/folo-dogfood3/r6', model: withPath },
  { name: 'mixed precision', config: { ...base, amp: true }, data: 'D:/r6' },
  { name: 'end to end', config: { ...base, head: 'o2o' }, data: 'D:/r6' },
  { name: 'device auto', config: { ...base, device: 'auto' }, data: 'D:/r6' },
  { name: 'device cpu', config: { ...base, device: 'cpu' }, data: 'D:/r6' },
  { name: 'device gpu:0', config: { ...base, device: 'gpu:0' }, data: 'D:/r6' },
  { name: 'device gpu:1', config: { ...base, device: 'gpu:1' }, data: 'D:/r6' },
  { name: 'adamw at a custom lr0', config: { ...base, optimizer: 'adamw', lr0: 0.002 }, data: 'D:/r6' },
  { name: 'sgd at a custom lr0', config: { ...base, optimizer: 'sgd', lr0: 0.05 }, data: 'D:/r6' },
  {
    name: 'augmentation and seed',
    config: { ...base, arch: 'folo-m', batch: 8, imgsz: 960, epochs: 300, seed: 42, augment: { mosaic: 0.5, mixup: 0.1, hsv: [0, 0.5, 0.3], fliplr: 0, scale: 0.25, translate: 0.2, close_mosaic: 0 } },
    data: 'D:/r6',
  },
  { name: 'a dataset path with spaces', config: { ...base, arch: 'folo-n' }, data: 'C:/Users/me/My Datasets/r6' },
  { name: 'a Windows path', config: { ...base, arch: 'folo-l' }, data: 'D:\\data\\shapes' },
  { name: 'an apostrophe in the dataset path and the name', config: { ...base, name: "Bob's r6" }, data: "D:/O'Brien/data" },
  { name: 'a name that starts with a dash', config: { ...base, name: '--amp' }, data: 'D:/r6' },
  { name: 'an apostrophe in the weights path', config: ft(), data: 'D:/r6', model: model({ path: "C:/Users/O'Brien/FOLO models/fasteners-n-best.folo" }) },
  {
    name: 'fine-tune from published weights',
    config: ft({ weights: 'folo-n-coco' }),
    data: 'D:/folo-dogfood3/r6',
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
