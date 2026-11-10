// The `folo train` command for a TrainConfig: the inverse of the CLI's own mapping (crates/folo-cli/src/train/config.rs
// builds a TrainConfig from TrainArgs, whose flags and defaults are crates/folo-cli/src/cli.rs). FLAGS is the one table
// of that mapping: a flag, the config value it carries and the clap default it can leave out. The commands it makes are
// parsed by the real clap parser in crates/folo-cli/tests/studio_cli.rs (fixture: routes/runnew/cli-fixtures.json).

import type { Model, TrainConfig } from '$lib/api/types';

/** Where the run's weights come from in the command: a file (or published name) the CLI opens, or a placeholder. */
export type WeightsRef = { kind: 'path'; value: string } | { kind: 'placeholder'; value: string; file: string };

/**
 * The model's weights as `folo train --weights` takes them: published weights by their name (`folo weights pull` caches
 * them where the CLI looks, as the server's own command names them), else its file on the server (`Model.path`, an
 * additive field), else a placeholder naming the file. The CLI takes a path or a published name, never a server id.
 */
export function weightsRef(id: string, model: Model | undefined): WeightsRef {
  if (model?.source === 'published') return { kind: 'path', value: model.id };
  const path = modelPath(model);
  if (path) return { kind: 'path', value: path };
  const file = `${model?.name ?? id}.folo`;
  return { kind: 'placeholder', value: `<path to ${file}>`, file };
}

/** The server's path of a model's file; the one place that reads it (`Model.path`, missing from older servers). */
export function modelPath(model: Model | undefined): string | null {
  return model?.path || null;
}

/** What a flag's value is: a word, several, a switch (`true` is the bare flag, `false` leaves it out) or nothing. */
type Value = string | number | readonly number[] | boolean | null | undefined;

interface Flag {
  flag: string;
  get: (c: TrainConfig, ctx: Context) => Value;
  /** The clap default (cli.rs, TrainArgs): a value equal to it is left out, unless the flag is `always` shown. */
  default?: string;
  always?: boolean;
}

interface Context {
  data: string;
  weights: WeightsRef | null;
}

/** `folo train`'s flags in the order the command lists them. */
export const FLAGS: readonly Flag[] = [
  { flag: '--data', get: (_, x) => x.data, always: true },
  // `--weights` takes the file's architecture; with an explicit `--model` it is a usage error (conflicts_with).
  { flag: '--model', get: (c, x) => (x.weights ? null : c.arch), default: 'folo-n', always: true },
  { flag: '--weights', get: (_, x) => x.weights?.value },
  { flag: '--epochs', get: (c) => c.epochs, default: '100', always: true },
  { flag: '--batch', get: (c) => c.batch, default: '16', always: true },
  { flag: '--imgsz', get: (c) => c.imgsz, default: '640', always: true },
  { flag: '--device', get: (c) => c.device, default: 'auto' },
  { flag: '--optimizer', get: (c) => c.optimizer, default: 'auto' },
  { flag: '--lr0', get: (c) => c.lr0, default: '0.01' },
  { flag: '--amp', get: (c) => !!c.amp },
  // The CLI's default head is the --weights file's own, the API's `o2m`: a fine-tune says which.
  { flag: '--head', get: (c, x) => (c.head === 'o2o' ? 'o2o' : x.weights ? 'o2m' : null) },
  { flag: '--mosaic', get: (c) => c.augment.mosaic, default: '1' },
  { flag: '--mixup', get: (c) => c.augment.mixup, default: '0' },
  // Three values (`num_args = 3`): clap does not split one `h,s,v` word into them.
  { flag: '--hsv', get: (c) => c.augment.hsv, default: '0.015 0.7 0.4' },
  { flag: '--fliplr', get: (c) => c.augment.fliplr, default: '0.5' },
  { flag: '--scale', get: (c) => c.augment.scale, default: '0.5' },
  { flag: '--translate', get: (c) => c.augment.translate, default: '0.1' },
  { flag: '--close-mosaic', get: (c) => c.augment.close_mosaic, default: '10' },
  { flag: '--seed', get: (c) => c.seed, default: '0' },
  { flag: '--name', get: (c) => c.name.trim() || null },
];

/**
 * The shell the command is written for, as New run labels it ("In PowerShell:", like ui/Remedy's commands): FOLO's
 * servers are Windows machines first, and paths are written as they are (`D:\data`, `D:/data`).
 */
export const SHELL = 'PowerShell';

/**
 * A word as PowerShell reads it: bare when it is plain, else in single quotes, where a quote is doubled (`'O''Brien'`).
 * PowerShell takes the typographic single quotes (‘ ’ ‚ ‛) for `'` too, so they are doubled as well. Not bare: a comma
 * (an array to PowerShell, passed as separate words), `@` (splatting), `$`, `#`, backticks, brackets and the rest,
 * and `--%`, which would stop PowerShell parsing the line.
 */
export function quote(word: string): string {
  return /^[\w%+=:./\\-]+$/.test(word) && word !== '--%' ? word :`'${word.replace(/['‘’‚‛]/g, '$&$&')}'`;
}

export interface TrainCommand {
  /** The arguments after `folo`, as the CLI receives them. */
  argv: string[];
  /** One line per flag, quoted: `folo train`, `--data D:/x`, … (joined with spaces, or with PowerShell's backtick continuations). */
  parts: string[];
  /** The command line. */
  line: string;
  /** The weights file the command names by placeholder (the command does not run as it is), else null. */
  placeholder: string | null;
}

/**
 * `folo train` for `config`. `data` is the dataset's path on the server; `model` the model `config.weights` names,
 * where known (its `path` is the file to fine-tune from).
 */
export function trainCommand(config: TrainConfig, data: string, model?: Model): TrainCommand {
  const weights = config.weights ? weightsRef(config.weights, model) : null;
  const ctx: Context = { data, weights };
  const argv = ['train'];
  const parts = ['folo train'];
  for (const f of FLAGS) {
    const v = f.get(config, ctx);
    if (v === null || v === undefined || v === false) continue;
    if (v === true) {
      argv.push(f.flag);
      parts.push(f.flag);
      continue;
    }
    const words = Array.isArray(v) ? v.map(String) : [String(v)];
    if (!f.always && words.join(' ') === f.default) continue;
    // A value that starts with `-` (a run named `--amp`) is attached, as the server's command does, so clap never
    // reads it as a flag.
    const word = words[0];
    if (words.length === 1 && word?.startsWith('-')) {
      argv.push(`${f.flag}=${word}`);
      parts.push(quote(`${f.flag}=${word}`));
      continue;
    }
    argv.push(f.flag, ...words);
    // A placeholder stays readable (`<path to x.folo>`): it is there to be replaced, not run.
    const shown = f.flag === '--weights' && weights?.kind === 'placeholder' ? words : words.map(quote);
    parts.push(`${f.flag} ${shown.join(' ')}`);
  }
  return { argv, parts, line: parts.join(' '), placeholder: weights?.kind === 'placeholder' ? weights.file : null };
}

/**
 * The command a server built for the run (an additive field of the estimate; null where it sends none): preferred
 * over Studio's own, since only the server knows its files. The one place that reads it.
 */
export function serverCommand(est: { cli?: string | null } | null | undefined): string | null {
  return est?.cli || null;
}
