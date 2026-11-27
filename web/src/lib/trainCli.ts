// The `folo train` command for a TrainConfig: the inverse of the CLI's own mapping (crates/folo-cli/src/train/config.rs
// builds a TrainConfig from TrainArgs, whose flags and defaults are crates/folo-cli/src/cli.rs). FLAGS is the one table
// of that mapping: a flag, the config value it carries and the clap default it can leave out. The commands it makes are
// parsed by the real clap parser in crates/folo-cli/tests/studio_cli.rs (fixture: routes/runnew/cli-fixtures.json).

import type { Model, TrainConfig } from '$lib/api/types';
import { DEFAULT_MAX_PREEMPT_RETRIES, validPreemptRetries } from '$lib/cloud';

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
  // `--device` conflicts with `--cloud` in `folo train` (the cloud worker picks its own GPU).
  { flag: '--device', get: (c) => (c.compute?.connection_id ? null : c.device), default: 'auto' },
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
  // Cloud compute flags (`crates/folo-cli/src/cli.rs`, TrainArgs): `--offer`, `--max-hours`, `--disk-gb`,
  // and `--max-preempt-retries` all require `--cloud`.
  { flag: '--cloud', get: (c) => c.compute?.connection_id || null },
  { flag: '--offer', get: (c) => (c.compute?.connection_id ? (c.compute.offer_id || null) : null) },
  { flag: '--max-hours', get: (c) => (c.compute?.connection_id ? (c.compute.max_hours ?? null) : null), default: '12' },
  { flag: '--disk-gb', get: (c) => (c.compute?.connection_id ? (c.compute.disk_gb ?? null) : null) },
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

/** CLI override flag enabling the GPU VRAM-budget auto-batch prober (`TrainConfig.auto_batch`, docs/VISION.md). */
export const AUTO_BATCH_FLAG = '--set auto_batch=true';

/**
 * Splits a PowerShell CLI command line into top-level flag segments (`folo train`, `--data ...`, `--set ...`),
 * ignoring ` --` inside PowerShell single-quoted strings (`'...'`, including typographic single quotes).
 */
export function splitCliFlags(line: string): string[] {
  const parts: string[] = [];
  let start = 0;
  let inQuote = false;
  const isSingleQuote = (c: string) => c === "'" || c === '\u2018' || c === '\u2019' || c === '\u201A' || c === '\u201B';
  for (let i = 0; i < line.length; i++) {
    const ch = line[i]!;
    if (isSingleQuote(ch)) {
      if (inQuote && i + 1 < line.length && isSingleQuote(line[i + 1]!)) {
        i++;
      } else {
        inQuote = !inQuote;
      }
      continue;
    }
    if (!inQuote && ch === ' ' && line.startsWith('--', i + 1)) {
      parts.push(line.slice(start, i));
      start = i + 1;
    }
  }
  parts.push(line.slice(start));
  return parts;
}

/** Appends `--set auto_batch=true` to a structured CLI preview (`parts` and `line`) without mutating inputs. */
export function withAutoBatch(parts: readonly string[], line: string): { parts: string[]; line: string } {
  const hasAutoBatch = parts.some((p) => {
    const t = p.trim();
    return t === AUTO_BATCH_FLAG || t === '--set=auto_batch=true';
  });
  if (hasAutoBatch) {
    return { parts: [...parts], line };
  }
  return { parts: [...parts, AUTO_BATCH_FLAG], line: `${line} ${AUTO_BATCH_FLAG}` };
}

/**
 * Resolves the `--server` URL for CLI handoffs that reference IDs on the connected FOLO server.
 * Uses the user's explicit `prefs.serverUrl` when set, else `location.origin` while connected to a live server
 * (omitted in mock mode with no configured server URL).
 */
export function cliServerUrl(
  configuredUrl: string,
  mode: string,
  origin = typeof location !== 'undefined' ? location.origin : '',
): string | null {
  const explicit = configuredUrl.trim().replace(/\/+$/, '');
  if (explicit) return explicit;
  if (mode === 'live' && /^https?:\/\//i.test(origin)) return origin.replace(/\/+$/, '');
  return null;
}

/**
 * `folo train` for `config`. `data` is the dataset's path on the server; `model` the model `config.weights` names,
 * where known (its `path` is the file to fine-tune from locally; `--cloud` takes the store model id directly).
 * Pass `{ autoBatch: true }` to preview CLI-only GPU VRAM-budget auto-batching (`--set auto_batch=true`, local runs
 * only) without changing `TrainConfig`, `{ maxPreemptRetries }` for cloud spot-preemption retry overrides
 * (`--max-preempt-retries N`, omitted at default 3), and `{ serverUrl }` to pass `--server <URL>` for cloud runs.
 */
export function trainCommand(
  config: TrainConfig,
  data: string,
  model?: Model,
  opts?: { autoBatch?: boolean; maxPreemptRetries?: number | null; serverUrl?: string | null },
): TrainCommand {
  const isCloud = !!config.compute?.connection_id;
  // `folo train --cloud --weights <W>` resolves `<W>` through the server store by model id or published name
  // (`crates/folo-cli/src/train/cloud.rs`), never a local `.folo` filesystem path.
  const weights: WeightsRef | null = config.weights
    ? isCloud
      ? { kind: 'path', value: config.weights }
      : weightsRef(config.weights, model)
    : null;
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
  const retries = opts?.maxPreemptRetries;
  if (isCloud && typeof retries === 'number' && validPreemptRetries(retries) && retries !== DEFAULT_MAX_PREEMPT_RETRIES) {
    const val = String(retries);
    argv.push('--max-preempt-retries', val);
    parts.push(`--max-preempt-retries ${val}`);
  }
  const server = opts?.serverUrl?.trim();
  if (isCloud && server) {
    if (server.startsWith('-')) {
      argv.push(`--server=${server}`);
      parts.push(quote(`--server=${server}`));
    } else {
      argv.push('--server', server);
      parts.push(`--server ${quote(server)}`);
    }
  }
  // `--set` conflicts with `--cloud` in `folo train`, so auto-batch preview applies only to local runs.
  if (opts?.autoBatch && !isCloud) {
    argv.push('--set', 'auto_batch=true');
    parts.push(AUTO_BATCH_FLAG);
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
