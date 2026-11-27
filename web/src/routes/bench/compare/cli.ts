/**
 * Pure CLI command builder for `folo bench compare` (crates/folo-cli/src/cli/bench.rs):
 * compares a benchmark's active baseline against a selected candidate entry, with optional
 * output format (`table` / `markdown` / `csv`) and CI regression-gate flags
 * (`--fail-on-regression` and `--min-delta POINTS`).
 */
import { quote } from '$lib/trainCli';

export type BenchCompareCliFormat = 'table' | 'markdown' | 'csv';

export interface BenchCompareCliOptions {
  benchId: string;
  baselineId: string;
  candidateId: string;
  format?: BenchCompareCliFormat;
  failOnRegression?: boolean;
  /** Optional `--min-delta` points as raw user input (empty omits the flag, using the CLI default `0.0`). */
  minDelta?: string;
  /** Optional FOLO server URL (`--server <URL>`) so the command targets the same server from CI or another host. */
  serverUrl?: string | null;
}

export interface BenchCompareCliResult {
  /** Arguments after `folo`, matching `crates/folo-cli/src/cli/bench.rs` verbatim. */
  argv: string[];
  /** Shell-safe command line (bare tokens when plain; PowerShell single-quoted otherwise). */
  line: string;
  /** Validation message when `--min-delta` holds non-numeric / non-finite input, else null. */
  minDeltaError: string | null;
}

/** Validates an optional `--min-delta` input string (empty = default `0.0`, finite numbers including negatives allowed). */
export function parseMinDelta(raw: string | undefined): { value: number | null; error: string | null } {
  const trimmed = (raw ?? '').trim();
  if (!trimmed) return { value: null, error: null };
  const n = Number(trimmed);
  if (!Number.isFinite(n)) {
    return { value: null, error: 'Enter a finite score delta in points (e.g. 0 or -0.5).' };
  }
  return { value: Object.is(n, -0) ? 0 : n, error: null };
}

/**
 * Builds a shell-safe `folo bench compare` invocation matching `BenchCompareArgs` in `crates/folo-cli/src/cli/bench.rs`.
 * Negative `--min-delta` values are attached (`--min-delta=-0.5`) so PowerShell and clap both parse them unambiguously.
 */
export function buildBenchCompareCli(opts: BenchCompareCliOptions): BenchCompareCliResult {
  const format = opts.format ?? 'table';
  const fail = !!opts.failOnRegression;
  const parsed = fail ? parseMinDelta(opts.minDelta) : { value: null, error: null };

  const argv: string[] = ['bench', 'compare'];
  const parts: string[] = ['folo bench compare'];

  const positionals = [opts.benchId, opts.baselineId, opts.candidateId];
  const needsEndOfFlags = positionals.some((p) => p.startsWith('-'));
  if (!needsEndOfFlags) {
    for (const p of positionals) {
      argv.push(p);
      parts.push(quote(p));
    }
  }

  if (format !== 'table') {
    argv.push('--format', format);
    parts.push(`--format ${quote(format)}`);
  }

  if (fail) {
    argv.push('--fail-on-regression');
    parts.push('--fail-on-regression');
    if (parsed.value !== null) {
      const text = String(parsed.value);
      if (text.startsWith('-')) {
        const attached = `--min-delta=${text}`;
        argv.push(attached);
        parts.push(quote(attached));
      } else {
        argv.push('--min-delta', text);
        parts.push(`--min-delta ${quote(text)}`);
      }
    }
  }

  const server = opts.serverUrl?.trim();
  if (server) {
    if (server.startsWith('-')) {
      const attached = `--server=${server}`;
      argv.push(attached);
      parts.push(quote(attached));
    } else {
      argv.push('--server', server);
      parts.push(`--server ${quote(server)}`);
    }
  }

  if (needsEndOfFlags) {
    argv.push('--', ...positionals);
    parts.push('--', ...positionals.map(quote));
  }

  return {
    argv,
    line: parts.join(' '),
    minDeltaError: parsed.error,
  };
}
