/**
 * Collapsible `folo bench compare` CI regression-gate CLI handoff inside BenchCompare's Side-by-side panel:
 * builds a copyable, shell-safe command comparing the active baseline against a chosen comparison entry,
 * with format selection (`table` / `markdown` / `csv`) and optional `--fail-on-regression` / `--min-delta` flags.
 */
import Check from '$icons/check';
import ChevronRight from '$icons/chevron-right';
import { computed, createUniqueId, css, effect, signal, untrack } from 'fw';
import { hub } from '$lib/api/index';
import type { BenchEntry } from '$lib/api/types';
import { prefs } from '$lib/prefs';
import { cliServerUrl, SHELL } from '$lib/trainCli';
import CommandLine from '$lib/ui/CommandLine';
import Segmented from '$lib/ui/Segmented';
import Select from '$lib/ui/Select';
import { buildBenchCompareCli, type BenchCompareCliFormat } from './cli';

export interface CompareCliProps {
  benchId: string;
  /** Currently compared finished entries; index 0 is always the active baseline. */
  shown: readonly BenchEntry[];
  /** Flags matching `shown`: entries graded on an earlier benchmark fingerprint. */
  old: readonly boolean[];
  /** Distinct human-readable label for each entry. */
  labelOf: (e: BenchEntry) => string;
}

const FORMATS: { value: BenchCompareCliFormat; label: string }[] = [
  { value: 'table', label: 'Table' },
  { value: 'markdown', label: 'Markdown' },
  { value: 'csv', label: 'CSV' },
];

export default function CompareCli(props: CompareCliProps) {
  const uid = createUniqueId();
  const bodyId = `${uid}-body`;
  const candSelectId = `${uid}-cand`;
  const minDeltaId = `${uid}-min-delta`;
  const minDeltaErrId = `${uid}-min-delta-err`;

  const open = signal(false);
  const format = signal<BenchCompareCliFormat>('table');
  const failOnRegression = signal(true);
  const minDelta = signal('');
  const pickedCandidate = signal<string>('');

  const baseline = computed(() => props.shown[0]);
  const candidates = computed(() => props.shown.slice(1));

  // Keep the selected candidate synced to a valid non-baseline entry when entries are swapped or removed.
  effect(() => {
    const list = candidates.get();
    if (!list.length) return;
    const cur = untrack(() => pickedCandidate.get());
    if (!list.some((e) => e.id === cur)) {
      pickedCandidate.set(list[0]!.id);
    }
  });

  const candidate = computed(() => {
    const list = candidates.get();
    const cur = pickedCandidate.get();
    return list.find((e) => e.id === cur) ?? list[0];
  });

  const candidateOptions = computed(() =>
    candidates.get().map((e) => ({ value: e.id, label: props.labelOf(e) })),
  );

  const serverUrl = computed(() => cliServerUrl(prefs.serverUrl, hub.mode));

  const cli = computed(() => {
    const base = baseline.get();
    const cand = candidate.get();
    if (!base || !cand) return null;
    return buildBenchCompareCli({
      benchId: props.benchId,
      baselineId: base.id,
      candidateId: cand.id,
      format: format.get(),
      failOnRegression: failOnRegression.get(),
      minDelta: minDelta.get(),
      serverUrl: serverUrl.get(),
    });
  });

  const hasStalePair = computed(() => {
    const cand = candidate.get();
    if (!cand) return false;
    const candIdx = props.shown.findIndex((e) => e.id === cand.id);
    return !!props.old[0] || (candIdx >= 0 && !!props.old[candIdx]);
  });

  const copiedHint = computed(() =>
    failOnRegression.get()
      ? 'Run this command in CI or a terminal to fail if the candidate score regresses.'
      : 'Run this command in a terminal to print the head-to-head comparison.',
  );

  return (
    <div class="cmp-cli" data-bench-compare-cli>
      <button
        type="button"
        class="cmp-cli-toggle"
        aria-expanded={open.get()}
        aria-controls={open.get() ? bodyId : undefined}
        data-compare-cli-toggle
        onclick={() => open.set(!open.peek())}
      >
        <ChevronRight size={14} class={open.get() ? 'chev open' : 'chev'} aria-hidden="true" />
        <span class="cmp-cli-title">CI regression gate</span>
        <span class="cmp-cli-meta mono">folo bench compare</span>
      </button>
      {open.get() && baseline.get() && candidate.get() && cli.get() ? (
        <div class="cmp-cli-body" id={bodyId}>
          <div class="cmp-cli-head">
            <span class="cmp-cli-shell">In {SHELL}:</span>
            <Segmented label="Output format" size="sm" bind:value={format} options={FORMATS} />
          </div>

          <div class="cmp-cli-controls">
            {candidates.get().length > 1 ? (
              <div class="cmp-cli-field">
                <label class="cmp-cli-lbl" for={candSelectId}>
                  Candidate (vs {props.labelOf(baseline.get()!)})
                </label>
                <Select
                  id={candSelectId}
                  size="sm"
                  value={candidate.get()!.id}
                  options={candidateOptions.get()}
                  onchange={(eid) => eid && pickedCandidate.set(eid)}
                />
              </div>
            ) : (
              <p class="cmp-cli-pair">
                Comparing <span class="pair-name">{props.labelOf(candidate.get()!)}</span> against baseline{' '}
                <span class="pair-name">{props.labelOf(baseline.get()!)}</span>.
              </p>
            )}

            <div class="cmp-cli-gate">
              <label class="cmp-cli-check">
                <input
                  type="checkbox"
                  data-compare-cli-fail
                  checked={failOnRegression.get()}
                  onchange={(e) => failOnRegression.set((e.currentTarget as HTMLInputElement).checked)}
                />
                <span class="cbox" aria-hidden="true">
                  <Check size={11} strokeWidth={2.5} />
                </span>
                <span>
                  Fail on regression (<span class="mono">--fail-on-regression</span>)
                </span>
              </label>

              {failOnRegression.get() ? (
                <div class="cmp-cli-delta">
                  <label class="cmp-cli-lbl" for={minDeltaId}>
                    Min delta (pts)
                  </label>
                  <input
                    id={minDeltaId}
                    type="text"
                    inputmode="decimal"
                    class="delta-in num"
                    class:invalid={!!cli.get()!.minDeltaError}
                    placeholder="0.0"
                    aria-invalid={cli.get()!.minDeltaError ? 'true' : undefined}
                    aria-describedby={cli.get()!.minDeltaError ? minDeltaErrId : undefined}
                    data-compare-cli-min-delta
                    value={minDelta.get()}
                    oninput={(e) => minDelta.set((e.currentTarget as HTMLInputElement).value)}
                  />
                </div>
              ) : null}
            </div>
          </div>

          {cli.get()!.minDeltaError ? (
            <p class="cmp-cli-err" id={minDeltaErrId} role="alert">
              {cli.get()!.minDeltaError}
            </p>
          ) : (
            <CommandLine
              command={cli.get()!.line}
              label="Benchmark compare CLI command"
              copied={copiedHint.get()}
            />
          )}

          {hasStalePair.get() ? (
            <p class="cmp-cli-note">
              One of these entries is from an earlier benchmark fingerprint; <span class="mono">folo bench compare</span>{' '}
              will print a comparability warning.
            </p>
          ) : null}

          {serverUrl.get() ? (
            <p class="cmp-cli-note">
              Targets <span class="mono">{serverUrl.get()}</span>; if this server requires a token, set{' '}
              <span class="mono">FOLO_TOKEN</span> in your terminal or CI.
            </p>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}

css`
  .cmp-cli {
    border-top: 1px solid var(--line-subtle);
    padding: var(--space-3) var(--space-4);
    display: grid;
    gap: var(--space-2);
    min-width: 0;
  }
  .cmp-cli-toggle {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    width: 100%;
    min-height: var(--control-sm);
    padding: 0;
    color: var(--text-primary);
    text-align: left;
  }
  .cmp-cli-toggle :global(svg.chev) {
    flex-shrink: 0;
    color: var(--text-tertiary);
    transition: transform var(--dur-fast) var(--ease-standard);
  }
  .cmp-cli-toggle :global(svg.chev.open) {
    transform: rotate(90deg);
  }
  .cmp-cli-title {
    font-size: var(--type-sm-size);
    font-weight: var(--weight-medium);
    color: var(--text-secondary);
  }
  .cmp-cli-meta {
    margin-left: auto;
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
  }
  .cmp-cli-body {
    display: grid;
    gap: var(--space-3);
    padding-top: var(--space-1);
    min-width: 0;
  }
  .cmp-cli-head {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-2);
  }
  .cmp-cli-shell,
  .cmp-cli-pair,
  .cmp-cli-note {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-tertiary);
  }
  .pair-name {
    color: var(--text-secondary);
    font-weight: var(--weight-medium);
  }
  .cmp-cli-controls {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-3);
  }
  .cmp-cli-field {
    display: inline-flex;
    flex-wrap: wrap;
    align-items: center;
    gap: var(--space-2);
    min-width: 0;
  }
  .cmp-cli-lbl {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
  }
  .cmp-cli-gate {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: var(--space-3);
  }
  .cmp-cli-check {
    position: relative;
    display: inline-flex;
    align-items: center;
    gap: var(--space-2);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
    cursor: pointer;
    width: fit-content;
  }
  .cmp-cli-check input {
    position: absolute;
    opacity: 0;
    width: 1px;
    height: 1px;
  }
  .cbox {
    display: grid;
    place-items: center;
    width: 14px;
    height: 14px;
    border-radius: var(--radius-xs);
    border: 1px solid var(--line-control);
    background: var(--bg-inset);
    color: transparent;
    flex-shrink: 0;
  }
  .cmp-cli-check input:checked + .cbox {
    border-color: var(--text-primary);
    background: var(--text-primary);
    color: var(--bg-panel);
  }
  .cmp-cli-check input:focus-visible + .cbox {
    outline: var(--size-focus-ring) solid var(--focus);
    outline-offset: 2px;
  }
  .cmp-cli-delta {
    display: inline-flex;
    align-items: center;
    gap: var(--space-2);
  }
  .delta-in {
    width: 80px;
    height: var(--control-sm);
    padding: 0 var(--space-2);
    border-radius: var(--radius-sm);
    border: 1px solid var(--line-control);
    background: var(--bg-inset);
    color: var(--text-primary);
    font-size: var(--type-sm-size);
    font-family: var(--font-mono);
  }
  .delta-in:focus-visible {
    outline: var(--size-focus-ring) solid var(--focus);
    outline-offset: 1px;
  }
  .delta-in.invalid {
    border-color: var(--danger-solid);
  }
  .cmp-cli-err {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--danger-text);
  }
  .cmp-cli-note .mono,
  .cmp-cli-check .mono {
    color: var(--text-secondary);
  }
  @media (pointer: coarse) {
    .cmp-cli-toggle {
      min-height: 44px;
    }
    .cmp-cli-check {
      min-height: 44px;
    }
    .delta-in {
      height: 44px;
      font-size: var(--type-lg-size);
    }
  }
`;
