/** The same run from a terminal, under the Summary: the server's command, else Studio's, one flag a line, copyable. */
import ChevronRight from '$icons/chevron-right';
import Copy from '$icons/copy';
import TriangleAlert from '$icons/triangle-alert';
import { computed, css, Index, signal } from 'fw';
import type { Model, TrainConfig } from '$lib/api/types';
import { SHELL, splitCliFlags, trainCommand, withAutoBatch } from '$lib/trainCli';
import Button from '$lib/ui/Button';
import { toast } from '$lib/ui/toast';

/**
 * The command for New run's settings: the server's (`served`, where it builds one: only it knows its files), else
 * Studio's, from the CLI's own mapping ($lib/trainCli). A fine-tune names the weights file by path; where the server
 * does not report it, a placeholder and a note say what to fill in.
 */
export function cliOf(
  config: () => TrainConfig,
  data: () => string,
  model: () => Model | undefined,
  served: () => string | null,
  maxPreemptRetries?: () => number | null,
  serverUrl?: () => string | null,
) {
  const own = computed(() =>
    trainCommand(config(), data(), model(), {
      maxPreemptRetries: maxPreemptRetries?.() ?? null,
      serverUrl: serverUrl?.() ?? null,
    }),
  );
  return {
    parts: computed(() => {
      const s = served();
      return s ? splitCliFlags(s) : own.get().parts;
    }),
    line: computed(() => served() ?? own.get().line),
    placeholder: computed(() => (served() ? null : own.get().placeholder)),
  };
}

export interface CliCommandProps {
  cliParts: string[];
  cli: string;
  cliPlaceholder: string | null;
  /** True when the selected target is CPU (GPU VRAM auto-batch probing is skipped on CPU). */
  onCpu?: boolean;
  /** True when training on cloud compute (`--set auto_batch=true` conflicts with `--cloud`). */
  onCloud?: boolean;
  /** Prefill or linked weights still loading: hold the command until the run's real settings are resolved. */
  pending?: string | null;
  /** The Seed field holds no seed: held as Start is (lib/seed), never a command that would run with another seed. */
  seedMissing?: boolean;
  /** Cloud compute or retry policy incomplete/invalid: hold the command so it never drops `--cloud` or `--max-preempt-retries`. */
  cloudBlocked?: string | null;
}

export default function CliCommand(props: CliCommandProps) {
  /** CLI-preview-only toggle for `--set auto_batch=true` (never mutates `POST /api/runs` payload). */
  const autoBatch = signal(false);
  const autoBatchDisabled = computed(() => !!props.onCpu || !!props.onCloud);
  const autoBatchActive = computed(() => autoBatch.get() && !autoBatchDisabled.get());
  const shown = computed(() =>
    autoBatchActive.get() ? withAutoBatch(props.cliParts, props.cli) : { parts: props.cliParts, line: props.cli },
  );

  // page-level: the Summary rail's Copy command (the phone sheet holds no copy control).
  async function copy(text: string, done: string) {
    try {
      await navigator.clipboard.writeText(text);
      toast.success(done);
    } catch (err) {
      toast.error(err, 'Could not copy');
    }
  }

  // A line with a placeholder is a template, not a command: said above it, on its Copy and in the confirmation, so
  // nothing calls it runnable while it holds "<path to …>".
  // One flag a line, ended by PowerShell's line continuation (a backtick); a line too long for the rail wraps at a space
  // and hangs deeper than the flags.
  return (
    <details class="cli">
      <summary>
        <ChevronRight size={14} aria-hidden="true" />
        CLI command <span class="faint">the same run from a terminal</span>
      </summary>
      {props.pending ? (
        <p class="cli-note">{props.pending}</p>
      ) : props.seedMissing ? (
        <p class="cli-note">Enter a seed first: the command needs one.</p>
      ) : props.cloudBlocked ? (
        <p class="cli-note">{props.cloudBlocked}</p>
      ) : (
        <>
          <div class="cli-top">
            <p class="cli-shell">In {SHELL}:</p>
            <label class="cli-auto" class:off={autoBatchDisabled.get()} title={props.onCloud ? 'Auto-batch VRAM applies to local GPU runs only.' : undefined}>
              <input
                type="checkbox"
                data-auto-batch-toggle
                checked={autoBatchActive.get()}
                disabled={autoBatchDisabled.get()}
                onchange={(e) => autoBatch.set((e.currentTarget as HTMLInputElement).checked)}
              />
              <span>Auto-batch VRAM</span>
            </label>
          </div>
          {props.cliPlaceholder ? (
            <p class="cli-req">
              <TriangleAlert size={14} aria-hidden="true" />
              Weights path required
            </p>
          ) : null}
          <div class="cli-body">
            <code aria-describedby={props.cliPlaceholder ? 'cli-note' : undefined}>
              <Index each={shown.get().parts}>
                {(p, i) => (
                  <span class="ln" class:cont={i > 0}>
                    {props.cliPlaceholder && p() === `--weights <path to ${props.cliPlaceholder}>` ? (
                      <>
                        {'--weights '}
                        <span class="ph">
                          {'<path to '}
                          <span class="ph-file">{props.cliPlaceholder}</span>
                          {'>'}
                        </span>
                      </>
                    ) : (
                      p()
                    )}
                    {i < shown.get().parts.length - 1 ? ' `' : ''}
                  </span>
                )}
              </Index>
            </code>
            <span class="cli-copy">
              {props.cliPlaceholder ? (
                <Button
                  size="sm"
                  variant="ghost"
                  icon={Copy}
                  aria-label="Copy template"
                  title="Copy template"
                  onclick={() => copy(shown.get().line, 'Template copied: replace the weights path')}
                />
              ) : (
                <Button size="sm" variant="ghost" icon={Copy} aria-label="Copy command" title="Copy command" onclick={() => copy(shown.get().line, 'Command copied')} />
              )}
            </span>
          </div>
          {autoBatchActive.get() ? (
            <p class="cli-note" data-auto-batch-note>
              CLI-only: probes GPU VRAM budget (80% target) via <span class="mono">--set auto_batch=true</span>; Start training still uses the selected numeric batch size.
            </p>
          ) : null}
          {props.cliPlaceholder ? (
            <p class="cli-note" id="cli-note">
              Put the path of {props.cliPlaceholder} on the server in its place: this server does not report it.
            </p>
          ) : null}
        </>
      )}
    </details>
  );
}

css`
  .cli {
    margin-top: calc(-1 * var(--space-1));
    padding-top: var(--space-2);
    border-top: 1px solid var(--line-subtle);
  }
  .cli summary {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    height: var(--control-md);
    color: var(--text-secondary);
    font-size: var(--type-sm-size);
    font-weight: var(--weight-medium);
    cursor: pointer;
    list-style: none;
    border-radius: var(--radius-sm);
  }
  .cli summary::-webkit-details-marker {
    display: none;
  }
  .cli summary :global(svg) {
    transition: transform var(--dur-fast) var(--ease-standard);
  }
  .cli[open] summary :global(svg) {
    transform: rotate(90deg);
  }
  .cli summary:hover {
    color: var(--text-primary);
  }
  .cli summary .faint {
    font-weight: var(--weight-regular);
  }
  /* The shell the command is written for, as ui/Remedy names it. */
  .cli-top {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-2);
    margin-top: var(--space-2);
  }
  .cli-shell {
    color: var(--text-tertiary);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
  }
  .cli-auto {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
    cursor: pointer;
    user-select: none;
  }
  .cli-auto.off {
    color: var(--text-disabled);
    cursor: not-allowed;
  }
  .cli-auto input {
    accent-color: var(--accent-solid);
  }
  @media (pointer: coarse) {
    .cli-auto {
      min-height: 44px;
    }
  }
  .cli-body {
    position: relative;
    margin-top: var(--space-1);
  }
  .ln {
    display: block;
    padding-left: 4ch;
    text-indent: -4ch;
  }
  .ln.cont {
    padding-left: 6ch;
  }
  /* The weights file to fill in, in placeholder ink; its name moves to the next line whole where it fits there. */
  .ph {
    color: var(--text-tertiary);
  }
  .ph-file {
    display: inline-block;
    max-width: 100%;
    text-indent: 0;
  }
  /* "Weights path required": the template's state, above the line it qualifies, its mark in the warning hue. */
  .cli-req {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    margin-top: var(--space-1);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    font-weight: var(--weight-medium);
    color: var(--text-primary);
  }
  .cli-req > :global(svg) {
    flex-shrink: 0;
    color: var(--warning-text);
  }
  .cli-note {
    margin-top: var(--space-2);
    color: var(--text-tertiary);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
  }
  .cli-copy {
    position: absolute;
    top: var(--space-1);
    right: var(--space-1);
  }
  code {
    display: block;
    padding: var(--space-2) var(--space-8) var(--space-2) var(--space-3);
    border-radius: var(--radius-md);
    background: var(--bg-inset);
    font-family: var(--font-mono);
    font-size: var(--type-sm-size);
    line-height: 18px;
    color: var(--text-secondary);
    white-space: pre-wrap;
    overflow-wrap: anywhere;
  }
`;
