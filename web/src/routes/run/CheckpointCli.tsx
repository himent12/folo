/**
 * Collapsible Checkpoint CLI quick-copy block inside Run › Config: one-click copyable `folo val` and
 * `folo predict` commands for the run's best checkpoint (matching docs/CLI.md).
 */
import ChevronRight from '$icons/chevron-right';
import { computed, css, signal } from 'fw';
import { SHELL } from '$lib/trainCli';
import CommandLine from '$lib/ui/CommandLine';
import Segmented from '$lib/ui/Segmented';
import type { CheckpointCliBundle, CheckpointCliMode } from './runExport';

export interface CheckpointCliProps {
  bundle: CheckpointCliBundle;
}

const MODES: { value: CheckpointCliMode; label: string }[] = [
  { value: 'val', label: 'Validate' },
  { value: 'predict', label: 'Predict' },
];

export default function CheckpointCli(props: CheckpointCliProps) {
  const open = signal(false);
  const mode = signal<CheckpointCliMode>('val');
  const command = computed(() => (mode.get() === 'val' ? props.bundle.val : props.bundle.predict));
  const copied = computed(() =>
    mode.get() === 'val'
      ? 'Run this command in a terminal to evaluate the run checkpoint.'
      : 'Run this command in a terminal to predict with the run checkpoint.',
  );
  const placeholders = computed(() => {
    const list: string[] = [];
    if (props.bundle.modelPlaceholder) list.push(`<path to ${props.bundle.modelPlaceholder}>`);
    if (mode.get() === 'val' && props.bundle.dataPlaceholder) list.push(`<path to ${props.bundle.dataPlaceholder}>`);
    if (mode.get() === 'predict') list.push('<images>');
    return list;
  });

  return (
    <div class="ckpt-cli" data-checkpoint-cli>
      <button
        type="button"
        class="ckpt-toggle"
        aria-expanded={open.get()}
        onclick={() => open.set(!open.peek())}
      >
        <ChevronRight size={14} class={open.get() ? 'chev open' : 'chev'} aria-hidden="true" />
        <span class="ckpt-title">Checkpoint CLI</span>
        <span class="ckpt-meta">val &amp; predict</span>
      </button>
      {open.get() ? (
        <div class="ckpt-body">
          <div class="ckpt-head">
            <span class="ckpt-shell">In {SHELL}:</span>
            <Segmented label="Checkpoint CLI command" size="sm" bind:value={mode} options={MODES} />
          </div>
          <CommandLine
            command={command.get()}
            label={mode.get() === 'val' ? 'Validate checkpoint CLI command' : 'Predict checkpoint CLI command'}
            copied={copied.get()}
          />
          {placeholders.get().length > 0 ? (
            <p class="ckpt-hint">
              Replace{' '}
              {placeholders.get().map((ph, i) => (
                <>
                  {i > 0 ? ' and ' : null}
                  <span class="mono">{ph}</span>
                </>
              ))}{' '}
              with local paths{mode.get() === 'predict' && !props.bundle.modelPlaceholder ? ' (an image file or folder)' : ' on the server'}.
            </p>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}

css`
  .ckpt-cli {
    margin-top: var(--space-3);
    padding-top: var(--space-3);
    border-top: 1px solid var(--line-subtle);
    display: grid;
    gap: var(--space-2);
    min-width: 0;
  }
  .ckpt-toggle {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    width: 100%;
    min-height: var(--control-sm);
    padding: 0;
    color: var(--text-primary);
    text-align: left;
  }
  .ckpt-toggle :global(svg.chev) {
    flex-shrink: 0;
    color: var(--text-tertiary);
    transition: transform var(--dur-fast) var(--ease-standard);
  }
  .ckpt-toggle :global(svg.chev.open) {
    transform: rotate(90deg);
  }
  .ckpt-title {
    font-size: var(--type-sm-size);
    font-weight: var(--weight-medium);
    color: var(--text-secondary);
  }
  .ckpt-meta {
    margin-left: auto;
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
  }
  .ckpt-body {
    display: grid;
    gap: var(--space-2);
    padding-top: var(--space-1);
    min-width: 0;
  }
  .ckpt-head {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-2);
  }
  .ckpt-shell {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-tertiary);
  }
  .ckpt-hint {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-tertiary);
  }
  .ckpt-hint .mono {
    color: var(--text-secondary);
  }
  @media (pointer: coarse) {
    .ckpt-toggle {
      min-height: 44px;
    }
  }
`;
