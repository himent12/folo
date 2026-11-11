/** The same run from a terminal, under the Summary: the server's command, else Studio's, one flag a line, copyable. */
import ChevronRight from '$icons/chevron-right';
import Copy from '$icons/copy';
import TriangleAlert from '$icons/triangle-alert';
import { css, Index } from 'fw';
import { SHELL } from '$lib/trainCli';
import Button from '$lib/ui/Button';
import { toast } from '$lib/ui/toast';

export interface CliCommandProps {
  cliParts: string[];
  cli: string;
  cliPlaceholder: string | null;
}

export default function CliCommand(props: CliCommandProps) {
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
      <p class="cli-shell">In {SHELL}:</p>
      {props.cliPlaceholder ? (
        <p class="cli-req">
          <TriangleAlert size={14} aria-hidden="true" />
          Weights path required
        </p>
      ) : null}
      <div class="cli-body">
        <code aria-describedby={props.cliPlaceholder ? 'cli-note' : undefined}>
          <Index each={props.cliParts}>
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
                {i < props.cliParts.length - 1 ? ' `' : ''}
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
              onclick={() => copy(props.cli, 'Template copied: replace the weights path')}
            />
          ) : (
            <Button size="sm" variant="ghost" icon={Copy} aria-label="Copy command" title="Copy command" onclick={() => copy(props.cli, 'Command copied')} />
          )}
        </span>
      </div>
      {props.cliPlaceholder ? (
        <p class="cli-note" id="cli-note">
          Put the path of {props.cliPlaceholder} on the server in its place: this server does not report it.
        </p>
      ) : null}
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
  .cli-shell {
    margin-top: var(--space-2);
    color: var(--text-tertiary);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
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
