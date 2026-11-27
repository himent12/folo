/**
 * A command to run, as one line that never wraps (a shell command broken mid-token, `'vulkan` / `';`, cannot be
 * read or pasted): full width, it scrolls sideways with a fade on the clipped edge (`fadeScroll`), and Copy copies
 * it whole. Every command the app shows to be run goes through it (or ui/Remedy, which holds one), never a truncated
 * `<code>`.
 */
import Copy from '$icons/copy';
import { css } from 'fw';
import Button from './Button';
import { fadeScroll } from './fadeScroll';
import { feedback } from './feedback';

export interface CommandLineProps {
  command: string;
  label?: string;
  /** What the copied command does, said with the result. */
  copied?: string;
}

export default function CommandLine(props: CommandLineProps) {
  // The page's toaster, or the status region of the dialog this line sits in.
  const notify = feedback();

  async function copy() {
    const command = props.command;
    const copied = props.copied;
    try {
      await navigator.clipboard.writeText(command);
      notify.success('Command copied', copied);
    } catch (e) {
      notify.error(e, 'Could not copy the command');
    }
  }

  return (
    <div class="cmd">
      <pre class="line" data-scrolls aria-label={props.label ?? 'Command'} use:fadeScroll>
        <code>{props.command}</code>
      </pre>
      <Button size="sm" variant="ghost" icon={Copy} aria-label="Copy command" onclick={copy}>
        Copy
      </Button>
    </div>
  );
}

css`
  .cmd {
    display: flex;
    align-items: center;
    gap: var(--space-1);
    min-width: 0;
    padding: var(--space-1) var(--space-1) var(--space-1) var(--space-3);
    border-radius: var(--radius-md);
    background: var(--bg-inset);
    border: 1px solid var(--line-subtle);
  }
  .line {
    flex: 1;
    min-width: 0;
    overflow-x: auto;
    scrollbar-width: none;
    white-space: pre;
    font-family: var(--font-mono);
    font-size: var(--type-sm-size);
    line-height: var(--control-sm);
    color: var(--text-primary);
    border-radius: var(--radius-xs);
  }
  .line::-webkit-scrollbar {
    display: none;
  }
  .line:global([data-fade='end']) {
    mask-image: linear-gradient(90deg, #000 calc(100% - 32px), transparent);
  }
  .line:global([data-fade='start']) {
    mask-image: linear-gradient(90deg, transparent, #000 32px);
  }
  .line:global([data-fade='both']) {
    mask-image: linear-gradient(90deg, transparent, #000 32px, #000 calc(100% - 32px), transparent);
  }
  .line:focus-visible {
    outline: var(--size-focus-ring) solid var(--focus);
    outline-offset: 2px;
  }
`;
