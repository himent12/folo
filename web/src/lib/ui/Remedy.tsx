/**
 * One remediation block, the same wherever something is off and a command turns it on (Settings › System, New
 * run's Mixed precision): the cause in plain text (identifiers in backticks are set as code), the shell the command
 * is written for ("In PowerShell:"), the command as one full-width line (`CommandLine`: never wrapped, scrolls
 * sideways, Copy), then the server's own words folded under Driver report. `compact` (inside a form) tightens the
 * spacing; the parts and their order stay.
 */
import { computed, css, Index } from 'fw';
import CommandLine from './CommandLine';
import DriverReport from './DriverReport';

export interface RemedyProps {
  cause: string;
  /** The cause's id, for the control it explains (`aria-describedby`). */
  id?: string;
  shell?: string;
  command?: string;
  report?: string | null;
  compact?: boolean;
}

export default function Remedy(props: RemedyProps) {
  const parts = computed(() => props.cause.split('`').map((text, i) => ({ text, code: i % 2 === 1 })));
  return (
    <div class="remedy" class:compact={props.compact ?? false}>
      <p class="cause" id={props.id}>
        <Index each={parts.get()}>{(p) => <>{p().code ? <code>{p().text}</code> : p().text}</>}</Index>
      </p>
      {props.command ? (
        <div class="run">
          {props.shell ? <p class="shell">{`In ${props.shell}:`}</p> : null}
          <CommandLine command={props.command!} label={props.shell ? `${props.shell} command` : 'Command'} />
        </div>
      ) : null}
      {props.report ? <DriverReport report={props.report!} /> : null}
    </div>
  );
}

css`
  .remedy {
    display: grid;
    gap: var(--space-3);
    min-width: 0;
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
  }
  .remedy.compact {
    gap: var(--space-2);
  }
  /* Helper text keeps a readable measure (≤ 60 characters a line); the command under it runs the full width. */
  .cause {
    max-inline-size: var(--measure);
    color: var(--text-secondary);
    text-wrap: pretty;
  }
  .cause code {
    font-family: var(--font-mono);
    white-space: nowrap;
  }
  .run {
    display: grid;
    gap: var(--space-1);
    min-width: 0;
  }
  .shell {
    color: var(--text-tertiary);
  }
`;
