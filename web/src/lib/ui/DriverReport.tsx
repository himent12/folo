// fw port of DriverReport.svelte.
/**
 * The server's own words about a capability, verbatim, for an issue or a terminal: folded under "Driver report"
 * (closed until asked for, in every state) and copied whole.
 */
import ChevronRight from '$icons/chevron-right';
import Copy from '$icons/copy';
import { computed, css, Index } from 'fw';
import Button from './Button';
import { feedback } from './feedback';

export interface DriverReportProps {
  report: string;
}

export default function DriverReport(props: DriverReportProps) {
  // One record per line of the server's report (blank lines dropped: the gaps between blocks part them).
  const records = computed(() => props.report.split(/\r?\n/).filter((l) => l.trim() !== ''));
  // The page's toaster, or the status region of the dialog this report sits in.
  const notify = feedback();

  async function copy() {
    try {
      await navigator.clipboard.writeText(props.report);
      notify.success('Driver report copied');
    } catch (e) {
      notify.error(e, 'Could not copy the driver report');
    }
  }

  // Open, Copy sits on the heading's row at the right, so the report below takes the panel's full width; each of the
  // server's records (one line of its report) is a block of its own with a gap after it, and a wrapped line hangs
  // deeper than its first (DESIGN_DIRECTION 21).
  return (
    <details class="report">
      <summary>
        <ChevronRight size={12} strokeWidth={2} aria-hidden="true" />
        Driver report
      </summary>
      <span class="copy-at">
        <Button size="sm" variant="ghost" icon={Copy} aria-label="Copy driver report" onclick={copy}>
          Copy
        </Button>
      </span>
      <pre class="body" aria-label="Driver report">
        <Index each={records.get()}>
          {(r, i) => (
            <>
              {i ? '\n' : null}
              <span class="rec">{r()}</span>
            </>
          )}
        </Index>
      </pre>
    </details>
  );
}

css`
  .report {
    --sum-h: var(--control-sm);
    position: relative;
    min-width: 0;
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-tertiary);
  }
  summary {
    display: inline-flex;
    align-items: center;
    gap: var(--space-1);
    min-height: var(--sum-h);
    cursor: pointer;
    list-style: none;
    border-radius: var(--radius-sm);
  }
  summary::-webkit-details-marker {
    display: none;
  }
  summary:hover {
    color: var(--text-secondary);
  }
  summary :global(svg) {
    transition: transform var(--dur-fast) var(--ease-standard);
  }
  .report[open] summary :global(svg) {
    transform: rotate(90deg);
  }
  /* On the heading's row, centred on it, at the report's right edge. */
  .copy-at {
    position: absolute;
    top: 0;
    right: 0;
    display: flex;
    align-items: center;
    height: var(--sum-h);
  }
  /* The report is the server's prose, not a command: it wraps between words (inside a long one only if it must). */
  .body {
    display: grid;
    gap: var(--space-3);
    margin-top: var(--space-2);
    padding: var(--space-3);
    border-radius: var(--radius-md);
    background: var(--bg-inset);
    border: 1px solid var(--line-subtle);
    font-family: var(--font-mono);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
    /* The newline between records stays in the text (a copied selection keeps one record per line) and collapses
       here: the grid's gap parts them. */
    white-space: normal;
    overflow-wrap: break-word;
  }
  .rec {
    display: block;
    white-space: pre-wrap;
    padding-left: 2ch;
    text-indent: -2ch;
  }
  @media (pointer: coarse) {
    .report {
      --sum-h: 44px;
    }
  }
`;
