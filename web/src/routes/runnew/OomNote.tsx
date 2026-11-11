/** A retry of a run that ran out of GPU memory: what changed since, and whether the forecast says it now fits. */
import TriangleAlert from '$icons/triangle-alert';
import { css } from 'fw';
import type { Run } from '$lib/api/types';
import { bytes } from '$lib/format';
import { prefs } from '$lib/prefs';
import { href } from '$lib/router';
import { roughBytes } from './summary';

export interface OomNoteProps {
  oomRun: Run;
  oomChanges: { key: 'batch' | 'imgsz' | 'model' | 'device'; was: string; now: string }[];
  vramFit: 'fits' | 'no' | 'unknown' | null;
  vram: number;
  gpuMem: number | null;
  vramLevel: 'danger' | 'warning' | null;
}

export default function OomNote(props: OomNoteProps) {
  /** A sentence that starts with a change: "Batch 16, 480 px" (an architecture or a device keeps its own case). */
  const capital = (t: string) => t.replace(/^batch/, 'Batch');
  const now = () => props.oomChanges.map((x) => x.now).join(', ');

  // Why the form is not a plain copy: the run it retries ran out of memory, and what was changed for that.
  // "Now …" claims a fix only when the forecast above says it fits; otherwise it says the change is not enough, and the
  // rail's one primary ("Use batch 8") is the change that is. A GPU that does not report its memory gets the figure and
  // says the fit cannot be checked, never a fix it cannot vouch for.
  return (
    <p class="oom-note" role="status">
      <TriangleAlert size={16} aria-hidden="true" />
      <span>
        <a class="link" href={href(`/runs/${props.oomRun.id}`)}>
          {props.oomRun.name}
        </a>{' '}
        ran out of GPU memory at{' '}
        {props.oomChanges.length ? (
          <>
            {props.oomChanges.map((x) => x.was).join(', ')}.{' '}
            {props.vramFit === 'no' ? (
              <>
                {capital(now())} still won't fit: {roughBytes(props.vram)} of {bytes(props.gpuMem, prefs.units)}.
              </>
            ) : props.vramFit === 'fits' ? (
              <>
                Now{' '}
                <span class="nowrap">
                  {now()} · {roughBytes(props.vram)}
                </span>
                {props.vramLevel === 'warning' ? `, close to this GPU's ${bytes(props.gpuMem, prefs.units)}` : ''}.
              </>
            ) : props.vramFit === 'unknown' ? (
              <>
                Now{' '}
                <span class="nowrap">
                  {now()} · {roughBytes(props.vram)}
                </span>{' '}
                · this GPU doesn't report its memory, so the fit can't be checked.
              </>
            ) : (
              <>Now {now()}.</>
            )}
          </>
        ) : (
          <>
            batch {props.oomRun.config.batch}, {props.oomRun.config.imgsz} px. These are the same settings: lower the batch or image size, or pick a
            smaller model.
          </>
        )}
      </span>
    </p>
  );
}

css`
  /* The retry note spans the form and the summary; the mark carries the hue, the text stays in ink. */
  .oom-note {
    grid-column: 1 / -1;
    display: flex;
    align-items: flex-start;
    gap: var(--space-3);
    padding: var(--space-3) var(--space-4);
    background: var(--bg-panel);
    border: 1px solid var(--line-panel);
    border-radius: var(--radius-lg);
    color: var(--text-secondary);
  }
  .oom-note > :global(svg) {
    flex-shrink: 0;
    margin-top: 2px;
    color: var(--warning-text);
  }
  .oom-note .link {
    color: var(--text-primary);
    font-weight: var(--weight-medium);
    text-decoration: underline;
    text-decoration-color: var(--line-strong);
    text-underline-offset: 3px;
  }
  .oom-note .link:hover {
    text-decoration-color: var(--text-secondary);
  }
`;
