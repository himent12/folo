/** Phones with an image: the classes and the timing as two disclosures under the detections, folded until asked for. */
import ChevronRight from '$icons/chevron-right';
import { css } from 'fw';
import type { Prediction } from '$lib/api/types';
import { ms } from '$lib/format';

export interface PhoneDetailsProps {
  result: { boxes: Prediction['boxes']; ms: number } | null;
  counts: [number, number][];
  hidden: Set<number>;
  pred: Prediction | null;
  where: string;
  showAllButton: () => JSX.Element;
  classList: () => JSX.Element;
  timingBody: () => JSX.Element;
}

export default function PhoneDetails(props: PhoneDetailsProps) {
  // Phones: classes and timing as two disclosures under the detections, folded until asked for.
  return (
    <div class="discs">
      <details class="disc">
        <summary>
          <ChevronRight size={14} aria-hidden="true" />
          <span class="disc-title">Classes</span>
          {props.result ? (
            <span class="disc-meta num">
              {props.counts.length} found{props.hidden.size ? ` · ${props.hidden.size} hidden` : ''}
            </span>
          ) : null}
        </summary>
        <div class="disc-body">
          {props.hidden.size ? <div class="disc-act">{props.showAllButton}</div> : null}
          {props.classList}
        </div>
      </details>
      <details class="disc">
        <summary>
          <ChevronRight size={14} aria-hidden="true" />
          <span class="disc-title">Timing</span>
          {props.pred ? (
            <span class="disc-meta num">
              {ms(props.pred.timing.total_ms)} · {props.where}
            </span>
          ) : null}
        </summary>
        <div class="disc-body">{props.timingBody}</div>
      </details>
    </div>
  );
}

css`
  .discs {
    display: grid;
    gap: var(--space-2);
    order: 3;
  }
  .disc {
    border: 1px solid var(--line-panel);
    border-radius: var(--radius-lg);
    background: var(--bg-panel);
  }
  .disc summary {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    min-height: 44px;
    padding: 0 var(--space-4);
    list-style: none;
    cursor: pointer;
    border-radius: inherit;
  }
  .disc summary::-webkit-details-marker {
    display: none;
  }
  .disc summary :global(svg) {
    flex-shrink: 0;
    color: var(--text-tertiary);
    transition: transform var(--dur-fast) var(--ease-standard);
  }
  .disc[open] summary :global(svg) {
    transform: rotate(90deg);
  }
  .disc summary:focus-visible {
    outline-offset: -2px;
  }
  .disc-title {
    font-weight: var(--weight-medium);
  }
  .disc-meta {
    margin-left: auto;
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
  }
  .disc-body {
    padding: 0 var(--space-4) var(--space-4);
  }
  .disc-act {
    display: flex;
    justify-content: flex-end;
    margin-bottom: var(--space-1);
  }
`;
