/**
 * The 320 px rail beside the image from 1024 px (design/UNIFIED.md §4 Playground), one panel in this order: the model
 * and where it runs; the thresholds; what is drawn on the image; the detections (with the classes found and the JSON
 * as tabs); the timing. As tall as the image panel at most, it scrolls inside itself when it holds more.
 */
import { css } from 'fw';

export interface RailProps {
  pickers: () => JSX.Element;
  thresholds: () => JSX.Element;
  resetButton: () => JSX.Element;
  /** What is drawn on the image (Labels, Scores, Ground truth); none before an image is in. */
  toggles: (() => JSX.Element) | null;
  /** Optional A/B comparison summary shown above Detections in Compare mode. */
  compareSummary?: (() => JSX.Element) | null;
  detections: () => JSX.Element;
  timing: () => JSX.Element;
  /** Where the last prediction ran ("server", "browser · WebGPU"), after the Timing heading. */
  where: string | null;
}

export default function Rail(props: RailProps) {
  return (
    <aside class="rail" aria-label="Model and results">
      <section class="sec" aria-label="Model">
        {props.pickers}
      </section>
      <section class="sec" aria-labelledby="pg-thresholds">
        <div class="sec-head">
          <h2 id="pg-thresholds">Thresholds</h2>
          {props.resetButton}
        </div>
        {props.thresholds}
      </section>
      {props.toggles ? (
        <section class="sec show" role="group" aria-label="Show on the image">
          {props.toggles}
        </section>
      ) : null}
      {props.compareSummary ? (
        <section class="sec" aria-labelledby="pg-compare-summary">
          <div class="sec-head">
            <h2 id="pg-compare-summary">A/B Summary</h2>
          </div>
          {props.compareSummary}
        </section>
      ) : null}
      <section class="sec det">{props.detections}</section>
      <section class="sec" aria-labelledby="pg-timing">
        <div class="sec-head">
          <h2 id="pg-timing">
            Timing
            {props.where ? <span class="meta">{props.where}</span> : null}
          </h2>
        </div>
        {props.timing}
      </section>
    </aside>
  );
}

css`
  .rail {
    display: flex;
    flex-direction: column;
    max-height: var(--fit-h);
    overflow-y: auto;
    overscroll-behavior: contain;
    background: var(--bg-panel);
    border: 1px solid var(--line-panel);
    border-radius: var(--radius-lg);
  }
  .sec {
    padding: var(--space-4);
  }
  .sec + .sec {
    border-top: 1px solid var(--line-subtle);
  }
  /* The detections' tabs are the section's heading: they sit on its top edge, the list runs to its sides. */
  .sec.det {
    padding: 0 0 var(--space-4);
  }
  .sec.show {
    display: flex;
    flex-wrap: wrap;
    gap: 2px;
    padding: var(--space-2) var(--space-4);
  }
  .sec-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-2);
    min-height: var(--control-sm);
    margin: calc(-1 * var(--space-1)) 0 var(--space-3);
  }
  h2 {
    display: flex;
    align-items: baseline;
    gap: var(--space-2);
    font-size: var(--type-base-size);
    line-height: var(--type-base-line);
    font-weight: var(--weight-medium);
    color: var(--text-primary);
  }
  .meta {
    font-size: var(--type-sm-size);
    font-weight: var(--weight-regular);
    color: var(--text-tertiary);
  }
`;
