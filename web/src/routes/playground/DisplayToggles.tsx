/** What is drawn on the image: Labels, Scores and (a dataset image) Ground truth, each a pressed / unpressed toggle. */
import Check from '$icons/check';
import { bindable, css } from 'fw';

export interface DisplayTogglesProps {
  labels?: boolean;
  scores?: boolean;
  truth?: boolean;
  /** A dataset image: its ground truth can be drawn too. */
  dataset: boolean;
}

export default function DisplayToggles(props: DisplayTogglesProps) {
  const labels = bindable(props, 'labels', true);
  const scores = bindable(props, 'scores', true);
  const truth = bindable(props, 'truth', false);
  const toggle = (label: string, on: () => boolean, set: (v: boolean) => void) => (
    <button type="button" class="tog" class:on={on()} aria-pressed={on()} onclick={() => set(!on())}>
      <span class="tog-box" aria-hidden="true">
        {on() ? <Check size={10} strokeWidth={3} /> : null}
      </span>
      {label}
    </button>
  );
  return (
    <>
      {toggle('Labels', () => labels.get(), (v) => labels.set(v))}
      {toggle('Scores', () => scores.get(), (v) => scores.set(v))}
      {props.dataset ? toggle('Ground truth', () => truth.get(), (v) => truth.set(v)) : null}
    </>
  );
}

css`
  .tog {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    height: var(--control-sm);
    padding: 0 var(--space-2);
    border-radius: var(--radius-sm);
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
    white-space: nowrap;
    transition:
      background-color var(--dur-fast) var(--ease-standard),
      color var(--dur-fast) var(--ease-standard);
  }
  .tog:hover {
    background: var(--bg-hover);
    color: var(--text-primary);
  }
  .tog.on {
    color: var(--text-primary);
  }
  .tog-box {
    display: grid;
    place-items: center;
    width: 12px;
    height: 12px;
    border-radius: 3px;
    /* The empty box's edge is line.control (>= 3:1); checked is neutral (lime is for actions, focus and live). */
    box-shadow: inset 0 0 0 1px var(--line-control);
  }
  .tog.on .tog-box {
    background: var(--text-primary);
    color: var(--text-inverse);
    box-shadow: none;
  }
  /* In a row of its own, the first toggle's box lines up with the headings and slider labels above. */
  :global(.show) .tog:first-child {
    margin-left: calc(-1 * var(--space-2));
  }
  @media (pointer: coarse) {
    :global(.more-toggles) .tog {
      height: 44px;
    }
  }
`;
