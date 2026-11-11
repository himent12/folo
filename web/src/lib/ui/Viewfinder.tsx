/**
 * A drop target's frame (the Playground's empty state, an empty labeling job): four corner marks that close in while
 * a file is dragged over it, and its message centred under the live dot, then its actions. It fills the box it is
 * placed in (a grid item stretches); the parent sets the room inside it with `--vf-pad`.
 */
import { css } from 'fw';

export interface ViewfinderProps {
  dragging?: boolean;
  title: string;
  /** One sentence under the title. */
  description: JSX.Element;
  actions?: JSX.Element;
}

export default function Viewfinder(props: ViewfinderProps) {
  return (
    <div class="vf" class:dragging={props.dragging ?? false}>
      <span class="corner tl" aria-hidden="true"></span>
      <span class="corner tr" aria-hidden="true"></span>
      <span class="corner bl" aria-hidden="true"></span>
      <span class="corner br" aria-hidden="true"></span>
      <div class="center">
        <p class="title">{props.title}</p>
        <p class="desc">{props.description}</p>
        {props.actions ? <div class="actions">{props.actions}</div> : null}
      </div>
    </div>
  );
}

css`
  .vf {
    position: relative;
    display: grid;
    place-items: center;
    min-width: 0;
    padding: var(--vf-pad, var(--space-6));
  }
  .corner {
    position: absolute;
    width: 28px;
    height: 28px;
    border: 0 solid var(--line-strong);
    transition:
      border-color var(--dur-fast) var(--ease-standard),
      inset var(--dur-base) var(--ease-out);
  }
  .tl {
    top: 0;
    left: 0;
    border-top-width: 2px;
    border-left-width: 2px;
  }
  .tr {
    top: 0;
    right: 0;
    border-top-width: 2px;
    border-right-width: 2px;
  }
  .bl {
    bottom: 0;
    left: 0;
    border-bottom-width: 2px;
    border-left-width: 2px;
  }
  .br {
    bottom: 0;
    right: 0;
    border-bottom-width: 2px;
    border-right-width: 2px;
  }
  .dragging .corner {
    border-color: var(--accent-line);
  }
  .dragging .tl {
    top: var(--space-3);
    left: var(--space-3);
  }
  .dragging .tr {
    top: var(--space-3);
    right: var(--space-3);
  }
  .dragging .bl {
    bottom: var(--space-3);
    left: var(--space-3);
  }
  .dragging .br {
    bottom: var(--space-3);
    right: var(--space-3);
  }
  .center {
    position: relative;
    display: grid;
    justify-items: center;
    gap: var(--space-2);
    text-align: center;
    max-width: 420px;
  }
  .center::before {
    content: '';
    width: 10px;
    height: 10px;
    margin-bottom: var(--space-3);
    border-radius: 50%;
    background: var(--accent-line);
    box-shadow: 0 0 0 6px var(--accent-muted);
  }
  .title {
    font-size: var(--type-lg-size);
    line-height: var(--type-lg-line);
    letter-spacing: var(--type-lg-tracking);
    font-weight: var(--weight-medium);
  }
  .desc {
    color: var(--text-secondary);
  }
  .actions {
    display: flex;
    gap: var(--space-2);
    margin-top: var(--space-3);
    flex-wrap: wrap;
    justify-content: center;
  }
  @media (max-width: 767px) {
    .corner {
      width: 20px;
      height: 20px;
    }
  }
`;
