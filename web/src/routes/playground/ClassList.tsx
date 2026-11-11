/** The classes found, each row showing or hiding its class (the Classes panel, and its phone disclosure). */
import Eye from '$icons/eye';
import EyeOff from '$icons/eye-off';
import { css, For } from 'fw';
import type { Prediction } from '$lib/api/types';
import { classColor } from '$lib/design/tokens';
import Swatch from './Swatch';

export interface ClassListProps {
  result: { boxes: Prediction['boxes']; ms: number } | null;
  counts: [number, number][];
  pred: Prediction | null;
  hidden: Set<number>;
  classes: string[];
  maxCount: number;
  toggleClass: (c: number) => void;
}

export default function ClassList(props: ClassListProps) {
  return (
    <>
      {!props.result ? (
        <p class="hint">Per-class counts; each row shows or hides its class.</p>
      ) : props.counts.length === 0 ? (
        <p class="hint">{props.pred?.boxes.length ? 'Nothing above the confidence threshold.' : 'The model found nothing in this image.'}</p>
      ) : (
        /* A visibility toggle per class, never a static legend: every row wears its state (an eye, shown by default; a
           struck eye and a dimmed row when hidden), the whole row is the button (aria-pressed: shown), and Show all
           brings every class back. */
        <ul class="counts" aria-label="Detected classes">
          <For each={props.counts} key={(e) => e[0]}>
            {(e) => {
              const c = () => e()[0];
              const n = () => e()[1];
              const off = () => props.hidden.has(c());
              const name = () => props.classes[c()] ?? `class ${c()}`;
              return (
                <li>
                  <button type="button" class:off={off()} aria-pressed={!off()} onclick={() => props.toggleClass(c())} aria-label={`Show ${name()}, ${n()} found`}>
                    <span class="vis" aria-hidden="true">
                      {off() ? <EyeOff size={14} strokeWidth={1.75} /> : <Eye size={14} strokeWidth={1.75} />}
                    </span>
                    <Swatch color={classColor(c())} />
                    <span class="c-name truncate">{name()}</span>
                    <span class="c-bar" aria-hidden="true">
                      <span style:transform={`scaleX(${n() / props.maxCount})`} style:background={classColor(c())}></span>
                    </span>
                    <span class="c-n num">{n()}</span>
                  </button>
                </li>
              );
            }}
          </For>
        </ul>
      )}
    </>
  );
}

css`
  /* Class marks keep the palette hue in both themes; a hairline edge holds pale hues on light surfaces. */
  :global([data-theme='light']) .c-bar span {
    box-shadow: inset 0 0 0 1px #0a0e1433;
  }
  .c-bar {
    position: relative;
    width: 72px;
    height: 4px;
    border-radius: var(--radius-full);
    background: var(--bg-active);
    overflow: hidden;
  }
  .c-bar span {
    position: absolute;
    inset: 0;
    transform-origin: left;
    border-radius: inherit;
  }
  .hint {
    color: var(--text-tertiary);
    font-size: var(--type-sm-size);
  }
  .counts {
    display: grid;
    margin: 0 calc(-1 * var(--space-2));
  }
  .counts button {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    width: 100%;
    height: 28px;
    padding: 0 var(--space-2);
    border-radius: var(--radius-sm);
    transition:
      background-color var(--dur-fast) var(--ease-standard),
      opacity var(--dur-fast) var(--ease-standard);
  }
  .counts button:hover {
    background: var(--bg-hover);
  }
  .counts button.off {
    opacity: 0.4;
  }
  /* The row's state, before its swatch: an eye in ink while shown, a struck one (dimmed with the row) when hidden. */
  .vis {
    display: flex;
    flex: none;
    color: var(--text-secondary);
  }
  .off .vis {
    color: var(--text-tertiary);
  }
  .counts button.off .c-name {
    text-decoration: line-through;
  }
  .counts button:focus-visible {
    outline-offset: -2px;
  }
  .c-name {
    flex: 1;
    min-width: 0;
    text-align: left;
  }
  .c-bar {
    width: 64px;
  }
  .c-n {
    min-width: 24px;
    text-align: right;
    font-size: var(--type-sm-size);
  }
  @media (pointer: coarse) {
    .counts button {
      height: 44px;
    }
  }
`;
