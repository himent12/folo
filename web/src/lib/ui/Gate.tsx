// fw port of Gate.svelte.
/**
 * A control that is disabled for a reason ("Server offline", "Select 2–4 runs to compare"). The reason is a tooltip
 * on hover and the control's accessible description, since a disabled button cannot take focus to show it. Render
 * the control in the snippet with `disabled` while `reason` is set, and pass it the id: `aria-describedby={why()}`.
 */
import { createUniqueId, css, untrack } from 'fw';
import { tooltip } from './tooltip';

export interface GateProps {
  reason: string | null | undefined;
  /** The control; `why()` is the id of the reason while there is one (its `aria-describedby`). */
  children: (why: () => string | undefined) => JSX.Element;
}

export default function Gate(props: GateProps) {
  const uid = createUniqueId();
  const why = () => (props.reason ? `${uid}-why` : undefined);
  // The snippet is called once (again only if `children` itself changes): `why` is read where it is used.
  const body = () => {
    const children = props.children;
    return untrack(() => children(why));
  };

  return (
    <>
      <span class="gate" class:off={!!props.reason} use:tooltip={props.reason}>
        {body()}
      </span>
      {props.reason ? (
        <span class="sr-only" id={`${uid}-why`}>
          {props.reason}
        </span>
      ) : null}
    </>
  );
}

css`
  .gate {
    display: inline-flex;
  }
  /* A disabled button takes no pointer events in some browsers: the wrapper takes the hover, so the reason shows. */
  .off {
    cursor: not-allowed;
  }
  .off > :global(:disabled) {
    pointer-events: none;
  }
`;
