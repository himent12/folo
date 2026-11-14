/**
 * One setting (design/UNIFIED.md §4, Settings): its name and one line of description on the left, its control or its
 * fact on the right, at least 56 px high with the panel's 16 px sides, a hairline between rows. `below` spans the row
 * under both (a fix, a test result, an error). Where the text would be under 240 px (phones) the control drops under it.
 */
import { css } from 'fw';

export interface RowProps {
  name: JSX.Element;
  desc?: JSX.Element;
  /** The control, or the fact the row states. */
  control?: JSX.Element;
  /** A fact sits on the name's line; a control is centred on the row. */
  fact?: boolean;
  below?: JSX.Element;
  /** The name as a label for the control's id. */
  for?: string;
  /** The description's id, for the control's aria-describedby. */
  descId?: string;
  class?: string;
}

export default function Row(props: RowProps) {
  return (
    <div class={`s-row ${props.class ?? ''}`} class:fact={!!props.fact}>
      <div class="s-text">
        {props.for ? (
          <label class="s-name" for={props.for}>
            {props.name}
          </label>
        ) : (
          <p class="s-name">{props.name}</p>
        )}
        {props.desc ? (
          <p class="s-desc" id={props.descId}>
            {props.desc}
          </p>
        ) : null}
      </div>
      {props.control ? <div class="s-ctl">{props.control}</div> : null}
      {props.below ? <div class="s-below">{props.below}</div> : null}
    </div>
  );
}

css`
  /* Global: a row the page writes itself (a switch, the driver report, About's facts) takes the same frame. */
  :global(.s-row) {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-2) var(--space-6);
    min-height: 56px;
    padding: var(--space-2) var(--space-4);
  }
  /* Rows in a list (.s-rows, e.g. the capabilities) separate the same way, from each other and from the rows around. */
  :global(.s-row) + :global(.s-row),
  :global(.s-row) + :global(.s-rows),
  :global(.s-rows) + :global(.s-row),
  :global(.s-rows) > :global(*) + :global(*) {
    border-top: 1px solid var(--line-subtle);
  }
  :global(.s-row.fact) {
    align-items: baseline;
  }
  /* The text gives way (wrapping its description) before the control drops under it. */
  .s-text {
    flex: 1 1 240px;
    min-width: 0;
    display: grid;
    gap: 2px;
  }
  .s-name {
    font-weight: var(--weight-medium);
    color: var(--text-primary);
  }
  /* One line where it fits: the 880 px column and the control bound it, not a measure of its own. */
  .s-desc {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-tertiary);
  }
  .s-ctl {
    flex-shrink: 0;
    display: flex;
    align-items: center;
    justify-content: flex-end;
    gap: var(--space-2);
    min-width: 0;
    max-width: 100%;
  }
  .s-below {
    flex: 1 0 100%;
    min-width: 0;
  }
`;
