/**
 * A note after a value: what the run resolved a setting to beside its request ("0.000833 (requested 0.01)",
 * "AdamW (auto)", "last 7 epochs (requested 10)"). One rendering wherever a value carries a note (Run's Config,
 * Compare's table; DESIGN_DIRECTION §2.3): a real, breakable space before it, so value and note never touch (round
 * 23's "0.000833(requested 0.01)") and a narrow cell breaks between them, never inside the note. 12 px tertiary sans.
 */
import { css } from 'fw';

export interface NoteProps {
  text: string;
  fact?: boolean;
}

export default function Note(props: NoteProps) {
  return (
    <>
      {' '}
      <span class="ui-note" data-fact={props.fact ? '' : undefined}>
        {props.text}
      </span>
    </>
  );
}

css`
  .ui-note {
    white-space: nowrap;
    font-family: var(--font-sans);
    font-size: var(--type-sm-size);
    font-weight: var(--weight-regular);
    color: var(--text-tertiary);
  }
`;
