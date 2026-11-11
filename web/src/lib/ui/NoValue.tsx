/**
 * A value that does not exist yet (a queued run's score): the dash in `text.tertiary` at body weight, so it never
 * reads as a minus sign or a value, and named for assistive tech ("no score yet"). Every list cell, row end and KPI
 * that has no figure uses it instead of a bare "—" in the value's own ink (DESIGN_DIRECTION §2.23).
 */
import { css } from 'fw';

export interface NoValueProps {
  label?: string;
}

export default function NoValue(props: NoValueProps) {
  return (
    <span class="no-value">
      <span aria-hidden="true">—</span>
      <span class="sr-only">{props.label ?? 'no score yet'}</span>
    </span>
  );
}

css`
  .no-value {
    color: var(--text-tertiary);
    font-weight: var(--weight-regular);
  }
`;
