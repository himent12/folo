/**
 * A meta line: its child elements separated by " · " (a tertiary dot with 8 px on each side), the one separator
 * every meta line uses (DESIGN_DIRECTION §2.3). Each child is one item; give it an element of its own (a span),
 * never a bare text node, and never write the dot by hand.
 *
 * The dot hangs at the start of every item, so a line that wraps never ends on a "·": the dot that would lead the
 * next line falls outside the line's left edge and is clipped. Every item is a fact (a count, a unit, a status, an
 * identifier), and a fact is never cut (DESIGN_DIRECTION §2.3): a line wraps between items, never inside one. `line`
 * keeps a desktop table row's items on one line (cut with an ellipsis only as a last resort, which the shoot's QA
 * reports); on phones (< 768 px) it wraps whole items like any meta line, so a phone row grows a line instead of
 * losing its last fact.
 */
import { css } from 'fw';

export interface MetaProps {
  children: JSX.Element;
  line?: boolean;
  class?: string;
  title?: string;
}

export default function Meta(props: MetaProps) {
  return (
    <span class={`ui-meta ${props.class ?? ''}`} class:line={props.line ?? false} title={props.title}>
      {props.children}
    </span>
  );
}

css`
  /* Wrapping: a flex row shifted one separator (20 px) left and clipped back at the content edge, so each line's
     leading dot is cut off; focus rings keep 4 px of room on that side. */
  .ui-meta {
    display: flex;
    flex-wrap: wrap;
    align-items: baseline;
    row-gap: var(--space-1);
    min-width: 0;
    margin-left: -20px;
    clip-path: inset(-8px -8px -8px 16px);
  }
  /* A line breaks between items, never inside one; only a free-text name longer than the whole line may truncate
     itself (with its full text reachable), never a fact. */
  .ui-meta > :global(*) {
    min-width: 0;
    white-space: nowrap;
  }
  .ui-meta > :global(*)::before {
    content: '·';
    display: inline-block;
    flex: none;
    width: 20px;
    text-align: center;
    font-family: var(--font-sans);
    font-weight: var(--weight-regular);
    color: var(--text-tertiary);
    text-decoration: none;
  }
  /* An item's content is inline (it follows its dot); a meta nested in an item sits after that dot too, and on phones
     wraps its own items whole. */
  :global(.ui-meta) .ui-meta.line {
    display: inline-flex;
    max-width: calc(100% - 20px);
    vertical-align: top;
  }
  /* From 768 px up (table rows): one line, inline flow; the first item has no dot. Phones keep the wrapping flow
     above: a fact is never cut, the row grows a line instead. */
  @media (min-width: 768px) {
    :global(.ui-meta) .ui-meta.line {
      display: inline-block;
    }
    .ui-meta.line {
      display: block;
      margin-left: 0;
      clip-path: none;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .ui-meta.line > :global(*:first-child)::before {
      content: none;
    }
  }
`;
