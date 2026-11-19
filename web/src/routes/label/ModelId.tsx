/**
 * The run's model as a meta item (ui/Meta) on the job page, the one place a person reads which model ran: whole, never
 * cut and never only in a tooltip. A short name sits in the line like any item; a model id the agent does not list (a
 * Bedrock profile's ARN, up to 100 characters with no spaces) takes a line of its own and wraps inside it instead of
 * widening the rail and the page.
 */
import { css } from 'fw';

export default function ModelId(props: { id: string }) {
  return (
    <span class="model">
      <span>{props.id}</span>
    </span>
  );
}

css`
  /* A flex item of its own: its dot (ui/Meta's ::before) keeps its 20 px and the id wraps beside it, so every wrapped
     line starts after the dot, inside the meta line's clip, like the first. */
  :global(.ui-meta) > .model {
    display: flex;
    white-space: normal;
  }
  .model > span {
    min-width: 0;
    overflow-wrap: anywhere;
  }
`;
