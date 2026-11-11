// fw port of Swatch.svelte.
/** A class's colour mark: the detections table, the picked box's line and the class list. */
import { css } from 'fw';

export default function Swatch(props: { color: string }) {
  return <span class="sw" style:background={props.color} aria-hidden="true"></span>;
}

css`
  .sw {
    width: 8px;
    height: 8px;
    border-radius: 2px;
    flex-shrink: 0;
  }
  /* Class marks keep the palette hue in both themes; a hairline edge holds pale hues on light surfaces. */
  :global([data-theme='light']) .sw {
    box-shadow: inset 0 0 0 1px #0a0e1433;
  }
`;
