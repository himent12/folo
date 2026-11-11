/** A series' key in the legend and the tooltip: a 12 px stroke in its ink and its line's rhythm. */
import { css } from 'fw';
import { inkOf, swatchDash, type Series } from './lineSeries';

export default function SeriesSwatch(props: { s: Series }) {
  return (
    <svg class="swatch" width="14" height="6" viewBox="0 0 14 6" aria-hidden="true">
      <line x1="1" x2="13" y1="3" y2="3" style:stroke={inkOf(props.s)} style:stroke-dasharray={swatchDash(props.s)} />
    </svg>
  );
}

css`
  .swatch {
    flex-shrink: 0;
    overflow: visible;
  }
  .swatch line {
    stroke-width: 2;
    stroke-linecap: round;
  }
  svg {
    display: block;
    overflow: visible;
    touch-action: pan-y;
  }
`;
