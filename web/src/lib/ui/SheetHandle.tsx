// fw port of SheetHandle.svelte.
/**
 * The grab handle of a bottom sheet (DESIGN_DIRECTION 12): a 36 × 4 px `line.strong` bar centred 8 px from the sheet's
 * top, on the grab area that `ui/sheetDrag` moves. Decoration for the eye: the drag is a shortcut, and every sheet
 * keeps its labelled Close. Its parent is the positioned grab area (the sheet's header).
 */
import { css } from 'fw';

export default function SheetHandle() {
  return <span class="sheet-handle" aria-hidden="true"></span>;
}

css`
  .sheet-handle {
    position: absolute;
    top: var(--space-2);
    left: 50%;
    width: 36px;
    height: 4px;
    transform: translateX(-50%);
    border-radius: var(--radius-full);
    background: var(--line-strong);
    pointer-events: none;
  }
`;
