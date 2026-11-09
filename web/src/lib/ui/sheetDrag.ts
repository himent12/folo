// A bottom sheet follows the hand down from its grab area (the header with `ui/SheetHandle`) and closes once it has
// been pulled past `SHEET_CLOSE_PX`; a shorter pull springs back. One behaviour for every sheet: `ui/Dialog` on phones
// and the Playground's thresholds sheet. The labelled Close stays: a drag is a shortcut, never the only way out.

/** How far a sheet must be pulled down before letting go closes it. */
export const SHEET_CLOSE_PX = 48;

export interface SheetDragOptions {
  /** The sheet that moves (the dialog or panel), not the grab area. */
  sheet: () => HTMLElement | null | undefined;
  /** Closes the sheet; called once a pull passes `SHEET_CLOSE_PX`. */
  close: () => void;
  /** Whether dragging applies now (a dialog is a sheet on phones only). */
  enabled?: () => boolean;
}

/** `use:sheetDrag` on the grab area. Presses on a control inside it (Close, Reset) stay clicks. */
export function sheetDrag(node: HTMLElement, opts: SheetDragOptions) {
  let o = opts;
  let from: number | null = null;
  let dy = 0;
  let pointer = -1;

  const move = (y: number) => {
    const el = o.sheet();
    if (!el) return;
    dy = Math.max(0, y);
    el.style.transform = dy ? `translateY(${dy}px)` : '';
  };
  function down(e: PointerEvent) {
    if (o.enabled && !o.enabled()) return;
    if (e.button !== 0 || (e.target as Element).closest('button, a, input, select, textarea, [role="tab"]')) return;
    from = e.clientY;
    pointer = e.pointerId;
    node.setPointerCapture?.(e.pointerId);
    const el = o.sheet();
    // Follow the finger exactly: no transition while dragging.
    if (el) el.style.transition = 'none';
  }
  function moved(e: PointerEvent) {
    if (from === null || e.pointerId !== pointer) return;
    move(e.clientY - from);
  }
  /** Ends the drag: the sheet back in place, the pointer released. Returns how far it had been pulled. */
  function end(e: PointerEvent): number | null {
    if (from === null || e.pointerId !== pointer) return null;
    const pulled = dy;
    from = null;
    pointer = -1;
    if (node.hasPointerCapture?.(e.pointerId)) node.releasePointerCapture(e.pointerId);
    const el = o.sheet();
    if (el) el.style.transition = '';
    move(0);
    dy = 0;
    return pulled;
  }
  function up(e: PointerEvent) {
    const pulled = end(e);
    if (pulled !== null && pulled > SHEET_CLOSE_PX) o.close();
  }
  // The browser or the system took the pointer (a scroll, a call, a lost capture): the sheet springs back and stays
  // open whatever the pull, since nobody let go.
  function cancel(e: PointerEvent) {
    end(e);
  }
  node.addEventListener('pointerdown', down);
  node.addEventListener('pointermove', moved);
  node.addEventListener('pointerup', up);
  node.addEventListener('pointercancel', cancel);
  node.addEventListener('lostpointercapture', cancel);
  return {
    update(next: SheetDragOptions) {
      o = next;
    },
    destroy() {
      node.removeEventListener('pointerdown', down);
      node.removeEventListener('pointermove', moved);
      node.removeEventListener('pointerup', up);
      node.removeEventListener('pointercancel', cancel);
      node.removeEventListener('lostpointercapture', cancel);
    },
  };
}
