// Touch readouts for charts: a tap or a sideways scrub pins the cursor (and with it the legend readout) until it is
// dismissed by tapping it again, tapping outside or pressing Escape. A gesture that turns into a vertical page
// scroll changes nothing, so the chart never traps the scroll (the plot is `touch-action: pan-y`).

/** What a touch gesture asks of the chart's cursor. */
export type PinAction = { kind: 'set'; px: number } | { kind: 'clear' } | { kind: 'none' };

/** Sideways travel (px) before a touch counts as a scrub; below it the gesture is still a tap or a scroll. */
export const SCRUB_SLOP = 6;
/** A tap this close (px) to the pinned cursor dismisses it; further away it moves the cursor there. */
export const TAP_HIT = 24;

/** One touch gesture on the plot, from pointerdown to pointerup or pointercancel. */
export class TouchGesture {
  private scrubbing = false;

  /**
   * @param x0 / y0 where the finger went down (px)
   * @param pinnedPx the pinned cursor's position (px) when the gesture started, or null when nothing is pinned
   */
  constructor(
    private readonly x0: number,
    private readonly y0: number,
    private readonly pinnedPx: number | null,
  ) {}

  /** The finger moved: once it travels mostly sideways the gesture scrubs, and the cursor follows it. */
  move(px: number, py: number): PinAction {
    if (!this.scrubbing) {
      const dx = Math.abs(px - this.x0);
      const dy = Math.abs(py - this.y0);
      if (dx < SCRUB_SLOP || dx <= dy) return { kind: 'none' };
      this.scrubbing = true;
    }
    return { kind: 'set', px };
  }

  /** The finger lifted: a scrub leaves the cursor where it ended; a tap pins it, or dismisses it when tapped again. */
  up(px: number): PinAction {
    if (this.scrubbing) return { kind: 'none' };
    if (this.pinnedPx !== null && Math.abs(px - this.pinnedPx) <= TAP_HIT) return { kind: 'clear' };
    return { kind: 'set', px };
  }

  /** The browser took the gesture over (a vertical scroll): the cursor stays as it was. */
  cancel(): PinAction {
    return { kind: 'none' };
  }
}
