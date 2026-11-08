import { describe, expect, it } from 'vitest';
import { SCRUB_SLOP, TAP_HIT, TouchGesture } from './pin';

describe('TouchGesture', () => {
  it('pins the cursor where a tap lands', () => {
    const g = new TouchGesture(100, 50, null);
    expect(g.up(100)).toEqual({ kind: 'set', px: 100 });
  });

  it('dismisses a pinned cursor tapped again, and moves it when tapped elsewhere', () => {
    expect(new TouchGesture(120, 50, 110).up(120)).toEqual({ kind: 'clear' });
    expect(new TouchGesture(110 + TAP_HIT + 1, 50, 110).up(110 + TAP_HIT + 1)).toEqual({ kind: 'set', px: 110 + TAP_HIT + 1 });
  });

  it('scrubs once the finger travels sideways, and keeps the cursor where the scrub ends', () => {
    const g = new TouchGesture(100, 50, null);
    expect(g.move(100 + SCRUB_SLOP - 1, 50)).toEqual({ kind: 'none' });
    expect(g.move(100 + SCRUB_SLOP, 51)).toEqual({ kind: 'set', px: 100 + SCRUB_SLOP });
    // Once scrubbing, vertical wobble does not stop it.
    expect(g.move(140, 80)).toEqual({ kind: 'set', px: 140 });
    expect(g.up(140)).toEqual({ kind: 'none' });
  });

  it('leaves the page scroll alone: a mostly vertical move never touches the cursor', () => {
    const g = new TouchGesture(100, 50, 60);
    expect(g.move(108, 70)).toEqual({ kind: 'none' });
    expect(g.cancel()).toEqual({ kind: 'none' });
  });
});
