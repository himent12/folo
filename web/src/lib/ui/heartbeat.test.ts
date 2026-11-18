import { afterEach, describe, expect, it, vi } from 'vitest';
import { beatAt, installHeartbeat, RESUME_EARLY, stillFrom } from './heartbeat';

// base.css's live-pulse: moves for the first 33% of 2.4 s, then holds the 33% keyframe's values to the end.
const PULSE: Keyframe[] = [
  { offset: 0, computedOffset: 0, boxShadow: '0 0 0 0 a' },
  { offset: 0.33, computedOffset: 0.33, boxShadow: '0 0 0 4px transparent' },
  { offset: 1, computedOffset: 1, boxShadow: '0 0 0 4px transparent' },
];

describe('stillFrom', () => {
  it('finds where the keyframes stop changing', () => {
    expect(stillFrom(PULSE)).toBe(0.33);
    expect(stillFrom([{ offset: 0, opacity: 1 }, { offset: 1, opacity: 0 }])).toBe(1);
    expect(stillFrom([])).toBe(1);
  });
});

describe('beatAt', () => {
  const b = { start: 1000, period: 2400, stillFrom: 792 };
  const resumeAt = 2400 - RESUME_EARLY;
  it('runs while it moves and pauses inside the still part, resuming well before it moves again', () => {
    expect(RESUME_EARLY).toBe(250);
    expect(beatAt(b, 1000)).toEqual({ pause: false, next: 1000 + 852 });
    expect(beatAt(b, 1000 + 852)).toEqual({ pause: true, next: 1000 + resumeAt });
    expect(beatAt(b, 1000 + resumeAt)).toEqual({ pause: false, next: 1000 + 2400 + 852 });
    // Any later period, at the same phase.
    expect(beatAt(b, 1000 + 5 * 2400 + 1000)).toEqual({ pause: true, next: 1000 + 5 * 2400 + resumeAt });
  });
  it('never pauses an animation with no still part long enough', () => {
    expect(beatAt({ start: 0, period: 2400, stillFrom: 2100 }, 2160).pause).toBe(false);
  });
});

/** A CSS animation as the browser exposes it, on a fake document timeline. */
function fakeAnimation(name: string, clock: { now: number }, start: number) {
  let paused: number | null = null;
  let startTime: number | null = start;
  let idle = false;
  const target = { isConnected: true };
  const anim = {
    animationName: name,
    get playState() {
      return idle ? 'idle' : paused !== null ? 'paused' : 'running';
    },
    /** CSS removed it (`animation: none`): the browser cancels it. */
    cancel: () => {
      idle = true;
      paused = null;
      startTime = null;
    },
    get startTime() {
      return startTime;
    },
    set startTime(v: number | null) {
      startTime = v;
    },
    pause: vi.fn(() => {
      paused = clock.now - (startTime ?? 0);
      startTime = null;
    }),
    play: vi.fn(() => {
      startTime = clock.now - (paused ?? 0);
      paused = null;
    }),
    effect: {
      target,
      getComputedTiming: () => ({ duration: 2400, iterations: Infinity }),
      getKeyframes: () => PULSE,
    },
  };
  return { anim, target };
}

function fakeDocument(clock: { now: number }) {
  const listeners = new Map<string, ((e: Event) => void)[]>();
  const doc = {
    visibilityState: 'visible' as DocumentVisibilityState,
    timeline: {
      get currentTime() {
        return clock.now;
      },
    },
    addEventListener: (t: string, f: (e: Event) => void) => listeners.set(t, [...(listeners.get(t) ?? []), f]),
    removeEventListener: (t: string, f: (e: Event) => void) => listeners.set(t, (listeners.get(t) ?? []).filter((g) => g !== f)),
  };
  return {
    doc: doc as unknown as Document,
    fire: (animationName: string, anim: unknown) =>
      (listeners.get('animationstart') ?? []).forEach((f) => f({ animationName, target: { getAnimations: () => [anim] } } as unknown as Event)),
    visibility: (v: DocumentVisibilityState) => {
      doc.visibilityState = v;
      (listeners.get('visibilitychange') ?? []).forEach((f) => f(new Event('visibilitychange')));
    },
  };
}

describe('installHeartbeat', () => {
  afterEach(() => void vi.useRealTimers());

  /** The timeline and the timers moving together, a millisecond at a time. */
  const stepper = (clock: { now: number }) => (ms: number) => {
    for (let i = 0; i < ms; i++) {
      clock.now += 1;
      vi.advanceTimersByTime(1);
    }
  };

  it('pauses a heartbeat for its still part and resumes it at its own phase', () => {
    vi.useFakeTimers();
    const clock = { now: 500 };
    const advance = stepper(clock);
    const { doc, fire } = fakeDocument(clock);
    const stop = installHeartbeat(doc);
    const { anim, target } = fakeAnimation('fw-abc12-end-beat', clock, 500);
    fire('fw-abc12-end-beat', anim);
    expect(anim.playState).toBe('running');
    advance(852);
    expect(anim.playState).toBe('paused');
    advance(2400 - RESUME_EARLY - 852);
    expect(anim.playState).toBe('running');
    // Resumed as if it had never stopped: its start time is the original one.
    expect(anim.startTime).toBe(500);
    advance(RESUME_EARLY + 1000);
    expect(anim.pause).toHaveBeenCalledTimes(2);
    expect(anim.playState).toBe('paused');
    // Its element left: let go of it, and of the timer.
    target.isConnected = false;
    advance(2400);
    expect(anim.play).toHaveBeenCalledTimes(1);
    expect(vi.getTimerCount()).toBe(0);
    stop();
  });

  it('lets go of a heartbeat CSS removed (animation: none) without touching it', () => {
    vi.useFakeTimers();
    const clock = { now: 0 };
    const advance = stepper(clock);
    const { doc, fire } = fakeDocument(clock);
    const stop = installHeartbeat(doc);
    const { anim } = fakeAnimation('live-pulse', clock, 0);
    fire('live-pulse', anim);
    advance(1000);
    expect(anim.playState).toBe('paused');
    // Reduced motion, a held status: the browser cancels the animation while it is paused.
    anim.cancel();
    advance(2400);
    expect([anim.playState, anim.play.mock.calls.length, anim.pause.mock.calls.length, vi.getTimerCount()]).toEqual(['idle', 0, 1, 0]);
    stop();
  });

  it('pauses nothing while the page is hidden, and places each beat from its own clock when it is shown', () => {
    vi.useFakeTimers();
    const clock = { now: 0 };
    const advance = stepper(clock);
    const { doc, fire, visibility } = fakeDocument(clock);
    const stop = installHeartbeat(doc);
    const { anim } = fakeAnimation('live-pulse', clock, 0);
    fire('live-pulse', anim);
    advance(1000);
    expect(anim.playState).toBe('paused');
    // Hidden in its still part: running at once, at its own phase, and no timer left (a hidden tab's run late).
    visibility('hidden');
    expect([anim.playState, anim.startTime, vi.getTimerCount()]).toEqual(['running', 0, 0]);
    // A minute later, shown inside a still part (25 beats on, at the same phase): paused again, resumed before the next.
    advance(60_000);
    expect(anim.playState).toBe('running');
    visibility('visible');
    expect(anim.playState).toBe('paused');
    advance(2400 - RESUME_EARLY - 1000);
    expect([anim.playState, anim.startTime]).toEqual(['running', 0]);
    stop();
  });

  it('leaves other animations alone', () => {
    vi.useFakeTimers();
    const clock = { now: 0 };
    const { doc, fire } = fakeDocument(clock);
    const stop = installHeartbeat(doc);
    const { anim } = fakeAnimation('spin', clock, 0);
    fire('spin', anim);
    clock.now += 1000;
    vi.advanceTimersByTime(1000);
    expect(anim.pause).not.toHaveBeenCalled();
    expect(vi.getTimerCount()).toBe(0);
    stop();
  });
});
