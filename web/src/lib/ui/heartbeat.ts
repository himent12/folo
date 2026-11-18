// The live heartbeats (base.css `live-pulse`: StatusBadge's running dot, the sidebar's and the label tiles'; LineEnds'
// `end-beat`, a live chart line's end) move for the first third of every 2.4 s and hold still for the rest. A running
// CSS animation costs a style recalc every frame even while its value holds, which kept a page with a training run
// restyling at display rate while nothing on it moved. This pauses each heartbeat for most of its still part (a paused
// animation holds its value and costs nothing) and resumes it at the phase it would have reached, its start time
// unchanged. The CSS stays the source of what is animated and when (reduced motion, a held status: `animation: none`
// still removes the animation, and this lets it go).
//
// What it guarantees: every frame shows what it showed without this, unless the timer that resumes a heartbeat runs more
// than RESUME_EARLY (250 ms) late: the main thread busy for that long at that moment, or the browser throttling the
// page's timers. Then the first frames of that beat show the still value (the ring transparent) until the timer runs,
// and the beat goes on from its own phase, never shifted. Measured on the Run page of a live run, the latest such timer
// ran 8 ms late, 57 ms with the CPU throttled 4x and 121 ms at 6x (PR #306). A hidden tab's timers are throttled to
// once a second or a minute: nothing is paused while the page is hidden, and when it is shown again each heartbeat is
// placed from its own clock (its start time) before anything is paused again.
//
// Limitation: once script has paused or played a CSS animation, the browser no longer applies `animation-play-state`
// to it (Web Animations). No rule sets that property on a heartbeat; one that does needs a change here.

/** The keyframes (by name, a component's scoped ones by suffix) whose still part is skipped. */
const BEATS = /^(?:fw-[0-9a-z]+-)?(?:live-pulse|end-beat)$/;
/** Paused this long after the still part starts: a pause that lands late, or a frame drawn just before it, is still. */
const PAUSE_AFTER = 60;
/**
 * Resumed this long before the still part ends, so a resume timer that runs up to this late still resumes before the
 * beat moves (header: the guarantee and what it is sized from).
 */
export const RESUME_EARLY = 250;

/** A heartbeat's timing: its period and when in each period it holds still (ms). */
export interface Beat {
  start: number;
  period: number;
  stillFrom: number;
}

/**
 * What a heartbeat should be doing at timeline time `now`: paused inside its still part (PAUSE_AFTER after it starts,
 * RESUME_EARLY before it ends), running otherwise; and when that next changes.
 */
export function beatAt(b: Beat, now: number): { pause: boolean; next: number } {
  const t = now - b.start;
  const k = Math.floor(t / b.period);
  const p = t - k * b.period;
  const from = b.stillFrom + PAUSE_AFTER;
  const to = b.period - RESUME_EARLY;
  const base = b.start + k * b.period;
  if (from >= to) return { pause: false, next: Infinity };
  if (p < from) return { pause: false, next: base + from };
  if (p < to) return { pause: true, next: base + to };
  return { pause: false, next: base + b.period + from };
}

/**
 * Where an animation's still tail starts, as a fraction of its period: the earliest keyframe from which every later
 * one has the same values as the last; 1 when it has none.
 */
export function stillFrom(frames: Keyframe[]): number {
  const values = (f: Keyframe) =>
    JSON.stringify(
      Object.entries(f)
        .filter(([k]) => k !== 'offset' && k !== 'computedOffset' && k !== 'easing' && k !== 'composite')
        .sort(([a], [b]) => (a < b ? -1 : 1)),
    );
  const last = frames.at(-1);
  if (!last) return 1;
  let from = 1;
  for (let i = frames.length - 1; i >= 0; i--) {
    const f = frames[i]!;
    if (values(f) !== values(last)) break;
    from = (f.computedOffset as number | undefined) ?? f.offset ?? from;
  }
  return from;
}

interface Tracked extends Beat {
  anim: CSSAnimation;
}

/** Starts pausing the document's heartbeats in their still part. Returns a stop function (tests). */
export function installHeartbeat(doc: Document = document): () => void {
  const tracked = new Set<Tracked>();
  let timer: ReturnType<typeof setTimeout> | undefined;
  const timeline = () => (doc.timeline.currentTime as number | null) ?? 0;
  const hidden = () => doc.visibilityState === 'hidden';

  /** Running again at the phase it would have had had it never stopped. */
  const resume = (b: Tracked) => {
    if (b.anim.playState !== 'paused') return;
    b.anim.play();
    b.anim.startTime = b.start;
  };

  const tick = () => {
    clearTimeout(timer);
    timer = undefined;
    const now = timeline();
    let next = Infinity;
    for (const b of tracked) {
      const a = b.anim;
      // Gone: CSS removed it (`animation: none`, the element left), or something else cancelled it.
      if (a.playState === 'idle' || a.playState === 'finished' || !(a.effect as KeyframeEffect | null)?.target?.isConnected) {
        tracked.delete(b);
        continue;
      }
      // Hidden: its timers may run a minute late, so nothing stays paused (and a hidden page draws no frames anyway).
      if (hidden()) {
        resume(b);
        continue;
      }
      // Running: its own clock says where its beat is.
      if (a.playState === 'running' && a.startTime !== null) b.start = Number(a.startTime);
      const want = beatAt(b, now);
      if (want.pause && a.playState === 'running') a.pause();
      else if (!want.pause) resume(b);
      next = Math.min(next, want.next);
    }
    if (next < Infinity) timer = setTimeout(tick, Math.max(0, next - timeline()));
  };

  const onStart = (e: Event) => {
    const { animationName } = e as AnimationEvent;
    if (!BEATS.test(animationName)) return;
    const el = e.target as Element;
    const anim = el.getAnimations().find((a): a is CSSAnimation => (a as Partial<CSSAnimation>).animationName === animationName);
    const timing = anim?.effect?.getComputedTiming();
    if (!anim || anim.startTime === null || !timing || timing.iterations !== Infinity) return;
    const period = Number(timing.duration);
    const still = stillFrom((anim.effect as KeyframeEffect).getKeyframes());
    if (!(period > 0) || still >= 1) return;
    for (const b of tracked) if (b.anim === anim) return;
    tracked.add({ anim, start: Number(anim.startTime), period, stillFrom: still * period });
    tick();
  };

  doc.addEventListener('animationstart', onStart, true);
  doc.addEventListener('visibilitychange', tick);
  return () => {
    doc.removeEventListener('animationstart', onStart, true);
    doc.removeEventListener('visibilitychange', tick);
    clearTimeout(timer);
    for (const b of tracked) resume(b);
    tracked.clear();
  };
}
