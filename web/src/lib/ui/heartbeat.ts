// The live heartbeats (base.css `live-pulse`: StatusBadge's running dot, the sidebar's and the label tiles'; LineEnds'
// `end-beat`, a live chart line's end) move for the first third of every 2.4 s and hold still for the rest. A running
// CSS animation costs a style recalc (and for the SVG ring a layout) every frame even while its value holds, which kept
// a page with a training run busy at display rate while nothing on it moved: on an idle Overview, about 190 style
// recalcs and 85 layouts a second, most of the main thread's work. This pauses each heartbeat for its still part and
// resumes it at the phase it would have reached, so every frame shows what it showed before (a paused animation holds
// its value and costs nothing); the CSS stays the source of what is animated and when (reduced motion, a held status:
// `animation: none` still removes the animation, and this lets it go).

/** The keyframes (by name, a component's scoped ones by suffix) whose still part is skipped. */
const BEATS = /^(?:fw-[0-9a-z]+-)?(?:live-pulse|end-beat)$/;
/** Paused this long after the still part starts (and resumed this long before it ends), so a late timer never pauses
 * a frame of movement, and a resumed animation has a frame or two to run before it moves again. */
const MARGIN = 60;

/** A heartbeat's timing: its period and when in each period it holds still (ms). */
export interface Beat {
  start: number;
  period: number;
  stillFrom: number;
}

/**
 * What a heartbeat should be doing at timeline time `now`: paused while inside its still part (with MARGIN on both
 * sides), running otherwise; and when that next changes.
 */
export function beatAt(b: Beat, now: number): { pause: boolean; next: number } {
  const t = now - b.start;
  const k = Math.floor(t / b.period);
  const p = t - k * b.period;
  const from = b.stillFrom + MARGIN;
  const to = b.period - MARGIN;
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

  const tick = () => {
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
      const want = beatAt(b, now);
      if (want.pause && a.playState === 'running') a.pause();
      else if (!want.pause && a.playState === 'paused') {
        a.play();
        // Back at the phase it would have had had it never stopped.
        a.startTime = b.start;
      }
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
    clearTimeout(timer);
    tick();
  };

  doc.addEventListener('animationstart', onStart, true);
  return () => {
    doc.removeEventListener('animationstart', onStart, true);
    clearTimeout(timer);
    tracked.clear();
  };
}
