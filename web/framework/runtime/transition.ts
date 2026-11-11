// Transitions (port of svelte/transition and its engine). `transition:fly={params}`, `in:fade={params}` and
// `out:fn={params}` on an element are local, as in Svelte 5: the intro plays when the nearest block (`For` row, `Show`,
// `Key`) adds the element after that block first rendered; the outro plays when that block removes it, and the nodes
// stay in the DOM until it ends. The animation is the same as Svelte's: `css(t, u)` sampled at 60 fps into Web
// Animations keyframes, with a dummy animation for the delay. A duration of 0 (`motion(ms)` under reduced motion)
// skips it.

import { effect, untrack } from './core';
import { setLeave } from './dom';

export type EasingFunction = (t: number) => number;

export interface TransitionConfig {
  delay?: number;
  duration?: number;
  easing?: EasingFunction;
  css?: (t: number, u: number) => string;
  tick?: (t: number, u: number) => void;
}

export type TransitionFn<P = unknown> = (node: Element, params: P, options: { direction: 'in' | 'out' | 'both' }) => TransitionConfig;

/** `t` unchanged. */
export const linear: EasingFunction = (t) => t;
/** Svelte's `cubicOut`. */
export const cubicOut: EasingFunction = (t) => {
  const f = t - 1.0;
  return f * f * f + 1.0;
};
/** Svelte's `cubicInOut`. */
export const cubicInOut: EasingFunction = (t) => (t < 0.5 ? 4.0 * t * t * t : 0.5 * Math.pow(2.0 * t - 2.0, 3.0) + 1.0);

function splitUnit(value: number | string): [number, string] {
  const split = typeof value === 'string' && value.match(/^\s*(-?[\d.]+)([^\s]*)\s*$/);
  return split ? [parseFloat(split[1]!), split[2] || 'px'] : [value as number, 'px'];
}

export interface FadeParams {
  delay?: number;
  duration?: number;
  easing?: EasingFunction;
}

/** Fades opacity from 0 (port of svelte/transition `fade`). */
export function fade(node: Element, { delay = 0, duration = 400, easing = linear }: FadeParams = {}): TransitionConfig {
  const o = +getComputedStyle(node).opacity;
  return { delay, duration, easing, css: (t) => `opacity: ${t * o}` };
}

export interface FlyParams {
  delay?: number;
  duration?: number;
  easing?: EasingFunction;
  x?: number | string;
  y?: number | string;
  opacity?: number;
}

/** Slides from an offset while fading (port of svelte/transition `fly`). */
export function fly(node: Element, { delay = 0, duration = 400, easing = cubicOut, x = 0, y = 0, opacity = 0 }: FlyParams = {}): TransitionConfig {
  const style = getComputedStyle(node);
  const targetOpacity = +style.opacity;
  const transform = style.transform === 'none' ? '' : style.transform;
  const od = targetOpacity * (1 - opacity);
  const [xValue, xUnit] = splitUnit(x);
  const [yValue, yUnit] = splitUnit(y);
  return {
    delay,
    duration,
    easing,
    css: (t, u) => `
			transform: ${transform} translate(${(1 - t) * xValue}${xUnit}, ${(1 - t) * yValue}${yUnit});
			opacity: ${targetOpacity - od * u}`,
  };
}

// ---------------------------------------------------------------------------------------------------------------------
// Blocks

interface Manager {
  out?: (done: () => void) => void;
}

/** A block's branch (a `For` row, a `Show` or `Key` content): the local transitions inside it. */
export interface Branch {
  t: Manager[] | null;
}

let branch: Branch | null = null;
let intro = false;

/** Internal: runs `fn` (which creates a branch's content) with `b` as the current branch; `introOk` lets intros play. */
export function inBranch<T>(b: Branch | null, introOk: boolean, fn: () => T): T {
  const pb = branch;
  const pi = intro;
  branch = b;
  intro = introOk;
  try {
    return fn();
  } finally {
    branch = pb;
    intro = pi;
  }
}

function collect(value: unknown, out: Node[]): void {
  if (value instanceof Node) out.push(value);
  else if (Array.isArray(value)) for (const v of value) collect(v, out);
}

/**
 * Internal: `b`'s content is being removed. If it has outros, its top nodes (`value`) wait for them before leaving
 * the DOM. Returns whether any outro was set up.
 */
export function leaveBranch(b: Branch | null, value: unknown): boolean {
  const outs = b?.t?.filter((m) => m.out);
  if (!outs || outs.length === 0) return false;
  const nodes: Node[] = [];
  collect(value, nodes);
  if (nodes.length === 0) return false;
  let remaining = outs.length;
  let started = false;
  let finished = false;
  const dones: (() => void)[] = [];
  const start = () => {
    if (started) return;
    started = true;
    for (const m of outs)
      m.out!(() => {
        if (--remaining === 0) {
          finished = true;
          for (const d of dones) d();
        }
      });
  };
  for (const n of nodes)
    setLeave(n, (done) => {
      if (finished) done();
      else dones.push(done);
      start();
    });
  return true;
}

// ---------------------------------------------------------------------------------------------------------------------
// The engine

interface Animation {
  abort(): void;
  deactivate(): void;
  t(): number;
}

const noop = () => {};

function toKeyframe(css: string): Keyframe {
  const keyframe: Keyframe = {};
  for (const part of css.split(';')) {
    const [property, value] = part.split(':');
    if (!property || value === undefined) break;
    keyframe[property.trim().replace(/-([a-z])/g, (_, c: string) => c.toUpperCase())] = value.trim();
  }
  return keyframe;
}

function animate(el: Element, options: TransitionConfig, counterpart: Animation | undefined, t2: number, onFinish: () => void): Animation {
  const isIntro = t2 === 1;
  let aborted = false;
  counterpart?.deactivate();
  if (!options.duration && !options.delay) {
    onFinish();
    return { abort: noop, deactivate: noop, t: () => t2 };
  }
  const { delay = 0, css, tick, easing = linear } = options;
  let animation: globalThis.Animation | undefined;
  let getT = () => 1 - t2;
  let finish = onFinish;
  queueMicrotask(() => {
    if (aborted) return;
    if (typeof el.animate !== 'function') {
      // No Web Animations (a test DOM): end at once.
      getT = () => t2;
      tick?.(t2, 1 - t2);
      finish();
      return;
    }
    const first: Keyframe[] = [];
    if (isIntro && counterpart === undefined) {
      tick?.(0, 1);
      if (css) {
        const k = toKeyframe(css(0, 1));
        first.push(k, k);
      }
    }
    animation = el.animate(first, { duration: delay, fill: 'forwards' });
    animation.onfinish = () => {
      animation!.cancel();
      const t1 = counterpart?.t() ?? 1 - t2;
      counterpart?.abort();
      const delta = t2 - t1;
      const duration = (options.duration ?? 0) * Math.abs(delta);
      const frames: Keyframe[] = [];
      if (duration > 0 && css) {
        const n = Math.ceil(duration / (1000 / 60));
        for (let i = 0; i <= n; i += 1) {
          const t = t1 + delta * easing(i / n);
          frames.push(toKeyframe(css(t, 1 - t)));
        }
      }
      if (duration > 0) getT = () => t1 + delta * easing((animation!.currentTime as number) / duration);
      animation = el.animate(frames, { duration, fill: 'forwards' });
      animation.onfinish = () => {
        getT = () => t2;
        tick?.(t2, 1 - t2);
        finish();
      };
    };
  });
  return {
    abort() {
      aborted = true;
      if (animation) {
        animation.cancel();
        animation.effect = null;
        animation.onfinish = noop;
      }
    },
    deactivate() {
      finish = noop;
    },
    t: () => getT(),
  };
}

const IN = 1;
const OUT = 2;

/**
 * Internal: `transition:fn={params}` (flags 3), `in:fn` (1) or `out:fn` (2) on `el`. Registers with the current
 * branch; plays the intro after mount when the branch allows it.
 */
export function transition<P>(el: Element, fn: TransitionFn<P>, params: (() => P) | undefined, flags: number): void {
  const direction = flags === (IN | OUT) ? 'both' : flags & IN ? 'in' : 'out';
  let options: TransitionConfig | undefined;
  let introAnim: Animation | undefined;
  let outroAnim: Animation | undefined;
  const html = el as HTMLElement;
  const inert = html.inert;
  const getOptions = () => (options ??= untrack(() => fn(el, (params?.() ?? {}) as P, { direction })));
  const manager: Manager = {};
  if (flags & OUT) {
    manager.out = (done) => {
      html.inert = true;
      outroAnim = animate(el, getOptions(), introAnim, 0, () => {
        el.dispatchEvent(new CustomEvent('outroend'));
        done();
      });
    };
  }
  if (branch !== null) (branch.t ??= []).push(manager);
  if (flags & IN && intro) {
    effect(() => {
      untrack(() => {
        html.inert = inert;
        introAnim?.abort();
        introAnim = animate(el, getOptions(), outroAnim, 1, () => {
          el.dispatchEvent(new CustomEvent('introend'));
          introAnim?.abort();
          introAnim = options = undefined;
        });
      });
    });
  }
}
