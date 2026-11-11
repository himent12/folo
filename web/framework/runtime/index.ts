// fw: FOLO Studio's UI runtime. Compile-time JSX (templates cloned, dynamic parts reached by sibling walks) on
// fine-grained signals. See ../README.md (design, API) and ../PORTING.md (Svelte → fw).

export {
  signal,
  state,
  computed,
  effect,
  renderEffect,
  batch,
  flush,
  tick,
  untrack,
  isTracking,
  onCleanup,
  onDestroy,
  onMount,
  createRoot,
  getOwner,
  runWithOwner,
  dispose,
  setContext,
  getContext,
  hasContext,
  createContext,
  memo,
  type Signal,
  type ReadSignal,
  type SignalOptions,
  type Context,
  type Owner,
  type Cleanup,
  type Reader,
} from './core';
export { mergeProps, splitProps, defaults, bindable, field, createUniqueId, createUniqueId as propsId, type Component, type ComponentProps } from './component';
export { Show, For, Index, Switch, Match, Key, Dynamic, Portal, type ShowProps, type ForProps, type IndexProps } from './flow';
export { render } from './render';
export { on, onWindow, onDocument } from './events';
export { deep, snapshot, ReactiveMap, ReactiveSet, MediaQuery, mediaQuery, createSubscriber } from './reactivity';
export { fade, fly, linear, cubicOut, cubicInOut, type TransitionConfig, type TransitionFn, type EasingFunction, type FadeParams, type FlyParams } from './transition';
export { Tween, type TweenOptions } from './tween';
export { clsx, type ClassValue, type StyleValue } from './dom';
export type { Action, ActionReturn, Bindable, Ref } from './bind';
export type { IconComponent, IconProps } from './icon';

/** The attributes an intrinsic element takes (port of svelte/elements' `HTMLButtonAttributes` and co.): `ElementProps<'button'>`. */
export type ElementProps<K extends keyof JSX.IntrinsicElements> = JSX.IntrinsicElements[K];

/**
 * Scoped styles for this file (port of a component's `<style>`): `css\`...\``. The compiler extracts the CSS at build
 * time, scopes it with the file's class exactly as Svelte 5 would, and removes the call. Write it once per file, at
 * the top level, with no `${}` in it.
 */
export function css(_strings: TemplateStringsArray): void {
  throw new Error('fw: css`` must be compiled (is the fw Vite plugin installed, and is this a .tsx file?)');
}
