// JSX types for fw (`jsx: "preserve"`, the global `JSX` namespace). Intrinsic elements take Svelte 5's attribute
// names and types (svelte/elements: HTML names such as `class`, `for`, `tabindex`, lowercase `onclick` handlers with a
// typed `currentTarget`), plus fw's directives: `class:name`, `style:prop`, `use:action`, `transition:fn`, `in:fn`,
// `out:fn`, `bind:value|checked|group|open|this|clientWidth|...`, `ref`, `use`, and `style` as a string or an object.
// Components are typed by their props; any prop `x` may instead be bound with `bind:x={signal}`.
//
// svelte/elements is imported as types only: it stays a dev dependency until the last Svelte component is gone, when
// these declarations get vendored.

import type { SvelteHTMLElements } from 'svelte/elements';
import type { Action, Bindable, Ref } from './runtime/bind';
import type { StyleValue } from './runtime/dom';
import type { TransitionFn } from './runtime/transition';

type Tags = HTMLElementTagNameMap & Omit<SVGElementTagNameMap, keyof HTMLElementTagNameMap> & { a: HTMLAnchorElement; title: HTMLTitleElement };
type ElementOf<K> = K extends keyof Tags ? Tags[K] : Element;

/** `use={...}`: an action, `[action, param]`, or a list of those. */
type UseValue<E extends Element> = Action<E, any> | readonly [Action<E, any>, unknown?] | readonly (Action<E, any> | readonly [Action<E, any>, unknown?])[];

/** What fw adds to (or changes in) every intrinsic element's attributes. */
interface FwAttributes<E extends Element> {
  children?: JSX.Element;
  /** Port of `bind:this`: a callback, or a signal set to the element (and back to `undefined` when it goes). */
  ref?: Ref<E>;
  use?: UseValue<E>;
  style?: StyleValue | null;
  'bind:this'?: Ref<E>;
  'bind:value'?: Bindable<any>;
  'bind:checked'?: Bindable<boolean>;
  'bind:group'?: Bindable<any>;
  'bind:open'?: Bindable<boolean>;
  'bind:clientWidth'?: { set(v: number): void };
  'bind:clientHeight'?: { set(v: number): void };
  'bind:offsetWidth'?: { set(v: number): void };
  'bind:offsetHeight'?: { set(v: number): void };
  [directive: `class:${string}`]: unknown;
  [directive: `style:${string}`]: string | number | null | undefined | false;
  [directive: `use:${string}`]: unknown;
  [directive: `transition:${string}` | `in:${string}` | `out:${string}`]: unknown;
}

type Intrinsic<K extends Tag> = Omit<SvelteHTMLElements[K], 'children' | 'style' | `bind:${string}` | `on:${string}`> &
  FwAttributes<ElementOf<K>>;

/** The element tags SvelteHTMLElements names (not its catch-all index signature, nor svelte:*). */
type Tag = keyof { [K in keyof SvelteHTMLElements as string extends K ? never : K extends `svelte:${string}` ? never : K]: 0 };

/** Props that can be bound: all but children. */
type BindKeys<P> = Exclude<keyof P, 'children'>;
type RequiredKeys<P> = { [K in BindKeys<P>]-?: {} extends Pick<P, K> ? never : K }[BindKeys<P>];
/** What `bind:x` takes for a prop of type T: anything readable as T and writable with a T. */
interface BindTarget<T> {
  get(): T | undefined;
  set(v: Exclude<T, undefined>): void;
}
/** For each required prop `x`: `x` or `bind:x` (the union per key, intersected over keys). */
type EitherRequired<P, R extends keyof P> = { [K in R]: (x: { [Q in K]: P[K] } | { [Q in K as Q extends string ? `bind:${Q}` : never]: BindTarget<P[K]> }) => void }[R] extends (x: infer I) => void ? I : never;
/** A component's props, where each prop `x` can be given as `x={...}` or bound as `bind:x={signal}`. */
type Bindified<P> = Omit<P, RequiredKeys<P>> & {
  [K in BindKeys<P> as K extends string ? `bind:${K}` : never]?: BindTarget<P[K]>;
} & ([RequiredKeys<P>] extends [never] ? unknown : EitherRequired<P, RequiredKeys<P>>);

declare global {
  namespace JSX {
    /** What JSX evaluates to and what a component returns: nodes, text, lists, functions read reactively, nothing. */
    type Element = Node | ArrayElement | FunctionElement | (string & {}) | number | bigint | boolean | null | undefined;
    interface ArrayElement extends Array<Element> {}
    interface FunctionElement {
      (): Element;
    }
    type ElementType = keyof IntrinsicElements | ((props: any) => Element);
    interface ElementChildrenAttribute {
      children: {};
    }
    interface IntrinsicAttributes {}
    type LibraryManagedAttributes<_C, P> = Bindified<P>;
    type IntrinsicElements = { [K in Tag]: Intrinsic<K> };
  }
}


export type { TransitionFn };
