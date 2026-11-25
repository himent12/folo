// The DOM layer: cloned templates, `insert` (text, nodes, arrays, reactive functions) with a keyed array reconciler,
// and attribute, property, class and style setters with Svelte 5's semantics.

/* eslint-disable @typescript-eslint/no-explicit-any */

import { isReader, memo, renderEffect } from './core';

// ---------------------------------------------------------------------------------------------------------------------
// Templates

/**
 * Internal: a cloner for a static HTML template (the compiler hoists one per JSX root). Parsed on first use.
 * `svg`: the HTML is an SVG fragment, wrapped in `<svg>` to parse in the SVG namespace.
 */
export function template(html: string, svg?: 1): () => Node {
  let node: Node | undefined;
  return () => {
    if (node === undefined) {
      const t = document.createElement('template');
      t.innerHTML = svg ? `<svg>${html}</svg>` : html;
      const first = t.content.firstChild!;
      node = svg ? first.firstChild! : first;
    }
    return node.cloneNode(true);
  };
}

// ---------------------------------------------------------------------------------------------------------------------
// Outros: a node can ask to leave later (an outro transition); `insert` then lets it finish before removing it.

type Leaving = Node & { $$leave?: (done: () => void) => void };
/** Nodes with a pending `$$leave`: while any exist, clearing never takes the `textContent = ''` shortcut. */
let leaving = 0;

/** Internal: `node` will run `fn(done)` instead of being removed at once. */
export function setLeave(node: Node, fn: ((done: () => void) => void) | undefined): void {
  const n = node as Leaving;
  if (n.$$leave && !fn) leaving--;
  else if (!n.$$leave && fn) leaving++;
  n.$$leave = fn;
}

function removeNode(node: Node): void {
  const n = node as Leaving;
  const leave = n.$$leave;
  if (leave !== undefined) {
    n.$$leave = undefined;
    leaving--;
    leave(() => n.parentNode?.removeChild(n));
  } else n.parentNode?.removeChild(n);
}

// ---------------------------------------------------------------------------------------------------------------------
// insert

/** What `insert` keeps between runs: a string (sole text content), a node, or the array of nodes in the DOM. */
type Current = unknown;

/**
 * Internal: renders `value` into `parent` before `marker` (`null`: at the end; `undefined`: as the parent's only
 * content). A function is reactive: re-evaluated in a render effect whenever what it reads changes. A function it
 * returns is a reader (read in the same effect) or a snippet (called in an effect of its own, so what the snippet
 * creates is not re-created when the outer function re-reads).
 */
export function insert(parent: Node, value: unknown, marker?: Node | null, initial?: Current): void {
  if (marker !== undefined && initial === undefined) initial = [];
  if (typeof value !== 'function' || isReader(value)) {
    if (typeof value === 'function') renderEffect<Current>((current) => insertExpression(parent, value, current, marker), initial);
    else insertExpression(parent, value, initial, marker);
    return;
  }
  // An expression (`{props.children}`, `{row()}`): re-evaluated when what it reads changes. A reader or snippet it
  // returns is read in an effect of its own, so a change inside a branch never re-runs the expression (which would
  // create the branch again).
  renderEffect<Current>((current) => insertExpression(parent, (value as () => unknown)(), current, marker, false, true), initial);
}

/**
 * Internal: applies `value` over `current` (what was rendered before) and returns what is rendered now. `expression`:
 * `value` is what an expression returned, so a reader it returned is read in an effect of its own.
 */
export function insertExpression(parent: Node, value: unknown, current: Current, marker?: Node | null, unwrap?: boolean, expression?: boolean): Current {
  while (typeof current === 'function') current = current();
  if (value === current) return current;
  const t = typeof value;
  const multi = marker !== undefined;
  // A fragment moved elsewhere (a portal, a reordered row) renders where its nodes are now.
  if (multi && Array.isArray(current) && current.length > 0) parent = (current[0] as Node).parentNode ?? parent;

  if (t === 'string' || t === 'number' || t === 'bigint') {
    const text = String(value);
    if (multi) {
      let node = Array.isArray(current) ? (current[0] as Node | undefined) : undefined;
      if (node !== undefined && node.nodeType === 3 && (current as Node[]).length >= 1) {
        if ((node as Text).data !== text) (node as Text).data = text;
      } else node = document.createTextNode(text);
      return cleanChildren(parent, current as Node[], marker, node);
    }
    if (typeof current === 'string' && current !== '' && parent.firstChild !== null && parent.firstChild.nodeType === 3) {
      (parent.firstChild as Text).data = text;
    } else if (leaving === 0 || !Array.isArray(current)) parent.textContent = text;
    else {
      cleanChildren(parent, current as Node[], null, document.createTextNode(text));
    }
    return text;
  }

  if (value == null || t === 'boolean') {
    return cleanChildren(parent, current, marker);
  }

  if (t === 'function') {
    // A reader read by `insert` itself is read here, in the effect that is running. A snippet is called once (a memo
    // owns what it creates) and read, like a reader an expression returned, in an effect of its own.
    if (isReader(value) && !expression) return insertExpression(parent, value(), current, marker, unwrap);
    const read = isReader(value) ? value : memo(value as () => unknown);
    let cur: Current = current;
    renderEffect<Current>((c) => (cur = insertExpression(parent, read(), c, marker)), current);
    return () => cur;
  }

  if (Array.isArray(value)) {
    const array: Node[] = [];
    const currentArray = Array.isArray(current) ? (current as Node[]) : null;
    if (normalize(array, value, currentArray, unwrap)) {
      // Reactive parts: resolve them in an effect of their own, so the code that built the array does not re-run.
      let cur: Current = current;
      renderEffect<Current>((c) => (cur = insertExpression(parent, array, c, marker, true)), current);
      return () => cur;
    }
    if (array.length === 0) {
      const r = cleanChildren(parent, current, marker);
      if (multi) return r;
    } else if (currentArray !== null) {
      if (currentArray.length === 0) appendNodes(parent, array, marker);
      else reconcileArrays(parent, currentArray, array, marker);
    } else {
      if (current != null && current !== '') cleanChildren(parent, current, undefined);
      appendNodes(parent, array, marker);
    }
    return array;
  }

  // A node.
  const node = value as Node;
  if (Array.isArray(current)) {
    if (multi) return cleanChildren(parent, current, marker, node);
    cleanChildren(parent, current, null, node);
  } else if (current == null || current === '' || parent.firstChild === null) {
    parent.appendChild(node);
  } else parent.replaceChild(node, parent.firstChild);
  return node;
}

/**
 * Flattens `value` into nodes; returns true when it holds functions (reactive parts) that were not resolved. Before
 * resolving (`unwrap` false), a snippet in the array is wrapped in a memo once, so the array's effect only reads.
 */
function normalize(out: Node[], value: unknown[], current: Node[] | null, unwrap?: boolean): boolean {
  let dynamic = false;
  for (let i = 0, len = value.length; i < len; i++) {
    let item = value[i];
    const prev = current?.[out.length];
    if (item == null || item === true || item === false) continue;
    const t = typeof item;
    if (t === 'object' && (item as Node).nodeType !== undefined) out.push(item as Node);
    else if (Array.isArray(item)) dynamic = normalize(out, item, current, unwrap) || dynamic;
    else if (t === 'function') {
      if (!isReader(item)) item = memo(item as () => unknown);
      if (unwrap) {
        while (typeof item === 'function') item = isReader(item) ? item() : memo(item as () => unknown)();
        dynamic = normalize(out, Array.isArray(item) ? item : [item], current, unwrap) || dynamic;
      } else {
        out.push(item as unknown as Node);
        dynamic = true;
      }
    } else {
      const text = String(item);
      if (prev !== undefined && prev.nodeType === 3 && (prev as Text).data === text) out.push(prev);
      else out.push(document.createTextNode(text));
    }
  }
  return dynamic;
}

function appendNodes(parent: Node, array: Node[], marker: Node | null | undefined): void {
  const m = marker ?? null;
  for (let i = 0, len = array.length; i < len; i++) parent.insertBefore(array[i]!, m);
}

/**
 * Removes what `current` rendered, leaving `replacement` (or an empty text node, which keeps the position of a
 * multi-node insert) in its place.
 */
function cleanChildren(parent: Node, current: Current, marker: Node | null | undefined, replacement?: Node): Current {
  if (marker === undefined) {
    if (leaving === 0) parent.textContent = '';
    else if (Array.isArray(current)) for (const n of current as Node[]) removeNode(n);
    else if (current instanceof Node) removeNode(current);
    else parent.textContent = '';
    if (replacement !== undefined) parent.appendChild(replacement);
    return replacement === undefined ? [] : [replacement];
  }
  const node = replacement ?? document.createTextNode('');
  const cur = Array.isArray(current) ? (current as Node[]) : current instanceof Node ? [current] : [];
  if (cur.length > 0) {
    let inserted = false;
    for (let i = cur.length - 1; i >= 0; i--) {
      const el = cur[i]!;
      if (node !== el) {
        const isParent = el.parentNode === parent;
        if (!inserted && i === 0) {
          if (isParent) {
            if ((el as Leaving).$$leave !== undefined) {
              parent.insertBefore(node, el);
              removeNode(el);
            } else parent.replaceChild(node, el);
          } else parent.insertBefore(node, marker);
        } else if (isParent) removeNode(el);
      } else inserted = true;
    }
  } else parent.insertBefore(node, marker);
  return [node];
}

type Focusable = Element & { focus?: (options?: FocusOptions) => void };

/** The active element if it is a focusable descendant inside `parent`, computed once per reconcile before moving nodes. */
function findFocused(parent: Node): Focusable | null {
  if (typeof document === 'undefined') return null;
  const active = document.activeElement;
  return active != null && active !== document.body && active !== parent && parent.contains(active)
    ? (active as Focusable)
    : null;
}

/**
 * Moves a node that is already in `parent`, keeping focus where it was. With `insertBefore`, as Svelte moves rows:
 * Chrome's `moveBefore` would keep focus by itself, but a node moved with it stops restyling its descendants when
 * its own attributes change afterwards (a chip moved first, then pressed, kept its unpressed key).
 */
function move(parent: Node, node: Node, ref: Node | null, active: Focusable | null): Focusable | null {
  const hadFocus = active !== null && node.contains(active);
  parent.insertBefore(node, ref);
  if (!hadFocus) return active;
  if (document.activeElement !== active) active.focus?.({ preventScroll: true });
  return document.activeElement === active ? null : findFocused(parent);
}

/**
 * Updates the nodes `a` (in the DOM) to `b`: skips a common prefix and suffix, then reuses nodes by identity and moves
 * only those outside the longest increasing subsequence of their old positions (the minimal set of moves).
 */
function reconcileArrays(parent: Node, a: Node[], b: Node[], marker: Node | null | undefined): void {
  let aStart = 0;
  let bStart = 0;
  let aEnd = a.length;
  let bEnd = b.length;
  let active: Focusable | null | undefined;
  const after = a[aEnd - 1]!.nextSibling;
  for (;;) {
    while (aStart < aEnd && bStart < bEnd && a[aStart] === b[bStart]) {
      aStart++;
      bStart++;
    }
    while (aEnd > aStart && bEnd > bStart && a[aEnd - 1] === b[bEnd - 1]) {
      aEnd--;
      bEnd--;
    }
    // Two rows swapped end for end (udomdiff's swap): two moves, no map.
    if (aEnd - aStart > 1 && bEnd - bStart > 1 && a[aStart] === b[bEnd - 1] && a[aEnd - 1] === b[bStart]) {
      if (active === undefined) active = findFocused(parent);
      const next = a[aEnd - 1]!.nextSibling;
      active = move(parent, b[bStart]!, a[aStart]!.nextSibling, active);
      active = move(parent, b[bEnd - 1]!, next, active);
      a[aEnd - 1] = b[bEnd - 1]!;
      a[aStart] = b[bStart]!;
      continue;
    }
    break;
  }
  if (aStart === aEnd) {
    // Only insertions.
    const ref = bEnd < b.length ? b[bEnd]! : after;
    while (bStart < bEnd) parent.insertBefore(b[bStart++]!, ref);
    return;
  }
  if (bStart === bEnd) {
    // Only removals.
    while (aStart < aEnd) removeNode(a[aStart++]!);
    return;
  }
  const n = bEnd - bStart;
  const index = new Map<Node, number>();
  for (let i = bStart; i < bEnd; i++) index.set(b[i]!, i);
  const sources = new Int32Array(n).fill(-1);
  let reused = 0;
  let moved = false;
  let last = 0;
  for (let i = aStart; i < aEnd; i++) {
    const j = index.get(a[i]!);
    if (j === undefined) continue;
    sources[j - bStart] = i;
    reused++;
    if (j < last) moved = true;
    else last = j;
  }
  if (reused === 0 && marker === undefined && aStart === 0 && aEnd === a.length && leaving === 0) {
    // Nothing kept and the insert owns the parent: one clear, then append.
    parent.textContent = '';
    appendNodes(parent, b, null);
    return;
  }
  if (reused < aEnd - aStart) {
    for (let i = aStart; i < aEnd; i++) if (!index.has(a[i]!)) removeNode(a[i]!);
    active = undefined;
  }
  if (moved) {
    const seq = lis(sources);
    let k = seq.length - 1;
    for (let i = n - 1; i >= 0; i--) {
      const node = b[bStart + i]!;
      const ref = bStart + i + 1 < b.length ? b[bStart + i + 1]! : after;
      if (sources[i] === -1) {
        parent.insertBefore(node, ref);
        active = undefined;
      } else if (k < 0 || seq[k] !== i) {
        if (active === undefined) active = findFocused(parent);
        active = move(parent, node, ref, active);
      } else k--;
    }
  } else if (reused !== n) {
    for (let i = n - 1; i >= 0; i--) {
      if (sources[i] !== -1) continue;
      const ref = bStart + i + 1 < b.length ? b[bStart + i + 1]! : after;
      parent.insertBefore(b[bStart + i]!, ref);
    }
  }
}

/** Indices (into `arr`) of a longest strictly increasing subsequence, ignoring -1 entries. */
function lis(arr: Int32Array): number[] {
  const n = arr.length;
  const prev = new Int32Array(n);
  const tails: number[] = [];
  for (let i = 0; i < n; i++) {
    const v = arr[i]!;
    if (v === -1) continue;
    let lo = 0;
    let hi = tails.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (arr[tails[mid]!]! < v) lo = mid + 1;
      else hi = mid;
    }
    prev[i] = lo > 0 ? tails[lo - 1]! : -1;
    tails[lo] = i;
  }
  const out: number[] = new Array(tails.length);
  let k = tails.length > 0 ? tails[tails.length - 1]! : -1;
  for (let i = tails.length - 1; i >= 0; i--) {
    out[i] = k;
    k = prev[k]!;
  }
  return out;
}

// ---------------------------------------------------------------------------------------------------------------------
// Attributes and properties (Svelte 5 semantics)

/**
 * Attribute names whose DOM property is spelled differently (Svelte's ATTRIBUTE_ALIASES). The compiler knows the full
 * list of attributes set as properties; at runtime (spreads) a non-string value goes to a property whenever the element
 * has a setter of that name, as Svelte's set_attributes does.
 */
const ALIASES: Record<string, string> = {
  formnovalidate: 'formNoValidate',
  ismap: 'isMap',
  nomodule: 'noModule',
  playsinline: 'playsInline',
  readonly: 'readOnly',
  defaultvalue: 'defaultValue',
  defaultchecked: 'defaultChecked',
  srcobject: 'srcObject',
  novalidate: 'noValidate',
  allowfullscreen: 'allowFullscreen',
  disablepictureinpicture: 'disablePictureInPicture',
  disableremoteplayback: 'disableRemotePlayback',
};

/** The DOM property an attribute name is set through, if any (`readonly` → `readOnly`). */
export function propName(name: string): string | undefined {
  return ALIASES[name.toLowerCase()];
}

const settersCache = new Map<string, Set<string>>();
function setters(el: Element): Set<string> {
  const key = el.getAttribute('is') || el.nodeName;
  let set = settersCache.get(key);
  if (set !== undefined) return set;
  settersCache.set(key, (set = new Set()));
  let proto: object | null = el;
  const stop = Element.prototype;
  while (proto !== null && proto !== stop) {
    const descriptors = Object.getOwnPropertyDescriptors(proto);
    for (const k in descriptors) {
      if (descriptors[k]!.set && k !== 'innerHTML' && k !== 'textContent' && k !== 'innerText') set.add(k);
    }
    proto = Object.getPrototypeOf(proto);
  }
  return set;
}

/** Whether `el` has a DOM setter called `name` (cached per tag). */
export function hasSetter(el: Element, name: string): boolean {
  return setters(el).has(name);
}

/** Sets an attribute as Svelte does: `null`/`undefined` remove it; a non-string with a matching DOM setter is a property. */
export function setAttribute(el: Element, name: string, value: unknown): void {
  if (value == null) el.removeAttribute(name);
  else if (typeof value !== 'string' && setters(el).has(name)) (el as any)[name] = value;
  else if (name.startsWith('xlink:')) el.setAttributeNS('http://www.w3.org/1999/xlink', name, String(value));
  else el.setAttribute(name, value as string);
}

/** Sets a DOM property (`disabled`, `readOnly`, `open`, ...). */
export function setProperty(el: Element, name: string, value: unknown): void {
  (el as any)[name] = value;
}

/** `value` as Svelte sets it: skipped when unchanged, `null`/`undefined` → `''` (a `<progress>` always takes 0). */
export function setValue(el: Element, value: unknown): void {
  const e = el as HTMLInputElement & { __value?: unknown };
  if ((e.__value === (e.__value = value ?? undefined)) || ((e.value as unknown) === value && (value !== 0 || el.nodeName !== 'PROGRESS'))) return;
  e.value = (value ?? '') as string;
}

/** `checked` as Svelte sets it (skipped when unchanged). */
export function setChecked(el: Element, value: unknown): void {
  const e = el as HTMLInputElement & { __checked?: unknown };
  if (e.__checked === (e.__checked = value ?? undefined)) return;
  e.checked = value as boolean;
}

/** Internal: a reactive attribute. */
export function attr(el: Element, name: string, fn: () => unknown): void {
  renderEffect<unknown>((prev) => {
    const v = fn();
    if (v !== prev) setAttribute(el, name, v);
    return v;
  }, UNSET);
}

/** Internal: a reactive DOM property. */
export function prop(el: Element, name: string, fn: () => unknown): void {
  renderEffect<unknown>((prev) => {
    const v = fn();
    if (v !== prev) {
      if (name === 'value') setValue(el, v);
      else if (name === 'checked') setChecked(el, v);
      else (el as any)[name] = v;
    }
    return v;
  }, UNSET);
}

const UNSET = {};

// ---------------------------------------------------------------------------------------------------------------------
// Classes

/** `class` values as Svelte 5 takes them: a string, or clsx-style arrays and objects. */
export type ClassValue = string | number | bigint | boolean | null | undefined | ClassValue[] | Record<string, unknown>;

/** Joins class values (clsx). */
export function clsx(value: ClassValue): string {
  if (typeof value !== 'object' || value === null) return value == null || value === false || value === true ? '' : String(value);
  let out = '';
  if (Array.isArray(value)) {
    for (const v of value) {
      const s = clsx(v);
      if (s) out = out ? out + ' ' + s : s;
    }
  } else {
    for (const k in value) if (value[k]) out = out ? out + ' ' + k : k;
  }
  return out;
}

const WS = ' \t\n\r\f \u000b﻿';

/** Svelte's `to_class`: the value, the scope class, then directive classes added or removed. */
export function toClass(value: unknown, scope: string | undefined, directives?: Record<string, unknown>): string | null {
  let c = value == null ? '' : '' + value;
  if (scope) c = c ? c + ' ' + scope : scope;
  if (directives) {
    for (const key of Object.keys(directives)) {
      if (directives[key]) c = c ? c + ' ' + key : key;
      else if (c.length) {
        const len = key.length;
        let a = 0;
        while ((a = c.indexOf(key, a)) >= 0) {
          const b = a + len;
          if ((a === 0 || WS.includes(c[a - 1]!)) && (b === c.length || WS.includes(c[b]!))) c = (a === 0 ? '' : c.substring(0, a)) + c.substring(b + 1);
          else a = b;
        }
      }
    }
  }
  return c === '' ? null : c;
}

function writeClass(el: Element, c: string | null): void {
  if (c === null) el.removeAttribute('class');
  else if (el.namespaceURI === 'http://www.w3.org/1999/xhtml') (el as HTMLElement).className = c;
  else el.setAttribute('class', c);
}

/**
 * Internal: a reactive `class` (plus the file's scope class, plus `class:` directives when the element has both).
 * Directive-only changes toggle just those classes, as Svelte does.
 */
export function className(el: Element, fn: () => unknown, scope?: string, dirs?: () => Record<string, unknown>): void {
  let prevValue: unknown = UNSET;
  let prevDirs: Record<string, unknown> | undefined;
  renderEffect(() => {
    const value = clsx(fn() as ClassValue);
    const d = dirs?.();
    if (value !== prevValue) writeClass(el, toClass(value, scope, d));
    else if (d !== undefined) {
      for (const k in d) if (prevDirs === undefined || !d[k] !== !prevDirs[k]) el.classList.toggle(k, !!d[k]);
    }
    prevValue = value;
    prevDirs = d;
  });
}

/** Internal: a `class:name={on}` directive on an element whose class is otherwise static. */
export function toggleClass(el: Element, name: string, fn: () => unknown): void {
  renderEffect<unknown>((prev) => {
    const on = !!fn();
    if (on !== prev) el.classList.toggle(name, on);
    return on;
  }, UNSET);
}

// ---------------------------------------------------------------------------------------------------------------------
// Styles: Svelte 5's `set_style` / `to_style`, step for step. The first application (and any change of the `style`
// attribute's value) writes the whole attribute text (`cssText`) with the directives appended; later changes of a
// directive alone go through `setProperty`. Chrome stores a value written either way slightly differently (an exact half
// pixel serializes as `17.2813px` one way and `17.2812px` the other), so the mechanism is the one Svelte uses.

/** `style` as a string (`cssText`) or an object of properties (`{ color: 'red', '--x': '2px' }`, as `style:` directives). */
export type StyleValue = string | null | undefined | false | Record<string, string | number | null | undefined | false>;

type Styles = Record<string, unknown>;

const kebab = new Map<string, string>();
/** A style object's key as a property name (`aspectRatio` → `aspect-ratio`; custom properties as they are). */
function cssName(k: string): string {
  if (k.charCodeAt(0) === 45 /* - */ || !/[A-Z]/.test(k)) return k;
  let n = kebab.get(k);
  if (n === undefined) kebab.set(k, (n = k.replace(/[A-Z]/g, (m) => '-' + m.toLowerCase())));
  return n;
}

/** A style object as directives (`false` is no value). */
function objectStyles(o: Record<string, unknown>): Styles {
  const out: Styles = {};
  for (const k in o) {
    const v = o[k];
    out[cssName(k)] = v === false ? undefined : v;
  }
  return out;
}

function appendStyles(styles: Styles, important: boolean): string {
  const separator = important ? ' !important;' : ';';
  let css = '';
  for (const key of Object.keys(styles)) {
    const value = styles[key];
    if (value != null && value !== '') css += ' ' + key + ': ' + value + separator;
  }
  return css;
}

function toCssName(name: string): string {
  return name[0] !== '-' || name[1] !== '-' ? name.toLowerCase() : name;
}

/** Svelte's `to_style`: the attribute's declarations minus those a directive sets, then the directives. */
export function toStyle(value: unknown, styles?: Styles, important?: Styles): string | null {
  if (!styles && !important) return value == null ? null : String(value);
  let out = '';
  if (value) {
    const text = String(value).replaceAll(/\/\*.*?\*\//g, '').trim();
    let inStr: false | string = false;
    let inApo = 0;
    let inComment = false;
    const reserved: string[] = [];
    if (styles) reserved.push(...Object.keys(styles).map(toCssName));
    if (important) reserved.push(...Object.keys(important).map(toCssName));
    let start = 0;
    let nameAt = -1;
    const len = text.length;
    for (let i = 0; i < len; i++) {
      const c = text[i]!;
      if (inComment) {
        if (c === '/' && text[i - 1] === '*') inComment = false;
      } else if (inStr) {
        if (inStr === c) inStr = false;
      } else if (c === '/' && text[i + 1] === '*') inComment = true;
      else if (c === '"' || c === "'") inStr = c;
      else if (c === '(') inApo++;
      else if (c === ')') inApo--;
      if (!inComment && inStr === false && inApo === 0) {
        if (c === ':' && nameAt === -1) nameAt = i;
        else if (c === ';' || i === len - 1) {
          if (nameAt !== -1) {
            const name = toCssName(text.substring(start, nameAt).trim());
            if (!reserved.includes(name)) {
              if (c !== ';') i++;
              out += ' ' + text.substring(start, i).trim() + ';';
            }
          }
          start = i + 1;
          nameAt = -1;
        }
      }
    }
  }
  if (styles) out += appendStyles(styles, false);
  if (important) out += appendStyles(important, true);
  out = out.trim();
  return out === '' ? null : out;
}

function updateStyles(el: HTMLElement, prev: Styles | undefined, next: Styles, priority?: string): void {
  for (const key in next) {
    const value = next[key];
    if ((prev === undefined ? undefined : prev[key]) !== value) {
      if (value == null) el.style.removeProperty(key);
      else el.style.setProperty(key, value as string, priority);
    }
  }
  // Keys a style object no longer has (directives always have the same keys).
  if (prev !== undefined) for (const key in prev) if (!(key in next) && prev[key] != null) el.style.removeProperty(key);
}

type StyleCached = HTMLElement & { $$style?: unknown };

/**
 * Svelte's `set_style`: `value` is the `style` attribute's value (`''` when the element only has directives), `next`
 * and `nextImportant` the directives (and style objects). The attribute text is rewritten when `value` changes (and
 * the first time); otherwise only the directives that changed are set.
 */
export function setStyle(el: Element, value: unknown, prev: Styles | undefined, next: Styles | undefined, prevImportant?: Styles, nextImportant?: Styles): void {
  const e = el as StyleCached;
  if (e.$$style !== value) {
    const text = toStyle(value, next, nextImportant);
    if (text == null) el.removeAttribute('style');
    else e.style.cssText = text;
    e.$$style = value;
    return;
  }
  if (next) updateStyles(e, prev, next);
  if (nextImportant) updateStyles(e, prevImportant, nextImportant, 'important');
}

/** Internal: sets one static `style:prop={value}` directive without an effect (ignores `null`/`undefined`/`''`). */
export function setStyleProp(el: Element, prop: string, value: unknown, important?: boolean): void {
  if (value != null && value !== '') (el as HTMLElement).style.setProperty(prop, String(value), important ? 'important' : undefined);
}

/** Internal: a single reactive `style:prop={fn}` directive on an element with no `style` attribute or spread. */
export function styleProp(el: Element, prop: string, fn: () => unknown, important?: boolean): void {
  const style = (el as HTMLElement).style;
  const priority = important ? 'important' : undefined;
  renderEffect<string | null>((prev) => {
    const raw = fn();
    const v = raw == null || raw === '' ? null : String(raw);
    if (v !== prev) {
      if (v === null) style.removeProperty(prop);
      else style.setProperty(prop, v, priority);
    }
    return v;
  }, null);
}

/**
 * Internal: an element's `style` (a string, an object, or `''` when it has only directives) with its `style:`
 * directives (`dirs`, and `important` for `|important` ones), as Svelte's `set_style` in one render effect.
 */
export function bindStyle(el: Element, value: (() => StyleValue) | string | null, dirs?: () => Styles, important?: () => Styles): void {
  let prev: Styles | undefined;
  let prevImportant: Styles | undefined;
  renderEffect(() => {
    let v: unknown = typeof value === 'function' ? value() : value;
    let next = dirs?.();
    if (v !== null && typeof v === 'object') {
      const o = objectStyles(v as Record<string, unknown>);
      next = next === undefined ? o : Object.assign(o, next);
      v = '';
    } else if (v === false) v = null;
    const nextImportant = important?.();
    setStyle(el, v, prev, next, prevImportant, nextImportant);
    prev = next;
    prevImportant = nextImportant;
  });
}

/** Internal (spreads): a spread's `style` value with the element's directives, as `bindStyle` takes them. */
export function styleParts(value: unknown, dirs: Styles | undefined): [unknown, Styles | undefined] {
  if (value !== null && typeof value === 'object') {
    const o = objectStyles(value as Record<string, unknown>);
    return ['', dirs === undefined ? o : Object.assign(o, dirs)];
  }
  return [value === false ? null : value, dirs];
}
