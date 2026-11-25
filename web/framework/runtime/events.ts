// Events. Common bubbling events are delegated, as Svelte 5 does: one listener per event type on each render root and
// on the document, which walks the event's path and calls the `$$<type>` handler stored on each element (with
// `currentTarget` set to it), stopping at `stopPropagation`. Other events use addEventListener on the element.
// The propagation walk is a port of Svelte's `handle_event_propagation`, so handler order, disabled elements and
// nested roots behave the same.

import { onCleanup, runWithOwner } from './core';

/** Event types handled by delegation (Svelte's list). */
export const DELEGATED = new Set([
  'beforeinput',
  'click',
  'change',
  'dblclick',
  'contextmenu',
  'focusin',
  'focusout',
  'input',
  'keydown',
  'keyup',
  'mousedown',
  'mousemove',
  'mouseout',
  'mouseover',
  'mouseup',
  'pointerdown',
  'pointermove',
  'pointerout',
  'pointerover',
  'pointerup',
  'touchend',
  'touchmove',
  'touchstart',
]);

const PASSIVE = new Set(['touchstart', 'touchmove']);
/** The event types some code has asked for. */
const registered = new Set<string>();
/** Per root (and the document): listener counts per event type. */
const roots = new Map<EventTarget, Map<string, number>>();
const handled = new WeakMap<Event, EventTarget>();

type Handler = (this: Element, e: Event) => unknown;
type Delegating = Element & Record<string, Handler | undefined>;

function listen(node: EventTarget, type: string): void {
  let counts = roots.get(node);
  if (counts === undefined) roots.set(node, (counts = new Map()));
  const n = counts.get(type);
  if (n === undefined) {
    node.addEventListener(type, propagate, { passive: PASSIVE.has(type) });
    counts.set(type, 1);
  } else counts.set(type, n + 1);
}

function unlisten(node: EventTarget, type: string): void {
  const counts = roots.get(node);
  const n = counts?.get(type);
  if (counts === undefined || n === undefined) return;
  if (n <= 1) {
    node.removeEventListener(type, propagate);
    counts.delete(type);
    if (counts.size === 0) roots.delete(node);
  } else counts.set(type, n - 1);
}

/** The roots that handle delegated events, ref-counted per EventTarget with the types currently attached. */
const rootEntries = new Map<EventTarget, { refs: number; types: Set<string> }>();

/** Internal: makes `root` (a render target) handle delegated events; returns an idempotent undo. */
export function delegateRoot(root: EventTarget): () => void {
  let entry = rootEntries.get(root);
  if (entry === undefined) {
    entry = { refs: 0, types: new Set<string>() };
    rootEntries.set(root, entry);
  }
  entry.refs++;
  if (entry.refs === 1) {
    for (const t of registered) addRootType(root, entry.types, t);
  }
  let undone = false;
  return () => {
    if (undone) return;
    undone = true;
    entry.refs--;
    if (entry.refs === 0) {
      for (const t of entry.types) {
        unlisten(root, t);
        unlisten(document, t);
      }
      entry.types.clear();
      rootEntries.delete(root);
    }
  };
}

function addRootType(root: EventTarget, types: Set<string>, type: string): void {
  if (types.has(type)) return;
  types.add(type);
  listen(root, type);
  listen(document, type);
}

/** Internal: makes sure delegated listeners for `types` exist (the compiler emits one call per module). */
export function delegateEvents(types: string[]): void {
  for (const t of types) {
    if (registered.has(t)) continue;
    registered.add(t);
    for (const [root, entry] of rootEntries) addRootType(root, entry.types, t);
  }
}

function hasHigherDelegatedTarget(path: EventTarget[], handlerElement: EventTarget, type: string): boolean {
  const idx = path.indexOf(handlerElement);
  if (idx === -1) return false;
  for (let i = idx + 1; i < path.length; i++) {
    if (roots.get(path[i]!)?.has(type)) return true;
  }
  return false;
}

function propagate(this: EventTarget, event: Event): void {
  const handlerElement = this;
  const type = event.type;
  const path = event.composedPath?.() ?? [];
  let pathIdx = 0;
  const handledAt = handled.get(event);
  if (handledAt !== undefined && handledAt !== handlerElement) {
    const atIdx = path.indexOf(handledAt);
    if (atIdx !== -1 && (handlerElement === document || handlerElement === (window as EventTarget))) {
      // The document fallback after a root already handled it: nothing to do, clean up unless another delegated root sits above.
      if (hasHigherDelegatedTarget(path, handlerElement, type)) handled.set(event, handlerElement);
      else handled.delete(event);
      return;
    }
    const handlerIdx = path.indexOf(handlerElement);
    if (handlerIdx === -1) return;
    if (atIdx !== -1 && atIdx <= handlerIdx) pathIdx = atIdx;
  }
  let current = (path[pathIdx] ?? event.target) as Element | null;
  if (current === handlerElement) return;
  const key = '$$' + type;
  let defined = false;
  try {
    let error: unknown;
    while (current !== null) {
      if (current === handlerElement) break;
      const h = (current as Delegating)[key];
      if (h != null && (!(current as HTMLButtonElement).disabled || event.target === current)) {
        if (!defined) {
          const ownerDocument = (handlerElement as Node).ownerDocument ?? (handlerElement as Document);
          Object.defineProperty(event, 'currentTarget', {
            configurable: true,
            get() {
              return current || ownerDocument;
            },
          });
          defined = true;
        }
        try {
          runWithOwner(null, () => h.call(current!, event));
        } catch (e) {
          if (error === undefined) error = e;
          else queueMicrotask(() => {
            throw e;
          });
        }
      }
      if (event.cancelBubble) break;
      pathIdx++;
      current = pathIdx < path.length ? (path[pathIdx] as Element) : null;
    }
    if (error !== undefined) throw error;
  } finally {
    if (event.cancelBubble || !event.bubbles || !hasHigherDelegatedTarget(path, handlerElement, type)) {
      handled.delete(event);
    } else {
      handled.set(event, handlerElement);
    }
    if (defined) delete (event as unknown as { currentTarget?: unknown }).currentTarget;
  }
}

/** Internal: sets a delegated handler (`onclick` and co.) on an element. */
export function setDelegated(el: Element, type: string, handler: unknown): void {
  (el as Delegating)['$$' + type] = (handler ?? undefined) as Handler | undefined;
  if (handler != null && !registered.has(type)) delegateEvents([type]);
}

/**
 * Adds a listener and removes it with the current owner. Pointer, touch and wheel listeners attach after a microtask
 * (Chrome drops them on cloned nodes that are not in the document yet), as Svelte does.
 * Returns a function that removes it at once.
 */
export function on<E extends Event = Event>(
  target: EventTarget,
  type: string,
  handler: (this: EventTarget, e: E) => unknown,
  options?: AddEventListenerOptions | boolean,
): () => void {
  let removed = false;
  const fn = function (this: EventTarget, e: Event) {
    return runWithOwner(null, () => handler.call(this, e as E));
  };
  const add = () => {
    if (!removed) target.addEventListener(type, fn, options);
  };
  if (target instanceof Element && (type.startsWith('pointer') || type.startsWith('touch') || type === 'wheel')) queueMicrotask(add);
  else add();
  const off = () => {
    removed = true;
    target.removeEventListener(type, fn, options);
  };
  onCleanup(off);
  return off;
}

/** A window listener for the current component's life (port of `<svelte:window on...>`). */
export function onWindow<K extends keyof WindowEventMap>(type: K, handler: (e: WindowEventMap[K]) => unknown, options?: AddEventListenerOptions | boolean): () => void {
  return on(window, type, handler as (e: Event) => unknown, options);
}

/** A document listener for the current component's life (port of `<svelte:document on...>`). */
export function onDocument<K extends keyof DocumentEventMap>(type: K, handler: (e: DocumentEventMap[K]) => unknown, options?: AddEventListenerOptions | boolean): () => void {
  return on(document, type, handler as (e: Event) => unknown, options);
}

/**
 * Internal: an element's `on<type>` attribute. Delegated types go through the root; `capture` (from `on<type>capture`)
 * and other types use a real listener that reads the current handler at event time.
 */
export function setEvent(el: Element, type: string, handler: unknown, capture?: boolean): void {
  if (!capture && DELEGATED.has(type)) {
    setDelegated(el, type, handler);
    return;
  }
  if (handler == null) return;
  on(el, type, handler as (e: Event) => unknown, capture ? { capture: true } : PASSIVE.has(type) ? { passive: true } : undefined);
}
