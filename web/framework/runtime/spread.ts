// `{...props}` on an element (port of Svelte's `set_attributes`): every key is applied as Svelte would, reactively;
// keys that disappear are removed.

/* eslint-disable @typescript-eslint/no-explicit-any */

import { renderEffect } from './core';
import { clsx, hasSetter, propName, setAttribute, setStyle, styleParts, toClass, type ClassValue } from './dom';
import { DELEGATED, on, setDelegated } from './events';

const SKIP = new Set(['children', 'ref', '$$scope']);

/**
 * Internal: applies `props` (an object, or a function returning one) to `el`; `scope` is the file's CSS scope class,
 * `dirs` the element's `class:` directives (kept on when the spread sets `class`), `styleDirs` / `styleImportant` its
 * `style:` directives (applied with the spread's `style`, as Svelte's `set_style`).
 */
export function spread(
  el: Element,
  props: object | (() => object),
  scope?: string,
  dirs?: () => Record<string, unknown>,
  styleDirs?: () => Record<string, unknown>,
  styleImportant?: () => Record<string, unknown>,
): void {
  let prevStyles: Record<string, unknown> | undefined;
  let prevImportant: Record<string, unknown> | undefined;
  let prev: Record<string, unknown> = {};
  const handlers: Record<string, unknown> = {};
  const listening = new Set<string>();
  let classSet = false;
  renderEffect(() => {
    const src = (typeof props === 'function' ? (props as () => object)() : props) as Record<string, unknown>;
    const next: Record<string, unknown> = {};
    for (const key in src) if (!SKIP.has(key)) next[key] = src[key];
    for (const key in prev) if (!(key in next)) next[key] = undefined;
    const d = dirs?.();
    if (!classSet || 'class' in next || d !== undefined) {
      // The scope class (and directives) stay on even without a `class` key.
      const value = next.class === undefined ? prev.class : next.class;
      if (!classSet || value !== prev.class || d !== undefined) {
        const c = toClass(clsx(value as ClassValue), scope, d);
        if (c === null) el.removeAttribute('class');
        else if (el.namespaceURI === 'http://www.w3.org/1999/xhtml') (el as HTMLElement).className = c;
        else el.setAttribute('class', c);
        classSet = true;
      }
    }
    for (const key in next) {
      if (key === 'class') continue;
      const value = next[key];
      if (key === 'style') continue;
      if (value === prev[key] && !(value === undefined && el.hasAttribute(key))) continue;
      if (key[0] === 'o' && key[1] === 'n') {
        let type = key.slice(2);
        const capture = type.endsWith('capture');
        if (capture) type = type.slice(0, -7);
        if (!capture && DELEGATED.has(type)) {
          setDelegated(el, type, value);
        } else {
          handlers[key] = value;
          if (value != null && !listening.has(key)) {
            listening.add(key);
            on(el, type, function (this: EventTarget, e: Event) {
              (handlers[key] as ((this: EventTarget, e: Event) => unknown) | undefined)?.call(this, e);
            }, capture ? { capture: true } : undefined);
          }
        }
        continue;
      }
      if (key === 'value' || key === 'checked') {
        if (value == null) {
          const input = el as HTMLInputElement;
          if (key === 'value') {
            const d = input.defaultValue;
            input.removeAttribute('value');
            input.defaultValue = d;
            input.value = prev[key] === undefined ? d : '';
          } else {
            const d = input.defaultChecked;
            input.removeAttribute('checked');
            input.defaultChecked = d;
            input.checked = prev[key] === undefined ? d : false;
          }
        } else if (key === 'value') {
          if (prev[key] == null || (el as HTMLInputElement).value !== value) (el as HTMLInputElement).value = value as string;
        } else (el as HTMLInputElement).checked = !!value;
        continue;
      }
      const name = el.namespaceURI === 'http://www.w3.org/1999/xhtml' ? (propName(key) ?? key) : key;
      if (value == null) el.removeAttribute(key);
      else if (typeof value !== 'string' && hasSetter(el, name)) (el as any)[name] = value;
      else if (typeof value !== 'function') setAttribute(el, name, value);
    }
    prev = next;
    const sd = styleDirs?.();
    const si = styleImportant?.();
    if ('style' in next || sd !== undefined || si !== undefined) {
      // No `style` key but directives: `null`, as Svelte forces the call.
      const [value, styles] = styleParts(next.style === undefined && !('style' in src) ? null : next.style, sd);
      setStyle(el, value, prevStyles, styles, prevImportant, si);
      prevStyles = styles;
      prevImportant = si;
    }
  });
}
