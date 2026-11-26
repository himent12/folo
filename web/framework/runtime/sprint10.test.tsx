// @vitest-environment happy-dom
// Sprint 10 regression tests for fw runtime: text-run shared-dependency splits, Dynamic ref forwarding,
// bindValue stringified equality, setStyle initial undefined, and multi-node insert marker ordering.
import { describe, expect, it } from 'vitest';
import { flush, signal } from './core';
import { bindValue } from './bind';
import { insert, setStyle } from './dom';
import { Dynamic } from './flow';
import { run } from './text';
import { render } from '../testing';

describe('fw Sprint 10 runtime hardening', () => {
  it('preserves shared signal dependencies when a text run splits on a later expression', () => {
    const active = signal(true);
    const badge = () => {
      const strong = document.createElement('strong');
      strong.textContent = 'live';
      return strong;
    };
    const { container, unmount } = render(() => {
      const p = document.createElement('p');
      const text = document.createTextNode('');
      p.appendChild(text);
      run(text, [() => (active.get() ? 'on' : 'off'), ': ', () => (active.get() ? badge() : 'idle')]);
      return p;
    });
    flush();
    expect(container.textContent).toBe('on: live');
    expect(container.querySelector('strong')?.textContent).toBe('live');

    active.set(false);
    flush();
    expect(container.textContent).toBe('off: idle');
    expect(container.querySelector('strong')).toBeNull();

    active.set(true);
    flush();
    expect(container.textContent).toBe('on: live');
    expect(container.querySelector('strong')?.textContent).toBe('live');
    unmount();
  });

  it('forwards direct ref (signal with cleanup and callback) on string Dynamic elements across tag swaps and unmount', () => {
    const tag = signal<'button' | 'a'>('button');
    const elRef = signal<Element | undefined>(undefined);
    const cbSeen: string[] = [];

    const view1 = render(() => (
      <Dynamic component={tag.get()} ref={elRef}>
        go
      </Dynamic>
    ));
    flush();
    expect(elRef.peek()?.tagName).toBe('BUTTON');

    tag.set('a');
    flush();
    expect(elRef.peek()?.tagName).toBe('A');

    view1.unmount();
    expect(elRef.peek()).toBeUndefined();

    const view2 = render(() => (
      <Dynamic component="span" ref={(el: Element) => cbSeen.push(el.tagName)}>
        hi
      </Dynamic>
    ));
    flush();
    expect(cbSeen).toEqual(['SPAN']);
    view2.unmount();
  });

  it('does not rewrite text input.value when bindValue receives an equivalent non-string or nullish value', () => {
    const val = signal<unknown>(42);
    const input = document.createElement('input');
    input.type = 'text';
    input.value = '42';
    const desc = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')!;
    let writes = 0;
    Object.defineProperty(input, 'value', {
      configurable: true,
      get() {
        return desc.get!.call(this);
      },
      set(v: string) {
        writes++;
        desc.set!.call(this, v);
      },
    });
    const view = render(() => {
      bindValue(input, val);
      return input;
    });
    flush();
    expect(writes).toBe(0);

    val.set(null);
    flush();
    expect(input.value).toBe('');
    expect(writes).toBe(1);

    // Switching null -> undefined while already '' must not rewrite input.value.
    val.set(undefined);
    flush();
    expect(writes).toBe(1);
    view.unmount();
  });

  it('synchronizes setStyle when initial value is undefined over an element with existing inline style', () => {
    const el = document.createElement('div');
    el.setAttribute('style', 'color: red');
    setStyle(el, undefined, undefined, undefined);
    expect(el.hasAttribute('style')).toBe(false);
  });

  it('respects marker in insert() when replacing a non-array current value with a single node', () => {
    const parent = document.createElement('div');
    const lead = document.createElement('i');
    const marker = document.createComment('end');
    parent.appendChild(lead);
    parent.appendChild(marker);

    // Non-undefined scalar initial (null) bypasses insert()'s `initial === undefined -> []` normalization.
    const node = document.createElement('span');
    node.textContent = 'x';
    insert(parent, node, marker, null);
    expect([...parent.childNodes]).toEqual([lead, node, marker]);

    // Replacing a previous single node in a marked region must also keep `lead` intact and stay before `marker`.
    const node2 = document.createElement('b');
    node2.textContent = 'y';
    insert(parent, node2, marker, node);
    expect([...parent.childNodes]).toEqual([lead, node2, marker]);
  });
});
