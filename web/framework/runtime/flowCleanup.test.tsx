// @vitest-environment happy-dom
// Disposal resilience tests for For and Index when a removed row's cleanup throws.
import { describe, expect, it } from 'vitest';
import { flush, onCleanup, signal } from './core';
import { For, Index } from './flow';
import { render } from '../testing';

describe('For / Index throwing row cleanup disposal', () => {
  it('For disposes all remaining doomed rows when clearing, truncating, or removing middle rows throws in cleanup', () => {
    const cleaned: string[] = [];
    let failOn = 'a';
    const items = signal(['a', 'b', 'c']);
    const view = render(() => (
      <div>
        <For each={items.get()} key={(x) => x}>
          {(item) => {
            const v = item();
            onCleanup(() => {
              cleaned.push(v);
              if (v === failOn) {
                failOn = '';
                throw new Error(`boom:${v}`);
              }
            });
            return <span>{v}</span>;
          }}
        </For>
      </div>
    ));
    flush();
    expect(view.container.textContent).toBe('abc');

    // Clear all: 'a' throws in cleanup, yet 'b' and 'c' must still be disposed and DOM updated immediately.
    items.set([]);
    expect(() => flush()).toThrow('boom:a');
    expect(cleaned).toEqual(['a', 'b', 'c']);
    expect(view.container.textContent).toBe('');

    items.set(['x', 'y', 'z']);
    flush();
    expect(view.container.textContent).toBe('xyz');

    // Truncate: 'y' throws in cleanup, yet 'z' must still be disposed and DOM truncated immediately.
    cleaned.length = 0;
    failOn = 'y';
    items.set(['x']);
    expect(() => flush()).toThrow('boom:y');
    expect(cleaned).toEqual(['y', 'z']);
    expect(view.container.textContent).toBe('x');

    items.set(['x', 'm', 'n', 'w']);
    flush();
    expect(view.container.textContent).toBe('xmnw');

    // Pure middle removal: 'm' throws in cleanup, yet 'n' must still be disposed and DOM updated immediately.
    cleaned.length = 0;
    failOn = 'm';
    items.set(['x', 'w']);
    expect(() => flush()).toThrow('boom:m');
    expect(cleaned).toEqual(['m', 'n']);
    expect(view.container.textContent).toBe('xw');

    items.set(['x', 'w', 'q']);
    flush();
    expect(view.container.textContent).toBe('xwq');
    view.unmount();
  });

  it('Index disposes all remaining doomed rows when shrinking throws in cleanup and reconciles DOM immediately', () => {
    const cleaned: number[] = [];
    let failIdx = 1;
    const items = signal(['a', 'b', 'c', 'd']);
    const view = render(() => (
      <div>
        <Index each={items.get()}>
          {(item, i) => {
            onCleanup(() => {
              cleaned.push(i);
              if (i === failIdx) {
                failIdx = -1;
                throw new Error(`idx-cleanup:${i}`);
              }
            });
            return <span>{item()}</span>;
          }}
        </Index>
      </div>
    ));
    flush();
    expect(view.container.textContent).toBe('abcd');

    items.set(['a']);
    expect(() => flush()).toThrow('idx-cleanup:1');
    expect(cleaned).toEqual([1, 2, 3]);
    expect(view.container.textContent).toBe('a');

    items.set(['a', 'e']);
    flush();
    expect(view.container.textContent).toBe('ae');
    view.unmount();
  });
});
