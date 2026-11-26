// @vitest-environment happy-dom
// Tabs keyboard navigation and roving tabindex edge cases.
import { describe, expect, it } from 'vitest';
import { flush, signal } from 'fw';
import { render } from '../../../framework/testing';
import Tabs from './Tabs';

describe('Tabs', () => {
  it('keeps the first tab tabbable and handles ArrowLeft/ArrowRight when value matches no tab', () => {
    const val = signal('missing');
    const { container, unmount } = render(() => (
      <Tabs
        label="Views"
        bind:value={val}
        tabs={[
          { value: 'a', label: 'A' },
          { value: 'b', label: 'B' },
          { value: 'c', label: 'C' },
        ]}
      />
    ));
    flush();
    const tabs = [...container.querySelectorAll<HTMLButtonElement>('[role="tab"]')];
    expect(tabs.map((t) => t.tabIndex)).toEqual([0, -1, -1]);
    const list = container.querySelector('[role="tablist"]')!;
    list.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowLeft', bubbles: true, cancelable: true }));
    flush();
    expect(val.get()).toBe('c');
    expect(tabs.map((t) => t.tabIndex)).toEqual([-1, -1, 0]);

    val.set('missing');
    flush();
    list.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true, cancelable: true }));
    flush();
    expect(val.get()).toBe('a');
    unmount();
  });

  it('ignores keyboard navigation safely when tabs is empty', () => {
    const val = signal('');
    const { container, unmount } = render(() => <Tabs label="Empty" bind:value={val} tabs={[]} />);
    flush();
    const ev = new KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true, cancelable: true });
    container.querySelector('[role="tablist"]')!.dispatchEvent(ev);
    expect(ev.defaultPrevented).toBe(false);
    expect(val.get()).toBe('');
    unmount();
  });
});
