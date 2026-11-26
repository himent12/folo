// @vitest-environment happy-dom
// Tabs keyboard navigation and roving tabindex edge cases.
import { describe, expect, it } from 'vitest';
import { flush, signal } from 'fw';
import { render } from '../../../framework/testing';
import Tabs from './Tabs';

describe('Tabs', () => {
  it('keeps the first tab tabbable and advances from the focused tab button when value matches no tab', () => {
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

    // Tab lands on the first tab ("a"); pressing ArrowRight on that button moves to "b".
    tabs[0]!.focus();
    tabs[0]!.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true, cancelable: true }));
    flush();
    expect(val.get()).toBe('b');
    expect(document.activeElement).toBe(tabs[1]);
    expect(tabs.map((t) => t.tabIndex)).toEqual([-1, 0, -1]);

    // When value is missing and focus is on the first tab, ArrowLeft wraps to the last tab ("c").
    val.set('missing');
    flush();
    tabs[0]!.focus();
    tabs[0]!.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowLeft', bubbles: true, cancelable: true }));
    flush();
    expect(val.get()).toBe('c');
    expect(document.activeElement).toBe(tabs[2]);
    expect(tabs.map((t) => t.tabIndex)).toEqual([-1, -1, 0]);

    // Dispatched on container when no tab button is focused: ArrowRight lands on first tab ("a").
    val.set('missing');
    flush();
    (document.activeElement as HTMLElement | null)?.blur();
    container.querySelector('[role="tablist"]')!.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true, cancelable: true }));
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
