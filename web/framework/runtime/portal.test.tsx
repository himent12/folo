// @vitest-environment happy-dom
// Portal lifecycle cleanup: error teardown, reactive mount moves, and detached marker cleanup.
import { describe, expect, it, vi } from 'vitest';
import { flush, onCleanup, Portal, signal } from 'fw';
import { render } from '../testing';

describe('Portal lifecycle', () => {
  it('cleans up markers, partial DOM, and child effects when children throw during mount', () => {
    const m = document.body.appendChild(document.createElement('div'));
    m.append('stay');
    const cleaned = vi.fn();
    const Good = () => {
      onCleanup(cleaned);
      return <b>partial</b>;
    };
    const Bad = () => {
      throw new Error('portal boom');
    };
    expect(() =>
      render(() => (
        <Portal mount={m}>
          <Good />
          <Bad />
        </Portal>
      )),
    ).toThrow('portal boom');
    expect(cleaned).toHaveBeenCalledOnce();
    expect(m.childNodes).toHaveLength(1);
    expect(m.textContent).toBe('stay');
    m.remove();
  });

  it('moves nodes cleanly when mount changes reactively and cleans up end marker if start is detached', () => {
    const m1 = document.body.appendChild(document.createElement('div'));
    const m2 = document.body.appendChild(document.createElement('div'));
    const target = signal<Node>(m1);
    const { unmount } = render(() => (
      <Portal mount={target.get()}>
        <b>moved</b>
      </Portal>
    ));
    expect(m1.textContent).toBe('moved');
    expect(m2.textContent).toBe('');

    target.set(m2);
    flush();
    expect(m1.childNodes).toHaveLength(0);
    expect(m2.textContent).toBe('moved');

    // Externally detaching the start comment must not leak the end comment on unmount.
    m2.firstChild!.remove();
    unmount();
    expect([...m2.childNodes].filter((n) => n.nodeType === 8)).toHaveLength(0);
    m1.remove();
    m2.remove();
  });
});
