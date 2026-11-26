// @vitest-environment happy-dom
// Portal lifecycle cleanup: error teardown, reactive mount moves, and detached marker cleanup.
import { describe, expect, it, vi } from 'vitest';
import { flush, onCleanup, Portal, Show, signal } from 'fw';
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

  it('moves nodes cleanly when mount changes reactively', () => {
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

    unmount();
    expect(m2.childNodes).toHaveLength(0);
    m1.remove();
    m2.remove();
  });

  it('removes owned portal content (including outgoing nodes) and preserves unrelated siblings when start or end marker is detached', () => {
    const slow = () => ({ duration: 50, css: (t: number) => `opacity: ${t}` });

    // Case 1: start marker detached externally — portal nodes and outgoing nodes still tear down, siblings survive.
    const m1 = document.body.appendChild(document.createElement('div'));
    m1.append('before ');
    const shown1 = signal(true);
    const r1 = render(() => (
      <Portal mount={m1}>
        <Show when={shown1.get()}>
          <b out:slow={{}}>leaving</b>
        </Show>
        <i>live</i>
      </Portal>
    ));
    m1.append(' after');
    shown1.set(false);
    flush();
    expect(m1.querySelector('b')).not.toBeNull();
    // Detach start comment (node after 'before ').
    m1.childNodes[1]!.remove();
    r1.unmount();
    expect(m1.querySelector('b')).toBeNull();
    expect(m1.querySelector('i')).toBeNull();
    expect([...m1.childNodes].filter((n) => n.nodeType === 8)).toHaveLength(0);
    expect(m1.textContent).toBe('before  after');
    m1.remove();

    // Case 2: end marker detached externally — portal nodes tear down without deleting unrelated following siblings.
    const m2 = document.body.appendChild(document.createElement('div'));
    m2.append('before ');
    const r2 = render(() => (
      <Portal mount={m2}>
        <b>owned</b>
      </Portal>
    ));
    const sibling = m2.appendChild(document.createElement('span'));
    sibling.textContent = 'keep-me';
    // Detach end comment (the comment node right before `sibling`).
    sibling.previousSibling!.remove();
    r2.unmount();
    expect(m2.querySelector('b')).toBeNull();
    expect(m2.querySelector('span')).toBe(sibling);
    expect([...m2.childNodes].filter((n) => n.nodeType === 8)).toHaveLength(0);
    expect(m2.textContent).toBe('before keep-me');
    m2.remove();
  });
});
