// @vitest-environment happy-dom
import { describe, expect, it, vi } from 'vitest';
import { render } from '../testing';

describe('delegated events', () => {
  it('schedules neither setTimeout nor queueMicrotask and handles repeated dispatches cleanly', async () => {
    const timeoutSpy = vi.spyOn(globalThis, 'setTimeout');
    const microtaskSpy = vi.spyOn(globalThis, 'queueMicrotask');
    let calls = 0;
    const { container, unmount } = render(() => (
      <button
        onclick={() => {
          calls++;
        }}
      >
        go
      </button>
    ));
    const btn = container.querySelector('button')!;

    // Re-dispatching the same Event object synchronously and across microtasks schedules no timers or microtasks.
    const ev = new MouseEvent('click', { bubbles: true });
    btn.dispatchEvent(ev);
    btn.dispatchEvent(ev);
    await Promise.resolve();
    btn.dispatchEvent(ev);
    expect(calls).toBe(3);
    expect(timeoutSpy).not.toHaveBeenCalled();
    expect(microtaskSpy).not.toHaveBeenCalled();

    // Sealing an event after dispatch does not break cleanup or poison subsequent dispatches;
    // a pre-sealed event with no delegated handler on its path does not throw because currentTarget is lazy.
    const sealedAfter = new MouseEvent('click', { bubbles: true });
    btn.dispatchEvent(sealedAfter);
    Object.seal(sealedAfter);
    btn.dispatchEvent(ev);
    expect(calls).toBe(5);
    expect(timeoutSpy).not.toHaveBeenCalled();
    expect(microtaskSpy).not.toHaveBeenCalled();

    timeoutSpy.mockRestore();
    microtaskSpy.mockRestore();
    unmount();
  });

  it('handles synchronous re-dispatch on a detached render root and across sibling roots', () => {
    const detached = document.createElement('div');
    let detachedCalls = 0;
    const rDetached = render(
      () => (
        <button
          onclick={() => {
            detachedCalls++;
          }}
        >
          detached
        </button>
      ),
      detached,
    );
    const detachedBtn = detached.querySelector('button')!;
    const ev = new MouseEvent('click', { bubbles: true });
    detachedBtn.dispatchEvent(ev);
    detachedBtn.dispatchEvent(ev);
    expect(detachedCalls).toBe(2);
    rDetached.unmount();

    // Sibling roots: if root1 recorded handledAt and propagation stopped before document,
    // re-dispatching the same Event on root2 (atIdx === -1) or root1 (handledAt === handlerElement)
    // must invoke target handlers once (neither skip nor double-fire).
    const root1El = document.body.appendChild(document.createElement('div'));
    const root2El = document.body.appendChild(document.createElement('div'));
    let root1Listener: EventListener | undefined;
    let root2Listener: EventListener | undefined;
    const spy1 = vi.spyOn(root1El, 'addEventListener').mockImplementation(function (this: EventTarget, type, listener, opts) {
      if (type === 'click' && typeof listener === 'function') root1Listener = listener;
      return HTMLElement.prototype.addEventListener.call(this, type, listener, opts);
    });
    const spy2 = vi.spyOn(root2El, 'addEventListener').mockImplementation(function (this: EventTarget, type, listener, opts) {
      if (type === 'click' && typeof listener === 'function') root2Listener = listener;
      return HTMLElement.prototype.addEventListener.call(this, type, listener, opts);
    });
    let c1 = 0;
    let c2 = 0;
    const r1 = render(() => <button onclick={() => c1++}>one</button>, root1El);
    const r2 = render(() => <button onclick={() => c2++}>two</button>, root2El);
    spy1.mockRestore();
    spy2.mockRestore();
    const b1 = root1El.querySelector('button')!;
    const b2 = root2El.querySelector('button')!;

    // Simulate dispatch stopped between root1 and document (so handled still maps ev -> root1El).
    const path1 = [b1, root1El, document.body, document.documentElement, document, window];
    const path2 = [b2, root2El, document.body, document.documentElement, document, window];
    let activePath: EventTarget[] = path1;
    const reused = {
      type: 'click',
      target: b1,
      bubbles: true,
      cancelBubble: false,
      composedPath: () => activePath,
    } as unknown as Event;

    root1Listener!.call(root1El, reused);
    expect(c1).toBe(1);
    // Re-dispatch on root1 itself (handledAt === handlerElement): must not skip.
    root1Listener!.call(root1El, reused);
    expect(c1).toBe(2);
    // Re-dispatch on sibling root2 (atIdx === -1): must invoke b2 once, not twice.
    activePath = path2;
    (reused as unknown as { target: EventTarget }).target = b2;
    root2Listener!.call(root2El, reused);
    expect(c2).toBe(1);

    r1.unmount();
    r2.unmount();
    root1El.remove();
    root2El.remove();
  });

  it('defines currentTarget lazily only when a delegated handler matches, and once across ancestors', () => {
    const defSpy = vi.spyOn(Object, 'defineProperty');
    const seen: string[] = [];
    const { container, unmount } = render(() => (
      <section>
        <p>
          <em id="plain">no handler</em>
        </p>
        <div onclick={(e) => seen.push(`div:${(e.currentTarget as Element).tagName}`)}>
          <button onclick={(e) => seen.push(`button:${(e.currentTarget as Element).tagName}`)}>
            <span id="handled">hit</span>
          </button>
        </div>
      </section>
    ));

    // Clicking an element with no delegated handlers on its path never defines `currentTarget`.
    defSpy.mockClear();
    const plainEvent = new MouseEvent('click', { bubbles: true });
    container.querySelector('#plain')!.dispatchEvent(plainEvent);
    expect(defSpy.mock.calls.filter(([obj, prop]) => obj === plainEvent && prop === 'currentTarget')).toHaveLength(0);

    // Clicking through two delegated handlers defines `currentTarget` once and updates it for each ancestor.
    defSpy.mockClear();
    const handledEvent = new MouseEvent('click', { bubbles: true });
    container.querySelector('#handled')!.dispatchEvent(handledEvent);
    expect(seen).toEqual(['button:BUTTON', 'div:DIV']);
    expect(defSpy.mock.calls.filter(([obj, prop]) => obj === handledEvent && prop === 'currentTarget')).toHaveLength(1);
    expect(Object.getOwnPropertyDescriptor(handledEvent, 'currentTarget')).toBeUndefined();

    defSpy.mockRestore();
    unmount();
  });

  it('cleans up currentTarget on throwing handlers and aborts cleanly on non-configurable currentTarget', () => {
    const seen: string[] = [];
    const { container, unmount } = render(() => (
      <div onclick={(e) => seen.push(`div:${(e.currentTarget as Element).tagName}`)}>
        <button
          id="throws"
          onclick={(e) => {
            seen.push(`btn:${(e.currentTarget as Element).tagName}`);
            throw new Error('boom');
          }}
        >
          err
        </button>
      </div>
    ));

    const errEv = new MouseEvent('click', { bubbles: true });
    expect(() => container.querySelector('#throws')!.dispatchEvent(errEv)).toThrow('boom');
    expect(seen).toEqual(['btn:BUTTON', 'div:DIV']);
    expect(Object.getOwnPropertyDescriptor(errEv, 'currentTarget')).toBeUndefined();

    // If currentTarget is non-configurable, defineProperty fails immediately without running ancestor handlers
    // or throwing a secondary delete error in finally.
    seen.length = 0;
    const locked = new MouseEvent('click', { bubbles: true });
    Object.defineProperty(locked, 'currentTarget', { value: null, configurable: false });
    expect(() => container.querySelector('#throws')!.dispatchEvent(locked)).toThrow(TypeError);
    expect(seen).toEqual([]);

    unmount();
  });

  it('handles re-entrant delegated events without double-firing or clobbering currentTarget', () => {
    const log: string[] = [];
    let otherBtn!: HTMLButtonElement;
    const { container, unmount } = render(() => (
      <div onclick={(e) => log.push(`div:${(e.currentTarget as Element).tagName}`)}>
        <button
          id="first"
          onclick={(e) => {
            log.push(`first-before:${(e.currentTarget as Element).id}`);
            otherBtn.click();
            log.push(`first-after:${(e.currentTarget as Element).id}`);
          }}
        >
          first
        </button>
        <button
          id="second"
          ref={(el) => (otherBtn = el as HTMLButtonElement)}
          onclick={(e) => {
            log.push(`second:${(e.currentTarget as Element).id}`);
            e.stopPropagation();
          }}
        >
          second
        </button>
      </div>
    ));

    container.querySelector<HTMLButtonElement>('#first')!.click();
    expect(log).toEqual(['first-before:first', 'second:second', 'first-after:first', 'div:DIV']);
    unmount();
  });

  it('propagates across nested render roots once per handler and respects stopPropagation', () => {
    const log: string[] = [];
    let innerMount!: HTMLDivElement;
    const outer = render(() => (
      <div id="outer" onclick={(e) => log.push(`outer:${(e.currentTarget as Element).id}`)}>
        <div ref={(el) => (innerMount = el as HTMLDivElement)} />
      </div>
    ));
    const inner = render(
      () => (
        <div id="inner">
          <button id="bubble" onclick={(e) => log.push(`inner:${(e.currentTarget as Element).id}`)}>
            bubble
          </button>
          <button
            id="stop"
            onclick={(e) => {
              log.push(`stop:${(e.currentTarget as Element).id}`);
              e.stopPropagation();
            }}
          >
            stop
          </button>
        </div>
      ),
      innerMount,
    );

    innerMount.querySelector<HTMLButtonElement>('#bubble')!.click();
    innerMount.querySelector<HTMLButtonElement>('#stop')!.click();
    expect(log).toEqual(['inner:bubble', 'outer:outer', 'stop:stop']);

    inner.unmount();
    outer.unmount();
  });

  it('skips delegated handlers on disabled ancestors without defining currentTarget', () => {
    const defSpy = vi.spyOn(Object, 'defineProperty');
    const clicks = vi.fn();
    const { container, unmount } = render(() => (
      <button disabled onclick={clicks}>
        <span id="child">disabled</span>
      </button>
    ));

    defSpy.mockClear();
    const ev = new MouseEvent('click', { bubbles: true });
    container.querySelector('#child')!.dispatchEvent(ev);
    expect(clicks).not.toHaveBeenCalled();
    expect(defSpy.mock.calls.filter(([obj, prop]) => obj === ev && prop === 'currentTarget')).toHaveLength(0);

    defSpy.mockRestore();
    unmount();
  });

  it('does not double-fire when a browser microtask checkpoint drains between root and document listeners', async () => {
    let rootListener: EventListener | undefined;
    let docListener: EventListener | undefined;
    const rootEl = document.body.appendChild(document.createElement('div'));
    const rootSpy = vi.spyOn(rootEl, 'addEventListener').mockImplementation(function (this: EventTarget, type, listener, opts) {
      if (type === 'click' && typeof listener === 'function') rootListener = listener;
      return HTMLElement.prototype.addEventListener.call(this, type, listener, opts);
    });
    const docSpy = vi.spyOn(document, 'addEventListener').mockImplementation(function (this: EventTarget, type, listener, opts) {
      if (type === 'click' && typeof listener === 'function') docListener = listener;
      return Document.prototype.addEventListener.call(this, type, listener, opts);
    });
    let calls = 0;
    let unmount!: () => void;
    try {
      ({ unmount } = render(
        () => (
          <button
            onclick={() => {
              calls++;
            }}
          >
            trusted
          </button>
        ),
        rootEl,
      ));
    } finally {
      rootSpy.mockRestore();
      docSpy.mockRestore();
    }

    expect(rootListener).toBeTypeOf('function');
    expect(docListener).toBeTypeOf('function');

    const btn = rootEl.querySelector('button')!;
    const path = [btn, rootEl, document.body, document.documentElement, document, window];
    const syntheticTrusted = {
      type: 'click',
      target: btn,
      bubbles: true,
      cancelBubble: false,
      composedPath: () => path,
    } as unknown as Event;

    // 1. Root listener runs during bubbling.
    rootListener!.call(rootEl, syntheticTrusted);
    expect(calls).toBe(1);
    // 2. Trusted browser click drains microtasks before invoking the document listener.
    await Promise.resolve();
    // 3. Document listener runs for the same bubbling event: must not double-fire.
    docListener!.call(document, syntheticTrusted);
    expect(calls).toBe(1);

    unmount();
    rootEl.remove();
  });
});
