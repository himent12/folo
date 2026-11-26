// @vitest-environment happy-dom
// Sprint 10 regression tests for fw runtime: text-run shared-dependency splits, Dynamic ref forwarding,
// bindValue stringified equality, setStyle initial undefined, and multi-node insert marker ordering.
import { describe, expect, it } from 'vitest';
import { createContext, flush, onCleanup, signal } from './core';
import { bindValue } from './bind';
import { defaults, mergeProps, splitProps } from './component';
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

  it('memoizes and disposes JSX props checked in conditionals without leaking unowned actions or double-mounting', () => {
    const CTX = createContext('missing');
    let mounts = 0;
    let destroys = 0;
    let ctxSeen = '';
    const act = () => {
      mounts++;
      return {
        destroy: () => {
          destroys++;
        },
      };
    };
    function Leaf() {
      ctxSeen = CTX.get();
      return <em use:act>ok</em>;
    }
    function RowBox(props: { control?: JSX.Element; extra?: string }) {
      CTX.set('from-row');
      return <section data-extra={props.extra}>{props.control ? <div class="ctl">{props.control}</div> : null}</section>;
    }

    const extra = signal('a');
    const view = render(() => <RowBox {...{ extra: extra.get() }} control={<Leaf />} />);
    flush();
    expect(mounts).toBe(1);
    expect(destroys).toBe(0);
    expect(ctxSeen).toBe('from-row');
    expect(view.container.querySelector('.ctl em')?.textContent).toBe('ok');

    view.unmount();
    expect(destroys).toBe(1);

    // Toggled branch: disposes when hidden and recreates cleanly when shown again.
    mounts = 0;
    destroys = 0;
    const open = signal(false);
    function GateBox(props: { body?: JSX.Element }) {
      return <div>{open.get() ? <div class="body">{props.body}</div> : null}</div>;
    }
    function Tracked() {
      mounts++;
      onCleanup(() => destroys++);
      return <span>hi</span>;
    }
    const view2 = render(() => <GateBox body={<Tracked />} />);
    flush();
    expect(mounts).toBe(0);
    open.set(true);
    flush();
    expect(mounts).toBe(1);
    expect(destroys).toBe(0);
    open.set(false);
    flush();
    expect(destroys).toBe(1);
    open.set(true);
    flush();
    expect(mounts).toBe(2);
    view2.unmount();
    expect(destroys).toBe(2);
  });

  it('keeps distinct DOM nodes for multi-insertion JSX props and preserves callback values in conditional JSX props', () => {
    function Dual(props: { slot: JSX.Element }) {
      return (
        <>
          <div class="a">{props.slot}</div>
          <div class="b">{props.slot}</div>
        </>
      );
    }
    const v1 = render(() => <Dual slot={<em>slot</em>} />);
    flush();
    expect(v1.container.querySelector('.a em')?.textContent).toBe('slot');
    expect(v1.container.querySelector('.b em')?.textContent).toBe('slot');
    v1.unmount();

    // Presence check + dual insertion: both containers render distinct nodes and share the cond evaluation for the first.
    function DualCond(props: { slot?: JSX.Element }) {
      return (
        <>
          {props.slot ? (
            <>
              <div class="a">{props.slot}</div>
              <div class="b">{props.slot}</div>
            </>
          ) : null}
        </>
      );
    }
    const v2 = render(() => <DualCond slot={<em>both</em>} />);
    flush();
    expect(v2.container.querySelector('.a em')?.textContent).toBe('both');
    expect(v2.container.querySelector('.b em')?.textContent).toBe('both');
    v2.unmount();

    // Conditional JSX prop returning a parameterized callback must not invoke it eagerly.
    const show = signal(false);
    const seen: boolean[] = [];
    const cb = (arg: { ok: boolean }) => {
      seen.push(arg.ok);
      return <span>{arg.ok ? 'yes' : 'no'}</span>;
    };
    function Consumer(props: { slot: JSX.Element | ((arg: { ok: boolean }) => JSX.Element) }) {
      return <div>{typeof props.slot === 'function' ? props.slot({ ok: true }) : props.slot}</div>;
    }
    const v3 = render(() => <Consumer slot={show.get() ? <em>elem</em> : cb} />);
    flush();
    expect(seen).toEqual([true]);
    expect(v3.container.textContent).toBe('yes');
    show.set(true);
    flush();
    expect(v3.container.querySelector('em')?.textContent).toBe('elem');
    v3.unmount();
  });

  it('updates reactive propJsx once without double-mounting and propagates owner/context through Dynamic, splitProps, and defaults', () => {
    const mounts: string[] = [];
    const destroys: string[] = [];
    function Tag(props: { name: string }) {
      mounts.push(props.name);
      onCleanup(() => destroys.push(props.name));
      return <b>{props.name}</b>;
    }

    // Direct insertion {props.body} + toggled insertion {open.get() ? <div>{props.body}</div> : null} + bare branch {open.get() ? props.body : null}.
    const flag = signal(false);
    const open = signal(true);
    function Card(props: { body: JSX.Element }) {
      return <div class="card">{props.body}</div>;
    }
    const v1 = render(() => <Card body={flag.get() ? <Tag name="B" /> : <Tag name="A" />} />);
    flush();
    expect(mounts).toEqual(['A']);
    expect(destroys).toEqual([]);
    flag.set(true);
    flush();
    expect(mounts).toEqual(['A', 'B']);
    expect(destroys).toEqual(['A']);
    expect(v1.container.textContent).toBe('B');
    v1.unmount();
    expect(destroys).toEqual(['A', 'B']);

    mounts.length = 0;
    destroys.length = 0;
    flag.set(false);
    function BareGate(props: { body: JSX.Element }) {
      return <div>{open.get() ? props.body : null}</div>;
    }
    const v2 = render(() => <BareGate body={flag.get() ? <Tag name="B" /> : <Tag name="A" />} />);
    flush();
    expect(mounts).toEqual(['A']);
    flag.set(true);
    flush();
    expect(mounts).toEqual(['A', 'B']);
    expect(destroys).toEqual(['A']);
    open.set(false);
    flush();
    expect(destroys).toEqual(['A', 'B']);
    v2.unmount();

    // Forwarding through Dynamic + splitProps + defaults preserves child context, disposes on component swap, and hides internal symbols.
    const CTX = createContext('missing');
    let ctxSeen = '';
    let leafMounts = 0;
    let leafDestroys = 0;
    function Leaf() {
      ctxSeen = CTX.get();
      leafMounts++;
      onCleanup(() => leafDestroys++);
      return <i>leaf</i>;
    }
    function RowChild(props: { control?: JSX.Element; label?: string }) {
      CTX.set('from-row');
      const p = defaults(props, { label: 'def' });
      const [local, rest] = splitProps(p, ['label']);
      expect(Reflect.ownKeys(mergeProps(p)).filter((k) => typeof k === 'symbol')).toEqual([]);
      expect(Reflect.ownKeys(rest).filter((k) => typeof k === 'symbol')).toEqual([]);
      return (
        <section data-label={local.label}>
          {rest.control ? <div class="ctl">{rest.control}</div> : null}
        </section>
      );
    }
    const showDyn = signal(true);
    const v3 = render(() => <Dynamic component={showDyn.get() ? RowChild : null} control={<Leaf />} />);
    flush();
    expect(ctxSeen).toBe('from-row');
    expect(leafMounts).toBe(1);
    expect(leafDestroys).toBe(0);
    showDyn.set(false);
    flush();
    expect(leafDestroys).toBe(1);
    showDyn.set(true);
    flush();
    expect(leafMounts).toBe(2);
    expect(v3.container.querySelector('.ctl i')?.textContent).toBe('leaf');
    v3.unmount();
    expect(leafDestroys).toBe(2);
  });
});
