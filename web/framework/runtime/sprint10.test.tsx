// @vitest-environment happy-dom
// Sprint 10 regression tests for fw runtime: text-run shared-dependency splits, Dynamic ref forwarding,
// bindValue stringified equality, setStyle initial undefined, and multi-node insert marker ordering.
import { describe, expect, it } from 'vitest';
import { computed, createContext, flush, onCleanup, signal } from './core';
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

  it('disposes hidden aliased/nested-cond JSX props, keeps distinct nodes in [props.body, props.body], and isolates nested/multi-hop forwarding', () => {
    // 1. Aliased JSX.Element type in `{open.get() ? props.value : null}` disposes immediately when hidden and remounts on show.
    type Slot = JSX.Element;
    let mounts = 0;
    let destroys = 0;
    function Tracked() {
      mounts++;
      onCleanup(() => destroys++);
      return <em>x</em>;
    }
    const open = signal(true);
    function AliasedGate(props: { value: Slot }) {
      return <div>{open.get() ? props.value : null}</div>;
    }
    const v1 = render(() => <AliasedGate value={<Tracked />} />);
    flush();
    expect([mounts, destroys]).toEqual([1, 0]);
    open.set(false);
    flush();
    expect([mounts, destroys]).toEqual([1, 1]);
    open.set(true);
    flush();
    expect([mounts, destroys]).toEqual([2, 1]);
    v1.unmount();
    expect(destroys).toBe(2);

    // 2. Nested cond() inside a conditional branch disposes when outer branch hides.
    mounts = 0;
    destroys = 0;
    const outerOpen = signal(true);
    function NestedCond(props: { body?: JSX.Element }) {
      return <div>{outerOpen.get() ? <section>{props.body ? <b>{props.body}</b> : null}</section> : null}</div>;
    }
    const v2 = render(() => <NestedCond body={<Tracked />} />);
    flush();
    expect([mounts, destroys]).toEqual([1, 0]);
    outerOpen.set(false);
    flush();
    expect([mounts, destroys]).toEqual([1, 1]);
    outerOpen.set(true);
    flush();
    expect([mounts, destroys]).toEqual([2, 1]);
    v2.unmount();
    expect(destroys).toBe(2);

    // 3. Two reads in a single insertion expression `[props.body, props.body]` render two distinct DOM nodes.
    mounts = 0;
    destroys = 0;
    function ArrayDual(props: { body: JSX.Element }) {
      return <div>{[props.body, props.body]}</div>;
    }
    const v3 = render(() => <ArrayDual body={<Tracked />} />);
    flush();
    expect(v3.container.querySelectorAll('em')).toHaveLength(2);
    expect([mounts, destroys]).toEqual([2, 0]);
    v3.unmount();
    expect(destroys).toBe(2);

    // 4. Multi-hop forwarding through Dynamic + dynamic spread + splitProps + local mergeProps preserves innermost context & disposal.
    const CTX = createContext('missing');
    let seenCtx = '';
    mounts = 0;
    destroys = 0;
    function CtxLeaf() {
      seenCtx = CTX.get();
      mounts++;
      onCleanup(() => destroys++);
      return <i>ok</i>;
    }
    function InnerConsumer(props: { control?: JSX.Element }) {
      CTX.set('inner-ctx');
      const m = mergeProps({ extra: 'def' }, props);
      return <section>{m.control ? <div class="c">{m.control}</div> : null}</section>;
    }
    function MidForward(props: { control?: JSX.Element; tag?: string }) {
      CTX.set('mid-ctx');
      const [, rest] = splitProps(props, ['tag']);
      return <InnerConsumer {...rest} />;
    }
    const showDyn = signal(true);
    const dynExtra = signal('e1');
    const v4 = render(() => <Dynamic {...{ extra: dynExtra.get() }} component={showDyn.get() ? MidForward : null} control={<CtxLeaf />} />);
    flush();
    expect(seenCtx).toBe('inner-ctx');
    expect([mounts, destroys]).toEqual([1, 0]);
    showDyn.set(false);
    flush();
    expect(destroys).toBe(1);
    v4.unmount();

    // 5. Nested JSX props `<Outer control={<Inner badge={<Deep />} />} />` do not leak Outer's owner/context into Deep,
    // and `{props.control ? <PassChild {...props} /> : null}` shares the cond memo without double-mounting.
    const NEST = createContext('none');
    let deepCtx = '';
    mounts = 0;
    destroys = 0;
    function Deep() {
      deepCtx = NEST.get();
      mounts++;
      onCleanup(() => destroys++);
      return <u>deep</u>;
    }
    function InnerBadge(props: { badge?: JSX.Element }) {
      NEST.set('from-inner');
      return <span>{props.badge ? <b>{props.badge}</b> : null}</span>;
    }
    function PassChild(props: { control?: JSX.Element }) {
      return <div class="pass">{props.control}</div>;
    }
    function OuterBox(props: { control?: JSX.Element }) {
      NEST.set('from-outer');
      return <article>{props.control ? <PassChild {...props} /> : null}</article>;
    }
    const v5 = render(() => <OuterBox {...{ extra: dynExtra.get() }} control={<InnerBadge badge={<Deep />} />} />);
    flush();
    expect(deepCtx).toBe('from-inner');
    expect([mounts, destroys]).toEqual([1, 0]);
    v5.unmount();
    expect(destroys).toBe(1);
  });

  it('preserves sibling cleanups on unmount, recovers computeds after throwing propJsx cleanup, and handles chained/nested ternary conds', () => {
    // 1. Sibling cleanups must not be skipped when unsubscribing an earlier JSX prop triggers _onEmpty teardown.
    let siblingCleanups = 0;
    function Leaf() {
      onCleanup(() => siblingCleanups++);
      return <span>leaf</span>;
    }
    function SiblingHost(props: { slot: JSX.Element }) {
      return (
        <>
          <div>{props.slot}</div>
          <div>{props.slot}</div>
          <Leaf />
        </>
      );
    }
    const v1 = render(() => <SiblingHost slot={<Leaf />} />);
    flush();
    expect(v1.container.querySelectorAll('span')).toHaveLength(3);
    v1.unmount();
    expect(siblingCleanups).toBe(3);

    // 2. A throwing cleanup on a hidden JSX prop must not leave a reading computed stuck in RUNNING.
    const open = signal(true);
    let throwOnce = true;
    function FlakyLeaf() {
      onCleanup(() => {
        if (throwOnce) {
          throwOnce = false;
          throw new Error('cleanup boom');
        }
      });
      return <em>flaky</em>;
    }
    function CompHost(props: { slot: JSX.Element }) {
      const val = computed(() => (open.get() ? props.slot : null));
      return <div>{val.get()}</div>;
    }
    const v2 = render(() => <CompHost slot={<FlakyLeaf />} />);
    flush();
    expect(v2.container.querySelector('em')?.textContent).toBe('flaky');
    open.set(false);
    expect(() => flush()).toThrow('cleanup boom');
    open.set(true);
    expect(() => flush()).not.toThrow();
    expect(v2.container.querySelector('em')?.textContent).toBe('flaky');
    v2.unmount();

    // 3. Chained else-if and directly nested ternaries anchor their conds to the enclosing owner, never a disposed branch root.
    let mounts = 0;
    let destroys = 0;
    function Tracked() {
      mounts++;
      onCleanup(() => destroys++);
      return <i>ok</i>;
    }
    const a = signal(false);
    const showBody = signal(true);
    function ElseIfHost(props: { body?: JSX.Element }) {
      return <div>{a.get() ? <span>A</span> : props.body ? <b>{props.body}</b> : null}</div>;
    }
    const v3 = render(() => <ElseIfHost body={showBody.get() ? <Tracked /> : null} />);
    flush();
    expect([mounts, destroys]).toEqual([1, 0]);
    a.set(true);
    flush();
    expect([mounts, destroys]).toEqual([1, 1]);
    a.set(false);
    flush();
    expect([mounts, destroys]).toEqual([2, 1]);
    showBody.set(false);
    flush();
    expect([mounts, destroys]).toEqual([2, 2]);
    showBody.set(true);
    flush();
    expect([mounts, destroys]).toEqual([3, 2]);
    expect(v3.container.querySelector('b i')?.textContent).toBe('ok');
    v3.unmount();
    expect(destroys).toBe(3);

    mounts = 0;
    destroys = 0;
    const outer = signal(true);
    function DirectNested(props: { body?: JSX.Element }) {
      return <div>{outer.get() ? (props.body ? <b>{props.body}</b> : null) : null}</div>;
    }
    const v4 = render(() => <DirectNested body={<Tracked />} />);
    flush();
    expect([mounts, destroys]).toEqual([1, 0]);
    outer.set(false);
    flush();
    expect([mounts, destroys]).toEqual([1, 1]);
    outer.set(true);
    flush();
    expect([mounts, destroys]).toEqual([2, 1]);
    v4.unmount();
    expect(destroys).toBe(2);

    // 4. Downstream computed recovers cleanly after upstream propJsx cleanup throws inside needsToRecompute.
    const chainOpen = signal(true);
    let chainThrow = true;
    function ChainFlaky() {
      onCleanup(() => {
        if (chainThrow) {
          chainThrow = false;
          throw new Error('chain cleanup boom');
        }
      });
      return <em>chain</em>;
    }
    function ChainHost(props: { slot: JSX.Element }) {
      const inner = computed(() => (chainOpen.get() ? props.slot : null));
      const outerC = computed(() => inner.get());
      return <div>{outerC.get()}</div>;
    }
    const v5 = render(() => <ChainHost slot={<ChainFlaky />} />);
    flush();
    expect(v5.container.querySelector('em')?.textContent).toBe('chain');
    chainOpen.set(false);
    expect(() => flush()).toThrow('chain cleanup boom');
    chainOpen.set(true);
    expect(() => flush()).not.toThrow();
    expect(v5.container.querySelector('em')?.textContent).toBe('chain');
    v5.unmount();

    // 5. A throwing cleanup in one propJsx source does not prevent sibling propJsx sources from unsubscribing and cleaning up.
    const pairOpen = signal(true);
    let secondCleaned = 0;
    function BadFirst() {
      onCleanup(() => {
        throw new Error('first boom');
      });
      return <b>first</b>;
    }
    function GoodSecond() {
      onCleanup(() => secondCleaned++);
      return <i>second</i>;
    }
    function PairHost(props: { first: JSX.Element; second: JSX.Element }) {
      const both = computed(() => [props.first, props.second]);
      return <section>{pairOpen.get() ? <div>{both.get()}</div> : null}</section>;
    }
    const v6 = render(() => <PairHost first={<BadFirst />} second={<GoodSecond />} />);
    flush();
    expect(v6.container.querySelector('b')?.textContent).toBe('first');
    expect(v6.container.querySelector('i')?.textContent).toBe('second');
    pairOpen.set(false);
    expect(() => flush()).toThrow('first boom');
    expect(secondCleaned).toBe(1);
    v6.unmount();

    // 6. Repeated reads of the same JSX prop inside a single computed produce distinct DOM nodes and clean both up.
    let repDestroys = 0;
    function RepLeaf() {
      onCleanup(() => repDestroys++);
      return <em>leaf</em>;
    }
    function RepComputed(props: { slot: JSX.Element }) {
      const both = computed(() => [props.slot, props.slot]);
      return <div>{both.get()}</div>;
    }
    const v7 = render(() => <RepComputed slot={<RepLeaf />} />);
    flush();
    expect(v7.container.querySelectorAll('em')).toHaveLength(2);
    v7.unmount();
    expect(repDestroys).toBe(2);
  });
});
