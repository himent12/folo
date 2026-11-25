// @vitest-environment happy-dom
import { describe, expect, it, vi } from 'vitest';
import { createRoot, deep, flush, getOwner, mergeProps, signal, type Owner } from 'fw';
import { createIcon } from './icon';
import { render } from '../testing';

const SampleIcon = createIcon({
  name: 'check',
  aliases: ['done'],
  node: [
    ['path', { d: 'M20 6 9 17l-5-5' }],
    ['circle', { cx: '12', cy: '12', r: '10' }],
  ],
});

const SmallBoxIcon = createIcon({
  name: 'dot',
  size: 16,
  node: [['circle', { cx: '8', cy: '8', r: '2' }]],
});

describe('createIcon', () => {
  it('clones baked template attributes and children for static default props without creating a renderEffect', () => {
    let rootOwner: Owner | null = null;
    const svg = createRoot(() => {
      rootOwner = getOwner();
      return SampleIcon({}) as SVGElement;
    });

    expect(svg.tagName.toLowerCase()).toBe('svg');
    expect(svg.namespaceURI).toBe('http://www.w3.org/2000/svg');
    expect(svg.getAttribute('xmlns')).toBe('http://www.w3.org/2000/svg');
    expect(svg.getAttribute('width')).toBe('24');
    expect(svg.getAttribute('height')).toBe('24');
    expect(svg.getAttribute('viewBox')).toBe('0 0 24 24');
    expect(svg.getAttribute('fill')).toBe('none');
    expect(svg.getAttribute('stroke')).toBe('currentColor');
    expect(svg.getAttribute('stroke-width')).toBe('2');
    expect(svg.getAttribute('stroke-linecap')).toBe('round');
    expect(svg.getAttribute('stroke-linejoin')).toBe('round');
    expect(svg.getAttribute('class')).toBe('lucide lucide-check lucide-done lucide-icon');
    expect(svg.getAttribute('aria-hidden')).toBe('true');
    expect(svg.children).toHaveLength(2);
    expect(svg.children[0]!.tagName.toLowerCase()).toBe('path');
    expect(svg.children[0]!.getAttribute('d')).toBe('M20 6 9 17l-5-5');
    expect(svg.children[1]!.getAttribute('r')).toBe('10');
    // Static props skip renderEffect: no child owners attached to rootOwner.
    expect(rootOwner!._first).toBeNull();
  });

  it('applies static custom props, overrides, nonScalingStroke, a11y, and events accurately', () => {
    const clicked = vi.fn();
    const { container, unmount } = render(() => (
      <div>
        <SampleIcon
          size={16}
          color="#f00"
          strokeWidth={1.5}
          absoluteStrokeWidth
          nonScalingStroke
          class="extra"
          fill="blue"
          aria-label="Confirmed"
          data-id="ok"
          onclick={clicked}
        />
        <SampleIcon aria-hidden={false} />
        <SmallBoxIcon size={8} strokeWidth={2} absoluteStrokeWidth />
      </div>
    ));

    const svgs = container.querySelectorAll('svg');
    const first = svgs[0]!;
    expect(first.getAttribute('width')).toBe('16');
    expect(first.getAttribute('height')).toBe('16');
    expect(first.getAttribute('stroke')).toBe('#f00');
    // (1.5 * 24) / 16 = 2.25
    expect(first.getAttribute('stroke-width')).toBe('2.25');
    expect(first.getAttribute('class')).toBe('lucide lucide-check lucide-done lucide-icon extra');
    expect(first.getAttribute('fill')).toBe('blue');
    expect(first.getAttribute('aria-label')).toBe('Confirmed');
    expect(first.hasAttribute('aria-hidden')).toBe(false);
    expect(first.getAttribute('data-id')).toBe('ok');
    expect(first.children[0]!.getAttribute('vector-effect')).toBe('non-scaling-stroke');
    expect(first.children[1]!.getAttribute('vector-effect')).toBe('non-scaling-stroke');
    first.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    expect(clicked).toHaveBeenCalledOnce();

    // Explicit aria-hidden={false} is preserved (not removed by a11y detection).
    const second = svgs[1]!;
    expect(second.getAttribute('aria-hidden')).toBe('false');
    expect(second.children[0]!.hasAttribute('vector-effect')).toBe(false);

    // Custom box size (16) with absoluteStrokeWidth: (2 * 16) / 8 = 4
    const third = svgs[2]!;
    expect(third.getAttribute('viewBox')).toBe('0 0 16 16');
    expect(third.getAttribute('stroke-width')).toBe('4');

    unmount();
  });

  it('updates reactively when passed getter props or merged PROXY props', () => {
    const size = signal(14);
    const cls = signal<string | undefined>('a');
    const label = signal<string | undefined>(undefined);
    const extra = signal<Record<string, unknown>>({ 'data-step': '1', fill: 'green' });

    const { container, unmount } = render(() => (
      <div>
        <SampleIcon size={size.get()} class={cls.get()} aria-label={label.get()} />
        <SampleIcon {...extra.get()} />
      </div>
    ));

    const [reactiveSvg, spreadSvg] = container.querySelectorAll('svg');
    expect(reactiveSvg!.getAttribute('width')).toBe('14');
    expect(reactiveSvg!.getAttribute('class')).toBe('lucide lucide-check lucide-done lucide-icon a');
    expect(reactiveSvg!.hasAttribute('aria-hidden')).toBe(false);

    size.set(20);
    cls.set(undefined);
    label.set('Saved');
    flush();
    expect(reactiveSvg!.getAttribute('width')).toBe('20');
    expect(reactiveSvg!.getAttribute('class')).toBe('lucide lucide-check lucide-done lucide-icon');
    expect(reactiveSvg!.getAttribute('aria-label')).toBe('Saved');

    // Spread PROXY props add, override, and remove attributes reactively.
    expect(spreadSvg!.getAttribute('data-step')).toBe('1');
    expect(spreadSvg!.getAttribute('fill')).toBe('green');
    expect(spreadSvg!.getAttribute('aria-hidden')).toBe('true');

    extra.set({ role: 'img' });
    flush();
    expect(spreadSvg!.hasAttribute('data-step')).toBe(false);
    expect(spreadSvg!.getAttribute('fill')).toBe('none');
    expect(spreadSvg!.getAttribute('role')).toBe('img');
    expect(spreadSvg!.hasAttribute('aria-hidden')).toBe(false);

    extra.set({});
    flush();
    expect(spreadSvg!.hasAttribute('role')).toBe(false);
    expect(spreadSvg!.getAttribute('aria-hidden')).toBe('true');

    // Also verify mergeProps directly produces a reactive icon when given a getter source.
    const sw = signal(1);
    const merged = mergeProps({ size: 18 }, () => ({ strokeWidth: sw.get() }));
    const svg = createRoot(() => SampleIcon(merged) as SVGElement);
    expect(svg.getAttribute('stroke-width')).toBe('1');
    sw.set(3);
    flush();
    expect(svg.getAttribute('stroke-width')).toBe('3');

    unmount();
  });

  it('tracks deep() reactive props mutated in place, preserves explicit aria-hidden order, restores nullish first-render overrides, and cleans up static non-delegated listeners', () => {
    // 1. deep() proxy props mutated in place
    const dp = deep<Record<string, unknown>>({ size: 16, role: 'img' });
    let deepSvg!: SVGElement;
    const disposeDeep = createRoot((d) => {
      deepSvg = SampleIcon(dp) as SVGElement;
      return d;
    });
    expect(deepSvg.getAttribute('width')).toBe('16');
    expect(deepSvg.getAttribute('role')).toBe('img');
    expect(deepSvg.hasAttribute('aria-hidden')).toBe(false);
    dp.size = 32;
    dp.fill = 'orange';
    delete dp.role;
    flush();
    expect(deepSvg.getAttribute('width')).toBe('32');
    expect(deepSvg.getAttribute('fill')).toBe('orange');
    expect(deepSvg.hasAttribute('role')).toBe(false);
    expect(deepSvg.getAttribute('aria-hidden')).toBe('true');
    disposeDeep();

    // 2. Explicit aria-hidden attribute ordering matches rest-prop order in both static and reactive paths
    const attrNames = (el: Element) => [...el.attributes].map((a) => a.name);
    const staticOrder = createRoot(() => SampleIcon({ role: 'img', 'aria-hidden': false }) as SVGElement);
    const sNames = attrNames(staticOrder);
    expect(sNames.indexOf('role')).toBeLessThan(sNames.indexOf('aria-hidden'));
    expect(staticOrder.getAttribute('aria-hidden')).toBe('false');

    const reactiveOrder = createRoot(() =>
      SampleIcon({
        get role() {
          return 'img';
        },
        'aria-hidden': 'true',
      }) as SVGElement,
    );
    const rNames = attrNames(reactiveOrder);
    expect(rNames.indexOf('role')).toBeLessThan(rNames.indexOf('aria-hidden'));
    expect(reactiveOrder.getAttribute('aria-hidden')).toBe('true');

    // 3. First-render nullish overrides remove baked defaults and restore them when keys are dropped
    const overrides = signal<Record<string, unknown>>({ fill: null, viewBox: undefined, 'stroke-width': null });
    const focused = vi.fn();
    const { container, unmount } = render(() => (
      <div>
        <SampleIcon {...overrides.get()} />
        <SampleIcon fill={undefined} viewBox={null} onfocus={focused} width={12} height={18} size={20} />
      </div>
    ));
    const [reactNull, staticNull] = container.querySelectorAll('svg');
    expect(reactNull!.hasAttribute('fill')).toBe(false);
    expect(reactNull!.hasAttribute('viewBox')).toBe(false);
    expect(reactNull!.hasAttribute('stroke-width')).toBe(false);
    overrides.set({});
    flush();
    expect(reactNull!.getAttribute('fill')).toBe('none');
    expect(reactNull!.getAttribute('viewBox')).toBe('0 0 24 24');
    expect(reactNull!.getAttribute('stroke-width')).toBe('2');

    // Static nullish overrides + explicit width/height + non-delegated onfocus cleanup on unmount
    expect(staticNull!.hasAttribute('fill')).toBe(false);
    expect(staticNull!.hasAttribute('viewBox')).toBe(false);
    expect(staticNull!.getAttribute('width')).toBe('12');
    expect(staticNull!.getAttribute('height')).toBe('18');
    staticNull!.dispatchEvent(new FocusEvent('focus'));
    expect(focused).toHaveBeenCalledTimes(1);
    unmount();
    staticNull!.dispatchEvent(new FocusEvent('focus'));
    expect(focused).toHaveBeenCalledTimes(1);
  });
});
