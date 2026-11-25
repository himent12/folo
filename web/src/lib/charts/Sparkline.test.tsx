// @vitest-environment happy-dom
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { flush } from 'fw';
import { afterEach, describe, expect, it } from 'vitest';
import { render } from '../../../framework/testing';
import Sparkline, { sparklinePaths } from './Sparkline';

let mounted: { unmount(): void } | undefined;

afterEach(() => {
  mounted?.unmount();
  mounted = undefined;
  document.body.innerHTML = '';
});

describe('sparklinePaths', () => {
  it('computes linear line and area SVG paths without d3 imports', () => {
    const src = readFileSync(resolve(process.cwd(), 'src/lib/charts/Sparkline.tsx'), 'utf8');
    expect(src).not.toMatch(/from ['"]d3-/);

    const g = sparklinePaths([0, 5, 10], 80, 24);
    expect(g).not.toBeNull();
    expect(g!.d).toBe('M1,22L39,12L77,2');
    expect(g!.a).toBe('M1,22L39,12L77,2L77,24L1,24Z');
    expect(g!.last).toEqual([77, 2]);
  });

  it('respects explicit domain and planned length for live runs', () => {
    const g = sparklinePaths([0.25, 0.75], 80, 24, [0, 1], 5);
    expect(g).not.toBeNull();
    // xSpan = 4 -> index 0 at 1, index 1 at 1 + (1/4)*76 = 20
    expect(g!.last[0]).toBe(20);
    // y=0.75 in [0,1] on height 24 (range [22, 2]): 22 - 0.75*20 = 7
    expect(g!.last[1]).toBe(7);
  });

  it('handles flat series and non-finite gaps cleanly', () => {
    const flat = sparklinePaths([4, 4, 4], 80, 24);
    expect(flat).not.toBeNull();
    expect(flat!.d).toBe('M1,22L39,22L77,22');

    const gapped = sparklinePaths([0, 10, Number.NaN, 5, 10], 80, 24);
    expect(gapped).not.toBeNull();
    expect(gapped!.d).toContain('M1,22L20,2M58,12L77,2');
    expect(gapped!.d).not.toContain('NaN');
    expect(gapped!.a).not.toContain('NaN');

    expect(sparklinePaths([1], 80, 24)).toBeNull();
    expect(sparklinePaths([Number.NaN, 3, Number.POSITIVE_INFINITY], 80, 24)).toBeNull();
  });
});

describe('Sparkline component', () => {
  it('renders paths and end dot when enough finite points exist, or a dashed fallback line otherwise', () => {
    const host = document.createElement('div');
    document.body.appendChild(host);
    mounted = render(() => <Sparkline values={[1, 3, 2]} width={80} height={24} endColor="var(--status-live)" />, host);
    flush();

    expect(host.querySelector('path.l')?.getAttribute('d')).toBe('M1,22L39,2L77,12');
    expect(host.querySelector('path.a')?.getAttribute('d')).toBe('M1,22L39,2L77,12L77,24L1,24Z');
    expect(host.querySelector('circle')?.getAttribute('cx')).toBe('77');
    expect(host.querySelector('circle')?.getAttribute('cy')).toBe('12');

    mounted.unmount();
    mounted = render(() => <Sparkline values={[1]} width={80} height={24} />, host);
    flush();
    expect(host.querySelector('line.none')).toBeTruthy();
    expect(host.querySelector('path.l')).toBeNull();
  });
});
