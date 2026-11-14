// @vitest-environment happy-dom
/// <reference types="node" />
// A KPI tile's value (design/UNIFIED.md §2, KPI strip): the unit and the secondary part of the value small, in
// tertiary, on the value's baseline ("314 img/s", "38 / 100", "12m 02s"), a pair's slash between two full-size values.
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { flush as flushSync } from 'fw';
import { render } from '../../../framework/testing';
import { describe, expect, it } from 'vitest';
import Stat, { figureParts } from './Stat';

describe('Stat', () => {
  it('splits a duration into figures and small units, a pair at its slash, and leaves a plain figure whole', () => {
    expect(figureParts('12m 02s')).toEqual([
      { t: '12', k: 'n' },
      { t: 'm', k: 'u' },
      { t: ' 02', k: 'n' },
      { t: 's', k: 'u' },
    ]);
    expect(figureParts('1h 04m').map((p) => p.k)).toEqual(['n', 'u', 'n', 'u']);
    expect(figureParts('7m')).toEqual([
      { t: '7', k: 'n' },
      { t: 'm', k: 'u' },
    ]);
    expect(figureParts('0.849 / 0.762')).toEqual([
      { t: '0.849', k: 'n' },
      { t: ' / ', k: 'slash' },
      { t: '0.762', k: 'n' },
    ]);
    for (const v of ['0.494', '1,234', '—', '2.62']) expect(figureParts(v)).toEqual([{ t: v, k: 'n' }]);
  });

  it('sets the unit and the duration units in the small step, in tertiary, whatever the value size', () => {
    const target = document.createElement('div');
    document.body.appendChild(target);
    const c = render(
      () => (
        <>
          <Stat label="Throughput" value="314" unit="img/s" />
          <Stat label="Time left" value="12m 02s" />
          <Stat label="Precision / Recall" value="0.849 / 0.762" size="lg" />
        </>
      ),
      target,
    );
    flushSync();
    const [ips, time, pr] = [...target.querySelectorAll('.value')];
    expect(ips!.textContent).toBe('314img/s');
    expect(ips!.querySelector('.unit')!.textContent).toBe('img/s');
    expect([...time!.querySelectorAll('.unit')].map((u) => u.textContent)).toEqual(['m', 's']);
    expect(pr!.querySelector('.slash')!.textContent).toBe(' / ');
    expect(pr!.textContent).toBe('0.849 / 0.762');
    expect(pr!.querySelector('.unit')).toBeNull();
    c.unmount();
    target.remove();
    // One rule in the source: the unit is `sm` tertiary at every size (no larger step for lg tiles).
    const src = readFileSync(resolve(process.cwd(), 'src/lib/ui/Stat.tsx'), 'utf8');
    const unit = src.match(/\n {2}\.unit \{([^}]*)\}/)![1]!;
    expect(unit).toContain('font-size: var(--type-sm-size)');
    expect(unit).toContain('color: var(--text-tertiary)');
    expect(src).not.toMatch(/\.(lg|sm) \.unit/);
  });
});
