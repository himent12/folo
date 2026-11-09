import { describe, expect, it } from 'vitest';
import { BestPulse, CompleteFocus } from './bestPulse';

describe('BestPulse', () => {
  it('pulses on each new best of a running run, never for the best the page opened with', () => {
    const p = new BestPulse();
    expect(p.see(12, true)).toBe(0);
    expect(p.see(12, true)).toBe(0);
    expect(p.see(13, true)).toBe(1);
    expect(p.see(13, true)).toBe(1);
    expect(p.see(15, true)).toBe(2);
  });

  it('pulses for the first best of a run that had none, but not for a run that is no longer running', () => {
    const p = new BestPulse();
    expect(p.see(null, true)).toBe(0);
    expect(p.see(1, true)).toBe(1);
    expect(p.see(2, false)).toBe(1);
  });
});

describe('CompleteFocus', () => {
  it('focuses once when a run seen training completes, never for a run that was complete when the page opened', () => {
    const f = new CompleteFocus();
    expect(f.see('completed')).toBe(0);
    expect(f.see('completed')).toBe(0);
    const g = new CompleteFocus();
    expect(g.see('running')).toBe(0);
    expect(g.see('running')).toBe(0);
    expect(g.see('completed')).toBe(1);
    expect(g.see('completed')).toBe(1);
    const h = new CompleteFocus();
    h.see('running');
    expect(h.see('failed')).toBe(0);
  });
});
