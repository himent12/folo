import { describe, expect, it } from 'vitest';
import type { LogLine } from '$lib/api/types';
import { wasResumed } from './resumes';

const line = (message: string): LogLine => ({ t: '2026-11-11T10:00:00Z', level: 'info', message });

describe('wasResumed', () => {
  it('reads the server’s resume lines, and nothing else', () => {
    expect(wasResumed([line('queued'), line('resuming from weights/last.folo')])).toBe(true);
    expect(wasResumed([line('no epoch finished before; starting over')])).toBe(true);
    expect(wasResumed([line('queued'), line('finished 5 epochs in 79.6 s')])).toBe(false);
  });
});
