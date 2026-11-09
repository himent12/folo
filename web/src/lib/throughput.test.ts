import { describe, expect, it } from 'vitest';
import { MockApi } from '$lib/api/mock';
import type { RunEstimate } from '$lib/api/types';
import { basisLine, confidenceLine, loadThroughput, measuredHere, measuredLatency, measuredRows, measuredTraining, timeRange } from './throughput';

describe('timeRange', () => {
  it('says a run takes a range of minutes, rounded outwards', () => {
    expect(timeRange(18.4 * 60, 23.2 * 60)).toBe('18–24 min');
  });
  it('switches to hours with a tenth once both ends pass an hour, and mixes units across it', () => {
    expect(timeRange(4300, 5800)).toBe('1.2–1.6 h');
    expect(timeRange(50 * 60, 4300)).toBe('50 min – 1.2 h');
  });
  it('collapses to one figure only when both ends round alike, and never says 0 min', () => {
    expect(timeRange(4300, 4310)).toBe('~1.2 h');
    expect(timeRange(10, 50)).toBe('under 1 min');
    expect(timeRange(10, 70)).toBe('1–2 min');
  });
});

describe('the throughput table', () => {
  it('reads nothing from a server without the feature, so pages keep the reference', async () => {
    expect(await loadThroughput(new MockApi('no-throughput', 0))).toBeNull();
    expect(await loadThroughput(new MockApi('error', 0))).toBeNull();
  });

  it("calls measured only the rows of this server's own adapter", async () => {
    const t = (await loadThroughput(new MockApi('normal', 0)))!;
    const m = measuredRows(t);
    expect(m.training.length).toBeGreaterThan(0);
    // The mock's folo-s predictions were measured on another adapter under gpu:0: listed, never "here".
    const other = m.inference.find((r) => r.measured_on !== t.device_name)!;
    expect(measuredHere(t, other)).toBe(false);
    expect(measuredLatency(t, 'folo-s')).toBeUndefined();
    expect(measuredLatency(t, 'folo-n')?.latency_ms).toBe(5.9);
    expect(measuredTraining(t, 'folo-n')?.imgs_per_s).toBe(212.4);
    // Reference rows never count as measured.
    expect(t.training.some((r) => r.source === 'reference' && measuredHere(t, r))).toBe(false);
  });
});

describe('what an estimate stands on', () => {
  const est = (o: Partial<RunEstimate['basis']>, e: Partial<RunEstimate> = {}) =>
    ({
      device: 'gpu:0',
      device_name: 'NVIDIA GeForce RTX 4080 SUPER',
      confidence: 'high',
      basis: {
        source: 'measured',
        arch: 'folo-n',
        imgsz: 640,
        batch: 16,
        amp: false,
        measured_on: 'NVIDIA GeForce RTX 4080 SUPER',
        samples: 99,
        updated: null,
        exact: true,
        note: '',
        ...o,
      },
      ...e,
    }) as RunEstimate;
  it('names the device and the sample count for an exact measured row', () => {
    expect(basisLine(est({}))).toBe('Measured on this GPU · 99 samples');
    expect(basisLine(est({ samples: 1 }, { device: 'cpu' }))).toBe('Measured on this CPU · 1 sample');
  });
  it('names the setting a carried-over estimate comes from', () => {
    expect(basisLine(est({ exact: false, arch: 'folo-s' }))).toBe('Scaled from folo-s at 640 px on this GPU');
  });
  it('says when the reference is not from this kind of device', () => {
    // A reference range is wide, and the line says why.
    expect(basisLine(est({ source: 'reference' }))).toBe('Reference benchmarks · no run measured here yet');
    expect(basisLine(est({ source: 'reference', measured_on: 'Intel Core i7-14700KF' }, { device: 'cpu', device_name: 'AMD Ryzen 9' }))).toBe(
      'Wide range: no run measured on this CPU yet',
    );
  });
  it('says what each confidence means for the range', () => {
    expect(confidenceLine(est({}, { confidence: 'low' }))).toContain('wide range');
  });
});
