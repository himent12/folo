// @vitest-environment happy-dom
// Tests for BenchCompare side-by-side CSV & Markdown export (overall score + axis deltas +
// per-stress-condition comparison rows) and SideBySide panel action buttons.
import { flush as flushSync } from 'fw';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { render, type Rendered } from '../../../framework/testing';
import { hub } from '$lib/api/index';
import { MockApi } from '$lib/api/mock';
import type { Bench, BenchEntry } from '$lib/api/types';
import { parse, router } from '$lib/router';
import { toast } from '$lib/ui/toast';
import BenchCompare from '../BenchCompare';
import { compareExport, compareMarkdown } from './export';

let target: HTMLElement;
let view: Rendered | undefined;
let downloads: Array<{ filename: string; text: string }> = [];

beforeEach(() => {
  hub.api = new MockApi('normal', 0);
  hub.mode = 'mock';
  toast.items = [];
  downloads = [];
  target = document.createElement('div');
  document.body.appendChild(target);

  const blobs = new Map<string, Blob>();
  vi.spyOn(URL, 'createObjectURL').mockImplementation((b) => {
    const url = `blob:bench-cmp/${blobs.size}`;
    blobs.set(url, b as Blob);
    return url;
  });
  vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {});
  vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function (this: HTMLAnchorElement) {
    const blob = blobs.get(this.getAttribute('href') ?? '');
    if (blob) {
      void blob.text().then((text) => downloads.push({ filename: this.download, text }));
    }
  });
});

afterEach(() => {
  view?.unmount();
  view = undefined;
  target.remove();
  toast.items = [];
  vi.restoreAllMocks();
});

async function settle(rounds = 16) {
  for (let i = 0; i < rounds; i++) {
    await new Promise((r) => setTimeout(r, 0));
    flushSync();
  }
}

describe('compareExport & compareMarkdown with stress conditions', () => {
  it('builds Markdown and CSV with overall score, axis deltas, per-stress-condition rows, and stale delta suppression', async () => {
    const bench = (await hub.api.bench('bn_3c90')) as Bench;
    const done = (await hub.api.benchEntries('bn_3c90')).filter((e) => e.status === 'done' && e.score);
    const e0: BenchEntry = { ...done[0]!, name: 'base | v1' };
    const e1: BenchEntry = { ...done[1]!, name: 'cand-v2' };

    const rows = [
      { key: 'score', label: 'Score', values: [72.4, 78.9], fmt: (v: number) => v.toFixed(1), d: 1 },
      { key: 'accuracy', label: 'Accuracy · mAP50-95', values: [0.612, 0.654], fmt: (v: number) => v.toFixed(3), d: 3 },
      { key: 'robustness', label: 'Robustness', values: [0.80, 0.85], fmt: (v: number) => v.toFixed(2), d: 2 },
    ];
    const conditions = [
      { key: 'blur', name: 'Motion blur', values: [0.78, 0.86] },
      { key: 'low_light', name: 'Low light', values: [0.82, null] },
    ];

    const payload = compareExport(
      bench,
      [e0, e1],
      rows,
      [false, false],
      (_k, vals) => (vals[1]! > vals[0]! ? new Set([1]) : new Set()),
      [{ text: 'cand-v2 ran 1 of 2 conditions.' }],
      [undefined, undefined],
      conditions,
    );

    // CSV includes score, axes, and stress condition rows with baseline deltas.
    expect(payload.rows).toHaveLength(5);
    expect(payload.rows[0]).toEqual(['Score', 72.4, 78.9, 6.5]);
    expect(payload.rows[3]).toEqual(['Condition · Motion blur', 0.78, 0.86, 0.08]);
    expect(payload.rows[4]).toEqual(['Condition · Low light', 0.82, null, null]);
    const json = payload.json as { conditions: Array<{ condition: string; values: Array<{ delta_printed: string | null }> }> };
    expect(json.conditions).toHaveLength(2);
    expect(json.conditions[0]!.values[1]!.delta_printed).toBe('+8 pp');
    expect(json.conditions[1]!.values[1]!.delta_printed).toBeNull();

    // Markdown includes escaped headers, bold winners, axis deltas, stress conditions table, and notes.
    const md = compareMarkdown(
      bench,
      [e0, e1],
      rows,
      [false, false],
      (_k, vals) => (vals[1]! > vals[0]! ? new Set([1]) : new Set()),
      [{ text: 'cand-v2 ran 1 of 2 conditions.' }],
      conditions,
    );
    expect(md).toContain(`# Benchmark comparison: ${bench.name}`);
    expect(md).toContain('base \\| v1 (baseline)');
    expect(md).toContain('| Score | 72.4 | **78.9** | +6.5 |');
    expect(md).toContain('| Accuracy · mAP50-95 | 0.612 | **0.654** | +0.042 |');
    expect(md).toContain('## Stress conditions (accuracy kept)');
    expect(md).toContain('| Motion blur | 78% | 86% | +8 pp |');
    expect(md).toContain('| Low light | 82% | — | — |');
    expect(md).toContain('- cand-v2 ran 1 of 2 conditions.');

    // Earlier-version entry suppresses condition and axis deltas in Markdown.
    const staleMd = compareMarkdown(bench, [e0, e1], rows, [false, true], () => new Set(), [], conditions);
    expect(staleMd).toContain('| Score | 72.4 | 78.9 | — |');
    expect(staleMd).toContain('| Motion blur | 78% | 86% | — |');
  });

  it('wires Copy Markdown and Download CSV buttons in SideBySide panel header on BenchCompare', async () => {
    const entries = (await hub.api.benchEntries('bn_3c90')).filter((e) => e.status === 'done' && e.score);
    const [e0, e1] = entries;
    router.route = parse(`#/benchmarks/bn_3c90/compare?entries=${e0!.id},${e1!.id}`);
    view = render(() => <BenchCompare />, target);
    await settle(20);

    const copyBtn = target.querySelector<HTMLButtonElement>('[data-compare-copy-md]')!;
    const csvBtn = target.querySelector<HTMLButtonElement>('[data-compare-download-csv]')!;
    expect(copyBtn).not.toBeNull();
    expect(csvBtn).not.toBeNull();

    const clipboardText: string[] = [];
    Object.defineProperty(navigator, 'clipboard', {
      value: { writeText: vi.fn(async (s: string) => void clipboardText.push(s)) },
      configurable: true,
    });

    copyBtn.click();
    await vi.waitFor(() => {
      flushSync();
      if (!clipboardText.length || copyBtn.textContent?.trim() !== 'Copied') throw new Error('waiting for clipboard write');
    });
    expect(clipboardText[0]).toContain('## Score & axes');
    expect(clipboardText[0]).toContain('## Stress conditions (accuracy kept)');
    expect(copyBtn.textContent?.trim()).toBe('Copied');

    csvBtn.click();
    await vi.waitFor(() => {
      flushSync();
      if (!downloads.length) throw new Error('waiting for CSV download');
    });
    expect(downloads[0]!.filename).toBe('benchmark-fasteners-val-compare.csv');
    expect(downloads[0]!.text).toContain('Score,');
    expect(downloads[0]!.text).toContain('Condition · ');
  });
});
