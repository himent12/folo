// @vitest-environment happy-dom
// Unit and component integration tests for RunCompare CSV & GitHub-Flavored Markdown export.
import { flush as flushSync } from 'fw';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { render, type Rendered } from '../../../framework/testing';
import { hub } from '$lib/api/index';
import { MockApi } from '$lib/api/mock';
import type { Run } from '$lib/api/types';
import { parse, router } from '$lib/router';
import { pendingBlobRevokeCount, revokePendingBlobUrls, toCsv } from '$lib/tableExport';
import { toast } from '$lib/ui/toast';
import RunCompare from '../RunCompare';
import { buildRunCompareCsv, buildRunCompareMarkdown, orderedRunsForExport, runExportLabel } from './export';

let target: HTMLElement;
let mounted: Rendered | undefined;

beforeEach(() => {
  target = document.createElement('div');
  document.body.appendChild(target);
  hub.api = new MockApi('normal', 0);
  hub.mode = 'mock';
});

afterEach(() => {
  mounted?.unmount();
  mounted = undefined;
  target.remove();
  revokePendingBlobUrls();
  for (const t of toast.items) toast.dismiss(t.id);
  vi.restoreAllMocks();
});

async function until<T>(get: () => T | null | undefined | false, what: string): Promise<T> {
  let v: T | null | undefined | false = undefined;
  await vi.waitFor(
    () => {
      flushSync();
      v = get();
      if (!v) throw new Error(`waiting for ${what}`);
    },
    { timeout: 3000, interval: 10 },
  );
  return v as T;
}

describe('compare/export (RunCompare CSV & Markdown)', () => {
  it('builds formula-safe RFC-4180 CSV and escaped GFM Markdown respecting baseline ordering, duplicate names, and dataset caveat', async () => {
    const api = hub.api as MockApi;
    const allRuns = (await api.runs()).filter((r) => r.status === 'completed');
    const datasets = await api.datasets();
    const rA: Run = { ...allRuns[0]!, id: 'run_aaaa', name: '=run|alpha' };
    const rB: Run = { ...allRuns[1]!, id: 'run_bbbb', name: '=run|alpha' };
    const mA = await api.runMetrics(allRuns[0]!.id);
    const mB = await api.runMetrics(allRuns[1]!.id);
    const metrics = new Map([
      [rA.id, mA],
      [rB.id, mB],
    ]);

    expect(orderedRunsForExport([rA, rB], rB.id).map((r) => r.id)).toEqual(['run_bbbb', 'run_aaaa']);
    expect(runExportLabel(rA, [rA, rB])).toBe('=run|alpha (aaaa)');

    const sameData = rA.config.dataset_id === rB.config.dataset_id;
    const input = {
      chosen: [rA, rB],
      baselineId: rB.id,
      datasets,
      metrics,
      sameData,
      startOf: () => null,
    };

    const csvObj = buildRunCompareCsv(input);
    expect(csvObj.headers).toEqual(['section', 'measure', 'differs', '=run|alpha (bbbb) (baseline)', '=run|alpha (aaaa)']);
    const csvText = toCsv(csvObj.headers, csvObj.rows);
    expect(csvText).toContain("'=run|alpha (bbbb) (baseline)");
    expect(csvText).toContain('result,Best mAP50-95,');
    expect(csvText).toContain('setting,Model,');

    const md = buildRunCompareMarkdown(input);
    expect(md).toContain('# Run comparison');
    expect(md).toContain('baseline: **=run\\|alpha (bbbb)**');
    expect(md).toContain('## Results');
    expect(md).toContain('| Best mAP50-95 |');
    if (!sameData) expect(md).toContain('> ');
  });

  it('wires Copy Markdown and Download CSV in CompareTable and updates when baseline swaps', async () => {
    const runs = (await hub.api.runs()).filter((r) => r.status === 'completed').slice(0, 2);
    router.route = parse(`#/runs/compare?ids=${runs.map((r) => r.id).join(',')}`);
    mounted = render(() => <RunCompare />, target);

    await until(() => target.querySelector('[data-run-compare-copy-md]'), 'export actions');
    const copyBtn = target.querySelector<HTMLButtonElement>('[data-run-compare-copy-md]')!;
    const csvBtn = target.querySelector<HTMLButtonElement>('[data-run-compare-download-csv]')!;

    const copied: string[] = [];
    Object.defineProperty(navigator, 'clipboard', {
      value: { writeText: vi.fn(async (text: string) => { copied.push(text); }) },
      configurable: true,
    });
    const blobs: Blob[] = [];
    vi.spyOn(URL, 'createObjectURL').mockImplementation((b) => {
      blobs.push(b as Blob);
      return `blob:runcmp-${blobs.length}`;
    });
    vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {});
    const click = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {});

    copyBtn.click();
    await until(() => copyBtn.textContent?.includes('Copied'), 'Copied button state');
    expect(copied).toHaveLength(1);
    expect(copied[0]).toContain(`baseline: **${runs[0]!.name}**`);

    // Swap baseline to second run and download CSV -> baseline column becomes run 2.
    target.querySelector<HTMLButtonElement>(`#cmp-table button[aria-label="Make ${runs[1]!.name} the baseline"]`)!.click();
    flushSync();
    csvBtn.click();
    expect(click).toHaveBeenCalledTimes(1);
    expect(pendingBlobRevokeCount()).toBe(1);
    const csvText = await blobs[0]!.text();
    expect(csvText).toContain(`${runs[1]!.name} (baseline)`);
    revokePendingBlobUrls();
  });
});
