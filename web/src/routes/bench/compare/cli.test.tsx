// @vitest-environment happy-dom
// Tests for `folo bench compare` CLI regression-gate builder (`cli.ts`) and collapsible `CompareCli`
// handoff inside `BenchCompare` (candidate picker for >2 entries, baseline swap sync, format selection,
// `--fail-on-regression` / `--min-delta` negative & invalid handling, quoting, and clipboard copy).
import { flush as flushSync } from 'fw';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { hub } from '$lib/api/index';
import { MockApi } from '$lib/api/mock';
import { prefs } from '$lib/prefs';
import { parse, router } from '$lib/router';
import { toast } from '$lib/ui/toast';
import { render, type Rendered } from '../../../../framework/testing';
import BenchCompare from '../../BenchCompare';
import { buildBenchCompareCli, parseMinDelta } from './cli';

let target: HTMLElement;
let view: Rendered | undefined;

beforeEach(() => {
  hub.api = new MockApi('normal', 0);
  hub.mode = 'mock';
  toast.items = [];
  target = document.createElement('div');
  document.body.appendChild(target);
});

afterEach(() => {
  view?.unmount();
  view = undefined;
  target.remove();
  toast.items = [];
  vi.restoreAllMocks();
});

async function settle(rounds = 18) {
  for (let i = 0; i < rounds; i++) {
    await new Promise((r) => setTimeout(r, 0));
    flushSync();
  }
}

describe('buildBenchCompareCli & parseMinDelta', () => {
  it('builds bare PowerShell/POSIX-safe commands for default table, markdown, csv, and regression gates', () => {
    const plain = buildBenchCompareCli({
      benchId: 'bn_3c90',
      baselineId: 'be_a1',
      candidateId: 'be_b2',
    });
    expect(plain.minDeltaError).toBeNull();
    expect(plain.argv).toEqual(['bench', 'compare', 'bn_3c90', 'be_a1', 'be_b2']);
    expect(plain.line).toBe('folo bench compare bn_3c90 be_a1 be_b2');

    const mdGate = buildBenchCompareCli({
      benchId: 'bn_3c90',
      baselineId: 'be_a1',
      candidateId: 'be_b2',
      format: 'markdown',
      failOnRegression: true,
    });
    expect(mdGate.argv).toEqual(['bench', 'compare', 'bn_3c90', 'be_a1', 'be_b2', '--format', 'markdown', '--fail-on-regression']);
    expect(mdGate.line).toBe('folo bench compare bn_3c90 be_a1 be_b2 --format markdown --fail-on-regression');

    // Negative --min-delta attaches with `=` so PowerShell and clap both parse it unambiguously.
    const negDelta = buildBenchCompareCli({
      benchId: 'bn_3c90',
      baselineId: 'be_a1',
      candidateId: 'be_b2',
      format: 'csv',
      failOnRegression: true,
      minDelta: '-0.5',
    });
    expect(negDelta.minDeltaError).toBeNull();
    expect(negDelta.argv).toEqual([
      'bench',
      'compare',
      'bn_3c90',
      'be_a1',
      'be_b2',
      '--format',
      'csv',
      '--fail-on-regression',
      '--min-delta=-0.5',
    ]);
    expect(negDelta.line).toBe('folo bench compare bn_3c90 be_a1 be_b2 --format csv --fail-on-regression --min-delta=-0.5');

    // Positive --min-delta passes as two tokens; -0 normalizes to 0.
    const posDelta = buildBenchCompareCli({
      benchId: 'bn_3c90',
      baselineId: 'be_a1',
      candidateId: 'be_b2',
      failOnRegression: true,
      minDelta: '1.25',
    });
    expect(posDelta.line).toBe('folo bench compare bn_3c90 be_a1 be_b2 --fail-on-regression --min-delta 1.25');
    expect(parseMinDelta('-0')).toEqual({ value: 0, error: null });
  });

  it('quotes unsafe characters for PowerShell, guards dash-prefixed positionals with --, and rejects non-finite minDelta', () => {
    const quoted = buildBenchCompareCli({
      benchId: "bn O'Brien",
      baselineId: 'be$1',
      candidateId: 'be;2',
      failOnRegression: true,
      minDelta: '0.25',
    });
    expect(quoted.line).toBe("folo bench compare 'bn O''Brien' 'be$1' 'be;2' --fail-on-regression --min-delta 0.25");

    const dashPos = buildBenchCompareCli({
      benchId: '-bn',
      baselineId: 'be_1',
      candidateId: 'be_2',
      format: 'markdown',
      serverUrl: 'http://gpu-box:7878',
    });
    expect(dashPos.argv).toEqual([
      'bench',
      'compare',
      '--format',
      'markdown',
      '--server',
      'http://gpu-box:7878',
      '--',
      '-bn',
      'be_1',
      'be_2',
    ]);
    expect(dashPos.line).toBe('folo bench compare --format markdown --server http://gpu-box:7878 -- -bn be_1 be_2');

    const bad = buildBenchCompareCli({
      benchId: 'bn_1',
      baselineId: 'be_1',
      candidateId: 'be_2',
      failOnRegression: true,
      minDelta: 'nope',
    });
    expect(bad.minDeltaError).toMatch(/finite/i);
    // Turning failOnRegression off ignores any stale minDelta text.
    const gateOff = buildBenchCompareCli({
      benchId: 'bn_1',
      baselineId: 'be_1',
      candidateId: 'be_2',
      failOnRegression: false,
      minDelta: 'nope',
    });
    expect(gateOff.minDeltaError).toBeNull();
    expect(gateOff.line).toBe('folo bench compare bn_1 be_1 be_2');
  });
});

describe('CompareCli inside BenchCompare', () => {
  it('supports candidate selection for >2 entries, baseline swap sync, format/gate toggles, validation, and copy', async () => {
    const entries = (await hub.api.benchEntries('bn_3c90')).filter((e) => e.status === 'done' && e.score);
    const [e0, e1, e2] = entries;
    router.route = parse(`#/benchmarks/bn_3c90/compare?entries=${e0!.id},${e1!.id},${e2!.id}`);
    view = render(() => <BenchCompare />, target);
    await settle(20);

    const toggle = target.querySelector<HTMLButtonElement>('[data-compare-cli-toggle]')!;
    expect(toggle).not.toBeNull();
    expect(toggle.getAttribute('aria-expanded')).toBe('false');
    toggle.click();
    flushSync();
    expect(toggle.getAttribute('aria-expanded')).toBe('true');

    const root = target.querySelector<HTMLElement>('[data-bench-compare-cli]')!;
    const cmdText = () => root.querySelector('code')?.textContent?.trim() ?? '';
    expect(cmdText()).toBe(`folo bench compare bn_3c90 ${e0!.id} ${e1!.id} --fail-on-regression`);

    // Switch candidate from e1 to e2 via the candidate <select>.
    const candSelect = root.querySelector<HTMLSelectElement>('select')!;
    expect(candSelect).not.toBeNull();
    candSelect.value = e2!.id;
    candSelect.dispatchEvent(new Event('change', { bubbles: true }));
    flushSync();
    expect(cmdText()).toBe(`folo bench compare bn_3c90 ${e0!.id} ${e2!.id} --fail-on-regression`);

    // Switch output format to Markdown and enter a negative --min-delta tolerance.
    const mdBtn = [...root.querySelectorAll<HTMLButtonElement>('[role="radio"]')].find((b) => b.textContent?.trim() === 'Markdown')!;
    mdBtn.click();
    flushSync();
    const deltaInput = root.querySelector<HTMLInputElement>('[data-compare-cli-min-delta]')!;
    deltaInput.value = '-0.5';
    deltaInput.dispatchEvent(new Event('input', { bubbles: true }));
    flushSync();
    expect(cmdText()).toBe(`folo bench compare bn_3c90 ${e0!.id} ${e2!.id} --format markdown --fail-on-regression --min-delta=-0.5`);

    // Copy command writes exact CLI line to clipboard.
    const writeText = vi.fn(async () => {});
    Object.defineProperty(navigator, 'clipboard', { value: { writeText }, configurable: true });
    root.querySelector<HTMLButtonElement>('button[aria-label="Copy command"]')!.click();
    await Promise.resolve();
    expect(writeText).toHaveBeenCalledWith(
      `folo bench compare bn_3c90 ${e0!.id} ${e2!.id} --format markdown --fail-on-regression --min-delta=-0.5`,
    );

    // Invalid min-delta surfaces inline alert and hides copyable command until fixed.
    deltaInput.value = 'abc';
    deltaInput.dispatchEvent(new Event('input', { bubbles: true }));
    flushSync();
    expect(root.querySelector('[role="alert"]')?.textContent).toMatch(/finite/i);
    expect(root.querySelector('code')).toBeNull();

    // Unchecking --fail-on-regression clears the error and removes gate flags.
    const failBox = root.querySelector<HTMLInputElement>('[data-compare-cli-fail]')!;
    failBox.checked = false;
    failBox.dispatchEvent(new Event('change', { bubbles: true }));
    flushSync();
    expect(root.querySelector('[role="alert"]')).toBeNull();
    expect(cmdText()).toBe(`folo bench compare bn_3c90 ${e0!.id} ${e2!.id} --format markdown`);

    // Promoting the currently selected candidate (e2) to baseline automatically re-picks a valid non-baseline candidate.
    const swapButtons = [...target.querySelectorAll<HTMLButtonElement>('button.swap')];
    const swapE2 = swapButtons.find((b) => b.getAttribute('aria-label')?.includes(e2!.name)) ?? swapButtons.at(-1)!;
    swapE2.click();
    flushSync();
    expect(cmdText()).not.toContain(`${e2!.id} ${e2!.id}`);
    expect(cmdText()).toMatch(/^folo bench compare bn_3c90 be_\w+ be_\w+ --format markdown$/);

    // Configured remote serverUrl is included via --server with a FOLO_TOKEN hint.
    const prevServer = prefs.serverUrl;
    prefs.serverUrl = 'http://gpu-box:7878';
    try {
      flushSync();
      expect(cmdText()).toContain('--server http://gpu-box:7878');
      expect(root.textContent).toContain('FOLO_TOKEN');
    } finally {
      prefs.serverUrl = prevServer;
    }
  });
});
