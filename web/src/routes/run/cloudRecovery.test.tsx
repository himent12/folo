// @vitest-environment happy-dom
// Tests for spot preemption & automatic checkpoint-resume detection (`cloudRecovery.ts`),
// CloudPanel recovery badges / status copy / attempt counter, manual-resume resets,
// failed replacement workers, structural memoization, and live Run page integration.
import { flush as flushSync } from 'fw';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { render, type Rendered } from '../../../framework/testing';
import { AppendLog } from '$lib/api/appendLog';
import type { RunEventHandlers } from '$lib/api/client';
import { hub } from '$lib/api/index';
import { MockApi } from '$lib/api/mock';
import type { LogLine, RunCloud } from '$lib/api/types';
import { parse, router } from '$lib/router';
import RunPage from '../Run';
import CloudPanel from './CloudPanel';
import { parseSpotRecovery, SpotRecoveryTracker } from './cloudRecovery';

const GiB = 2 ** 30;
const baseCloud = (extra: Partial<RunCloud> = {}): RunCloud => ({
  connection_id: 'cc_1',
  provider: 'runpod',
  offer: {
    id: 'o',
    gpu: 'RTX 4090',
    gpu_count: 1,
    vram_bytes: 24 * GiB,
    price_per_hour: 0.69,
    region: null,
    detail: 'Community Cloud',
    availability: 'high',
  },
  instance_id: null,
  phase: 'provisioning',
  progress: null,
  max_hours: 12,
  deadline: new Date(Date.now() + 3_600_000).toISOString(),
  cost_usd: 0.42,
  synced: new Date(Date.now() - 60_000).toISOString(),
  ...extra,
});

const log = (message: string, level: LogLine['level'] = 'warn'): LogLine => ({
  t: '2026-11-27T10:00:00Z',
  level,
  message,
});

let target: HTMLElement;
let view: Rendered | undefined;

beforeEach(() => {
  hub.api = new MockApi('normal', 0);
  hub.mode = 'mock';
  target = document.createElement('div');
  document.body.appendChild(target);
});

afterEach(() => {
  view?.unmount();
  view = undefined;
  target.remove();
  vi.restoreAllMocks();
});

describe('parseSpotRecovery & SpotRecoveryTracker', () => {
  it('returns null when no preemption or spot recovery occurred', () => {
    const res = parseSpotRecovery(baseCloud({ phase: 'training' }), { status: 'running', error: null }, [
      log('launching RTX 4090 at RunPod (30 GB disk)', 'info'),
      log('the worker answers: FOLO 0.9.0 training on gpu:0', 'info'),
    ]);
    expect(res).toBeNull();
  });

  it('keeps an attempt pending while releasing the old worker and only switches to resumed once the replacement takes the run', () => {
    const releasingLines = [
      log('releasing the preempted instance before resuming on a new worker', 'info'),
    ];
    // Even if Run.cloud.phase is still 'training' while the old worker is releasing, do NOT report Auto-resumed.
    const releasing = parseSpotRecovery(
      baseCloud({ phase: 'training', instance_id: 'ci_old' }),
      { status: 'running', error: null },
      releasingLines,
    )!;
    expect(releasing.kind).toBe('recovering');
    expect(releasing.summary).toContain('releasing the preempted worker');

    const lines = [
      ...releasingLines,
      log(
        'worker preempted or lost (the worker stopped answering (timeout) and its instance is gone); auto-resuming from synced checkpoint weights/last.folo (epoch 12, attempt 1/3 after 5.0s backoff)',
      ),
    ];

    const recovering = parseSpotRecovery(
      baseCloud({ phase: 'provisioning' }),
      { status: 'running', error: null },
      lines,
    )!;
    expect(recovering.kind).toBe('recovering');
    expect(recovering.attempts).toBe(1);
    expect(recovering.maxAttempts).toBe(3);
    expect(recovering.resumeEpoch).toBe(12);
    expect(recovering.backoffSeconds).toBe(5);
    expect(recovering.badge).toBe('Spot auto-resume 1/3');
    expect(recovering.summary).toContain('not a permanent failure');
    expect(recovering.summary).toContain('epoch 12');

    // Once the server logs that the replacement worker actually took the run, state switches to resumed.
    const resumed = parseSpotRecovery(
      baseCloud({ phase: 'training', instance_id: 'ci_2' }),
      { status: 'running', error: null },
      [...lines, log('the worker took the run as r_worker_2', 'info')],
    )!;
    expect(resumed.kind).toBe('resumed');
    expect(resumed.badge).toBe('Auto-resumed (1×)');
    expect(resumed.summary).toContain('1 spot preemption');
  });

  it('surfaces a failed replacement worker as Spot recovery failed rather than Auto-resumed, preserving earlier recovery counts', () => {
    const failedFirst = parseSpotRecovery(
      baseCloud({ phase: 'released', ended_in: 'booting' }),
      { status: 'failed', error_code: 'cloud_boot_failed', error: 'the instance did not start within 5 minutes' },
      [
        log('worker preempted or lost (gone); auto-resuming from synced checkpoint weights/last.folo (epoch 4, attempt 1/3)'),
      ],
    )!;
    expect(failedFirst.kind).toBe('failed');
    expect(failedFirst.badge).toBe('Spot recovery failed');
    expect(failedFirst.summary).toContain('attempt 1 of 3 failed');
    expect(failedFirst.summary).toContain('Permanent failure');

    // Earlier attempt succeeded (worker took the run), second replacement failed during sync.
    const failedSecond = parseSpotRecovery(
      baseCloud({ phase: 'released', ended_in: 'syncing' }),
      { status: 'failed', error_code: 'cloud_sync_failed', error: 'syncing dataset failed' },
      [
        log('worker preempted or lost (gone); auto-resuming from synced checkpoint weights/last.folo (epoch 4, attempt 1/3)'),
        log('the worker took the run as r_worker_2', 'info'),
        log('worker preempted or lost (gone); auto-resuming from synced checkpoint weights/last.folo (epoch 9, attempt 2/3)'),
      ],
    )!;
    expect(failedSecond.kind).toBe('failed');
    expect(failedSecond.attempts).toBe(2);
    expect(failedSecond.summary).toContain('after 1 earlier recovery');
  });

  it('reports permanent failure distinctly when auto-resume exhausts retries, lacks a checkpoint, or stops/hits deadline during recovery', () => {
    const exhausted = parseSpotRecovery(
      baseCloud({ phase: 'released', ended_in: 'training' }),
      {
        status: 'failed',
        error_code: 'cloud_instance_lost',
        error:
          'the worker stopped answering and its instance is gone; automatic checkpoint resume exhausted all 3 attempt(s) — the weights copied so far stay (resume the run to continue from them)',
      },
      [],
    )!;
    expect(exhausted.kind).toBe('exhausted');
    expect(exhausted.attempts).toBe(3);
    expect(exhausted.maxAttempts).toBe(3);
    expect(exhausted.badge).toBe('Auto-resume exhausted');
    expect(exhausted.summary).toContain('Permanent failure');

    const noCkpt = parseSpotRecovery(
      baseCloud({ phase: 'released', ended_in: 'training' }),
      {
        status: 'failed',
        error_code: 'cloud_instance_lost',
        error:
          'the worker no longer has the run; no synced checkpoint (weights/last.folo) is available to auto-resume from',
      },
      [],
    )!;
    expect(noCkpt.kind).toBe('no_checkpoint');
    expect(noCkpt.badge).toBe('Preempted · no checkpoint');
    expect(noCkpt.summary).toContain('no resumable synced checkpoint');

    const stopped = parseSpotRecovery(
      baseCloud({ phase: 'released', ended_in: 'provisioning' }),
      { status: 'stopped', error: null },
      [
        log('worker preempted or lost (gone); auto-resuming from synced checkpoint weights/last.folo (epoch 4, attempt 2/3)'),
        log('the time limit of 12 h ended the run during preemption recovery', 'info'),
      ],
    )!;
    expect(stopped.kind).toBe('stopped');
    expect(stopped.badge).toBe('Stopped during spot recovery');
    expect(stopped.summary).toContain('attempt 2 of 3');
  });

  it('resets prior preemption/exhausted/stopped history across manual-resume boundaries so unrelated later runs or errors are never poisoned', () => {
    const logs = [
      log('worker preempted or lost (gone); auto-resuming from synced checkpoint weights/last.folo (epoch 4, attempt 1/1)'),
      log('the worker stopped answering; automatic checkpoint resume exhausted all 1 attempt(s)'),
      log('resuming from weights/last.folo', 'info'),
      log('launching RTX 4090 at RunPod (30 GB disk)', 'info'),
    ];
    // During manual resume provisioning, must NOT show "Spot auto-resume 1/1".
    expect(parseSpotRecovery(baseCloud({ phase: 'provisioning' }), { status: 'running', error: null }, logs)).toBeNull();
    // If the manually resumed run later fails with an unrelated error (e.g. OOM), must NOT show "Auto-resume exhausted".
    expect(
      parseSpotRecovery(
        baseCloud({ phase: 'released', ended_in: 'training' }),
        { status: 'failed', error_code: 'oom', error: 'CUDA out of memory' },
        logs,
      ),
    ).toBeNull();
  });

  it('tracks appended SSE log lines incrementally, memoizes unchanged results to avoid 10 Hz DOM churn, and preserves recovery across same-run tail truncation', () => {
    const appendLog = new AppendLog<LogLine>();
    const tracker = new SpotRecoveryTracker();
    const cloud = baseCloud({ phase: 'provisioning' });
    const run = { id: 'r_spot_1', status: 'running' as const, error: null };

    expect(tracker.sync(appendLog, cloud, run)).toBeNull();
    appendLog.push(
      log('worker preempted or lost (gone); auto-resuming from synced checkpoint weights/last.folo (epoch 7, attempt 1/3)'),
    );
    const first = tracker.sync(appendLog, cloud, run)!;
    expect(first.attempts).toBe(1);

    // Repeating sync on 10 Hz training ticks without state change returns the exact same object reference.
    appendLog.push(log('step 100 loss 0.42', 'info'));
    expect(tracker.sync(appendLog, cloud, run)).toBe(first);

    appendLog.push(log('the worker took the run as r_w2', 'info'));
    const resumed = tracker.sync(appendLog, baseCloud({ phase: 'training', instance_id: 'ci_2' }), run)!;
    expect(resumed.kind).toBe('resumed');

    // Reconnect snapshot for the SAME run where old preemption lines aged out of the 2,000-line tail preserves history.
    appendLog.replace([log('step 5000 loss 0.12', 'info')]);
    expect(tracker.sync(appendLog, baseCloud({ phase: 'training', instance_id: 'ci_2' }), run)).toBe(resumed);

    // Switching to a different run resets cleanly.
    appendLog.replace([log('launching RTX 4090 at RunPod (30 GB disk)', 'info')]);
    expect(tracker.sync(appendLog, cloud, { id: 'r_spot_2', status: 'running', error: null })).toBeNull();
  });
});

describe('CloudPanel & Run spot preemption recovery UI', () => {
  const text = (el: Element | null | undefined) => el?.textContent?.replace(/\s+/g, ' ').trim() ?? '';
  const fact = (label: string) =>
    text([...target.querySelectorAll('#cloud .facts > div')].find((d) => text(d.querySelector('dt')) === label)?.querySelector('dd'));

  it('renders recovering badge, non-failure status copy, and attempt counter in CloudPanel', () => {
    const recovery = parseSpotRecovery(baseCloud({ phase: 'booting' }), { status: 'running', error: null }, [
      log('worker preempted or lost (gone); auto-resuming from synced checkpoint weights/last.folo (epoch 12, attempt 1/3 after 5.0s backoff)'),
    ])!;
    view = render(
      () => (
        <CloudPanel
          cloud={baseCloud({ phase: 'booting' })}
          provider="RunPod"
          run={{ status: 'running' }}
          recovery={recovery}
        />
      ),
      target,
    );
    flushSync();

    expect(text(target.querySelector('[data-spot-recovery-badge="recovering"]'))).toBe('Spot auto-resume 1/3');
    expect(text(target.querySelector('[data-spot-recovery="recovering"]'))).toContain('not a permanent failure');
    expect(fact('Spot recoveries')).toBe('1 of 3 · epoch 12');
  });

  it('renders permanent failure alerts distinctly for exhausted auto-resume and failed replacement workers', () => {
    const recovery = parseSpotRecovery(
      baseCloud({ phase: 'released', ended_in: 'booting' }),
      { status: 'failed', error_code: 'cloud_boot_failed', error: 'boot failed' },
      [log('worker preempted or lost (gone); auto-resuming from synced checkpoint weights/last.folo (epoch 5, attempt 1/3)')],
    )!;
    view = render(
      () => (
        <CloudPanel
          cloud={baseCloud({ phase: 'released', ended_in: 'booting' })}
          provider="RunPod"
          run={{ status: 'failed', error_code: 'cloud_boot_failed' }}
          recovery={recovery}
        />
      ),
      target,
    );
    flushSync();

    expect(text(target.querySelector('[data-spot-recovery-badge="failed"]'))).toBe('Spot recovery failed');
    const line = target.querySelector('[data-spot-recovery="failed"]')!;
    expect(line.getAttribute('role')).toBe('alert');
    expect(text(line)).toContain('Permanent failure');
    expect(fact('Spot recoveries')).toBe('1 of 3 · failed');
  });

  it('surfaces live SSE spot preemption recovery on the Run page and switches to resumed once the replacement takes the run', async () => {
    const api = new MockApi('cloud-training', 0);
    let handlers: RunEventHandlers = {};
    vi.spyOn(api, 'runEvents').mockImplementation((_id, h) => {
      handlers = h;
      return () => {};
    });
    const run = (await api.runs()).find((r) => r.cloud && r.status === 'running')!;
    hub.api = api;
    router.route = parse(`#/runs/${run.id}`);
    view = render(() => <RunPage />, target);

    await vi.waitFor(() => {
      flushSync();
      if (!target.querySelector('#cloud .phases')) throw new Error('waiting for CloudPanel');
    });

    // Server emits preemption log + transitions Run.cloud back to provisioning.
    handlers.cloud?.({ ...run.cloud!, phase: 'provisioning', instance_id: null, progress: null });
    handlers.log?.(
      log('worker preempted or lost (instance gone); auto-resuming from synced checkpoint weights/last.folo (epoch 5, attempt 1/3 after 5.0s backoff)'),
    );
    await vi.waitFor(() => {
      flushSync();
      if (!target.querySelector('[data-spot-recovery="recovering"]')) throw new Error('waiting for spot recovery UI');
    });
    expect([...target.querySelectorAll('.status-row .badge')].map(text)).toContain('Spot auto-resume 1/3');
    expect(fact('Spot recoveries')).toBe('1 of 3 · epoch 5');

    // Replacement worker takes the run & enters training -> header warning badge clears, CloudPanel shows Auto-resumed.
    handlers.log?.(log('the worker took the run as r_worker_replacement', 'info'));
    handlers.cloud?.({ ...run.cloud!, phase: 'training', instance_id: 'ci_replacement' });
    await vi.waitFor(() => {
      flushSync();
      if (!target.querySelector('[data-spot-recovery="resumed"]')) throw new Error('waiting for resumed UI');
    });
    expect([...target.querySelectorAll('.status-row .badge')].map(text)).not.toContain('Spot auto-resume 1/3');
    expect(text(target.querySelector('[data-spot-recovery-badge="resumed"]'))).toBe('Auto-resumed (1×)');
  });
});
