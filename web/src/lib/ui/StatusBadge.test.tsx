// @vitest-environment happy-dom
// A cloud run in its hand-over is not "Queued" (docs/CLOUD.md): while it is provisioned, boots or uploads its data the
// badge says that phase, in the neutral look of a waiting run, wherever a status is shown. Once its worker has it, or
// for a run of this machine, the badge says the status as before. What is not a run (a labeling job) passes its own
// `state`: the mark of a run status with its own word.
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { afterEach, describe, expect, it } from 'vitest';
import type { RunCloud, RunStatus } from '$lib/api/types';
import { render as mount, type Rendered } from '../../../framework/testing';
import StatusBadge, { statusWord, type StatusBadgeProps } from './StatusBadge';

const run = (status: RunStatus, phase?: string) => ({ status, cloud: phase === undefined ? undefined : ({ phase } as RunCloud) });

let c: Rendered | undefined;
afterEach(() => {
  if (c) c.unmount();
  c = undefined;
});

function render(props: StatusBadgeProps) {
  if (c) c.unmount();
  const target = document.createElement('div');
  c = mount(() => <StatusBadge {...props} />, target);
  return target.querySelector<HTMLElement>('.badge')!;
}

describe('StatusBadge', () => {
  it('says the phase of a cloud run in its hand-over, the status otherwise', () => {
    expect(statusWord(run('queued', 'provisioning'))).toBe('Provisioning');
    expect(statusWord(run('queued', 'booting'))).toBe('Booting');
    expect(statusWord(run('queued', 'syncing'))).toBe('Uploading data');
    expect(statusWord(run('queued', 'training'))).toBe('Queued');
    expect(statusWord(run('queued'))).toBe('Queued');
    expect(statusWord(run('running', 'training'))).toBe('Running');
    expect(statusWord(run('completed', 'released'))).toBe('Completed');
  });

  it("draws the hand-over as a waiting run with the phase's word, and another thing's state with its own word", () => {
    const pill = render({ run: run('queued', 'syncing') });
    expect(pill.classList.contains('queued')).toBe(true);
    expect(pill.textContent?.trim()).toBe('Uploading data');
    // A dot alone still says its word to assistive tech.
    expect(render({ run: run('queued', 'provisioning'), plain: true, dotOnly: true }).querySelector('.sr-only')?.textContent).toBe('Provisioning');
    const job = render({ state: { status: 'running', label: 'Labeling' }, plain: true });
    expect(job.classList.contains('running')).toBe(true);
    expect(job.textContent?.trim()).toBe('Labeling');
  });

  it('uses a compositor-only pseudo-element pulse on live running badges and stills it when held or under reduced motion', () => {
    const live = render({ run: run('running') });
    expect(live.classList.contains('running')).toBe(true);
    expect(live.classList.contains('held')).toBe(false);
    expect(live.querySelector('.dot')).not.toBeNull();

    const held = render({ run: run('running'), lastKnown: 'Last known: Running · 2m ago' });
    expect(held.classList.contains('running')).toBe(true);
    expect(held.classList.contains('held')).toBe(true);

    const cssText = readFileSync(resolve(import.meta.dirname, 'StatusBadge.tsx'), 'utf8').replace(/\/\*[\s\S]*?\*\//g, '');
    expect(cssText).not.toContain('live-pulse');
    expect(cssText).toContain('.running:not(.held) .dot::after');
    expect(cssText).toContain('@keyframes status-badge-pulse');
    expect(cssText).toContain('will-change: transform, opacity');
    expect(cssText).toMatch(/@media \(prefers-reduced-motion: reduce\)[\s\S]*content: none/);
  });
});
