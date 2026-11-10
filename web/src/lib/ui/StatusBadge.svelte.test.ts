// @vitest-environment happy-dom
// A cloud run in its hand-over is not "Queued" (docs/CLOUD.md): while it is provisioned, boots or uploads its data the
// badge says that phase, in the neutral look of a waiting run, wherever a status is shown. Once its worker has it, or
// for a run of this machine, the badge says the status as before. What is not a run (a labeling job) passes its own
// `state`: the mark of a run status with its own word.
import { mount, unmount, type ComponentProps } from 'svelte';
import { afterEach, describe, expect, it } from 'vitest';
import type { RunCloud, RunStatus } from '$lib/api/types';
import StatusBadge, { statusWord } from './StatusBadge.svelte';

const run = (status: RunStatus, phase?: string) => ({ status, cloud: phase === undefined ? undefined : ({ phase } as RunCloud) });

let c: ReturnType<typeof mount> | undefined;
afterEach(() => {
  if (c) unmount(c);
  c = undefined;
});

function render(props: ComponentProps<typeof StatusBadge>) {
  if (c) unmount(c);
  const target = document.createElement('div');
  c = mount(StatusBadge, { target, props });
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
});
