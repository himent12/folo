// @vitest-environment happy-dom
// A cloud run in its hand-over is not "Queued" (docs/CLOUD.md): while it is provisioned, boots or uploads its data the
// badge says that phase, in the neutral look of a waiting run, wherever a status is shown. Once its worker has it, or
// for a run of this machine, the badge says the status as before.
import { mount, unmount } from 'svelte';
import { afterEach, describe, expect, it } from 'vitest';
import type { RunCloud, RunStatus } from '$lib/api/types';
import StatusBadge, { statusWord } from './StatusBadge.svelte';

const run = (status: RunStatus, phase?: string) => ({ status, cloud: phase === undefined ? undefined : ({ phase } as RunCloud) });

let c: ReturnType<typeof mount> | undefined;
afterEach(() => {
  if (c) unmount(c);
  c = undefined;
});

function render(props: { run: ReturnType<typeof run>; plain?: boolean; dotOnly?: boolean; lastKnown?: string }) {
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

  it("draws the hand-over as a waiting run: neutral, the hollow dot, the phase's word", () => {
    const pill = render({ run: run('queued', 'syncing') });
    expect(pill.classList.contains('queued')).toBe(true);
    expect(pill.textContent?.trim()).toBe('Uploading data');
    unmount(c!);
    c = undefined;
    const cell = render({ run: run('queued', 'provisioning'), plain: true, dotOnly: true });
    expect(cell.classList.contains('queued')).toBe(true);
    expect(cell.querySelector('.sr-only')?.textContent).toBe('Provisioning');
  });

  it('keeps a last known status in its own words', () => {
    expect(render({ run: run('queued', 'booting'), lastKnown: 'Last known: Booting · 2m ago' }).textContent?.trim()).toBe('Last known: Booting · 2m ago');
  });
});
