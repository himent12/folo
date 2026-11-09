// @vitest-environment happy-dom
import { flushSync, mount, unmount } from 'svelte';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { hub } from '$lib/api/index.svelte';
import { MockApi, type Scenario } from '$lib/api/mock';
import { parse, router } from '$lib/router.svelte';
import Playground from './Playground.svelte';

let target: HTMLElement;
let c: ReturnType<typeof mount> | undefined;

async function until<T>(get: () => T | null | undefined | false, what = 'the condition'): Promise<T> {
  let v: T | null | undefined | false = undefined;
  await vi.waitFor(
    () => {
      flushSync();
      v = get();
      if (!v) throw new Error(`still waiting for ${what}`);
    },
    { timeout: 4000, interval: 10 },
  );
  return v as T;
}

/** Opens the Playground on `model` and runs it on the first sample image. */
async function run(scenario: Scenario, model: string) {
  const api = new MockApi(scenario, 0);
  api.imageUrl = async () => 'data:,';
  hub.api = api;
  hub.mode = 'mock';
  target = document.createElement('div');
  document.body.appendChild(target);
  router.route = parse(`#/playground?model=${model}`);
  c = mount(Playground, { target });
  (await until(() => target.querySelector<HTMLButtonElement>('button.sample'), 'a sample image')).click();
}

afterEach(() => {
  if (c) unmount(c);
  c = undefined;
  target?.remove();
});

describe('Playground: a failed prediction', () => {
  it('shows the error under the image, says so in the caption and moves focus to it', async () => {
    await run('oom', 'm_shapes_s');
    const notice = await until(() => target.querySelector('.notice'), 'the notice');
    // Under the image, not over it: the stage holds only the image.
    expect(target.querySelector('.stage .notice, .stage [role="alert"]')).toBeNull();
    expect(notice.previousElementSibling?.classList.contains('stage')).toBe(true);
    const title = notice.querySelector<HTMLElement>('.title')!;
    expect(title.textContent).toBe('Prediction failed');
    expect(document.activeElement).toBe(title);
    expect(target.querySelector('.vfoot')?.textContent?.replace(/\s+/g, ' ').trim()).toContain('Prediction failed·out of GPU memory');
    expect(target.querySelector('.vfoot')?.textContent).not.toContain('Waiting');
    // The ways out: this browser first (primary), then the smaller model, then Retry.
    expect([...notice.querySelectorAll('button')].map((b) => b.textContent?.trim()).filter((t) => t !== 'Technical details')).toEqual([
      'Run in this browser',
      'Use shapes-n-best folo-n',
      'Retry',
    ]);
  });
});

describe('Playground: detections', () => {
  it('shows scores with one precision in the table and the picked line', async () => {
    await run('normal', 'm_hard_s');
    const row = await until(() => target.querySelector<HTMLTableRowElement>('#pg-panel tbody tr'), 'the detections');
    const cell = row.querySelector('.score .num')!.textContent!;
    expect(cell).toMatch(/^0\.\d\d$/);
    row.click();
    flushSync();
    expect(row.getAttribute('aria-label')).toContain(`score ${cell},`);
    expect(target.querySelector('.pick-line')?.textContent).toContain(cell);
  });

  it('describes IoU as suppression', async () => {
    await run('normal', 'm_hard_s');
    await until(() => target.querySelector('#pg-panel tbody tr'), 'the detections');
    expect(target.querySelector('.thresholds')?.textContent).toContain('Suppress lower-scoring boxes of the same class above this overlap.');
    expect(target.querySelector('.thresholds')?.textContent).not.toContain('Merge');
  });
});
