// @vitest-environment happy-dom
import { flushSync, mount, unmount } from 'svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { hub } from '$lib/api/index.svelte';
import { MockApi } from '$lib/api/mock';
import { prefs } from '$lib/prefs.svelte';
import { parse, router } from '$lib/router.svelte';
import Settings from './Settings.svelte';

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
    { timeout: 3000, interval: 10 },
  );
  return v as T;
}

beforeEach(() => {
  target = document.createElement('div');
  document.body.appendChild(target);
  router.route = parse('#/settings');
  hub.mode = 'mock';
});

afterEach(() => {
  if (c) unmount(c);
  c = undefined;
  target.remove();
});

describe('Settings', () => {
  it('says which theme is current in its own card and in its radio’s name', () => {
    hub.api = new MockApi('normal', 0);
    prefs.theme = 'light';
    c = mount(Settings, { target });
    flushSync();
    const radios = [...target.querySelectorAll<HTMLInputElement>('input[name="theme"]')];
    expect(radios.map((r) => r.getAttribute('aria-label'))).toEqual(['Dark', 'Light, current', 'System']);
    // "Current" sits inside the chosen card's label, under its name.
    const current = target.querySelector('.theme .current')!;
    expect(current.closest('label')?.querySelector('input')?.value).toBe('light');
    expect(current.parentElement?.querySelector('.theme-name')?.textContent).toBe('Light');
  });

  it('gives each fix one remedy: the shell, the command on one unwrapped line, the driver report folded', async () => {
    hub.api = new MockApi('no-tensor-cores', 0);
    c = mount(Settings, { target });
    const fixes = await until(() => {
      const f = [...target.querySelectorAll('#system .cap-fix')];
      return f.length ? f : null;
    }, 'the System panel');
    const tc = fixes.find((f) => f.textContent?.includes('--tensor-cores'))!;
    expect(tc.querySelector('.cause')?.textContent).toBe('Restart the server on Vulkan to use them.');
    expect(tc.querySelector('.shell')?.textContent).toBe('In PowerShell:');
    const line = tc.querySelector('pre.line')!;
    expect(line.textContent).toBe("$env:FOLO_GPU_BACKEND='vulkan'; folo serve --tensor-cores");
    expect(line.hasAttribute('data-scrolls')).toBe(true);
    expect(tc.querySelector('button[aria-label="Copy command"]')).not.toBeNull();
    const report = target.querySelector<HTMLDetailsElement>('#system details.report')!;
    expect(report.open).toBe(false);
    expect(report.querySelector('summary')?.textContent).toBe('Driver report');
  });
});
