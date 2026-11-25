// @vitest-environment happy-dom
import { flush, signal } from 'fw';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { hub } from '$lib/api/index';
import { parse, router } from '$lib/router';
import { render } from '../../../framework/testing';
import OfflineNotice from './OfflineNotice';

let target: HTMLElement | undefined;
let view: { unmount(): void } | undefined;

afterEach(() => {
  view?.unmount();
  view = undefined;
  target?.remove();
  target = undefined;
  vi.restoreAllMocks();
});

function mount() {
  target = document.body.appendChild(document.createElement('div'));
  view = render(() => <OfflineNotice />, target);
  flush();
  return target;
}

describe('OfflineNotice', () => {
  it('keeps an empty live region mounted while online with no visible actions', () => {
    vi.spyOn(hub, 'offline', 'get').mockReturnValue(false);
    router.route = parse('#/runs');
    const el = mount();
    expect(el.querySelector('.notice')?.classList.contains('idle')).toBe(true);
    expect(el.querySelector('[role="status"]')?.textContent).toBe('');
    expect(el.querySelector('.actions')).toBeNull();
  });

  it('shows Retry and Settings outside the live region when offline, and hides Settings on /settings', () => {
    vi.spyOn(hub, 'offline', 'get').mockReturnValue(true);
    vi.spyOn(hub, 'refused', 'get').mockReturnValue(null);
    router.route = parse('#/runs');
    const el = mount();

    const status = el.querySelector('[role="status"]')!;
    expect(el.querySelector('.long')?.textContent).toBe("Can't reach folo serve. Retrying…");
    expect(el.querySelector('.short')?.textContent).toBe('Offline');
    const retry = el.querySelector<HTMLButtonElement>('button.action')!;
    expect(retry.textContent?.trim()).toBe('Retry');
    const settings = el.querySelector<HTMLAnchorElement>('a.action')!;
    expect(settings.textContent?.trim()).toBe('Settings');
    expect(settings.getAttribute('href')).toBe('#/settings?section=connection');
    // Interactive controls live outside role="status" so route/button state changes do not reannounce them.
    expect(status.contains(retry)).toBe(false);
    expect(status.contains(settings)).toBe(false);

    // Already on Settings: hide the Settings link, keep Retry.
    router.route = parse('#/settings');
    flush();
    expect(el.querySelector('a.action')).toBeNull();
    expect(el.querySelector<HTMLButtonElement>('button.action')?.textContent?.trim()).toBe('Retry');

    // Navigating away brings the Settings link back.
    router.route = parse('#/datasets');
    flush();
    expect(el.querySelector<HTMLAnchorElement>('a.action')?.getAttribute('href')).toBe('#/settings?section=connection');
  });

  it('keeps keyboard focus on Retry during hub.connect(), guards duplicates, and hands focus off on recovery', async () => {
    const offline = signal(true);
    vi.spyOn(hub, 'offline', 'get').mockImplementation(() => offline.get());
    vi.spyOn(hub, 'refused', 'get').mockReturnValue(null);
    let resolveConnect!: () => void;
    const connect = vi.spyOn(hub, 'connect').mockImplementation(
      () =>
        new Promise<void>((resolve) => {
          // Real hub.connect() clears offline while probing; notice should stay visible until it settles.
          offline.set(false);
          resolveConnect = resolve;
        }),
    );

    router.route = parse('#/runs');
    target = document.body.appendChild(document.createElement('div'));
    view = render(
      () => (
        <header>
          <OfflineNotice />
          <div class="actions">
            <button type="button" class="search">
              Search
            </button>
          </div>
        </header>
      ),
      target,
    );
    flush();

    const retry = target.querySelector<HTMLButtonElement>('.notice button.action')!;
    retry.focus();
    retry.click();
    flush();

    expect(connect).toHaveBeenCalledTimes(1);
    expect(target.querySelector('.notice')?.classList.contains('idle')).toBe(false);
    expect(retry.disabled).toBe(false);
    expect(retry.getAttribute('aria-disabled')).toBe('true');
    expect(retry.getAttribute('aria-busy')).toBe('true');
    expect(retry.textContent?.trim()).toBe('Retrying…');
    expect(document.activeElement).toBe(retry);

    // Duplicate click while busy is ignored.
    retry.click();
    expect(connect).toHaveBeenCalledTimes(1);

    // Still offline when connect finishes: Retry remains enabled and focused.
    offline.set(true);
    resolveConnect();
    await Promise.resolve();
    flush();
    expect(target.querySelector('.notice')?.classList.contains('idle')).toBe(false);
    expect(retry.hasAttribute('aria-disabled')).toBe(false);
    expect(retry.hasAttribute('aria-busy')).toBe(false);
    expect(retry.textContent?.trim()).toBe('Retry');
    expect(document.activeElement).toBe(retry);

    // Back online when connect finishes: hides notice and moves focus to a stable topbar control.
    retry.click();
    flush();
    resolveConnect();
    await Promise.resolve();
    flush();
    expect(target.querySelector('.notice')?.classList.contains('idle')).toBe(true);
    expect(target.querySelector('.notice .actions')).toBeNull();
    expect(document.activeElement).toBe(target.querySelector('button.search'));
  });

  it('preserves refusal wording across Retry while hub.connect() clears hub.refused mid-probe', async () => {
    const offline = signal(true);
    const refused = signal<string | null>('This server refuses access (403).');
    vi.spyOn(hub, 'offline', 'get').mockImplementation(() => offline.get());
    vi.spyOn(hub, 'refused', 'get').mockImplementation(() => refused.get());
    let resolveConnect!: () => void;
    vi.spyOn(hub, 'connect').mockImplementation(
      () =>
        new Promise<void>((resolve) => {
          offline.set(false);
          refused.set(null);
          resolveConnect = resolve;
        }),
    );

    router.route = parse('#/models');
    const el = mount();
    expect(el.querySelector('.long')?.textContent).toBe('This server refuses access (403). Retrying…');
    expect(el.querySelector('.short')?.textContent).toBe('Not connected');
    expect(el.querySelector('a.action')?.getAttribute('href')).toBe('#/settings?section=connection');

    // Mid-retry: refusal wording does not flicker back to "Can't reach folo serve" / "Offline".
    el.querySelector<HTMLButtonElement>('button.action')!.click();
    flush();
    expect(el.querySelector('.long')?.textContent).toBe('This server refuses access (403). Retrying…');
    expect(el.querySelector('.short')?.textContent).toBe('Not connected');

    offline.set(true);
    refused.set('This server refuses access (403).');
    resolveConnect();
    await Promise.resolve();
    flush();
    expect(el.querySelector('.long')?.textContent).toBe('This server refuses access (403). Retrying…');
  });
});
