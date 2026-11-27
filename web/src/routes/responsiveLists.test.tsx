// @vitest-environment happy-dom
// Verifies that list screens (Runs, Models, Datasets, Benchmarks, LabelJobs) mount only ONE responsive DOM tree
// (desktop table/cards OR phone rows) based on the reactive phone media query, swap cleanly on live resize,
// preserve empty/error state frames on phones, and avoid hidden mosaic fetches on phone Datasets.
import { flush as flushSync, type Component } from 'fw';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { render, type Rendered } from '../../framework/testing';
import { hub } from '$lib/api/index';
import { MockApi } from '$lib/api/mock';
import { parse, router } from '$lib/router';
import Benchmarks from './Benchmarks';
import Datasets from './Datasets';
import LabelJobs from './LabelJobs';
import Models from './Models';
import Runs from './Runs';

interface ControlledMedia {
  setPhone: (isPhone: boolean) => void;
  restore: () => void;
}

/** Stubs `window.matchMedia` with a live controllable `(max-width: 767px)` query that notifies listeners on change. */
function installMediaControl(initialPhone = false): ControlledMedia {
  const real = window.matchMedia;
  let phone = initialPhone;
  const listeners = new Set<() => void>();
  window.matchMedia = ((query: string) => {
    const isPhoneQuery = query.includes('max-width: 767px');
    return {
      get matches() {
        return isPhoneQuery ? phone : false;
      },
      media: query,
      addEventListener(_type: string, fn: () => void) {
        if (isPhoneQuery) listeners.add(fn);
      },
      removeEventListener(_type: string, fn: () => void) {
        if (isPhoneQuery) listeners.delete(fn);
      },
    };
  }) as unknown as typeof window.matchMedia;
  return {
    setPhone(next: boolean) {
      if (phone === next) return;
      phone = next;
      for (const fn of [...listeners]) fn();
      flushSync();
    },
    restore() {
      window.matchMedia = real;
    },
  };
}

let target: HTMLElement;
let c: Rendered | undefined;
let mediaCtl: ControlledMedia | undefined;

beforeEach(() => {
  hub.api = new MockApi('normal', 0);
  hub.mode = 'mock';
  target = document.createElement('div');
  document.body.appendChild(target);
});

afterEach(() => {
  c?.unmount();
  c = undefined;
  target.remove();
  mediaCtl?.restore();
  mediaCtl = undefined;
  vi.restoreAllMocks();
});

function mountScreen(Comp: Component<Record<string, never>>, path: string) {
  router.route = parse(`#${path}`);
  c = render(() => <Comp />, target);
  flushSync();
}

const until = <T,>(get: () => T | null | undefined | false, what: string) =>
  vi.waitFor(() => {
    flushSync();
    const v = get();
    if (!v) throw new Error(`still waiting for ${what}`);
    return v;
  });

const SCREENS: {
  name: string;
  Comp: Component<Record<string, never>>;
  path: string;
  desktopSel: string;
  phoneSel: string;
}[] = [
  { name: 'Runs', Comp: Runs, path: '/runs', desktopSel: 'table.runs tbody tr[data-row]', phoneSel: 'ul.rows li' },
  { name: 'Models', Comp: Models, path: '/models', desktopSel: 'table.models tbody tr[data-row]', phoneSel: 'div.rows section li' },
  { name: 'Datasets', Comp: Datasets, path: '/datasets', desktopSel: 'ul.cards > li.card', phoneSel: '.phone-list ul.rows li.m-row' },
  { name: 'Benchmarks', Comp: Benchmarks, path: '/benchmarks', desktopSel: 'table.benches tbody tr[data-row]', phoneSel: 'ul.rows li' },
  { name: 'LabelJobs', Comp: LabelJobs, path: '/datasets/label', desktopSel: 'table.jobs tbody tr[data-row]', phoneSel: 'ul.rows li.m-row' },
];

describe('single-DOM responsive lists', () => {
  for (const s of SCREENS) {
    it(`${s.name}: mounts only desktop OR phone list DOM and swaps cleanly on live resize`, async () => {
      mediaCtl = installMediaControl(false);
      mountScreen(s.Comp, s.path);

      await until(() => target.querySelector(s.desktopSel), `${s.name} desktop rows`);
      expect(target.querySelectorAll(s.desktopSel).length).toBeGreaterThan(0);
      expect(target.querySelector(s.phoneSel)).toBeNull();

      // Resize down to phone: desktop DOM unmounts and phone DOM mounts with the same items.
      mediaCtl.setPhone(true);
      await until(() => target.querySelector(s.phoneSel), `${s.name} phone rows`);
      expect(target.querySelector(s.desktopSel)).toBeNull();
      expect(target.querySelectorAll(s.phoneSel).length).toBeGreaterThan(0);

      // Resize back to desktop: phone DOM unmounts and desktop DOM returns.
      mediaCtl.setPhone(false);
      await until(() => target.querySelector(s.desktopSel), `${s.name} desktop rows restored`);
      expect(target.querySelector(s.phoneSel)).toBeNull();
    });

    it(`${s.name}: keeps its empty state visible on phones without mounting duplicate rows`, async () => {
      hub.api = new MockApi('empty', 0);
      mediaCtl = installMediaControl(true);
      mountScreen(s.Comp, s.path);

      await until(() => target.querySelector('.empty, .state'), `${s.name} empty state on phone`);
      expect(target.querySelector(s.phoneSel)).toBeNull();
      expect(target.querySelector(s.desktopSel)).toBeNull();
    });
  }

  it('Models: mounts only Segmented on desktop and only the Group Select picker on phone', async () => {
    mediaCtl = installMediaControl(false);
    mountScreen(Models, '/models');
    await until(() => target.querySelector('table.models tbody tr[data-row]'), 'models desktop');
    expect(target.querySelector('.lt-filter .seg')).not.toBeNull();
    expect(target.querySelector('.lt-filter .ds-pick')).toBeNull();

    mediaCtl.setPhone(true);
    await until(() => target.querySelector('.lt-filter .ds-pick select'), 'models phone picker');
    expect(target.querySelector('.lt-filter .seg')).toBeNull();
  });

  it('Datasets: avoids mounting desktop DatasetCard mosaics on phones (zero datasetImages calls)', async () => {
    const realIO = globalThis.IntersectionObserver;
    vi.stubGlobal('IntersectionObserver', undefined);
    try {
      const api = new MockApi('normal', 0);
      const imagesSpy = vi.spyOn(api, 'datasetImages');
      hub.api = api;
      mediaCtl = installMediaControl(true);
      mountScreen(Datasets, '/datasets');

      await until(() => target.querySelectorAll('.phone-list ul.rows li.m-row').length > 0, 'phone dataset rows');
      expect(target.querySelector('ul.cards')).toBeNull();
      expect(imagesSpy).not.toHaveBeenCalled();
    } finally {
      vi.stubGlobal('IntersectionObserver', realIO);
    }
  });

  it('Benchmarks: exposes a phone-only sort control with direction toggle that reorders rows and syncs URL', async () => {
    mediaCtl = installMediaControl(false);
    mountScreen(Benchmarks, '/benchmarks');
    await until(() => target.querySelector('table.benches tbody tr[data-row]'), 'benchmarks desktop');
    expect(target.querySelector('#benches-sort')).toBeNull();

    mediaCtl.setPhone(true);
    const select = await until(() => target.querySelector<HTMLSelectElement>('#benches-sort'), 'benchmarks phone sort select');
    const dirBtn = target.querySelector<HTMLButtonElement>('.lt-sort button')!;
    expect(dirBtn).not.toBeNull();

    const rowNames = () => [...target.querySelectorAll('ul.rows li .name')].map((el) => el.textContent?.trim());

    // Switch sort key to Images -> defaults to descending (most images first) and updates URL.
    select.value = 'images';
    select.dispatchEvent(new Event('change', { bubbles: true }));
    flushSync();
    expect(router.route.query.get('sort')).toBe('images');
    expect(router.route.query.get('dir')).toBe('desc');
    const descOrder = rowNames();
    expect(descOrder.length).toBeGreaterThan(1);

    // Toggle direction button -> flips to ascending (fewest images first) and moves the largest benchmark to the end.
    dirBtn.click();
    flushSync();
    expect(router.route.query.get('dir')).toBe('asc');
    const ascOrder = rowNames();
    expect(ascOrder).not.toEqual(descOrder);
    expect(ascOrder[ascOrder.length - 1]).toBe(descOrder[0]);
  });
});
