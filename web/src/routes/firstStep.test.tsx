// @vitest-environment happy-dom
// x2-newcomer-2: on an empty server the Runs, Models and Playground empty states asked for New run, which only led to
// "Add a dataset first". With no dataset they offer Add dataset (as Desktop's screens/first_step.rs); with one, New run.
import { flush as flushSync, type Component } from 'fw';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { render } from '../../framework/testing';
import { hub } from '$lib/api/index';
import { MockApi } from '$lib/api/mock';
import type { Dataset } from '$lib/api/types';
import { parse, router } from '$lib/router';
import { datasetsChanged } from './datasets/changes';
import Models from './Models';
import Overview from './Overview';
import Playground from './Playground';
import Runs from './Runs';

type Mounted = { unmount(): void };
let target: HTMLElement | undefined;
let c: Mounted | undefined;

afterEach(() => {
  c?.unmount();
  c = undefined;
  target?.remove();
});

function open(page: Component<Record<string, never>>, path: string, api: MockApi) {
  hub.api = api;
  hub.mode = 'mock';
  history.replaceState(history.state, '', `#${path}`);
  router.route = parse(`#${path}`);
  target = document.createElement('div');
  document.body.appendChild(target);
  const P = page;
  c = render(() => <P />, target);
  flushSync();
}

/** The empty state's action, once the page shows it. */
const action = () =>
  vi.waitFor(() => {
    flushSync();
    const el = target!.querySelector<HTMLAnchorElement>('.empty .actions a');
    if (!el) throw new Error('still waiting for the empty state');
    return el;
  });

/** Every answer already given is taken in: the microtasks it queued have run, and the page has redrawn. */
const settled = async () => {
  await new Promise((r) => setTimeout(r, 0));
  flushSync();
};

const PAGES: [string, Component<Record<string, never>>, string][] = [
  ['Runs', Runs, '/runs'],
  ['Models', Models, '/models'],
  ['Playground', Playground, '/playground'],
];

describe('the first step on an empty server', () => {
  for (const [name, page, path] of PAGES) {
    it(`${name}: no dataset yet asks for one, with the folder dialog`, async () => {
      open(page, path, new MockApi('empty', 0));
      const a = await vi.waitFor(async () => {
        const el = await action();
        if (!el.textContent?.includes('Add dataset')) throw new Error('the datasets have not answered yet');
        return el;
      });
      expect(a.getAttribute('href')).toBe('#/datasets?add=folder');
      expect(target!.textContent).toContain('dataset first');
    });

    it(`${name}: with a dataset, New run`, async () => {
      const api = new MockApi('empty', 0);
      const [one] = await new MockApi('normal', 0).datasets();
      let answered!: () => void;
      const asked = new Promise<void>((r) => (answered = r));
      api.datasets = async () => {
        answered();
        return [one!];
      };
      open(page, path, api);
      // The datasets have answered (with one) and the page has taken it in: the action is New run, not the loading one.
      await asked;
      await settled();
      const a = await action();
      expect(a.textContent).toContain('New run');
      expect(a.getAttribute('href')).toBe('#/runs/new');
    });

    // Review round 1: the runs and models answer before the datasets do; the action follows once those say "none".
    it(`${name}: datasets answering last still turn the action into Add dataset`, async () => {
      const api = new MockApi('empty', 0);
      let answer!: (v: never[]) => void;
      api.datasets = () => new Promise((r) => (answer = r));
      open(page, path, api);
      const before = await action();
      expect(before.textContent).toContain('New run');
      answer([]);
      await settled();
      const a = await action();
      expect(a.textContent).toContain('Add dataset');
      expect(a.getAttribute('href')).toBe('#/datasets?add=folder');
    });

    // Review round 1: the first dataset added in this tab or another one (the tab shown again) turns it into New run.
    for (const how of ['a dataset action in this tab', 'the tab shown again'] as const) {
      it(`${name}: the first dataset, after ${how}, turns Add dataset into New run`, async () => {
        const api = new MockApi('empty', 0);
        const [one] = await new MockApi('normal', 0).datasets();
        let list: Dataset[] = [];
        api.datasets = async () => list;
        open(page, path, api);
        await vi.waitFor(async () => {
          if (!(await action()).textContent?.includes('Add dataset')) throw new Error('the datasets have not answered yet');
        });
        list = [one!];
        if (how === 'the tab shown again') document.dispatchEvent(new Event('visibilitychange'));
        else datasetsChanged();
        const a = await vi.waitFor(async () => {
          const el = await action();
          if (!el.textContent?.includes('New run')) throw new Error('still Add dataset');
          return el;
        });
        expect(a.getAttribute('href')).toBe('#/runs/new');
      });
    }
  }
});

// Review round 1: with past runs but no dataset left, the header's create action of Runs and Overview asks for a
// dataset too (Desktop's screens/runs/list.rs and overview/page.rs gate theirs).
describe('the header action with runs but no dataset', () => {
  for (const [name, page, path] of [
    ['Runs', Runs, '/runs'],
    ['Overview', Overview, '/'],
  ] as const) {
    it(`${name}: Add dataset, not New run`, async () => {
      const api = new MockApi('normal', 0);
      api.datasets = async () => [];
      open(page as Component<Record<string, never>>, path, api);
      await vi.waitFor(() => {
        flushSync();
        if (!target!.querySelector('a[href="#/datasets?add=folder"]')) throw new Error('no Add dataset yet');
      });
      expect(target!.querySelector('a[href="#/runs/new"]')).toBeNull();
    });

    // Review round 2: the header follows datasets added or deleted in another tab once this one is shown again.
    for (const change of ['the last dataset deleted', 'the first dataset added'] as const) {
      it(`${name}: ${change} in another tab turns the header action over`, async () => {
        const api = new MockApi('normal', 0);
        const [one] = await new MockApi('normal', 0).datasets();
        const deleting = change === 'the last dataset deleted';
        let list: Dataset[] = deleting ? [one!] : [];
        let answered!: () => void;
        const asked = new Promise<void>((r) => (answered = r));
        api.datasets = async () => {
          answered();
          return list;
        };
        open(page as Component<Record<string, never>>, path, api);
        await asked;
        await settled();
        const header = (want: string) =>
          vi.waitFor(() => {
            flushSync();
            const hrefs = [...target!.querySelectorAll('header .actions a')].map((a) => a.getAttribute('href'));
            if (!hrefs.includes(want)) throw new Error(`header has ${hrefs.join(', ')}, not ${want}`);
            return hrefs;
          });
        await header(deleting ? '#/runs/new' : '#/datasets?add=folder');
        list = deleting ? [] : [one!];
        document.dispatchEvent(new Event('visibilitychange'));
        const after = await header(deleting ? '#/datasets?add=folder' : '#/runs/new');
        expect(after).not.toContain(deleting ? '#/runs/new' : '#/datasets?add=folder');
      });
    }
  }
});
