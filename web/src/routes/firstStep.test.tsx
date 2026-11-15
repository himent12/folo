// @vitest-environment happy-dom
// x2-newcomer-2: on an empty server the Runs, Models and Playground empty states asked for New run, which only led to
// "Add a dataset first". With no dataset they offer Add dataset (as Desktop's screens/first_step.rs); with one, New run.
import { flush as flushSync, type Component } from 'fw';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { render } from '../../framework/testing';
import { hub } from '$lib/api/index';
import { MockApi } from '$lib/api/mock';
import { parse, router } from '$lib/router';
import Models from './Models';
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
      api.datasets = async () => [one!];
      open(page, path, api);
      // Let the datasets answer (with one): the action stays New run.
      await action();
      await new Promise((r) => setTimeout(r, 50));
      const a = await action();
      expect(a.textContent).toContain('New run');
      expect(a.getAttribute('href')).toBe('#/runs/new');
    });
  }
});
