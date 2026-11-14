// @vitest-environment happy-dom
// The shell of design/UNIFIED.md §1 and the shared parts of §2 it owns: the Runs badge counts training and waiting
// runs (the icon rail and the tab bar mark a training run with a lime dot and name the counts), the top bar shows
// crumbs on a nested page only and never a New run, the foot's line says the data source and the version, a detail
// header puts its status badge on the title line, and an architecture is a name (sans) where an id stays mono.
import { flush } from 'fw';
import { afterEach, describe, expect, it } from 'vitest';
import { render } from '../../../framework/testing';
import { hub } from '$lib/api/index';
import { MockApi } from '$lib/api/mock';
import type { Run } from '$lib/api/types';
import { prefs } from '$lib/prefs';
import { parse, router } from '$lib/router';
import Ident, { identKind } from '$lib/ui/Ident';
import StatusBadge from '$lib/ui/StatusBadge';
import ConnectionStatus from './ConnectionStatus';
import { setCrumbs } from './crumbs';
import Page from './Page';
import { watched } from './runList';
import Sidebar from './Sidebar';
import { sidebar } from './sidebarState';
import TabBar from './TabBar';
import Topbar from './Topbar';

let view: { unmount(): void } | undefined;
let target: HTMLElement | undefined;
afterEach(() => {
  view?.unmount();
  view = undefined;
  target?.remove();
  watched.runs = undefined;
  prefs.sidebarCollapsed = false;
});

function mount(f: () => JSX.Element) {
  target = document.body.appendChild(document.createElement('div'));
  view = render(f, target);
  flush();
  return target;
}

async function someRuns(training: number, queued: number) {
  const [any] = await new MockApi('normal', 0).runs();
  const run = (i: number, status: Run['status']) => ({ ...any!, id: `r${i}`, name: `run-${i}`, status, cloud: undefined }) as Run;
  return [
    ...Array.from({ length: training }, (_, i) => run(i, 'running')),
    ...Array.from({ length: queued }, (_, i) => run(training + i, 'queued')),
    run(99, 'completed'),
  ];
}

describe('the Runs badge', () => {
  it('counts the training and the waiting runs at the end of the Runs row, and names them', async () => {
    router.route = parse('#/');
    sidebar.narrow = false;
    prefs.sidebarCollapsed = false;
    const shell = mount(() => <Sidebar />);
    const runsRow = () => shell.querySelector<HTMLAnchorElement>('a[href="#/runs"]')!;
    expect(runsRow().querySelector('.badge')).toBeNull();
    watched.runs = await someRuns(1, 2);
    flush();
    expect(runsRow().querySelector('.badge')!.textContent).toBe('3');
    expect(runsRow().getAttribute('aria-label')).toBe('Runs, 1 training, 2 queued');
    // No other row has one.
    expect(shell.querySelectorAll('.badge')).toHaveLength(1);
  });

  it('becomes a lime dot on the icon rail while a run trains (the count in the tooltip and the name)', async () => {
    router.route = parse('#/');
    sidebar.narrow = false;
    prefs.sidebarCollapsed = true;
    const shell = mount(() => <Sidebar />);
    watched.runs = await someRuns(0, 2);
    flush();
    const runsRow = () => shell.querySelector<HTMLAnchorElement>('a[href="#/runs"]')!;
    // Waiting is not "now": no lime dot for queued runs alone.
    expect(runsRow().querySelector('.live-dot')).toBeNull();
    expect(runsRow().getAttribute('aria-label')).toBe('Runs, 2 queued');
    watched.runs = await someRuns(1, 0);
    flush();
    expect(runsRow().querySelector('.live-dot')).not.toBeNull();
    expect(runsRow().getAttribute('aria-label')).toBe('Runs, 1 training');
  });

  it('marks the tab bar the same way', async () => {
    router.route = parse('#/');
    const shell = mount(() => <TabBar />);
    watched.runs = await someRuns(1, 1);
    flush();
    const tab = shell.querySelector<HTMLAnchorElement>('a[href="#/runs"]')!;
    expect(tab.querySelector('.live-dot')).not.toBeNull();
    expect(tab.getAttribute('aria-label')).toBe('Runs, 1 training, 1 queued');
  });
});

describe('the top bar', () => {
  it('shows the trail on a nested page only, and never a New run', () => {
    const shell = mount(() => <Topbar onsearch={() => {}} />);
    setCrumbs({ label: 'Runs' });
    flush();
    expect(shell.querySelector('.crumbs')!.classList.contains('top')).toBe(true);
    setCrumbs({ label: 'Runs', path: '/runs' }, { label: 'fasteners-s' });
    flush();
    expect(shell.querySelector('.crumbs')!.classList.contains('top')).toBe(false);
    expect([...shell.querySelectorAll('.crumbs li:not(.sep)')].map((li) => li.textContent?.trim())).toEqual(['Runs', 'fasteners-s']);
    expect([...shell.querySelectorAll('a, button')].some((b) => b.textContent?.includes('New run'))).toBe(false);
  });
});

describe('the sidebar foot', () => {
  it('says the data source and the version on one line', () => {
    hub.mode = 'mock';
    hub.version = '0.1.0';
    const shell = mount(() => <ConnectionStatus collapsed={false} />);
    expect(shell.querySelector('.conn .line')!.textContent).toBe('Demo data · v0.1.0');
    expect(shell.querySelector('.conn .label')!.textContent).toBe('Demo data');
  });
});

describe('the detail header', () => {
  it("puts a status badge on the title's line, after the name", () => {
    const shell = mount(() => (
      <Page title="fasteners-s" badge={<StatusBadge run={{ status: 'running' }} />}>
        <p>body</p>
      </Page>
    ));
    const line = shell.querySelector('.title-line')!;
    expect(line.querySelector('h1')!.textContent?.trim()).toBe('fasteners-s');
    expect(line.querySelector('h1 + .badge-slot .badge')!.textContent).toBe('Running');
  });
});

describe('names and identifiers', () => {
  it('sets an architecture, and a model named for one, in sans; paths, files, server ids and fingerprints in mono', () => {
    for (const v of ['folo-s', 'folo-n', 'yolo26n', 'folo-s-coco80', 'folo-n-best']) expect(identKind(v), v).toBe('name');
    for (const v of ['best.folo', 'D:/data/fasteners', 'run_7d1e', 'm_fast_n', '9f2c1a7b3e4d5c6f', 'folo-run_c3d5-4c1a']) expect(identKind(v), v).toBe('id');
    const shell = mount(() => (
      <>
        <Ident value="folo-s" />
        <Ident value="run_7d1e" />
        <Ident value="folo-s" kind="id" />
      </>
    ));
    expect([...shell.querySelectorAll('.ident')].map((e) => e.classList.contains('mono'))).toEqual([false, true, true]);
  });
});
