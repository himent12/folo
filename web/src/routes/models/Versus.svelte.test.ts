// @vitest-environment happy-dom
import { flushSync, mount, tick, unmount } from 'svelte';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { comparable, unlike, unlikeCondition, versusYolo26n } from '$lib/benchmarks';
import { MockApi } from '$lib/api/mock';
import Versus from './Versus.svelte';

let c: ReturnType<typeof mount> | undefined;
let target: HTMLElement;

beforeEach(() => {
  target = document.createElement('div');
  document.body.appendChild(target);
  window.matchMedia ??= (() => ({ matches: false })) as unknown as typeof window.matchMedia;
  Element.prototype.scrollIntoView ??= () => {};
});

afterEach(() => {
  if (c) unmount(c);
  c = undefined;
  target.remove();
});

describe('Engine speed table', () => {
  it('ties every value to its metric and engine through headers, and states a winner only on like-for-like rows', async () => {
    c = mount(Versus, { target });
    flushSync();
    const rows = [...target.querySelectorAll('tbody tr')];
    expect(rows).toHaveLength(versusYolo26n.length);
    rows.forEach((tr, i) => {
      const metric = tr.querySelector('th[scope="row"]');
      expect(metric?.textContent).toContain(versusYolo26n[i]!.metric);
      const cells = [...tr.querySelectorAll('td')];
      expect(cells.map((td) => target.querySelector(`#${(td.getAttribute('headers') ?? '').split(' ')[1]}`)?.textContent)).toEqual([
        'folo-n',
        'YOLO26n',
        'folo-n vs YOLO26n',
      ]);
      for (const td of cells) {
        const ids = (td.getAttribute('headers') ?? '').split(' ');
        expect(ids[0]).toBe(metric?.id);
        expect(target.querySelector(`#${ids[1]}`)?.getAttribute('scope')).toBe('col');
      }
    });

    // × for speed, % for memory; a row measured under different conditions names its difference and claims no ratio.
    const leads = [...target.querySelectorAll('tbody td.lead')].map((td) =>
      [...td.childNodes].map((n) => n.textContent?.replace(/ /g, ' ').trim()).filter(Boolean).join(' | '),
    );
    versusYolo26n.forEach((v, i) => {
      if (!comparable(v)) expect(leads[i]).toBe(`— | ${unlike[v.metric]!.differs} | — ${unlikeCondition(unlike[v.metric]!)}`);
      else if (v.unit === 'MiB') expect(leads[i]).toMatch(/^\d+% (less|more) memory$/);
      else expect(leads[i]).toMatch(/^\d+\.\d\d× (faster|slower)$/);
    });
    target.querySelectorAll('tbody tr.unlike').forEach((tr) => expect(tr.textContent).not.toMatch(/faster|slower|less memory|more memory/));

    // The legend's button opens How this was measured and moves focus there.
    const how = target.querySelector<HTMLDetailsElement>('details.how')!;
    expect(how.open).toBe(false);
    target.querySelector<HTMLButtonElement>('.legend button')!.click();
    await tick();
    expect(how.open).toBe(true);
    expect(document.activeElement).toBe(how.querySelector('summary'));
  });

  it("adds this server's own folo-n figures as a group of their own, compared with nothing, and only with a table", async () => {
    const table = await new MockApi('normal', 0).throughput();
    c = mount(Versus, { target, props: { table } });
    flushSync();
    const groups = target.querySelectorAll('table.vs tbody');
    expect(groups).toHaveLength(2);
    // The reference rows are unchanged.
    expect(groups[0]!.querySelectorAll('tr')).toHaveLength(versusYolo26n.length);
    const rows = [...groups[1]!.querySelectorAll('tr:not(.here-h)')];
    expect(rows.map((tr) => [...tr.querySelectorAll('td')].map((td) => td.lastChild?.textContent))).toEqual([
      ['5.90 ms', 'not run here', 'Measured here'],
      ['212 img/s', 'not run here', 'Measured here'],
    ]);
    // Each value still names its metric and column.
    for (const td of groups[1]!.querySelectorAll('td')) expect(target.querySelector(`#${td.getAttribute('headers')!.split(' ')[1]}`)?.getAttribute('scope')).toBe('col');

    // Older servers: the reference alone.
    unmount(c);
    c = mount(Versus, { target, props: { table: null } });
    flushSync();
    expect(target.querySelectorAll('table.vs tbody')).toHaveLength(1);
  });
});
