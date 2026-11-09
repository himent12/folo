// @vitest-environment happy-dom
import { flushSync, mount, tick, unmount } from 'svelte';
import { afterEach, describe, expect, it } from 'vitest';
import { comparable, unlike, versusYolo26n } from '$lib/benchmarks';
import { MockApi } from '$lib/api/mock';
import Versus from './Versus.svelte';

let c: ReturnType<typeof mount> | undefined;
let target: HTMLElement;

afterEach(() => {
  if (c) unmount(c);
  c = undefined;
  target.remove();
});

describe('Engine speed table', () => {
  it('ties every value to its metric and engine through headers, with the engine named in the markup', () => {
    target = document.createElement('div');
    document.body.appendChild(target);
    c = mount(Versus, { target });
    flushSync();
    const rows = [...target.querySelectorAll('tbody tr')];
    expect(rows).toHaveLength(versusYolo26n.length);
    rows.forEach((tr, i) => {
      const metric = tr.querySelector('th[scope="row"]');
      expect(metric?.textContent).toContain(versusYolo26n[i]!.metric);
      const cells = [...tr.querySelectorAll('td')];
      expect(cells.map((td) => td.querySelector('.cl')?.textContent)).toEqual(['folo-n', 'YOLO26n', 'folo-n vs YOLO26n']);
      for (const td of cells) {
        const ids = (td.getAttribute('headers') ?? '').split(' ');
        expect(ids[0]).toBe(metric?.id);
        const col = target.querySelector(`#${ids[1]}`);
        expect(col?.getAttribute('scope')).toBe('col');
        expect(col?.textContent).toBe(td.querySelector('.cl')?.textContent);
      }
    });
  });

  it('states a winner only on like-for-like rows: × for speed, % for memory', () => {
    target = document.createElement('div');
    document.body.appendChild(target);
    c = mount(Versus, { target });
    flushSync();
    const leads = [...target.querySelectorAll('tbody td.lead')].map((td) => [...td.childNodes].slice(1).map((n) => n.textContent?.trim()).filter(Boolean).join(' | '));
    versusYolo26n.forEach((v, i) => {
      if (!comparable(v)) expect(leads[i]).toBe(`Different conditions | ${unlike[v.metric]!.differs}`);
      else if (v.unit === 'MiB') expect(leads[i]).toMatch(/^\d+% (less|more) memory$/);
      else expect(leads[i]).toMatch(/^\d+\.\d\d× (faster|slower)$/);
    });
    // No row measured under different conditions claims a ratio anywhere in the table.
    target.querySelectorAll('tbody tr.unlike').forEach((tr) => expect(tr.textContent).not.toMatch(/faster|slower|less memory|more memory/));
  });

  it('qualifies the comparison beside it and names the specific difference on each affected row', async () => {
    target = document.createElement('div');
    document.body.appendChild(target);
    window.matchMedia ??= (() => ({ matches: false })) as unknown as typeof window.matchMedia;
    Element.prototype.scrollIntoView ??= () => {};
    c = mount(Versus, { target });
    flushSync();
    const flagged = [...target.querySelectorAll('tbody tr.unlike')].map((tr) => ({
      m: tr.querySelector('.m')?.textContent,
      why: tr.querySelector('.lead .why')?.textContent,
    }));
    expect(flagged).toEqual([
      { m: 'GPU forward throughput, batch 16', why: 'f32 vs TF32' },
      { m: 'GPU end to end, batch 1', why: 'less NMS work for FOLO' },
      { m: 'GPU end-to-end throughput, batch 16', why: 'f32 vs fp16' },
      { m: 'CPU forward, batch 1', why: 'FOLO under other load' },
      { m: 'CPU end to end, batch 1', why: 'less NMS work for FOLO' },
      { m: 'Training peak GPU memory', why: 'train step vs trainer' },
    ]);
    expect(target.querySelector('.lede .qual')?.textContent).toContain(`6 of ${versusYolo26n.length} rows ran under different conditions`);
    const how = target.querySelector<HTMLDetailsElement>('details.how')!;
    expect(how.open).toBe(false);
    target.querySelector<HTMLButtonElement>('.lede button')!.click();
    await tick();
    expect(how.open).toBe(true);
    expect(document.activeElement).toBe(how.querySelector('summary'));
  });

  it("adds this server's own folo-n figures as a group of their own, said to be measured here and compared with nothing", async () => {
    target = document.createElement('div');
    document.body.appendChild(target);
    const table = await new MockApi('normal', 0).throughput();
    c = mount(Versus, { target, props: { table } });
    flushSync();
    const groups = target.querySelectorAll('table.vs tbody');
    expect(groups).toHaveLength(2);
    // The reference rows are unchanged; the lede says which they are.
    expect(groups[0]!.querySelectorAll('tr')).toHaveLength(versusYolo26n.length);
    expect(target.querySelector('.lede')?.textContent).toContain("the reference figures, with this server's own folo-n figures last");
    const here = groups[1]!;
    expect(here.querySelector('.here-h')?.textContent?.replace(/\s+/g, ' ').trim()).toBe('Measured on this server RTX 4080 SUPER');
    const rows = [...here.querySelectorAll('tr:not(.here-h)')];
    expect(rows.map((tr) => [...tr.querySelectorAll('td')].map((td) => td.lastChild?.textContent))).toEqual([
      ['5.90 ms', 'not run here', 'Measured here'],
      ['212 img/s', 'not run here', 'Measured here'],
    ]);
    // Each value still names its metric and column.
    for (const td of here.querySelectorAll('td')) expect(target.querySelector(`#${td.getAttribute('headers')!.split(' ')[1]}`)?.getAttribute('scope')).toBe('col');
  });

  it('shows only the reference without a table (older servers)', () => {
    target = document.createElement('div');
    document.body.appendChild(target);
    c = mount(Versus, { target, props: { table: null } });
    flushSync();
    expect(target.querySelectorAll('table.vs tbody')).toHaveLength(1);
    expect(target.querySelector('.lede')?.textContent).not.toContain('this server');
  });
});
