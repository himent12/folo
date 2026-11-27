// @vitest-environment happy-dom
import { flush as flushSync, tick } from 'fw';
import { render } from '../../../framework/testing';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { comparable, unlike, unlikeCondition, versusYolo26n } from '$lib/benchmarks';
import { MockApi } from '$lib/api/mock';
import { pendingBlobRevokeCount, revokePendingBlobUrls } from '$lib/tableExport';
import { toast } from '$lib/ui/toast';
import Versus from './Versus';
import { versusCsv, versusMarkdown } from './versusExport';

/** Port of `mount(C, { target, props })`: renders into `target`; `unmount(c)` is `c.unmount()`. */
const mount = <P extends object>(C: (p: P) => JSX.Element, o: { target: HTMLElement; props?: P }) => render(() => C(o.props ?? ({} as P)), o.target);
const unmount = (c: { unmount(): void }) => c.unmount();

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
  revokePendingBlobUrls();
  toast.items = [];
  vi.useRealTimers();
  vi.restoreAllMocks();
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

  it('copies a complete Markdown report table and downloads CSV with safe blob cleanup', async () => {
    const table = await new MockApi('normal', 0).throughput();
    const md = versusMarkdown(table);
    expect(md).toContain('| Measured | folo-n | YOLO26n | folo-n vs YOLO26n | Notes |');
    expect(md).toContain('GPU forward, batch 1 (ms)');
    expect(md).toContain('## Measured on this server');
    expect(md).toContain('5.90 ms');

    const csv = versusCsv(table);
    expect(csv.headers).toEqual(['source', 'device', 'metric', 'unit', 'folo_n', 'yolo26n', 'comparison', 'note']);
    expect(csv.rows).toHaveLength(versusYolo26n.length + 2);
    expect(versusCsv(null).rows).toHaveLength(versusYolo26n.length);
    // Numeric cells in CSV stay raw numbers (no grouping commas or repeated units).
    expect(csv.rows[0]![4]).toBe(2.99);
    expect(csv.rows[1]![4]).toBe(1385);
    expect(typeof csv.rows.at(-2)![4]).toBe('number');
    expect(typeof csv.rows.at(-1)![4]).toBe('number');

    let copiedText = '';
    Object.defineProperty(navigator, 'clipboard', {
      configurable: true,
      value: {
        writeText: vi.fn(async (s: string) => {
          copiedText = s;
        }),
      },
    });
    const created: Blob[] = [];
    const revoked: string[] = [];
    vi.spyOn(URL, 'createObjectURL').mockImplementation((b) => {
      created.push(b as Blob);
      return `blob:versus/${created.length}`;
    });
    vi.spyOn(URL, 'revokeObjectURL').mockImplementation((u) => revoked.push(u));
    const click = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {});

    c = mount(Versus, { target, props: { table } });
    flushSync();

    const copyBtn = [...target.querySelectorAll<HTMLButtonElement>('header button')].find((b) => b.textContent?.includes('Copy Markdown'))!;
    copyBtn.click();
    await vi.waitFor(() => {
      flushSync();
      expect(copiedText).toBe(md);
      expect(copyBtn.textContent).toContain('Copied');
    });

    const csvBtn = [...target.querySelectorAll<HTMLButtonElement>('header button')].find((b) => b.textContent?.includes('Download CSV'))!;
    csvBtn.click();
    flushSync();
    expect(click).toHaveBeenCalledTimes(1);
    expect(created).toHaveLength(1);
    const csvText = await created[0]!.text();
    expect(csvText).toContain('folo_n,yolo26n');
    expect(csvText).toContain('1385,1454');
    expect(pendingBlobRevokeCount()).toBe(1);

    revokePendingBlobUrls();
    expect(pendingBlobRevokeCount()).toBe(0);
    expect(revoked).toEqual(['blob:versus/1']);

    // Hostile device_name starting with '=' is neutralized in CSV, and clipboard failure shows an error toast.
    unmount(c);
    const hostileTable = {
      ...table,
      device_name: '=CMD()|calc',
      inference: table.inference.map((r) => ({ ...r, measured_on: '=CMD()|calc' })),
      training: table.training.map((r) => ({ ...r, measured_on: '=CMD()|calc' })),
    };
    Object.defineProperty(navigator, 'clipboard', {
      configurable: true,
      value: { writeText: vi.fn(async () => Promise.reject(new Error('denied'))) },
    });
    c = mount(Versus, { target, props: { table: hostileTable } });
    flushSync();
    const copyBtn2 = [...target.querySelectorAll<HTMLButtonElement>('header button')].find((b) => b.textContent?.includes('Copy Markdown'))!;
    copyBtn2.click();
    await vi.waitFor(() => {
      flushSync();
      expect(toast.items.some((t) => t.title === 'Could not copy Markdown' && t.tone === 'error')).toBe(true);
    });
    const csvBtn2 = [...target.querySelectorAll<HTMLButtonElement>('header button')].find((b) => b.textContent?.includes('Download CSV'))!;
    csvBtn2.click();
    flushSync();
    expect(await created[1]!.text()).toContain("'=CMD()|calc");
    revokePendingBlobUrls();
    expect(pendingBlobRevokeCount()).toBe(0);

    // CPU-only server throughput reports CPU latency rather than mislabeling it as GPU in UI, Markdown, and CSV.
    unmount(c);
    const cpuTable = {
      ...table,
      device: 'cpu',
      device_name: 'Intel Core i7-14700KF',
      inference: table.inference.map((r) => ({ ...r, device: 'cpu', measured_on: 'Intel Core i7-14700KF' })),
      training: [],
    };
    expect(versusMarkdown(cpuTable)).toContain('CPU end to end, batch 1');
    expect(versusCsv(cpuTable).rows.at(-1)![2]).toBe('CPU end to end, batch 1');
    c = mount(Versus, { target, props: { table: cpuTable } });
    flushSync();
    expect(target.querySelector('tbody.here')?.textContent).toContain('CPU end to end, batch 1');
  });
});
