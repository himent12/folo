// @vitest-environment happy-dom
import { flushSync, mount, unmount } from 'svelte';
import { afterEach, describe, expect, it } from 'vitest';
import { versusYolo26n } from '$lib/benchmarks';
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
      expect(cells.map((td) => td.querySelector('.cl')?.textContent)).toEqual(['folo-n', 'YOLO26n', 'FOLO']);
      for (const td of cells) {
        const ids = (td.getAttribute('headers') ?? '').split(' ');
        expect(ids[0]).toBe(metric?.id);
        const col = target.querySelector(`#${ids[1]}`);
        expect(col?.getAttribute('scope')).toBe('col');
        expect(col?.textContent).toBe(td.querySelector('.cl')?.textContent);
      }
    });
  });
});
