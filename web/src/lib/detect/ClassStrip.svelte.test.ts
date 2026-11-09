// @vitest-environment happy-dom
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { flushSync, mount, unmount } from 'svelte';
import { describe, expect, it } from 'vitest';
import ClassStrip from './ClassStrip.svelte';

describe('ClassStrip', () => {
  it('is one "Class balance" disclosure that opens a name / count / share breakdown, with no native titles', () => {
    // Round 26: the segments' names and counts were only in native hover titles.
    const target = document.createElement('div');
    document.body.appendChild(target);
    const c = mount(ClassStrip, { target, props: { counts: [300, 0, 100, 600], classes: ['bolt', 'nut', 'washer', 'screw'] } });
    flushSync();
    const buttons = target.querySelectorAll('button');
    expect(buttons).toHaveLength(1);
    const b = buttons[0]!;
    expect(b.textContent).toContain('Class balance');
    expect(b.getAttribute('aria-expanded')).toBe('false');
    expect(target.querySelector('[title]')).toBeNull();
    b.click();
    flushSync();
    expect(b.getAttribute('aria-expanded')).toBe('true');
    const pop = document.getElementById(b.getAttribute('aria-controls')!)!;
    const rows = [...pop.querySelectorAll('[role=listitem]')].map((r) => r.textContent!.replace(/\s+/g, ' ').trim());
    // Largest first; a class with no boxes is not listed.
    expect(rows).toEqual(['screw 600 60%', 'bolt 300 30%', 'washer 100 10%']);
    unmount(c);
  });

  it('is never put inside a link or a label (a button cannot nest there)', () => {
    const root = resolve(__dirname, '../..');
    const files: string[] = [];
    const walk = (d: string) => {
      for (const f of readdirSync(d)) {
        const p = join(d, f);
        if (statSync(p).isDirectory()) walk(p);
        else if (f.endsWith('.svelte')) files.push(p);
      }
    };
    walk(root);
    const users = files.filter((f) => readFileSync(f, 'utf8').includes('<ClassStrip'));
    expect(users.length).toBeGreaterThanOrEqual(3);
    for (const f of users) {
      const src = readFileSync(f, 'utf8');
      for (let at = src.indexOf('<ClassStrip'); at >= 0; at = src.indexOf('<ClassStrip', at + 1)) {
        const before = src.slice(0, at);
        for (const tag of ['a', 'label']) {
          const open = Math.max(before.lastIndexOf(`<${tag} `), before.lastIndexOf(`<${tag}>`));
          const close = before.lastIndexOf(`</${tag}>`);
          expect(open < close || open < 0, `${f}: <ClassStrip> inside <${tag}>`).toBe(true);
        }
      }
    }
  });
});
