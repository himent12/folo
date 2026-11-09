// The status pill is the title form (DESIGN_DIRECTION 6): only beside the name of the one run a screen is about. Every
// other status (tables, lists, Compare) is dot + word (`plain`), so one job never has two looks.
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join, relative, resolve, sep } from 'node:path';
import { parse } from 'svelte/compiler';
import { describe, expect, it } from 'vitest';

const SRC = resolve(process.cwd(), 'src');
const TITLE_FORM = new Set(['routes/Run.svelte', 'routes/overview/LiveRunCard.svelte']);

function files(dir: string, out: string[] = []): string[] {
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) files(p, out);
    else if (name.endsWith('.svelte')) out.push(p);
  }
  return out;
}

interface Node {
  type?: string;
  name?: string;
  start?: number;
  end?: number;
  attributes?: Node[];
  value?: unknown;
  expression?: Node;
}

/**
 * Whether a `<StatusBadge>` is certainly plain: a bare `plain` or `plain={true}`. `plain={false}`, a value only known at
 * run time (`plain={flag}`, `{plain}`), a string, a spread that could set it, or no `plain` at all is a pill.
 */
function isPlain(badge: Node): boolean {
  const attrs = badge.attributes ?? [];
  if (attrs.some((a) => a.type === 'SpreadAttribute')) return false;
  const plain = attrs.find((a) => a.type === 'Attribute' && a.name === 'plain');
  if (!plain) return false;
  if (plain.value === true) return true;
  const v = plain.value as Node | Node[];
  const tag = Array.isArray(v) ? (v.length === 1 ? v[0] : undefined) : v;
  return tag?.type === 'ExpressionTag' && tag.expression?.type === 'Literal' && tag.expression.value === true;
}

/** Every `<StatusBadge>` in a component's markup that renders as a pill, as its source text. */
function pills(source: string): string[] {
  const out: string[] = [];
  const seen = new WeakSet<object>();
  const walk = (n: unknown) => {
    if (!n || typeof n !== 'object' || seen.has(n)) return;
    seen.add(n);
    if (Array.isArray(n)) return n.forEach(walk);
    const node = n as Node;
    if (node.type === 'Component' && node.name === 'StatusBadge' && !isPlain(node)) out.push(source.slice(node.start, node.end));
    for (const v of Object.values(node)) walk(v);
  };
  walk(parse(source, { modern: true }).fragment);
  return out;
}

describe('status pills', () => {
  it('appear only as the title form; every other status is dot + word', () => {
    const hits: string[] = [];
    for (const f of files(SRC)) {
      const rel = relative(SRC, f).split(sep).join('/');
      if (TITLE_FORM.has(rel)) continue;
      for (const p of pills(readFileSync(f, 'utf8'))) hits.push(`${rel}: ${p}`);
    }
    expect(hits).toEqual([]);
  });

  it('reads the value of `plain`, not just the word', () => {
    for (const ok of ['<StatusBadge status={s} plain />', '<StatusBadge status={s} plain dotOnly />', '<StatusBadge status={s} plain={true} />'])
      expect(pills(ok), ok).toEqual([]);
    for (const bad of [
      '<StatusBadge status={s} />',
      '<StatusBadge status={s} plain={false} />',
      '<StatusBadge status={s} plain={flag} />',
      '<StatusBadge status={s} {plain} />',
      '<StatusBadge status={s} plain="false" />',
      '<StatusBadge status={s} plain {...rest} />',
      '<StatusBadge status={s} title="plain" />',
      '<StatusBadge status={s}></StatusBadge>',
      '<td>{#if x}<StatusBadge status={s} plain={false} />{/if}</td>',
    ])
      expect(pills(bad), bad).toHaveLength(1);
  });
});
