// The status pill is the title form (DESIGN_DIRECTION 6): only beside the name of the one run a screen is about. Every
// other status (tables, lists, Compare) is dot + word (`plain`), so one job never has two looks. Either form, and every
// status word written as text, is read from the run itself, so a cloud run in its hand-over says its phase everywhere.
// What is not a run (a labeling job) passes its `state`: a status with its own word, never a run's bare status.
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join, relative, resolve, sep } from 'node:path';
import { parse } from 'svelte/compiler';
import { describe, expect, it } from 'vitest';

const SRC = resolve(process.cwd(), 'src');
const TITLE_FORM = new Set(['routes/Run.svelte', 'routes/overview/LiveRunCard.svelte']);

function files(dir: string, ext = /\.svelte$/, out: string[] = []): string[] {
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) files(p, ext, out);
    else if (ext.test(name)) out.push(p);
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

/** Every `<StatusBadge>` in a component's markup. */
function badges(source: string): Node[] {
  const out: Node[] = [];
  const seen = new WeakSet<object>();
  const walk = (n: unknown) => {
    if (!n || typeof n !== 'object' || seen.has(n)) return;
    seen.add(n);
    if (Array.isArray(n)) return n.forEach(walk);
    const node = n as Node;
    if (node.type === 'Component' && node.name === 'StatusBadge') out.push(node);
    for (const v of Object.values(node)) walk(v);
  };
  walk(parse(source, { modern: true }).fragment);
  return out;
}

/** Every `<StatusBadge>` in a component's markup that renders as a pill, as its source text. */
const pills = (source: string): string[] => badges(source).filter((b) => !isPlain(b)).map((b) => source.slice(b.start, b.end));

/** A badge attribute's value as one expression tag (`run={r}`, `{run}`), else undefined (a string, a bare attribute). */
function expressionOf(badge: Node, name: string): Node | undefined {
  const attr = (badge.attributes ?? []).find((a) => a.type === 'Attribute' && a.name === name);
  const v = attr?.value as Node | Node[] | true | undefined;
  const tag = Array.isArray(v) ? (v.length === 1 ? v[0] : undefined) : v === true ? undefined : v;
  return tag?.type === 'ExpressionTag' ? tag : undefined;
}

/**
 * Every `<StatusBadge>` not certainly given a run as it came from the API or another thing's own status: neither `run`
 * nor `state`, both, one rebuilt in place (`run={{ status: r.status }}`, `state={{ status: r.status, label: … }}`),
 * which drops the cloud hand-over and says "Queued" where it should say the phase, or a bare `status` (the props of
 * before, which took a run's). `state` itself is typed to need its own word, so a run cannot be passed as one.
 */
function rebuilt(source: string): string[] {
  return badges(source)
    .filter((b) => {
      const attrs = b.attributes ?? [];
      if (attrs.some((a) => a.type === 'SpreadAttribute' || (a.type === 'Attribute' && (a.name === 'status' || a.name === 'label')))) return true;
      const given = ['run', 'state'].map((name) => expressionOf(b, name)).filter((t) => t !== undefined);
      const named = attrs.filter((a) => a.type === 'Attribute' && (a.name === 'run' || a.name === 'state'));
      return named.length !== 1 || given.length !== 1 || given[0]!.expression?.type === 'ObjectExpression';
    })
    .map((b) => source.slice(b.start, b.end));
}

/** `statusWord` called on an object built in place rather than the run, as source text. */
const rebuiltWords = (source: string): string[] => [...source.matchAll(/\bstatusWord\(\s*\{[^)]*\)/g)].map((m) => m[0]);

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
    for (const ok of ['<StatusBadge run={r} plain />', '<StatusBadge run={r} plain dotOnly />', '<StatusBadge run={r} plain={true} />'])
      expect(pills(ok), ok).toEqual([]);
    for (const bad of [
      '<StatusBadge run={r} />',
      '<StatusBadge run={r} plain={false} />',
      '<StatusBadge run={r} plain={flag} />',
      '<StatusBadge run={r} {plain} />',
      '<StatusBadge run={r} plain="false" />',
      '<StatusBadge run={r} plain {...rest} />',
      '<StatusBadge run={r} title="plain" />',
      '<StatusBadge run={r}></StatusBadge>',
      '<td>{#if x}<StatusBadge run={r} plain={false} />{/if}</td>',
    ])
      expect(pills(bad), bad).toHaveLength(1);
  });

  it("reads every status from the run itself, so a cloud run in its hand-over says its phase (other things: their own word)", () => {
    const hits: string[] = [];
    for (const f of files(SRC, /\.(?:svelte|ts)$/)) {
      if (/\.test\.ts$/.test(f)) continue;
      const rel = relative(SRC, f).split(sep).join('/');
      const source = readFileSync(f, 'utf8');
      if (f.endsWith('.svelte')) for (const b of rebuilt(source)) hits.push(`${rel}: ${b}`);
      for (const w of rebuiltWords(source)) hits.push(`${rel}: ${w}`);
    }
    expect(hits).toEqual([]);
  });

  it('tells a run passed whole from one rebuilt without its cloud state', () => {
    for (const ok of [
      '<StatusBadge {run} />',
      '<StatusBadge run={r} plain />',
      '<StatusBadge run={live.run ?? r} plain />',
      '<StatusBadge plain state={st} />',
      '<StatusBadge plain state={jobStatus(j)} lastKnown={x} />',
    ])
      expect(rebuilt(ok), ok).toEqual([]);
    for (const bad of [
      '<StatusBadge run={{ status: r.status }} plain />',
      '<StatusBadge plain />',
      '<StatusBadge run="queued" plain />',
      '<StatusBadge {...r} plain />',
      '<StatusBadge plain status={r.status} />',
      '<StatusBadge plain status={r.status} label={statusWord(r)} />',
      '<StatusBadge plain state={{ status: r.status, label: "Queued" }} />',
      '<StatusBadge plain state="queued" />',
      '<StatusBadge run={r} state={st} plain />',
    ])
      expect(rebuilt(bad), bad).toHaveLength(1);
    expect(rebuiltWords('statusWord(x); statusWord({ status: x.status })')).toEqual(['statusWord({ status: x.status })']);
  });
});
