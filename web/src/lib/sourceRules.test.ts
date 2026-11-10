// @vitest-environment happy-dom
/// <reference types="node" />
// Source rules whose breach is a bug a person hits and that neither svelte-check, the linters of `npm run check` nor
// the layout QA can see: each reads the components as written. The sources are read once, for every rule.
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join, relative, resolve, sep } from 'node:path';
import { parse } from 'svelte/compiler';
import { describe, expect, it } from 'vitest';

const SRC = resolve(process.cwd(), 'src');

function walk(dir: string, out: string[] = []): string[] {
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) walk(p, out);
    else if (/\.(?:svelte|ts)$/.test(name) && !/\.test\.(?:ts|svelte)$/.test(name)) out.push(p);
  }
  return out;
}

/** Every component and module of the app (no tests), as its path under src/ and its text. */
const sources = walk(SRC).map((f) => ({ rel: relative(SRC, f).split(sep).join('/'), text: readFileSync(f, 'utf8') }));
const components = sources.filter((s) => s.rel.endsWith('.svelte'));

describe('a cloud run says its phase wherever its status shows', () => {
  // A cloud run in its hand-over is queued with a phase of its own (docs/CLOUD.md); a status rebuilt from `r.status`
  // drops the phase and says "Queued". So every `<StatusBadge>` takes the run whole (`run`) or another thing's own
  // status and word (`state`), and `statusWord` is called on the run, never on an object built in place.
  interface Node {
    type?: string;
    name?: string;
    start?: number;
    end?: number;
    attributes?: Node[];
    value?: unknown;
    expression?: Node;
  }

  function badges(source: string): Node[] {
    const out: Node[] = [];
    const seen = new WeakSet<object>();
    const visit = (n: unknown) => {
      if (!n || typeof n !== 'object' || seen.has(n)) return;
      seen.add(n);
      if (Array.isArray(n)) return n.forEach(visit);
      const node = n as Node;
      if (node.type === 'Component' && node.name === 'StatusBadge') out.push(node);
      for (const v of Object.values(node)) visit(v);
    };
    visit(parse(source, { modern: true }).fragment);
    return out;
  }

  /** A badge attribute's value as one expression tag (`run={r}`, `{run}`), else undefined (a string, a bare attribute). */
  function expressionOf(badge: Node, name: string): Node | undefined {
    const attr = (badge.attributes ?? []).find((a) => a.type === 'Attribute' && a.name === name);
    const v = attr?.value as Node | Node[] | true | undefined;
    const tag = Array.isArray(v) ? (v.length === 1 ? v[0] : undefined) : v === true ? undefined : v;
    return tag?.type === 'ExpressionTag' ? tag : undefined;
  }

  /** Every `<StatusBadge>` not given exactly one of `run` or `state` as an expression that is not an object literal. */
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

  const rebuiltWords = (source: string): string[] => [...source.matchAll(/\bstatusWord\(\s*\{[^)]*\)/g)].map((m) => m[0]);

  it('takes the run whole in every StatusBadge and statusWord', () => {
    const hits: string[] = [];
    for (const { rel, text } of sources) {
      if (rel.endsWith('.svelte') && text.includes('<StatusBadge')) for (const b of rebuilt(text)) hits.push(`${rel}: ${b}`);
      for (const w of rebuiltWords(text)) hits.push(`${rel}: ${w}`);
    }
    expect(hits).toEqual([]);
  });

  it('tells a run passed whole from one rebuilt without its cloud state', () => {
    for (const ok of ['<StatusBadge {run} />', '<StatusBadge run={live.run ?? r} plain />', '<StatusBadge plain state={jobStatus(j)} lastKnown={x} />'])
      expect(rebuilt(ok), ok).toEqual([]);
    for (const bad of [
      '<StatusBadge run={{ status: r.status }} plain />',
      '<StatusBadge plain />',
      '<StatusBadge run="queued" plain />',
      '<StatusBadge {...r} plain />',
      '<StatusBadge plain status={r.status} />',
      '<StatusBadge plain state={{ status: r.status, label: "Queued" }} />',
      '<StatusBadge run={r} state={st} plain />',
    ])
      expect(rebuilt(bad), bad).toHaveLength(1);
    expect(rebuiltWords('statusWord(x); statusWord({ status: x.status })')).toEqual(['statusWord({ status: x.status })']);
  });
});

it("keeps the space at a block's edge as {' '}, where Svelte would trim it", () => {
  // Svelte trims the whitespace a block's content starts or ends with: "served from{#if origin} (" rendered "served from(".
  const lead = /(?:\{(?:#if|:else if)\s[^{}]*\}|\{:else\}) +[^\s<{]/;
  const trail = /[^\s>}] +\{(?:\/if|:else if\s[^{}]*|:else|\/each|\/key)\}/;
  const hits: string[] = [];
  for (const { rel, text } of components) {
    text
      .replace(/<script[\s\S]*?<\/script>/g, (m) => m.replace(/[^\n]/g, ''))
      .split('\n')
      .forEach((line, i) => {
        if (lead.test(line) || trail.test(line)) hits.push(`${rel}:${i + 1}: ${line.trim()}`);
      });
  }
  expect(hits).toEqual([]);
});

it('never toasts from a component while its modal dialog is open', () => {
  // The toaster sits behind the modal's scrim, outside its focus trap: a toast fired while a dialog is open is missed
  // and cannot be reached. A toast in a component that owns a modal fires once it is closed (`if (!open)`, or an
  // `…open = false` / `confirm… = false` just above it), or a comment within the 8 lines above says `page-level:` and
  // why. Results inside a dialog go to its status region (`DialogStatus`, or `feedback()`).
  const call = /\btoast\.(success|info|error|show)\(/;
  const closed = /!open\b|\b\w*[oO]pen = false|\bconfirm\w* = false|page-level:/;
  const bad: string[] = [];
  for (const { rel, text } of components) {
    if (!/<Dialog\b|\.showModal\(/.test(text)) continue;
    const lines = text.split('\n');
    lines.forEach((line, i) => {
      if (!call.test(line) || line.trim().startsWith('//') || line.trim().startsWith('*')) return;
      if (!closed.test(lines.slice(Math.max(0, i - 8), i + 1).join('\n'))) bad.push(`${rel}:${i + 1}: ${line.trim()}`);
    });
  }
  expect(bad).toEqual([]);
});

it('never puts a ClassStrip (a button) inside a link or a label, where a click would also follow or toggle it', () => {
  const bad: string[] = [];
  for (const { rel, text } of components) {
    for (let at = text.indexOf('<ClassStrip'); at >= 0; at = text.indexOf('<ClassStrip', at + 1)) {
      const before = text.slice(0, at);
      for (const tag of ['a', 'label']) {
        const open = Math.max(before.lastIndexOf(`<${tag} `), before.lastIndexOf(`<${tag}>`));
        if (open > before.lastIndexOf(`</${tag}>`)) bad.push(`${rel}: <ClassStrip> inside <${tag}>`);
      }
    }
  }
  expect(bad).toEqual([]);
});

describe('touch text', () => {
  // On a coarse pointer every editable control's own text is at least 16 px (`type.lg`), or iOS zooms the page into
  // the field when it takes focus. Controls inherit 12–14 px elsewhere, so a component that renders an editable
  // `<input>`, `<select>` or `<textarea>` carries an `@media (pointer: coarse)` rule setting that element's font-size to
  // `--type-lg-size` or more. A rule counts only if its selector matches the control as rendered (round 27: a scan of
  // selector text alone passed `.unused input`): the markup is parsed into a DOM (Svelte's `{…}` reduced to
  // placeholders, `class:x` applied as classes) and each editable element must `matches()` one of the selectors.

  /** Input types a person types into or picks text in (the rest are toggles, sliders and pickers with no text). */
  const NOT_EDITABLE = /^(checkbox|radio|range|file|color|hidden|button|submit|reset|image)$/;

  /** The bodies of every `@media (pointer: coarse)` block in a stylesheet (braces matched). */
  function coarseBlocks(css: string): string[] {
    const out: string[] = [];
    for (const m of css.matchAll(/@media\s*\(pointer:\s*coarse\)\s*\{/g)) {
      let depth = 1;
      let i = m.index! + m[0].length;
      const start = i;
      for (; i < css.length && depth > 0; i++) {
        if (css[i] === '{') depth++;
        else if (css[i] === '}') depth--;
      }
      out.push(css.slice(start, i - 1));
    }
    return out;
  }

  /** The selectors (`:global()` unwrapped) of the coarse rules that set text to 16 px or more. */
  function touchSelectors(css: string): string[] {
    return coarseBlocks(css.replace(/\/\*[\s\S]*?\*\//g, '')).flatMap((b) =>
      [...b.matchAll(/([^{}]+)\{([^{}]*)\}/g)]
        .filter(([, , body]) => /font-size:\s*var\(--type-(lg|xl|2xl)-size\)/.test(body!))
        .flatMap(([, sel]) => sel!.split(','))
        .map((sel) => sel.replace(/:global\(([^()]*(?:\([^()]*\)[^()]*)*)\)/g, '$1').trim())
        .filter(Boolean),
    );
  }

  /** Every `{…}` group (braces balanced) of a component's markup, replaced by `_`: blocks, expressions, attribute values. */
  function placeholders(markup: string): string {
    let out = '';
    let depth = 0;
    for (const c of markup) {
      if (c === '{') {
        if (depth++ === 0) out += '_';
      } else if (c === '}') depth = Math.max(0, depth - 1);
      else if (depth === 0) out += c;
    }
    return out;
  }

  /**
   * The editable controls of a component's markup, rendered as a DOM: a component's tag renamed (`<Input>` is not an
   * `<input>`: HTML parsing lowercases it), a self-closing tag closed (a parser would nest its siblings inside it) and
   * every `class:x` directive applied as the class `x`.
   */
  function editables(markup: string): Element[] {
    const root = document.createElement('div');
    root.innerHTML = placeholders(markup)
      .replace(/<(\/?)([A-Z][\w.]*|svelte:[\w-]+)/g, '<$1c-$2')
      .replace(/<([A-Za-z][\w:.-]*)([^<>]*?)\/>/g, '<$1$2></$1>');
    for (const el of root.querySelectorAll('*'))
      for (const a of [...el.attributes]) if (a.name.startsWith('class:')) el.classList.add(a.name.slice('class:'.length));
    return [...root.querySelectorAll('input, select, textarea')].filter((el) => !(el.tagName === 'INPUT' && NOT_EDITABLE.test(el.getAttribute('type') ?? '')));
  }

  /** Whether a selector matches `el`; one happy-dom cannot parse matches nothing (and so covers nothing). */
  function matches(el: Element, sel: string): boolean {
    try {
      return el.matches(sel);
    } catch {
      return false;
    }
  }

  const covered = (css: string, markup: string) => {
    const sels = touchSelectors(css);
    return editables(markup).every((el) => sels.some((sel) => matches(el, sel)));
  };

  it('every editable control is at least 16 px on a coarse pointer, by a rule that matches it as rendered', () => {
    const misses: string[] = [];
    let seen = 0;
    for (const { rel, text } of components) {
      if (!/<(?:input|select|textarea)\b/.test(text)) continue;
      const style = /<style[^>]*>([\s\S]*?)<\/style>/.exec(text)?.[1] ?? '';
      // Markup only: a script's comment may name a tag ("Native <input type=range>").
      const markup = text.replace(/<(style|script)\b[\s\S]*?<\/\1>/g, '').replace(/<!--[\s\S]*?-->/g, '');
      const sels = touchSelectors(style);
      for (const el of editables(markup)) {
        seen++;
        if (!sels.some((sel) => matches(el, sel))) misses.push(`${rel}: <${el.tagName.toLowerCase()}> matches no coarse-pointer rule of --type-lg-size`);
      }
    }
    // The scan found the controls it guards (ui/Input, ui/Select, the palette's field, Add dataset's class field, …).
    expect(seen).toBeGreaterThanOrEqual(5);
    expect(misses).toEqual([]);
  });

  it('counts a rule only when its selector matches the rendered control, at 16 px or more', () => {
    const rule = (sel: string, size = 'lg') => `@media (pointer: coarse) { ${sel} { font-size: var(--type-${size}-size); } }`;
    const field = '<div class="chips" class:invalid={bad} onclick={(e) => e.x > 1}><input bind:value={q} /></div>';
    expect(covered(rule('.chips input'), field)).toBe(true);
    // Round 27's false pass: the right element named under a class nothing renders.
    expect(covered(rule('.unused input'), field)).toBe(false);
    // A class a directive adds counts; `:global()` is unwrapped.
    expect(covered(rule('.invalid input'), field)).toBe(true);
    expect(covered(rule(':global(.chips) input'), field)).toBe(true);
    expect(covered(rule('.chips input', 'md'), field)).toBe(false);
    expect(covered('@media (pointer: fine) { input { font-size: var(--type-lg-size); } }', field)).toBe(false);
    // A self-closing component before the control does not swallow it, and a component named Input is not an input.
    expect(covered(rule('.row > input'), '<div class="row"><Icon size={14} /><input /></div>')).toBe(true);
    expect(editables('<Input bind:value={q} /><Select /><input type="checkbox" /><input oninput={(e) => go(e)} type="range" />')).toHaveLength(0);
  });
});
