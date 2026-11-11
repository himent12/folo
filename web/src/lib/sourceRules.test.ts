/// <reference types="node" />
// Source rules whose breach is a bug a person hits and that neither tsc, the linters of `npm run check` nor the layout
// QA can see: each reads the components as written (`.tsx`, parsed with the compiler's parser). The sources are read
// once, for every rule.
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join, relative, resolve, sep } from 'node:path';
import { Window } from 'happy-dom';
import { parseSync } from 'oxc-parser';
import { describe, expect, it } from 'vitest';

// A DOM to render markup into (the node environment, where the compiler's parser loads natively).
const document = new Window().document as unknown as Document;

const SRC = resolve(process.cwd(), 'src');

function walk(dir: string, out: string[] = []): string[] {
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) walk(p, out);
    else if (/\.tsx?$/.test(name) && !/\.test\.tsx?$/.test(name) && !/\.testing\.tsx?$/.test(name)) out.push(p);
  }
  return out;
}

/** Every component and module of the app (no tests), as its path under src/ and its text. */
const sources = walk(SRC).map((f) => ({ rel: relative(SRC, f).split(sep).join('/'), text: readFileSync(f, 'utf8') }));
const components = sources.filter((s) => s.rel.endsWith('.tsx'));

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Node = any;

/** Every node of a parsed `.tsx` source, depth first. */
function* nodes(n: Node): Generator<Node> {
  if (!n || typeof n !== 'object') return;
  if (Array.isArray(n)) {
    for (const x of n) yield* nodes(x);
    return;
  }
  if (typeof n.type === 'string') yield n;
  for (const k in n) if (k !== 'parent') yield* nodes(n[k]);
}

const parse = (text: string, file = 'x.tsx') => parseSync(file, text, { sourceType: 'module', lang: 'tsx' } as never).program as Node;

const elementName = (n: Node): string => (n.type === 'JSXIdentifier' ? n.name : n.type === 'JSXMemberExpression' ? `${elementName(n.object)}.${n.property.name}` : n.type === 'JSXNamespacedName' ? `${n.namespace.name}:${n.name.name}` : '');

describe('a cloud run says its phase wherever its status shows', () => {
  // A cloud run in its hand-over is queued with a phase of its own (docs/CLOUD.md); a status rebuilt from `r.status`
  // drops the phase and says "Queued". So every `<StatusBadge>` takes the run whole (`run`) or another thing's own
  // status and word (`state`), and `statusWord` is called on the run, never on an object built in place.

  /** Every `<StatusBadge>` not given exactly one of `run` or `state` as an expression that is not an object literal. */
  function rebuilt(source: string): string[] {
    const out: string[] = [];
    for (const n of nodes(parse(source))) {
      if (n.type !== 'JSXOpeningElement' || elementName(n.name) !== 'StatusBadge') continue;
      const attrs: Node[] = n.attributes;
      const named = attrs.filter((a) => a.type === 'JSXAttribute' && (a.name.name === 'run' || a.name.name === 'state'));
      const bad =
        attrs.some((a) => a.type === 'JSXSpreadAttribute' || (a.type === 'JSXAttribute' && (a.name.name === 'status' || a.name.name === 'label'))) ||
        named.length !== 1 ||
        named[0].value?.type !== 'JSXExpressionContainer' ||
        named[0].value.expression.type === 'ObjectExpression';
      if (bad) out.push(source.slice(n.start, n.end));
    }
    return out;
  }

  const rebuiltWords = (source: string): string[] => [...source.matchAll(/\bstatusWord\(\s*\{[^)]*\)/g)].map((m) => m[0]);

  it('takes the run whole in every StatusBadge and statusWord', () => {
    const hits: string[] = [];
    for (const { rel, text } of sources) {
      if (rel.endsWith('.tsx') && text.includes('<StatusBadge')) for (const b of rebuilt(text)) hits.push(`${rel}: ${b}`);
      for (const w of rebuiltWords(text)) hits.push(`${rel}: ${w}`);
    }
    expect(hits).toEqual([]);
  });

  it('tells a run passed whole from one rebuilt without its cloud state', () => {
    const jsx = (s: string) => `const x = ${s};`;
    for (const ok of ['<StatusBadge run={run} />', '<StatusBadge run={live.run ?? r} plain />', '<StatusBadge plain state={jobStatus(j)} lastKnown={x} />'])
      expect(rebuilt(jsx(ok)), ok).toEqual([]);
    for (const bad of [
      '<StatusBadge run={{ status: r.status }} plain />',
      '<StatusBadge plain />',
      '<StatusBadge run="queued" plain />',
      '<StatusBadge {...r} plain />',
      '<StatusBadge plain status={r.status} />',
      '<StatusBadge plain state={{ status: r.status, label: "Queued" }} />',
      '<StatusBadge run={r} state={st} plain />',
    ])
      expect(rebuilt(jsx(bad)), bad).toHaveLength(1);
    expect(rebuiltWords('statusWord(x); statusWord({ status: x.status })')).toEqual(['statusWord({ status: x.status })']);
  });
});

it('never toasts from a component while its modal dialog is open', () => {
  // The toaster sits behind the modal's scrim, outside its focus trap: a toast fired while a dialog is open is missed
  // and cannot be reached. A toast in a component that owns a modal fires once it is closed (`if (!open)`, or an
  // `…open = false` / `confirm… = false` / `….set(false)` just above it), or a comment within the 8 lines above says
  // `page-level:` and why. Results inside a dialog go to its status region (`DialogStatus`, or `feedback()`).
  const call = /\btoast\.(success|info|error|show)\(/;
  const closed = /!open\b|\.get\(\)\s*\)\s*return|\b\w*[oO]pen(\.set\(false\)| = false)|\bconfirm\w*(\.set\(false\)| = false)|page-level:/;
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
    if (!text.includes('<ClassStrip')) continue;
    const walkJsx = (n: Node, inside: string[]) => {
      if (!n || typeof n !== 'object') return;
      if (Array.isArray(n)) return n.forEach((x) => walkJsx(x, inside));
      if (n.type === 'JSXElement') {
        const name = elementName(n.openingElement.name);
        if (name === 'ClassStrip') for (const tag of inside) bad.push(`${rel}: <ClassStrip> inside <${tag}>`);
        const next = name === 'a' || name === 'label' ? [...inside, name] : inside;
        for (const k in n) if (k !== 'openingElement') walkJsx(n[k], next);
        return;
      }
      // A function in the markup (a row, a snippet) renders where it is written.
      for (const k in n) walkJsx(n[k], inside);
    };
    walkJsx(parse(text, rel), []);
  }
  expect(bad).toEqual([]);
});

describe('touch text', () => {
  // On a coarse pointer every editable control's own text is at least 16 px (`type.lg`), or iOS zooms the page into
  // the field when it takes focus. Controls inherit 12–14 px elsewhere, so a component that renders an editable
  // `<input>`, `<select>` or `<textarea>` carries an `@media (pointer: coarse)` rule setting that element's font-size to
  // `--type-lg-size` or more. A rule counts only if its selector matches the control as rendered (round 27: a scan of
  // selector text alone passed `.unused input`): the JSX is built into a DOM (the static parts of each `class`, and
  // `class:x` applied as classes; what a `{…}` renders placed where it stands) and each editable element must
  // `matches()` one of the selectors.

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

  /** The static class names an attribute value can give: string literals and template text inside it. */
  function staticClasses(v: Node): string[] {
    if (!v) return [];
    if (v.type === 'Literal' || v.type === 'StringLiteral') return String(v.value).split(/\s+/).filter(Boolean);
    const out: string[] = [];
    for (const n of nodes(v)) {
      if ((n.type === 'Literal' || n.type === 'StringLiteral') && typeof n.value === 'string') out.push(...n.value.split(/\s+/));
      else if (n.type === 'TemplateElement') out.push(...String(n.value.cooked ?? '').split(/\s+/));
      // `class={[a, { on: x }]}` (clsx): an object's keys are classes too.
      else if (n.type === 'Property' && (n.key.type === 'Identifier' || n.key.type === 'Literal')) out.push(String(n.key.name ?? n.key.value));
    }
    return out.filter((c) => /^[\w-]+$/.test(c));
  }

  /**
   * The editable controls of a component's JSX, rendered as a DOM: a component is an element of its own (`<Input>` is
   * not an `<input>`), an element gets its static classes and `class:x` directives, and the JSX inside an expression
   * (a conditional, a list's row) is placed in the element around it.
   */
  function editables(source: string, file = 'x.tsx'): Element[] {
    const root = document.createElement('div');
    const build = (n: Node, parent: Element) => {
      if (!n || typeof n !== 'object') return;
      if (Array.isArray(n)) return n.forEach((x) => build(x, parent));
      if (n.type === 'JSXElement') {
        const name = elementName(n.openingElement.name);
        const el = document.createElement(/^[a-z]/.test(name) && !name.includes(':') ? name : `c-${name.replace(/[^\w-]/g, '-')}`);
        for (const a of n.openingElement.attributes) {
          if (a.type !== 'JSXAttribute') continue;
          const an = a.name.type === 'JSXNamespacedName' ? `${a.name.namespace.name}:${a.name.name.name}` : a.name.name;
          if (an === 'class') el.classList.add(...staticClasses(a.value?.type === 'JSXExpressionContainer' ? a.value.expression : a.value));
          else if (an.startsWith('class:')) el.classList.add(an.slice('class:'.length));
          else if (an === 'type' && (a.value?.type === 'Literal' || a.value?.type === 'StringLiteral')) el.setAttribute('type', String(a.value.value));
        }
        parent.appendChild(el);
        build(n.children, el);
        return;
      }
      if (n.type === 'JSXFragment') return build(n.children, parent);
      for (const k in n) build(n[k], parent);
    };
    // The outermost JSX of the module: a component's markup and its snippets.
    const visit = (n: Node) => {
      if (!n || typeof n !== 'object') return;
      if (Array.isArray(n)) return n.forEach(visit);
      if (n.type === 'JSXElement' || n.type === 'JSXFragment') return build(n, root);
      for (const k in n) visit(n[k]);
    };
    visit(parse(source, file));
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

  /** A component's css`...` (its styles). */
  const styleOf = (text: string) => /\ncss`([\s\S]*?)`;/.exec(text)?.[1] ?? '';

  const covered = (css: string, markup: string) => {
    const sels = touchSelectors(css);
    return editables(`const x = <>${markup}</>;`).every((el) => sels.some((sel) => matches(el, sel)));
  };

  it('every editable control is at least 16 px on a coarse pointer, by a rule that matches it as rendered', () => {
    const misses: string[] = [];
    let seen = 0;
    for (const { rel, text } of components) {
      if (!/<(?:input|select|textarea)\b/.test(text)) continue;
      const sels = touchSelectors(styleOf(text));
      for (const el of editables(text, rel)) {
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
    const field = '<div class="chips" class:invalid={bad()} onclick={(e) => e.x > 1}><input bind:value={q} /></div>';
    expect(covered(rule('.chips input'), field)).toBe(true);
    // Round 27's false pass: the right element named under a class nothing renders.
    expect(covered(rule('.unused input'), field)).toBe(false);
    // A class a directive adds counts; `:global()` is unwrapped; a template's static part is a class.
    expect(covered(rule('.invalid input'), field)).toBe(true);
    expect(covered(rule(':global(.chips) input'), field)).toBe(true);
    expect(covered(rule('.wrap input'), '<div class={`wrap ${size()}`}><input /></div>')).toBe(true);
    expect(covered(rule('.chips input', 'md'), field)).toBe(false);
    expect(covered('@media (pointer: fine) { input { font-size: var(--type-lg-size); } }', field)).toBe(false);
    // A control inside a conditional is the element's; a component named Input is not an input.
    expect(covered(rule('.row > input'), '<div class="row"><Icon size={14} />{on() ? <input /> : null}</div>')).toBe(true);
    expect(editables('const x = <><Input bind:value={q} /><Select /><input type="checkbox" /><input oninput={(e) => go(e)} type="range" /></>;')).toHaveLength(0);
  });
});
