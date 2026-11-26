// The JSX transform. Each outermost JSX expression becomes: a module-level template (static HTML, the file's scope
// class baked in) cloned once per use, a walk to the nodes with dynamic parts (`firstChild` / `nextSibling`), and one
// binding per dynamic part, in source order. Components become calls with getter props; their children a function.
//
// Only the JSX spans are rewritten (magic-string), and every embedded JS expression keeps its original text in place:
// generated code is written into the gaps between expressions, so the source map stays exact for all of them.

/* eslint-disable @typescript-eslint/no-explicit-any */

import MagicString from 'magic-string';
import { parseSync } from 'oxc-parser';
import {
  analyze,
  attrName,
  bare,
  containsJsx,
  eachChild,
  hasCall,
  isComponentTag,
  isDynamic,
  isFunctionExpr,
  isJsx,
  jsxWithin,
  literal,
  memberName,
  needsDerived,
  propKey,
  tagName,
  type Knowledge,
} from './ast.ts';
import { cleanJsxText, decodeEntities, SVG } from './html.ts';
import { Template } from './templates.ts';

type Node = any;
type Range = { s: number; e: number };
export type Chunk = string | Range;

export interface JsxResult {
  code: string;
  map: ReturnType<MagicString['generateMap']>;
  /** The file's `css\`...\`` content (unscoped), if any. */
  css?: string;
}

export interface JsxOptions {
  /** The file's scope class, when it has styles: added to every element its JSX creates. */
  scope?: string;
  /** Where the runtime's internal entry is imported from. */
  runtime?: string;
}

class CompileError extends Error {}

// ---------------------------------------------------------------------------------------------------------------------

/** A part of a text run: static text, or an expression. */
type RunPart = { text: string } | { node: Node };

/**
 * A child of an element or fragment after JSX whitespace and fragments are resolved. A `run` is what Svelte makes one
 * text node of: adjacent static text and expressions (`Epoch {a}/{b}`), keyed by its first expression.
 */
type Item = { kind: 'text'; text: string } | { kind: 'el'; node: Node } | { kind: 'dyn'; node: Node } | { kind: 'run'; node: Node; parts: RunPart[] };

export class Compiler {
  readonly s: MagicString;
  readonly code: string;
  readonly scope: string | undefined;
  readonly helpers = new Set<string>();
  readonly templates: string[] = [];
  readonly templateIds = new Map<string, string>();
  readonly events = new Set<string>();
  /** Text runs by their key (first expression), and every expression that belongs to a run. */
  readonly runs = new Map<Node, RunPart[]>();
  readonly runMembers = new Set<Node>();
  readonly known: Knowledge;
  uid = 0;
  constructor(code: string, s: MagicString, scope: string | undefined, known: Knowledge) {
    this.code = code;
    this.s = s;
    this.scope = scope;
    this.known = known;
  }

  /**
   * Whether an expression renders more than text, as far as the file shows: JSX in it, a snippet's call or a snippet
   * prop (`{link()}`, `{props.children}`, `{props.row(item)}`), or a conditional with one in a branch. Such an
   * expression is a block (Svelte's `{#if}` / `{@render}`), never part of a text run.
   */
  isBlock(n: Node): boolean {
    const b = bare(n);
    if (!b || typeof b !== 'object') return false;
    if (isJsx(b) || jsxWithin(b).length > 0) return true;
    switch (b.type) {
      case 'Identifier':
        return this.known.snippets.has(b.name);
      case 'CallExpression': {
        const callee = bare(b.callee);
        if (callee.type === 'Identifier') return this.known.snippets.has(callee.name);
        const m = memberName(callee);
        return m !== null && this.known.snippetProps.has(m);
      }
      case 'ChainExpression':
        return this.isBlock(b.expression);
      case 'ConditionalExpression':
        return this.isBlock(b.consequent) || this.isBlock(b.alternate);
      case 'LogicalExpression':
        return this.isBlock(b.left) || this.isBlock(b.right);
      case 'SequenceExpression':
        return this.isBlock(b.expressions[b.expressions.length - 1]);
    }
    const m = memberName(b);
    return m !== null && this.known.snippetProps.has(m);
  }

  /** A conditional's branch: JSX compiled, a dynamic value as a function (its own memo), so the branch's memo tracks only the condition. */
  branchValue(n: Node, svg: boolean): Chunk[] {
    const b = bare(n);
    if (isJsx(b)) return this.jsx(b, svg);
    return isDynamic(n) ? ['() => ', ...this.paren(n, svg)] : this.paren(n, svg);
  }

  h(name: string): string {
    this.helpers.add(name);
    return `_$${name}`;
  }

  /** Registers (or reuses) a module-level template for `(html, wrapSvg)` and returns its identifier. */
  template(html: string, wrapSvg: boolean): string {
    const key = `${wrapSvg ? '1' : '0'}:${html}`;
    let id = this.templateIds.get(key);
    if (id === undefined) {
      id = `_tmpl$${this.templates.length + 1}`;
      this.templateIds.set(key, id);
      this.templates.push(`const ${id} = /*#__PURE__*/${this.h('template')}(${JSON.stringify(html)}${wrapSvg ? ', 1' : ''});`);
    }
    return id;
  }

  fail(n: Node, msg: string): never {
    const before = this.code.slice(0, n.start);
    const line = before.split('\n').length;
    const col = n.start - before.lastIndexOf('\n');
    throw new CompileError(`${msg} (${line}:${col})`);
  }

  /** `n`'s source, with the JSX inside it compiled. */
  expr(n: Node, svg: boolean): Chunk[] {
    if (isJsx(n)) return this.jsx(n, svg);
    const out: Chunk[] = [];
    let pos = n.start;
    for (const j of jsxWithin(n)) {
      if (j.start > pos) out.push({ s: pos, e: j.start });
      out.push(...this.jsx(j, svg));
      pos = j.end;
    }
    if (n.end > pos) out.push({ s: pos, e: n.end });
    return out;
  }

  /** `(expr)`. */
  paren(n: Node, svg: boolean): Chunk[] {
    return ['(', ...this.expr(n, svg), ')'];
  }

  jsx(n: Node, svg: boolean): Chunk[] {
    if (n.type === 'JSXFragment') return this.fragment(n.children, svg);
    const opening = n.openingElement;
    if (isComponentTag(opening.name)) return this.component(n, svg);
    if (opening.name.type !== 'JSXIdentifier') this.fail(n, `<${tagName(opening.name)}> is not supported`);
    return this.element(n, svg);
  }

  /**
   * The children of an element or fragment, as items, with text runs merged as Svelte merges its text sequences:
   * adjacent static text and expressions become one text node (Chrome shapes each text node on its own, so this is
   * what keeps glyphs identical). An expression with JSX in it is a block (an `{#if}`, a list), never part of a run.
   */
  items(children: Node[]): Item[] {
    const out: Item[] = [];
    const flat = this.rawItems(children);
    const textLike = (it: Item) => it.kind === 'text' || (it.kind === 'dyn' && it.node.type !== 'JSXElement' && !this.isBlock(it.node));
    for (let i = 0; i < flat.length; ) {
      if (!textLike(flat[i]!)) {
        out.push(flat[i++]!);
        continue;
      }
      let j = i;
      while (j < flat.length && textLike(flat[j]!)) j++;
      const seg = flat.slice(i, j);
      const exprs = seg.filter((it) => it.kind === 'dyn');
      if (seg.length >= 2 && exprs.length > 0) {
        const parts: RunPart[] = seg.map((it) => (it.kind === 'text' ? { text: it.text } : { node: (it as { node: Node }).node }));
        const key = (exprs[0] as { node: Node }).node;
        this.runs.set(key, parts);
        for (const e of exprs) this.runMembers.add((e as { node: Node }).node);
        out.push({ kind: 'run', node: key, parts });
      } else out.push(...seg);
      i = j;
    }
    return out;
  }

  /** A text run's parts as an array: static text as strings, every expression as a function. */
  runParts(parts: RunPart[], svg: boolean): Chunk[] {
    const out: Chunk[] = ['['];
    parts.forEach((pt, i) => {
      if (i) out.push(', ');
      if ('text' in pt) out.push(JSON.stringify(pt.text));
      else out.push('() => ', ...this.paren(pt.node, svg));
    });
    out.push(']');
    return out;
  }

  rawItems(children: Node[]): Item[] {
    const out: Item[] = [];
    const pushText = (t: string) => {
      const last = out[out.length - 1];
      if (last?.kind === 'text') last.text += t;
      else out.push({ kind: 'text', text: t });
    };
    for (const c of children) {
      if (c.type === 'JSXText') {
        const t = cleanJsxText(c.value);
        if (t) pushText(this.decode(c, t));
      } else if (c.type === 'JSXExpressionContainer') {
        if (c.expression.type === 'JSXEmptyExpression') continue;
        const lit = literal(c.expression);
        if (lit && (typeof lit.value === 'string' || typeof lit.value === 'number')) pushText(String(lit.value));
        else if (lit && (lit.value === null || lit.value === undefined || typeof lit.value === 'boolean')) continue;
        else out.push({ kind: 'dyn', node: c.expression });
      } else if (c.type === 'JSXFragment') {
        for (const it of this.rawItems(c.children)) {
          if (it.kind === 'text') pushText(it.text);
          else out.push(it);
        }
      } else if (c.type === 'JSXElement') {
        if (isComponentTag(c.openingElement.name)) out.push({ kind: 'dyn', node: c });
        else out.push({ kind: 'el', node: c });
      } else if (c.type === 'JSXSpreadChild') this.fail(c, 'spread children are not supported');
    }
    return out;
  }

  decode(n: Node, raw: string): string {
    const bad = raw.match(/&([A-Za-z][A-Za-z0-9]*);/g)?.find((m) => decodeEntities(m) === m);
    if (bad) this.fail(n, `unknown HTML entity ${bad} (write the character, or {'\\u....'})`);
    return decodeEntities(raw);
  }

  /** A child value for `insert` or an array: static as is, dynamic as a function, conditionals memoized. */
  childValue(n: Node, svg: boolean): Chunk[] {
    const e = bare(n);
    if (isJsx(e)) return this.jsx(e, svg);
    // A conditional that renders blocks is an `{#if}`: a branch is created when the condition flips. One that renders
    // text is a text expression, as in Svelte (one effect); `a && b` renders nothing for any falsy `a`.
    if (e.type === 'ConditionalExpression' && isDynamic(e.test)) {
      return this.isBlock(e) ? this.conditional(e, svg) : ['() => ', ...this.paren(n, svg)];
    }
    // A static test (`i > 0` with a plain `i`) picks its branch once; the branch is a child value of its own.
    if (e.type === 'ConditionalExpression' && this.isBlock(e)) {
      return ['(', ...this.paren(e.test, svg), ' ? ', ...this.childValue(e.consequent, svg), ' : ', ...this.childValue(e.alternate, svg), ')'];
    }
    if (e.type === 'LogicalExpression' && e.operator === '&&' && isDynamic(e.left)) {
      if (!this.isBlock(e.right)) return ['() => (', ...this.paren(e.left, svg), ' ? ', ...this.paren(e.right, svg), ' : null)'];
      const c = `_c$${++this.uid}`;
      return [`(() => { const ${c} = ${this.h('cond')}(() => `, ...this.paren(e.left, svg), `); return ${this.h('memo')}(() => ${c}() ? `, ...this.branchValue(e.right, svg), ` : null); })()`];
    }
    if (isDynamic(n)) return ['() => ', ...this.paren(n, svg)];
    return this.paren(n, svg);
  }

  /** `a ? b : c ? d : e` as a memo that re-creates a branch only when a condition flips. */
  conditional(e: Node, svg: boolean): Chunk[] {
    const lets: string[] = [];
    const first = `_c$${++this.uid}`;
    const branch = (n: Node): Chunk[] => {
      const b = bare(n);
      if (b.type === 'ConditionalExpression' && isDynamic(b.test)) {
        const c = `_c$${++this.uid}`;
        lets.push(c);
        return [`(${c} ??= ${this.h('cond')}(() => `, ...this.paren(b.test, svg), `))() ? `, ...branch(b.consequent), ' : ', ...branch(b.alternate)];
      }
      return this.branchValue(n, svg);
    };
    const head: Chunk[] = [`const ${first} = ${this.h('cond')}(() => `, ...this.paren(e.test, svg), `); return ${this.h('memo')}(() => ${first}() ? `];
    const body: Chunk[] = [...branch(e.consequent), ' : ', ...branch(e.alternate), '); })()'];
    return [`(() => { ${lets.length ? `let ${lets.join(', ')}; ` : ''}`, ...head, ...body];
  }

  fragment(children: Node[], svg: boolean): Chunk[] {
    const items = this.items(children);
    const parts = items.map((it): Chunk[] => {
      if (it.kind === 'text') return [JSON.stringify(it.text)];
      if (it.kind === 'el') return this.element(it.node, svg);
      if (it.kind === 'run') return [`${this.h('runNode')}(`, ...this.runParts(it.parts, svg), ')'];
      return this.childValue(it.node, svg);
    });
    if (parts.length === 1) return parts[0]!;
    const out: Chunk[] = ['['];
    parts.forEach((p, i) => {
      if (i) out.push(', ');
      out.push(...p);
    });
    out.push(']');
    return out;
  }

  // -------------------------------------------------------------------------------------------------------------------
  // Components

  component(n: Node, svg: boolean): Chunk[] {
    const opening = n.openingElement;
    const segments: Chunk[][] = [];
    let obj: Chunk[] | null = null;
    const prop = (chunks: Chunk[]) => {
      if (obj === null) obj = ['{ '];
      else obj.push(', ');
      obj.push(...chunks);
    };
    const close = () => {
      if (obj !== null) {
        obj.push(' }');
        segments.push(obj);
        obj = null;
      }
    };
    const name = tagName(opening.name);
    for (const a of opening.attributes) {
      if (a.type === 'JSXSpreadAttribute') {
        close();
        // A spread with a call is memoized (Svelte's `$.derived`), so its keys read one evaluation.
        if (isDynamic(a.argument) && hasCall(a.argument) && !containsJsx(a.argument)) segments.push([`${this.h('derive')}(() => `, ...this.paren(a.argument, svg), ')']);
        else segments.push(isDynamic(a.argument) ? ['() => ', ...this.paren(a.argument, svg)] : this.paren(a.argument, svg));
        continue;
      }
      const { ns, name: key } = attrName(a);
      if (ns === 'bind') {
        if (!a.value || a.value.type !== 'JSXExpressionContainer') this.fail(a, 'bind: takes an expression: bind:x={signal}');
        close();
        segments.push([`${this.h('bound')}(${JSON.stringify(key)}, `, ...this.paren(a.value.expression, svg), ')']);
        continue;
      }
      if (ns !== null) this.fail(a, `${ns}:${key} is not supported on a component`);
      if (key === 'children') this.fail(a, 'pass children as JSX children');
      if (key === 'ref' && name !== 'Dynamic') this.fail(a, 'ref is not supported on a component (pass a callback prop)');
      const k = propKey(key);
      if (a.value === null || a.value === undefined) prop([`${k}: true`]);
      else if (a.value.type === 'Literal' || a.value.type === 'StringLiteral') prop([`${k}: ${JSON.stringify(this.decode(a.value, a.value.value))}`]);
      else if (a.value.type === 'JSXExpressionContainer') {
        const x = a.value.expression;
        if (x.type === 'JSXEmptyExpression') this.fail(a, 'empty attribute expression');
        if (isDynamic(x) && !containsJsx(x) && needsDerived(x)) prop([`get ${k}() { return ${this.h('propMemo')}(this, ${JSON.stringify(key)}, () => `, ...this.paren(x, svg), '); }']);
        else if (isDynamic(x) || isJsx(bare(x))) prop([`get ${k}() { return `, ...this.paren(x, svg), '; }']);
        else prop([`${k}: `, ...this.paren(x, svg)]);
      } else if (isJsx(a.value)) prop([`get ${k}() { return `, ...this.jsx(a.value, svg), '; }']);
    }
    if (name === 'Dynamic' && this.scope) prop([`$$scope: ${JSON.stringify(this.scope)}`]);
    const kids = this.componentChildren(n, svg);
    if (kids) prop(['children: ', ...kids]);
    close();
    const tag: Chunk = { s: opening.name.start, e: opening.name.end };
    let props: Chunk[];
    if (segments.length === 0) props = ['{}'];
    else if (segments.length === 1 && segments[0]![0] === '{ ') props = segments[0]!;
    else {
      props = [`${this.h('mergeProps')}(`];
      segments.forEach((sg, i) => {
        if (i) props.push(', ');
        props.push(...sg);
      });
      props.push(')');
    }
    return [`${this.h('createComponent')}(`, tag, ', ', ...props, ')'];
  }

  /** A component's `children` prop: a render function passed through, a static string, else a snippet function. */
  componentChildren(n: Node, svg: boolean): Chunk[] | null {
    const kids = n.children as Node[];
    const meaningful = kids.filter((c) => !(c.type === 'JSXText' && !cleanJsxText(c.value)) && !(c.type === 'JSXExpressionContainer' && c.expression.type === 'JSXEmptyExpression'));
    if (meaningful.length === 0) return null;
    if (meaningful.length === 1 && meaningful[0].type === 'JSXExpressionContainer') {
      const x = bare(meaningful[0].expression);
      if (isFunctionExpr(x) || x.type === 'Identifier') return this.paren(meaningful[0].expression, svg);
    }
    const items = this.items(kids);
    if (items.length === 1 && items[0]!.kind === 'text') return [JSON.stringify(items[0]!.text)];
    return ['() => ', ...wrapParen(this.fragment(kids, svg))];
  }

  // -------------------------------------------------------------------------------------------------------------------
  // Elements

  element(n: Node, svgCtx: boolean): Chunk[] {
    const tag = n.openingElement.name.name as string;
    const svg = svgCtx || (SVG.has(tag) && tag !== 'svg');
    const t = new Template(this, svg && tag !== 'svg');
    return t.build(n, svgCtx || tag === 'svg' || SVG.has(tag));
  }
}

function wrapParen(c: Chunk[]): Chunk[] {
  return ['(', ...c, ')'];
}

/** Finds `css\`...\`` statements (the tag imported from 'fw'): returns their content and removes them. */
function extractCss(program: Node, code: string, s: MagicString, runtime: string): string | undefined {
  let local: string | null = null;
  for (const st of program.body) {
    if (st.type === 'ImportDeclaration' && (st.source.value === 'fw' || st.source.value === runtime)) {
      for (const sp of st.specifiers) if (sp.type === 'ImportSpecifier' && (sp.imported.name ?? sp.imported.value) === 'css') local = sp.local.name;
    }
  }
  if (local === null) return undefined;
  const parts: string[] = [];
  for (const st of program.body) {
    const e = st.type === 'ExpressionStatement' ? st.expression : null;
    if (e?.type === 'TaggedTemplateExpression' && e.tag.type === 'Identifier' && e.tag.name === local) {
      if (e.quasi.expressions.length) throw new CompileError('css`` cannot contain ${...}');
      // The raw text with the template's own escapes undone (\` \${ \\), so the CSS is exactly what was written.
      parts.push(code.slice(e.quasi.start + 1, e.quasi.end - 1).replace(/\\(`|\\|\$\{)/g, '$1'));
      s.remove(st.start, st.end);
    }
  }
  return parts.length ? parts.join('\n') : undefined;
}

/** Writes chunks over the source span [start, end): kept ranges stay, generated text fills the gaps. */
function apply(s: MagicString, start: number, end: number, chunks: Chunk[]): void {
  let pos = start;
  let buf = '';
  for (const ch of chunks) {
    if (typeof ch === 'string') {
      buf += ch;
      continue;
    }
    if (ch.s < pos) throw new CompileError('internal: out-of-order chunk');
    if (ch.s > pos) {
      if (buf) s.overwrite(pos, ch.s, buf);
      else s.remove(pos, ch.s);
    } else if (buf) s.appendLeft(pos, buf);
    buf = '';
    pos = ch.e;
  }
  if (end > pos) {
    if (buf) s.overwrite(pos, end, buf);
    else s.remove(pos, end);
  } else if (buf) s.appendLeft(end, buf);
}

/** Compiles the JSX and `css` tags of a `.tsx` module. */
export function transformJsx(code: string, id: string, options: JsxOptions = {}): JsxResult {
  const runtime = options.runtime ?? 'fw/internal';
  const parsed = parseSync(id, code, { sourceType: 'module', lang: 'tsx' } as any);
  if (parsed.errors.length) {
    const e = parsed.errors[0] as any;
    throw new CompileError(`${id}: ${e.message}`);
  }
  const program = parsed.program as Node;
  const s = new MagicString(code);
  const css = extractCss(program, code, s, runtime);
  const c = new Compiler(code, s, css !== undefined ? options.scope : undefined, analyze(program));
  try {
    const roots: Node[] = [];
    const visit = (n: Node) => {
      if (isJsx(n)) roots.push(n);
      else eachChild(n, visit);
    };
    eachChild(program, visit);
    for (const r of roots) apply(s, r.start, r.end, c.jsx(r, false));
  } catch (e) {
    if (e instanceof CompileError) throw new Error(`${id}: ${e.message}`);
    throw e;
  }
  const head: string[] = [];
  if (c.helpers.size || c.events.size) {
    if (c.events.size) c.helpers.add('delegateEvents');
    head.push(`import { ${[...c.helpers].sort().map((h) => `${h} as _$${h}`).join(', ')} } from ${JSON.stringify(runtime)};`);
  }
  head.push(...c.templates);
  if (head.length) s.prepend(head.join('\n') + '\n');
  if (c.events.size) s.append(`\n_$delegateEvents(${JSON.stringify([...c.events].sort())});\n`);
  return { code: s.toString(), map: s.generateMap({ hires: true, source: id, includeContent: true }), css };
}
