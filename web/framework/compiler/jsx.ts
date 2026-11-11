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
  IDENT,
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
import { cleanJsxText, decodeEntities, escapeAttr, escapeText, RAW_TEXT, SVG, VOID } from './html.ts';
import { eventOf } from '../runtime/names.ts';

type Node = any;
type Range = { s: number; e: number };
type Chunk = string | Range;

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

/** Svelte's delegated events (see runtime/events.ts). */
const DELEGATED = new Set([
  'beforeinput', 'click', 'change', 'dblclick', 'contextmenu', 'focusin', 'focusout', 'input', 'keydown', 'keyup',
  'mousedown', 'mousemove', 'mouseout', 'mouseover', 'mouseup', 'pointerdown', 'pointermove', 'pointerout',
  'pointerover', 'pointerup', 'touchend', 'touchmove', 'touchstart',
]);

/** Attributes set as DOM properties on HTML elements (Svelte's DOM_PROPERTIES), attribute name → property. */
const PROPS: Record<string, string> = {
  allowfullscreen: 'allowFullscreen', async: 'async', autofocus: 'autofocus', autoplay: 'autoplay', checked: 'checked',
  controls: 'controls', default: 'default', disabled: 'disabled', formnovalidate: 'formNoValidate',
  indeterminate: 'indeterminate', inert: 'inert', ismap: 'isMap', loop: 'loop', multiple: 'multiple', muted: 'muted',
  nomodule: 'noModule', novalidate: 'noValidate', open: 'open', playsinline: 'playsInline', readonly: 'readOnly',
  required: 'required', reversed: 'reversed', selected: 'selected', value: 'value', volume: 'volume',
  defaultvalue: 'defaultValue', defaultchecked: 'defaultChecked', srcobject: 'srcObject', innerhtml: 'innerHTML',
  textcontent: 'textContent', innertext: 'innerText',
};

/**
 * Properties a template attribute does not set (no content attribute, or one that is not the property): set from
 * JS even when static (Svelte's NON_STATIC_PROPERTIES `muted`, `defaultValue`, `defaultChecked`, and those with no
 * attribute at all).
 */
const TEMPLATE_CANNOT_SET = new Set(['indeterminate', 'muted', 'volume', 'defaultValue', 'defaultChecked', 'srcObject', 'innerHTML', 'textContent', 'innerText']);

/** Svelte's DOM_BOOLEAN_ATTRIBUTES: present when true, absent when false. Other attributes stringify booleans. */
const BOOLEAN_ATTRS = new Set([
  'allowfullscreen', 'async', 'autofocus', 'autoplay', 'checked', 'controls', 'default', 'disabled', 'formnovalidate',
  'indeterminate', 'inert', 'ismap', 'loop', 'multiple', 'muted', 'nomodule', 'novalidate', 'open', 'playsinline',
  'readonly', 'required', 'reversed', 'seamless', 'selected', 'webkitdirectory', 'defer', 'disablepictureinpicture',
  'disableremoteplayback',
]);

const SIZE_BINDS = new Set(['clientWidth', 'clientHeight', 'offsetWidth', 'offsetHeight']);
/** Block elements a `<p>` cannot contain (the HTML parser would close the `<p>`, breaking the template walk). */
const NOT_IN_P = new Set(['address', 'article', 'aside', 'blockquote', 'details', 'dialog', 'div', 'dl', 'fieldset', 'figcaption', 'figure', 'footer', 'form', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'header', 'hgroup', 'hr', 'main', 'menu', 'nav', 'ol', 'p', 'pre', 'section', 'table', 'ul']);

class CompileError extends Error {}


// ---------------------------------------------------------------------------------------------------------------------

interface TNode {
  kind: 'el' | 'text' | 'comment';
  parent: TNode | null;
  children: TNode[];
  v?: string;
  jsx?: Node;
}

/** A part of a text run: static text, or an expression. */
type RunPart = { text: string } | { node: Node };

/**
 * A child of an element or fragment after JSX whitespace and fragments are resolved. A `run` is what Svelte makes one
 * text node of: adjacent static text and expressions (`Epoch {a}/{b}`), keyed by its first expression.
 */
type Item = { kind: 'text'; text: string } | { kind: 'el'; node: Node } | { kind: 'dyn'; node: Node } | { kind: 'run'; node: Node; parts: RunPart[] };

class Compiler {
  readonly s: MagicString;
  readonly code: string;
  readonly scope: string | undefined;
  readonly helpers = new Set<string>();
  readonly templates: string[] = [];
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
      if (key === 'ref') this.fail(a, 'ref is not supported on a component (pass a callback prop)');
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
    const svg = svgCtx || (SVG.has(tag) && tag !== 'svg') ;
    const t = new Template(this, svg && tag !== 'svg');
    return t.build(n, svgCtx || tag === 'svg' || SVG.has(tag));
  }
}

/**
 * Whether a static attribute is set as its DOM property from JS rather than written in the template: as Svelte, a
 * property given as an expression (`checked={true}`, `<option value={1}>`), and a property no template attribute sets.
 */
function propOnly(a: Node, name: string, svg: boolean): boolean {
  const prop = svg ? undefined : PROPS[name.toLowerCase()];
  return prop !== undefined && (a.value?.type === 'JSXExpressionContainer' || TEMPLATE_CANNOT_SET.has(prop));
}

function wrapParen(c: Chunk[]): Chunk[] {
  return ['(', ...c, ')'];
}

/** One template: its HTML, the node tree, and the bindings in source order. */
class Template {
  readonly c: Compiler;
  readonly wrapSvg: boolean;
  html = '';
  readonly decls: string[] = [];
  readonly ops: Chunk[][] = [];
  root!: TNode;
  constructor(c: Compiler, wrapSvg: boolean) {
    this.c = c;
    this.wrapSvg = wrapSvg;
  }

  ref(t: TNode): string {
    if (t.v) return t.v;
    const parent = t.parent!;
    const i = parent.children.indexOf(t);
    const expr = i > 0 ? `${this.ref(parent.children[i - 1]!)}.nextSibling` : `${this.ref(parent)}.firstChild`;
    t.v = `_el$${++this.c.uid}`;
    this.decls.push(`${t.v} = ${expr}`);
    return t.v;
  }

  build(n: Node, svg: boolean): Chunk[] {
    // Phase 1: shape (HTML and node tree); phase 2: bindings, in source order.
    const shapes = new Map<Node, { t: TNode; markers: Map<Node, TNode | null | undefined> }>();
    this.root = this.shape(n, null, svg, shapes);
    this.root.v = `_el$${++this.c.uid}`;
    const tmpl = `_tmpl$${this.c.templates.length + 1}`;
    this.c.templates.push(`const ${tmpl} = /*#__PURE__*/${this.c.h('template')}(${JSON.stringify(this.html)}${this.wrapSvg ? ', 1' : ''});`);
    this.emit(n, svg, shapes);
    if (this.ops.length === 0) return [`${tmpl}()`];
    const out: Chunk[] = [`(() => { const ${[`${this.root.v} = ${tmpl}()`, ...this.decls].join(', ')}; `];
    for (const op of this.ops) out.push(...op, ' ');
    out.push(`return ${this.root.v}; })()`);
    return out;
  }

  shape(n: Node, parent: TNode | null, svgCtx: boolean, shapes: Map<Node, { t: TNode; markers: Map<Node, TNode | null | undefined> }>): TNode {
    const c = this.c;
    const opening = n.openingElement;
    const tag = opening.name.name as string;
    const svg = svgCtx || tag === 'svg';
    const t: TNode = { kind: 'el', parent, children: [], jsx: n };
    if (parent) parent.children.push(t);
    const markers = new Map<Node, TNode | null | undefined>();
    shapes.set(n, { t, markers });

    // Attributes that go into the template.
    let html = `<${tag}`;
    const hasSpread = opening.attributes.some((a: Node) => a.type === 'JSXSpreadAttribute');
    let staticClass: string[] = [];
    let classDynamic = false;
    for (const a of opening.attributes) {
      if (a.type === 'JSXSpreadAttribute') continue;
      const { ns, name } = attrName(a);
      if (ns === 'class') {
        if (a.value == null || (a.value.type === 'JSXExpressionContainer' && literal(a.value.expression)?.value === true)) staticClass.push(name);
        continue;
      }
      if (ns !== null || hasSpread) continue;
      if (name === 'class' || name === 'className') {
        const v = this.staticValue(a);
        if (v === undefined) classDynamic = true;
        else if (v !== null && v !== false) staticClass = [String(v === true ? '' : v), ...staticClass].filter(Boolean);
        continue;
      }
      if (name.startsWith('on') || name === 'ref' || name === 'use' || name === 'key') continue;
      if (name === 'style' && this.hasStyleDirectives(opening)) continue;
      if (propOnly(a, name, svg)) continue;
      if (tag === 'select' && name === 'value') continue;
      if (tag === 'textarea' && name === 'value') continue;
      const v = this.staticValue(a);
      if (v === undefined || v === null || v === false) continue;
      if (v === true && a.value != null && !BOOLEAN_ATTRS.has(name.toLowerCase())) continue; // set at runtime (`"true"`)
      html += v === true ? ` ${name}` : ` ${name}="${escapeAttr(String(v))}"`;
    }
    if (!classDynamic && !hasSpread) {
      const cls = [...staticClass, ...(c.scope ? [c.scope] : [])].join(' ');
      if (cls) html += ` class="${escapeAttr(cls)}"`;
    }
    html += '>';
    this.html += html;
    if (VOID.has(tag)) {
      if (n.children.length) c.fail(n, `<${tag}> cannot have children`);
      return t;
    }

    // Children.
    const items = c.items(n.children);
    if (RAW_TEXT.has(tag) || tag === 'template') {
      for (const it of items) {
        if (it.kind !== 'text' && (RAW_TEXT.has(tag) || it.kind === 'dyn' || it.kind === 'run')) c.fail(n, `<${tag}> takes static content only`);
      }
    }
    let lastTemplate: TNode | null = null;
    let pending: Node[] = [];
    const place = (node: TNode) => {
      t.children.push(node);
      for (const p of pending) markers.set(p, node);
      pending = [];
      lastTemplate = node;
    };
    const childSvg = svg && tag !== 'foreignObject';
    items.forEach((it, i) => {
      if (it.kind === 'text') {
        if (lastTemplate?.kind === 'text') {
          this.html += '<!>';
          place({ kind: 'comment', parent: t, children: [] });
        }
        this.html += tag === 'script' || tag === 'style' ? it.text : escapeText(it.text);
        place({ kind: 'text', parent: t, children: [] });
      } else if (it.kind === 'run') {
        // One text node, its content set by the run's binding (a space so the parser keeps it). After static text the
        // parser would merge the two: a comment keeps them apart (Svelte's `<!>`).
        if (lastTemplate?.kind === 'text') {
          this.html += '<!>';
          place({ kind: 'comment', parent: t, children: [] });
        }
        this.html += ' ';
        const node: TNode = { kind: 'text', parent: t, children: [] };
        place(node);
        markers.set(it.node, node);
      } else if (it.kind === 'el') {
        const ctag = it.node.openingElement.name.name as string;
        if (tag === 'table' && ctag === 'tr') c.fail(it.node, '<tr> must be inside <thead>, <tbody> or <tfoot> (the HTML parser inserts one)');
        if (tag === 'p' && NOT_IN_P.has(ctag)) c.fail(it.node, `<${ctag}> cannot be inside <p> (the HTML parser closes the <p>)`);
        if (tag === 'a' && ctag === 'a') c.fail(it.node, '<a> cannot be inside <a>');
        const node = this.shape(it.node, null, childSvg, shapes);
        node.parent = t;
        place(node);
      } else {
        if (items.length === 1) {
          markers.set(it.node, undefined);
          return;
        }
        const next = items[i + 1];
        if (next === undefined) {
          markers.set(it.node, null);
          return;
        }
        if (next.kind === 'dyn' || (next.kind === 'text' && lastTemplate !== null && (lastTemplate as TNode).kind === 'text')) {
          this.html += '<!>';
          const comment: TNode = { kind: 'comment', parent: t, children: [] };
          t.children.push(comment);
          lastTemplate = comment;
          markers.set(it.node, comment);
          for (const p of pending) markers.set(p, comment);
          pending = [];
          return;
        }
        pending.push(it.node);
      }
    });
    this.html += `</${tag}>`;
    return t;
  }

  /** An attribute's static value (`true` for a bare attribute), or `undefined` when it is an expression. */
  staticValue(a: Node): unknown {
    if (a.value == null) return true;
    if (a.value.type === 'Literal' || a.value.type === 'StringLiteral') return this.c.decode(a.value, a.value.value);
    if (a.value.type === 'JSXExpressionContainer') {
      const lit = literal(a.value.expression);
      if (lit && (lit.value === null || ['string', 'number', 'boolean'].includes(typeof lit.value))) return lit.value;
      if (lit && lit.value === undefined) return null;
    }
    return undefined;
  }

  emit(n: Node, svgCtx: boolean, shapes: Map<Node, { t: TNode; markers: Map<Node, TNode | null | undefined> }>): void {
    const c = this.c;
    const { t, markers } = shapes.get(n)!;
    const opening = n.openingElement;
    const tag = opening.name.name as string;
    const svg = svgCtx || tag === 'svg';
    const el = () => this.ref(t);
    const op = (chunks: Chunk[]) => this.ops.push(chunks);
    const hasSpread = opening.attributes.some((a: Node) => a.type === 'JSXSpreadAttribute');
    const spreadParts: string[] = [];
    const classDirs: [string, string][] = [];
    let classAccessor: string | null = null;
    let groupValue: string | null = null;
    let groupSig: string | null = null;
    const styleDirs: [string, string][] = [];
    const styleImportant: [string, string][] = [];
    let styleAccessor: string | null = null;
    const accessor = (x: Node): string => {
      const v = `_v$${++c.uid}`;
      op([`const ${v} = () => `, ...c.paren(x, svg), ';']);
      return v;
    };
    // `bind:group` needs the element's `value`: find it first.
    const valueAttr = opening.attributes.find((a: Node) => a.type === 'JSXAttribute' && attrName(a).ns === null && attrName(a).name === 'value');
    const needsGroupValue = opening.attributes.some((a: Node) => a.type === 'JSXAttribute' && attrName(a).ns === 'bind' && attrName(a).name === 'group');

    for (const a of opening.attributes) {
      if (a.type === 'JSXSpreadAttribute') {
        const v = `_p$${++c.uid}`;
        op([`const ${v} = `, ...(isDynamic(a.argument) ? ['() => ', ...c.paren(a.argument, svg)] : c.paren(a.argument, svg)), ';']);
        spreadParts.push(v);
        continue;
      }
      const { ns, name } = attrName(a);
      const expr = a.value?.type === 'JSXExpressionContainer' ? a.value.expression : null;
      if (expr?.type === 'JSXEmptyExpression') c.fail(a, 'empty attribute expression');
      if (ns === 'class') {
        const dynamicClass = hasSpread || this.isClassDynamic(opening);
        if (a.value == null || (expr && literal(expr)?.value === true)) {
          // Always on: in the template, or a constant directive when the class is computed at runtime.
          if (dynamicClass) classDirs.push([name, '() => true']);
          continue;
        }
        if (!expr) c.fail(a, 'class:name takes an expression');
        const lit = literal(expr);
        if (lit && !lit.value) continue;
        if (dynamicClass) classDirs.push([name, accessor(expr)]);
        else op([`${c.h('toggleClass')}(${el()}, ${JSON.stringify(name)}, () => `, ...c.paren(expr, svg), ');']);
        continue;
      }
      if (ns === 'style') {
        const important = name.endsWith('|important');
        const prop = important ? name.slice(0, -10) : name;
        // `style:x={expr}`, or a string `style:x="1px"` (as Svelte allows).
        const sv = expr ? null : this.staticValue(a);
        if (!expr && typeof sv !== 'string') c.fail(a, 'style:prop takes a value: style:prop={expr} or style:prop="..."');
        const value: Chunk[] = expr ? c.paren(expr, svg) : [JSON.stringify(sv)];
        // Applied with the style attribute (Svelte's set_style with directives), in one binding after the attributes.
        const v = `_v$${++c.uid}`;
        op([`const ${v} = () => `, ...value, ';']);
        (important ? styleImportant : styleDirs).push([prop, v]);
        continue;
      }
      if (ns === 'use') {
        if (!IDENT.test(name)) c.fail(a, `use:${name}: the action must be an identifier in scope`);
        if (expr) op([`${c.h('use')}(${el()}, ${name}, () => `, ...c.paren(expr, svg), ');']);
        else op([`${c.h('use')}(${el()}, ${name});`]);
        continue;
      }
      if (ns === 'transition' || ns === 'in' || ns === 'out') {
        if (!IDENT.test(name)) c.fail(a, `${ns}:${name}: the transition must be an identifier in scope`);
        const flags = ns === 'transition' ? 3 : ns === 'in' ? 1 : 2;
        if (expr) op([`${c.h('transition')}(${el()}, ${name}, () => `, ...c.paren(expr, svg), `, ${flags});`]);
        else op([`${c.h('transition')}(${el()}, ${name}, undefined, ${flags});`]);
        continue;
      }
      if (ns === 'bind') {
        if (!expr) c.fail(a, 'bind:x takes an expression: bind:x={signal}');
        if (name === 'this') op([`${c.h('ref')}(${el()}, `, ...c.paren(expr, svg), ');']);
        else if (name === 'value') op([`${c.h('bindValue')}(${el()}, `, ...c.paren(expr, svg), ');']);
        else if (name === 'checked') op([`${c.h('bindChecked')}(${el()}, `, ...c.paren(expr, svg), ');']);
        else if (name === 'open') op([`${c.h('bindOpen')}(${el()}, `, ...c.paren(expr, svg), ');']);
        else if (name === 'group') {
          if (!valueAttr) c.fail(a, 'bind:group needs a value on the same element');
          groupSig = `_g$${++c.uid}`;
          op([`const ${groupSig} = `, ...c.paren(expr, svg), ';']);
        } else if (SIZE_BINDS.has(name)) op([`${c.h('bindSize')}(${el()}, ${JSON.stringify(name)}, `, ...c.paren(expr, svg), ');']);
        else c.fail(a, `bind:${name} is not supported on <${tag}>`);
        continue;
      }
      if (ns !== null) c.fail(a, `${ns}:${name} is not supported`);
      if (name === 'ref') {
        if (!expr) c.fail(a, 'ref takes an expression');
        op([`${c.h('ref')}(${el()}, `, ...c.paren(expr, svg), ');']);
        continue;
      }
      if (name === 'use') {
        if (!expr) c.fail(a, 'use takes an expression');
        const x = bare(expr);
        if (x.type === 'ArrayExpression' && x.elements.length === 2 && x.elements[0] && x.elements[0].type !== 'ArrayExpression' && x.elements[0].type !== 'SpreadElement' && x.elements[1]?.type !== 'SpreadElement') {
          op([`${c.h('use')}(${el()}, `, ...c.paren(x.elements[0], svg), ', () => ', ...c.paren(x.elements[1], svg), ');']);
        } else op([`${c.h('useValue')}(${el()}, `, ...c.paren(expr, svg), ');']);
        continue;
      }
      if (name === 'key') c.fail(a, 'key is not supported (use <For key={...}>)');
      if (name === 'children') c.fail(a, 'pass children as JSX children');
      if (hasSpread) {
        // Everything goes through the spread, in order.
        const v = `_p$${++c.uid}`;
        const k = propKey(name === 'className' ? 'class' : name);
        if (expr && (isDynamic(expr) || isJsx(bare(expr)))) op([`const ${v} = { get ${k}() { return `, ...c.paren(expr, svg), '; } };']);
        else if (expr) op([`const ${v} = { ${k}: `, ...c.paren(expr, svg), ' };']);
        else op([`const ${v} = { ${k}: ${JSON.stringify(this.staticValue(a))} };`]);
        spreadParts.push(v);
        continue;
      }
      if (name === 'class' || name === 'className') {
        if (this.staticValue(a) !== undefined) continue;
        if (this.hasClassDirectives(opening)) classAccessor = accessor(expr);
        else op([`${c.h('className')}(${el()}, () => `, ...c.paren(expr, svg), `${c.scope ? `, ${JSON.stringify(c.scope)}` : ''});`]);
        continue;
      }
      if (name.length > 2 && name.startsWith('on')) {
        const { type, capture } = eventOf(name.slice(2).toLowerCase());
        if (!expr) c.fail(a, `${name} takes a handler`);
        const handler: Chunk[] = isDynamic(expr) ? ['function (e) { const h = ', ...c.paren(expr, svg), '; return h == null ? undefined : h.call(this, e); }'] : c.paren(expr, svg);
        if (!capture && DELEGATED.has(type)) {
          c.events.add(type);
          op([`${el()}.$$${type} = `, ...handler, ';']);
        } else op([`${c.h('setEvent')}(${el()}, ${JSON.stringify(type)}, `, ...handler, `${capture ? ', true' : ''});`]);
        continue;
      }
      if (name === 'style') {
        const sv = this.staticValue(a);
        if (sv === undefined) styleAccessor = accessor(expr);
        else if (this.hasStyleDirectives(opening)) styleAccessor = sv === null || sv === false ? 'null' : JSON.stringify(String(sv));
        continue;
      }
      if (tag === 'select' && name === 'value') {
        // Selected after the options are in (a user effect).
        if (!expr) op([`${c.h('selectValue')}(${el()}, () => ${JSON.stringify(this.staticValue(a))});`]);
        else op([`${c.h('selectValue')}(${el()}, () => `, ...c.paren(expr, svg), ');']);
        continue;
      }
      const sv = this.staticValue(a);
      const prop = !svg ? PROPS[name.toLowerCase()] : undefined;
      if (typeof sv === 'boolean' && a.value != null && !BOOLEAN_ATTRS.has(name.toLowerCase())) {
        // `spellcheck={false}`, `aria-hidden={true}`: as Svelte's set_attribute (a property if the element has a
        // setter of that name, else the string "true"/"false"); only boolean attributes are dropped on false.
        op([`${c.h('setAttribute')}(${el()}, ${JSON.stringify(name)}, ${sv});`]);
        continue;
      }
      if (sv !== undefined && !(tag === 'textarea' && name === 'value')) {
        if (needsGroupValue && name === 'value') groupValue ??= `() => ${JSON.stringify(sv)}`;
        // In the template, unless only the property holds it (`indeterminate={true}`; `<option value={1}>`, whose
        // `__value` keeps the number a bound select compares with).
        if (!propOnly(a, name, svg)) continue;
      }
      if (needsGroupValue && name === 'value') {
        groupValue = accessor(expr);
        op([`${c.h('prop')}(${el()}, "value", ${groupValue});`]);
        continue;
      }
      const valueChunks: Chunk[] = expr ? c.paren(expr, svg) : [JSON.stringify(sv)];
      if (prop) {
        if (expr && isDynamic(expr)) op([`${c.h('prop')}(${el()}, ${JSON.stringify(prop)}, () => `, ...valueChunks, ');']);
        else if (prop === 'value') op([`${c.h('setValue')}(${el()}, `, ...valueChunks, ');']);
        else if (prop === 'checked') op([`${c.h('setChecked')}(${el()}, `, ...valueChunks, ');']);
        else op([`${el()}.${prop} = `, ...valueChunks, ';']);
      } else if (expr && isDynamic(expr)) op([`${c.h('attr')}(${el()}, ${JSON.stringify(name)}, () => `, ...valueChunks, ');']);
      else op([`${c.h('setAttribute')}(${el()}, ${JSON.stringify(name)}, `, ...valueChunks, ');']);
    }
    if (groupSig !== null) op([`${c.h('bindGroup')}(${el()}, ${groupSig}, ${groupValue ?? '() => undefined'});`]);
    const dirs = classDirs.length ? `() => ({ ${classDirs.map(([k, v]) => `${JSON.stringify(k)}: ${v}()`).join(', ')} })` : null;
    const styleObject = (list: [string, string][]) => (list.length ? `() => ({ ${list.map(([k, v]) => `${JSON.stringify(k)}: ${v}()`).join(', ')} })` : null);
    const sdirs = styleObject(styleDirs);
    const simp = styleObject(styleImportant);
    if (!spreadParts.length && (styleAccessor !== null || sdirs !== null || simp !== null)) {
      // Svelte's set_style: the attribute's value (`''` with only directives), then the directives.
      const args = [styleAccessor ?? '""', ...(sdirs !== null || simp !== null ? [sdirs ?? 'undefined'] : []), ...(simp !== null ? [simp] : [])];
      op([`${c.h('bindStyle')}(${el()}, ${args.join(', ')});`]);
    }
    if (spreadParts.length) {
      const props = spreadParts.length === 1 ? spreadParts[0]! : `${c.h('mergeProps')}(${spreadParts.join(', ')})`;
      const extra = simp ? `, ${dirs ?? 'undefined'}, ${sdirs ?? 'undefined'}, ${simp}` : sdirs ? `, ${dirs ?? 'undefined'}, ${sdirs}` : dirs ? `, ${dirs}` : '';
      op([`${c.h('spread')}(${el()}, ${props}, ${c.scope ? JSON.stringify(c.scope) : 'undefined'}${extra});`]);
    } else if (classAccessor !== null || (dirs && this.isClassDynamic(opening))) {
      op([`${c.h('className')}(${el()}, ${classAccessor ?? '() => undefined'}, ${c.scope ? JSON.stringify(c.scope) : 'undefined'}${dirs ? `, ${dirs}` : ''});`]);
    }

    // Children, in order.
    for (const child of n.children as Node[]) this.emitChild(child, t, markers, svg && tag !== 'foreignObject', shapes);
  }

  emitChild(child: Node, t: TNode, markers: Map<Node, TNode | null | undefined>, svg: boolean, shapes: Map<Node, { t: TNode; markers: Map<Node, TNode | null | undefined> }>): void {
    const c = this.c;
    if (child.type === 'JSXFragment') {
      for (const x of child.children) this.emitChild(x, t, markers, svg, shapes);
      return;
    }
    if (child.type === 'JSXElement' && !isComponentTag(child.openingElement.name)) {
      this.emit(child, svg, shapes);
      return;
    }
    const node = child.type === 'JSXExpressionContainer' ? child.expression : child;
    const run = c.runs.get(node);
    if (run !== undefined && markers.has(node)) {
      this.ops.push([`${c.h('run')}(${this.ref(markers.get(node)!)}, `, ...c.runParts(run, svg), ');']);
      return;
    }
    if (c.runMembers.has(node)) return; // the rest of a run
    if (!markers.has(node)) return; // static text, comment or nothing
    const marker = markers.get(node);
    const value = child.type === 'JSXElement' ? c.jsx(child, svg) : c.childValue(node, svg);
    const m = marker === undefined ? '' : marker === null ? ', null' : `, ${this.ref(marker)}`;
    this.ops.push([`${c.h('insert')}(${this.ref(t)}, `, ...value, `${m});`]);
  }

  isClassDynamic(opening: Node): boolean {
    return opening.attributes.some((a: Node) => a.type === 'JSXAttribute' && attrName(a).ns === null && (attrName(a).name === 'class' || attrName(a).name === 'className') && this.staticValue(a) === undefined);
  }

  hasStyleDirectives(opening: Node): boolean {
    return opening.attributes.some((a: Node) => a.type === 'JSXAttribute' && attrName(a).ns === 'style');
  }

  isStyleDynamic(opening: Node): boolean {
    return opening.attributes.some((a: Node) => a.type === 'JSXAttribute' && attrName(a).ns === null && attrName(a).name === 'style' && this.staticValue(a) === undefined);
  }

  hasClassDirectives(opening: Node): boolean {
    return opening.attributes.some((a: Node) => a.type === 'JSXAttribute' && attrName(a).ns === 'class');
  }
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
