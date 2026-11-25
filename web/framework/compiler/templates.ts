// Template emission for the JSX compiler: builds static HTML templates (with the file's scope class baked in),
// walks to dynamic nodes (`firstChild` / `nextSibling`), and emits bindings in source order.

/* eslint-disable @typescript-eslint/no-explicit-any */

import { attrName, bare, IDENT, isComponentTag, isDynamic, isJsx, literal, propKey, type Node } from './ast.ts';
import { escapeAttr, escapeText, RAW_TEXT, VOID } from './html.ts';
import { eventOf } from '../runtime/names.ts';
import type { Chunk, Compiler } from './jsx.ts';

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

interface TNode {
  kind: 'el' | 'text' | 'comment';
  parent: TNode | null;
  children: TNode[];
  v?: string;
  jsx?: Node;
}

/**
 * Whether a static attribute is set as its DOM property from JS rather than written in the template: as Svelte, a
 * property given as an expression (`checked={true}`, `<option value={1}>`), and a property no template attribute sets.
 */
function propOnly(a: Node, name: string, svg: boolean): boolean {
  const prop = svg ? undefined : PROPS[name.toLowerCase()];
  return prop !== undefined && (a.value?.type === 'JSXExpressionContainer' || TEMPLATE_CANNOT_SET.has(prop));
}

function cssPropKey(prop: string): string {
  return prop[0] !== '-' || prop[1] !== '-' ? prop.toLowerCase() : prop;
}

/** One template: its HTML, the node tree, and the bindings in source order. */
export class Template {
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
    const tmpl = this.c.template(this.html, this.wrapSvg);
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
    const inlineStyle = !hasSpread && !this.hasStyleAttr(opening) && !this.hasDuplicateStyleDirectives(opening);
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
    const dirValue = (x: Node): string => {
      if (isDynamic(x)) return `${accessor(x)}()`;
      const v = `_v$${++c.uid}`;
      op([`const ${v} = `, ...c.paren(x, svg), ';']);
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
          if (dynamicClass) classDirs.push([name, 'true']);
          continue;
        }
        if (!expr) c.fail(a, 'class:name takes an expression');
        const lit = literal(expr);
        if (lit && !lit.value) continue;
        if (dynamicClass) classDirs.push([name, dirValue(expr)]);
        else if (isDynamic(expr)) op([`${c.h('toggleClass')}(${el()}, ${JSON.stringify(name)}, () => `, ...c.paren(expr, svg), ');']);
        else op([`${el()}.classList.toggle(${JSON.stringify(name)}, !!`, ...c.paren(expr, svg), ');']);
        continue;
      }
      if (ns === 'style') {
        const important = name.endsWith('|important');
        const prop = important ? name.slice(0, -10) : name;
        // `style:x={expr}`, or a string `style:x="1px"` (as Svelte allows).
        const sv = expr ? null : this.staticValue(a);
        if (!expr && typeof sv !== 'string') c.fail(a, 'style:prop takes a value: style:prop={expr} or style:prop="..."');
        if (inlineStyle) {
          if (!expr) {
            if (sv !== '') op([`${el()}.style.setProperty(${JSON.stringify(prop)}, ${JSON.stringify(sv)}${important ? ', "important"' : ''});`]);
          } else if (isDynamic(expr)) {
            op([`${c.h('styleProp')}(${el()}, ${JSON.stringify(prop)}, () => `, ...c.paren(expr, svg), `${important ? ', true' : ''});`]);
          } else {
            const lit = literal(expr);
            if (lit && (lit.value == null || lit.value === '')) continue;
            op([`${c.h('setStyleProp')}(${el()}, ${JSON.stringify(prop)}, `, ...c.paren(expr, svg), `${important ? ', true' : ''});`]);
          }
          continue;
        }
        // Applied with the style attribute (Svelte's set_style with directives), in one binding after the attributes.
        const valRef = expr ? dirValue(expr) : JSON.stringify(sv);
        (important ? styleImportant : styleDirs).push([prop, valRef]);
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
    const dirs = classDirs.length ? `() => ({ ${classDirs.map(([k, v]) => `${JSON.stringify(k)}: ${v}`).join(', ')} })` : null;
    const styleObject = (list: [string, string][]) => (list.length ? `() => ({ ${list.map(([k, v]) => `${JSON.stringify(k)}: ${v}`).join(', ')} })` : null);
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

  hasStyleAttr(opening: Node): boolean {
    return opening.attributes.some((a: Node) => a.type === 'JSXAttribute' && attrName(a).ns === null && attrName(a).name === 'style');
  }

  hasStyleDirectives(opening: Node): boolean {
    return opening.attributes.some((a: Node) => a.type === 'JSXAttribute' && attrName(a).ns === 'style');
  }

  hasDuplicateStyleDirectives(opening: Node): boolean {
    const seen = new Set<string>();
    for (const a of opening.attributes) {
      if (a.type !== 'JSXAttribute') continue;
      const { ns, name } = attrName(a);
      if (ns !== 'style') continue;
      const prop = name.endsWith('|important') ? name.slice(0, -10) : name;
      const key = cssPropKey(prop);
      if (seen.has(key)) return true;
      seen.add(key);
    }
    return false;
  }

  hasClassDirectives(opening: Node): boolean {
    return opening.attributes.some((a: Node) => a.type === 'JSXAttribute' && attrName(a).ns === 'class');
  }
}
