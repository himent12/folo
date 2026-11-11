// Scoped CSS with Svelte 5's semantics, mirrored from svelte/src/compiler/phases/{2-analyze,3-transform}/css: the
// first scoped compound of a selector gets `.scope` (a +0-1-0 specificity bump, as Svelte's hash class), every later
// one `:where(.scope)`; nested rules under a local rule get `:where(.scope)` only; `:global(...)`, `:global` blocks and
// `:global` at either end of a selector stay unscoped; `:is/:where/:has` arguments are scoped like the selector,
// `:not` arguments only when they are complex; `@keyframes` names get the scope as a prefix (`-global-` opts out) and
// `animation` / `animation-name` follow.
//
// One difference by design: Svelte drops (comments out) selectors that match no element of its component, and leaves
// `:is()` alternatives that match nothing unscoped. We cannot see the elements, so we keep and scope them all. A
// dropped selector matched nothing in Svelte; here it is scoped, so it still only matches this file's elements.

import postcss, { type AtRule, type ChildNode, type Container, type Declaration, type Rule } from 'postcss';
import selectorParser from 'postcss-selector-parser';

type SNode = selectorParser.Node;
type Selector = selectorParser.Selector;
type Pseudo = selectorParser.Pseudo;

interface Rel {
  comb: selectorParser.Combinator | null;
  nodes: SNode[];
  global: boolean;
  globalLike: boolean;
  scoped: boolean;
}

interface RuleInfo {
  rule: Rule;
  parent: RuleInfo | null;
  complexes: { sel: Selector; rels: Rel[] }[];
  globalBlock: boolean;
  hasLocal: boolean;
}

const VIEW_TRANSITION = ['view-transition', 'view-transition-group', 'view-transition-old', 'view-transition-new', 'view-transition-image-pair'];
const PREFIX = /^-((webkit)|(moz)|(o)|(ms))-/;

const isPseudo = (n: SNode): n is Pseudo => n.type === 'pseudo';
const isElement = (n: SNode) => isPseudo(n) && n.value.startsWith('::');
const pname = (n: Pseudo) => n.value.replace(/^::?/, '').toLowerCase();
const hasArgs = (n: Pseudo) => n.nodes.length > 0;
const isGlobalPseudo = (n: SNode): n is Pseudo => isPseudo(n) && !isElement(n) && pname(n) === 'global';

/** Splits a complex selector into Svelte's relative selectors (a combinator and a compound). */
function split(sel: Selector): Rel[] {
  const rels: Rel[] = [];
  let cur: Rel = { comb: null, nodes: [], global: false, globalLike: false, scoped: false };
  for (const n of sel.nodes) {
    if (n.type === 'comment') continue;
    if (n.type === 'combinator') {
      if (cur.nodes.length || rels.length || cur.comb) rels.push(cur);
      cur = { comb: n, nodes: [], global: false, globalLike: false, scoped: false };
      continue;
    }
    cur.nodes.push(n);
  }
  rels.push(cur);
  return rels.filter((r) => r.nodes.length > 0);
}

function complexRels(sel: Selector): Rel[] {
  const rels = split(sel);
  for (const r of rels) analyzeRel(r);
  return rels;
}

/** Svelte's `is_unscoped_pseudo_class`. */
function unscopedPseudoClass(n: SNode): boolean {
  if (!isPseudo(n) || isElement(n)) return false;
  const name = pname(n);
  if (!hasArgs(n)) return true;
  if (name !== 'has' && name !== 'is' && name !== 'where' && (name !== 'not' || (n.nodes as Selector[]).every((c) => split(c).length === 1))) return true;
  return (n.nodes as Selector[]).every((c) => complexRels(c).every((r) => r.global));
}

/** Svelte's `is_global`: `:global` / `:global(...)` with only unscoped pseudos after it. */
function relIsGlobal(r: Rel): boolean {
  const first = r.nodes[0];
  if (!first || !isGlobalPseudo(first)) return false;
  return !hasArgs(first) || r.nodes.every((s) => unscopedPseudoClass(s) || isElement(s));
}

/** Svelte's `is_outer_global`. */
function relIsOuterGlobal(r: Rel): boolean {
  const first = r.nodes[0];
  if (!first || !isGlobalPseudo(first)) return false;
  return !hasArgs(first) || r.nodes.every((s) => isPseudo(s));
}

function analyzeRel(r: Rel): void {
  r.global = relIsGlobal(r);
  if (r.nodes.every(isPseudo)) {
    const first = r.nodes[0] as Pseudo;
    if ((!isElement(first) && pname(first) === 'host') || (isElement(first) && VIEW_TRANSITION.includes(pname(first)))) r.globalLike = true;
  }
  if (r.nodes.some((s) => isPseudo(s) && !isElement(s) && pname(s) === 'root') && !r.nodes.some((s) => isPseudo(s) && !isElement(s) && pname(s) === 'has')) r.globalLike = true;
}

/** Svelte's `truncate`: drops trailing global relatives (they do not scope). */
function truncate(rels: Rel[]): Rel[] {
  let i = rels.length - 1;
  while (i >= 0) {
    const r = rels[i]!;
    const first = r.nodes[0]!;
    if (!r.globalLike && !(isGlobalPseudo(first) && !hasArgs(first)) && !r.global) break;
    i--;
  }
  return rels.slice(0, i + 1);
}

function parseSelector(s: string): selectorParser.Root {
  return selectorParser().astSync(s, { lossless: true });
}

class Scoper {
  readonly hash: string;
  readonly keyframes: string[] = [];
  constructor(hash: string) {
    this.hash = hash;
  }

  modifier(bumped: boolean): SNode {
    const cls = selectorParser.className({ value: this.hash });
    if (!bumped) return cls;
    return selectorParser.pseudo({ value: ':where', nodes: [selectorParser.selector({ nodes: [cls], value: '' })] });
  }

  info(rule: Rule, parent: RuleInfo | null): RuleInfo {
    const root = parseSelector(rule.selector);
    const complexes = root.nodes.map((sel) => ({ sel, rels: complexRels(sel) }));
    let globalBlock = false;
    for (const { rels } of complexes) {
      let afterMarker = false;
      for (const r of rels) {
        if (afterMarker) r.globalLike = true;
        const idx = r.nodes.findIndex((n) => isGlobalPseudo(n) && !hasArgs(n));
        if (idx === 0) globalBlock = afterMarker = true;
      }
    }
    const hasLocal = complexes.some(({ rels }) => !rels.every((r) => r.global || r.globalLike));
    const info: RuleInfo = { rule, parent, complexes, globalBlock, hasLocal };
    (rule as Rule & { __root?: selectorParser.Root }).__root = root;
    return info;
  }

  run(root: postcss.Root): void {
    // Keyframes declared here (not -global-, not inside a :global block) are scoped.
    const collect = (node: Container, inGlobal: boolean) => {
      node.each((child: ChildNode) => {
        if (child.type === 'atrule') {
          if (child.name.replace(PREFIX, '') === 'keyframes') {
            const name = child.params.trim();
            if (!name.startsWith('-global-') && !inGlobal) this.keyframes.push(name);
            return;
          }
          if (child.nodes) collect(child, inGlobal);
        } else if (child.type === 'rule') {
          const sel = parseSelector(child.selector);
          const block = sel.nodes.some((s) => split(s).some((r) => isGlobalPseudo(r.nodes[0]!) && !hasArgs(r.nodes[0] as Pseudo)));
          collect(child, inGlobal || block);
        }
      });
    };
    collect(root, false);
    this.container(root, null, false);
  }

  container(node: Container, parent: RuleInfo | null, inGlobal: boolean): void {
    for (const child of [...(node.nodes ?? [])]) {
      if (child.type === 'atrule') this.atrule(child, parent, inGlobal);
      else if (child.type === 'rule') this.rule(child, parent, inGlobal);
      else if (child.type === 'decl') this.decl(child);
    }
  }

  atrule(node: AtRule, parent: RuleInfo | null, inGlobal: boolean): void {
    if (node.name.replace(PREFIX, '') === 'keyframes') {
      const name = node.params.trim();
      if (name.startsWith('-global-')) node.params = node.params.replace('-global-', '');
      else if (!inGlobal) node.params = node.params.replace(name, `${this.hash}-${name}`);
      return;
    }
    if (node.nodes) this.container(node, parent, inGlobal);
  }

  decl(node: Declaration): void {
    const prop = node.prop.toLowerCase().replace(PREFIX, '');
    if (prop !== 'animation' && prop !== 'animation-name') return;
    const value = node.value + ';';
    let out = '';
    let name = '';
    for (let i = 0; i < value.length; i++) {
      const ch = value[i]!;
      if (/^[\s,;}]$/.test(ch)) {
        out += (this.keyframes.includes(name) ? `${this.hash}-` : '') + name;
        if (ch === ';' || ch === '}') {
          out += value.slice(i, -1);
          break;
        }
        out += ch;
        name = '';
      } else name += ch;
    }
    node.value = out;
  }

  rule(rule: Rule, parent: RuleInfo | null, inGlobal: boolean): void {
    const info = this.info(rule, parent);
    const root = (rule as Rule & { __root?: selectorParser.Root }).__root!;
    if (inGlobal) {
      for (const { rels } of info.complexes) this.stripGlobals(rels);
      rule.selector = root.toString();
      this.container(rule, info, true);
      return;
    }
    const lone = info.complexes.length === 1 && info.complexes[0]!.rels.length === 1 && info.complexes[0]!.rels[0]!.nodes.length === 1 && info.globalBlock;
    if (lone) {
      // `:global { ... }`: its content, unscoped, in its place.
      this.container(rule, info, true);
      rule.replaceWith(...(rule.nodes ?? []));
      return;
    }
    let bumped = false;
    for (let r = parent; r !== null; r = r.parent) if (r.hasLocal) bumped = true;
    for (const { sel, rels } of info.complexes) {
      for (const r of truncate(rels)) if (!relIsOuterGlobal(r) && !r.globalLike) r.scoped = true;
      this.complex(sel, rels, { bumped }, parent !== null);
    }
    rule.selector = root.toString();
    this.container(rule, info, info.globalBlock);
  }

  /** Svelte's ComplexSelector visitor: modifiers on scoped relatives, `:global` removed, then pseudo arguments. */
  complex(sel: Selector, rels: Rel[], state: { bumped: boolean }, nested: boolean, inNot = false): void {
    const before = state.bumped;
    const pseudos: { p: Pseudo; name: string }[] = [];
    for (const r of rels)
      for (const n of r.nodes) {
        if (isPseudo(n) && !isElement(n) && hasArgs(n)) {
          const name = pname(n);
          if (name === 'is' || name === 'where' || name === 'has' || name === 'not') pseudos.push({ p: n, name });
        }
      }
    for (const r of rels) {
      if (r.global) {
        const g = r.nodes[0] as Pseudo;
        if (!hasArgs(g)) {
          if (nested && r.comb === null) {
            const amp = selectorParser.nesting({ value: '&' });
            g.parent!.insertBefore(g, amp);
          }
          if (r.comb && r.comb.value.trim() === '') r.comb.remove();
          g.remove();
          const next = r.nodes[1];
          if (next && !r.comb) next.spaces.before = g.spaces.before;
        } else this.unwrap(g, sel);
        continue;
      }
      if (r.scoped && !(r.nodes.length === 1 && isPseudo(r.nodes[0]!) && ['is', 'where'].includes(pname(r.nodes[0] as Pseudo))) && !r.nodes.some((n) => n.type === 'nesting')) {
        const mod = this.modifier(state.bumped);
        state.bumped = true;
        let i = r.nodes.length;
        while (i--) {
          const n = r.nodes[i]!;
          if (isPseudo(n)) {
            const name = pname(n);
            if (name !== 'root' && name !== 'host' && i === 0) {
              mod.spaces.before = n.spaces.before;
              n.spaces.before = '';
              n.parent!.insertBefore(n, mod);
            }
            continue;
          }
          if (n.type === 'universal' && !(n as { namespace?: unknown }).namespace) {
            mod.spaces.before = n.spaces.before;
            mod.spaces.after = n.spaces.after;
            n.replaceWith(mod);
          } else {
            mod.spaces.after = n.spaces.after;
            n.spaces.after = '';
            n.parent!.insertAfter(n, mod);
          }
          break;
        }
      }
      for (const n of [...r.nodes]) if (isGlobalPseudo(n) && n.parent) this.unwrap(n, sel);
    }
    for (const { p, name } of pseudos) {
      for (const arg of p.nodes as Selector[]) {
        const argRels = complexRels(arg);
        const t = truncate(argRels);
        // Svelte scopes `:not` arguments only when complex, and nothing nested inside a `:not` argument.
        if (!inNot && (name !== 'not' || argRels.length > 1)) for (const r of t) if (!relIsOuterGlobal(r) && !r.globalLike) r.scoped = true;
        this.complex(arg, argRels, state, nested, inNot || name === 'not');
      }
    }
    state.bumped = before;
  }

  /** `:global(x)` → `x` (in place). */
  unwrap(g: Pseudo, sel: Selector): void {
    if (!g.parent) return;
    if (!hasArgs(g)) {
      g.remove();
      return;
    }
    const args = g.nodes as Selector[];
    if (args.length > 1 && sel.nodes.filter((n) => n.type !== 'comment').length === 1) {
      // A standalone `:global(a, b)`: the selector list itself.
      const root = sel.parent!;
      for (const a of args) root.insertBefore(sel, a.clone() as Selector);
      sel.remove();
      return;
    }
    const inner = args[0]!.nodes.map((n) => n.clone());
    if (inner[0]) inner[0].spaces.before = g.spaces.before;
    const last = inner[inner.length - 1];
    if (last) last.spaces.after = g.spaces.after;
    let ref: SNode = g;
    for (const n of inner) {
      g.parent!.insertAfter(ref, n);
      ref = n;
    }
    g.remove();
  }

  stripGlobals(rels: Rel[]): void {
    for (const r of rels) {
      for (const n of [...r.nodes]) {
        if (!isGlobalPseudo(n) || !n.parent) continue;
        if (!hasArgs(n) && r.comb && r.comb.value.trim() === '' && n === r.nodes[0]) r.comb.remove();
        this.unwrap(n, n.parent as Selector);
      }
      for (const n of r.nodes) {
        if (isPseudo(n) && hasArgs(n) && ['is', 'where', 'has', 'not'].includes(pname(n))) for (const arg of n.nodes as Selector[]) this.stripGlobals(complexRels(arg));
      }
    }
  }
}

/** Scopes a component's CSS (the content of its `css\`\`` or `<style>`) with `hash` as Svelte 5 would. */
export function scopeCss(css: string, hash: string): string {
  const root = postcss.parse(css);
  new Scoper(hash).run(root);
  return root.toString();
}

/** Specificity [a, b, c] of one complex selector, per Selectors 4 (`:where` 0, `:is/:not/:has` their most specific). */
export function specificity(selector: string): [number, number, number] {
  const root = parseSelector(selector);
  const of = (sel: Selector): [number, number, number] => {
    let a = 0;
    let b = 0;
    let c = 0;
    for (const n of sel.nodes) {
      switch (n.type) {
        case 'id':
          a++;
          break;
        case 'class':
        case 'attribute':
          b++;
          break;
        case 'tag':
          c++;
          break;
        case 'pseudo': {
          const name = pname(n);
          if (isElement(n) || ['before', 'after', 'first-line', 'first-letter'].includes(name)) c++;
          else if (name === 'where') break;
          else if (['is', 'not', 'has', 'matches', 'any', 'nth-child', 'nth-last-child'].includes(name) && hasArgs(n)) {
            let best: [number, number, number] = [0, 0, 0];
            const args = name.startsWith('nth') ? [] : (n.nodes as Selector[]);
            for (const s of args) {
              const sp = of(s);
              if (cmp(sp, best) > 0) best = sp;
            }
            if (name.startsWith('nth')) b++;
            a += best[0];
            b += best[1];
            c += best[2];
          } else b++;
          break;
        }
      }
    }
    return [a, b, c];
  };
  return of(root.nodes[0]!);
}

function cmp(x: [number, number, number], y: [number, number, number]): number {
  return x[0] - y[0] || x[1] - y[1] || x[2] - y[2];
}
