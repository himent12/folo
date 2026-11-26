// AST queries the JSX transform needs (ESTree-shaped nodes from oxc-parser): JSX, reactivity (what is dynamic, what
// Svelte would memoize), literals and names, and what the file tells about snippets (`analyze`).

/* eslint-disable @typescript-eslint/no-explicit-any */

export type Node = any;

export function isJsx(n: Node): boolean {
  return n !== null && typeof n === 'object' && (n.type === 'JSXElement' || n.type === 'JSXFragment');
}

/** Calls `fn` on every child node of `n` (ESTree-shaped objects). */
export function eachChild(n: Node, fn: (c: Node) => void): void {
  for (const key in n) {
    if (key === 'type' || key === 'start' || key === 'end' || key === 'range' || key === 'loc') continue;
    const v = n[key];
    if (Array.isArray(v)) {
      for (const x of v) if (x && typeof x === 'object' && typeof x.type === 'string') fn(x);
    } else if (v && typeof v === 'object' && typeof v.type === 'string') fn(v);
  }
}

/** The outermost JSX nodes under `n` (not `n` itself), in source order. */
export function jsxWithin(n: Node): Node[] {
  const out: Node[] = [];
  const visit = (c: Node) => {
    if (isJsx(c)) out.push(c);
    else eachChild(c, visit);
  };
  eachChild(n, visit);
  return out.sort((a, b) => a.start - b.start);
}

export const FUNCTION_TYPES = new Set(['ArrowFunctionExpression', 'FunctionExpression', 'FunctionDeclaration', 'ClassExpression', 'ClassDeclaration']);

/**
 * Whether evaluating `n` may read reactive state: it calls something or reads a property (a getter, a signal). Plain
 * identifiers and literals are static; function bodies and nested JSX do not count (they are not evaluated here).
 */
export function isDynamic(n: Node): boolean {
  if (!n || typeof n !== 'object') return false;
  switch (n.type) {
    case 'CallExpression':
    case 'NewExpression':
    case 'MemberExpression':
    case 'TaggedTemplateExpression':
    case 'ChainExpression':
    case 'ImportExpression':
    case 'AwaitExpression':
    case 'YieldExpression':
    case 'StaticMemberExpression':
    case 'ComputedMemberExpression':
    case 'PrivateFieldExpression':
      return true;
    case 'JSXElement':
    case 'JSXFragment':
      return false;
  }
  if (FUNCTION_TYPES.has(n.type)) return false;
  let found = false;
  eachChild(n, (c) => {
    if (!found && isDynamic(c)) found = true;
  });
  return found;
}

/** Whether `n` calls something (outside nested functions and JSX): Svelte's `has_call`. */
export function hasCall(n: Node): boolean {
  if (!n || typeof n !== 'object') return false;
  if (n.type === 'CallExpression' || n.type === 'NewExpression' || n.type === 'TaggedTemplateExpression' || n.type === 'ImportExpression' || n.type === 'AwaitExpression') return true;
  if (FUNCTION_TYPES.has(n.type) || isJsx(n)) return false;
  let found = false;
  eachChild(n, (c) => {
    if (!found && hasCall(c)) found = true;
  });
  return found;
}

export function containsJsx(n: Node): boolean {
  return isJsx(bare(n)) || jsxWithin(n).length > 0;
}

/** Whether evaluating `n` directly creates JSX (outside nested function bodies). */
export function evaluatesJsx(n: Node): boolean {
  if (!n || typeof n !== 'object') return false;
  if (isJsx(n)) return true;
  if (FUNCTION_TYPES.has(n.type)) return false;
  let found = false;
  eachChild(n, (c) => {
    if (!found && evaluatesJsx(c)) found = true;
  });
  return found;
}

/**
 * What the file's code tells about expressions that render more than text (Svelte's `{@render}`, which never joins a
 * text sequence): `snippets`, names whose every binding in the file is JSX or a function with JSX in it
 * (`const link = () => <a/>`), and `snippetProps`, property names typed `JSX.Element` (or a function returning one) in
 * the file's types, plus `children`.
 */
export interface Knowledge {
  snippets: Set<string>;
  snippetProps: Set<string>;
}

export function patternNames(p: Node, out: string[]): void {
  if (!p || typeof p !== 'object') return;
  switch (p.type) {
    case 'Identifier':
      out.push(p.name);
      return;
    case 'AssignmentPattern':
      patternNames(p.left, out);
      return;
    case 'RestElement':
      patternNames(p.argument, out);
      return;
    case 'ArrayPattern':
      for (const e of p.elements) patternNames(e, out);
      return;
    case 'ObjectPattern':
      for (const pr of p.properties) patternNames(pr.type === 'RestElement' ? pr : pr.value, out);
      return;
    case 'TSParameterProperty':
      patternNames(p.parameter, out);
      return;
  }
}

export function isJsxType(t: Node): boolean {
  if (!t) return false;
  if (t.type === 'TSTypeAnnotation') return isJsxType(t.typeAnnotation);
  if (t.type === 'TSParenthesizedType') return isJsxType(t.typeAnnotation);
  if (t.type === 'TSTypeReference') {
    const n = t.typeName;
    return n.type === 'TSQualifiedName' && n.left.type === 'Identifier' && n.left.name === 'JSX' && n.right.name === 'Element';
  }
  if (t.type === 'TSFunctionType') return isJsxType(t.returnType);
  if (t.type === 'TSUnionType') {
    const rest = t.types.filter((x: Node) => x.type !== 'TSUndefinedKeyword' && x.type !== 'TSNullKeyword');
    return rest.length > 0 && rest.every(isJsxType);
  }
  return false;
}

export function analyze(program: Node): Knowledge {
  const bindings = new Map<string, boolean>();
  const bindName = (name: string, snippet: boolean) => bindings.set(name, (bindings.get(name) ?? true) && snippet);
  const snippetProps = new Set<string>(['children']);
  const isSnippetValue = (init: Node): boolean => {
    const b = bare(init);
    if (!b) return false;
    if (isJsx(b)) return true;
    if (isFunctionExpr(b)) return isJsx(bare(b.body)) || jsxWithin(b.body).length > 0;
    return false;
  };
  const visit = (n: Node) => {
    switch (n.type) {
      case 'VariableDeclarator': {
        const names: string[] = [];
        patternNames(n.id, names);
        const snippet = n.id.type === 'Identifier' && n.init != null && isSnippetValue(n.init);
        for (const name of names) bindName(name, snippet);
        break;
      }
      case 'FunctionDeclaration':
        if (n.id) bindName(n.id.name, jsxWithin(n.body).length > 0);
        break;
      case 'ClassDeclaration':
        if (n.id) bindName(n.id.name, false);
        break;
      case 'ImportSpecifier':
      case 'ImportDefaultSpecifier':
      case 'ImportNamespaceSpecifier':
        bindName(n.local.name, false);
        break;
      case 'CatchClause':
        if (n.param) {
          const names: string[] = [];
          patternNames(n.param, names);
          for (const name of names) bindName(name, false);
        }
        break;
      case 'TSPropertySignature':
        if (n.key?.type === 'Identifier' && isJsxType(n.typeAnnotation)) snippetProps.add(n.key.name);
        break;
      case 'TSMethodSignature':
        if (n.key?.type === 'Identifier' && isJsxType(n.returnType)) snippetProps.add(n.key.name);
        break;
    }
    if (FUNCTION_TYPES.has(n.type) && Array.isArray(n.params)) {
      const names: string[] = [];
      for (const p of n.params) patternNames(p, names);
      for (const name of names) bindName(name, false);
    }
    eachChild(n, visit);
  };
  visit(program);
  const snippets = new Set<string>();
  for (const [name, snippet] of bindings) if (snippet) snippets.add(name);
  return { snippets, snippetProps };
}

export function memberName(n: Node): string | null {
  if ((n.type === 'MemberExpression' || n.type === 'StaticMemberExpression') && !n.computed && n.property?.type === 'Identifier') return n.property.name;
  return null;
}

/**
 * Svelte 5's rule for a component prop: memoized when it calls something, or reads state through anything but a
 * plain identifier or member read (`active={i === index}`, `src={err ? null : src}`). For a template literal, its
 * expressions decide (they are Svelte's attribute chunks).
 */
export function needsDerived(x: Node): boolean {
  if (hasCall(x)) return true;
  const b = bare(x);
  if (b.type === 'TemplateLiteral') return b.expressions.some((e: Node) => isDynamic(e) && needsDerived(e));
  return b.type !== 'Identifier' && b.type !== 'MemberExpression';
}

export function isFunctionExpr(n: Node): boolean {
  return n.type === 'ArrowFunctionExpression' || n.type === 'FunctionExpression';
}

/** Unwraps TS-only wrappers and parentheses. */
export function bare(n: Node): Node {
  while (n && (n.type === 'TSAsExpression' || n.type === 'TSNonNullExpression' || n.type === 'TSSatisfiesExpression' || n.type === 'ParenthesizedExpression' || n.type === 'TSTypeAssertion')) n = n.expression;
  return n;
}

/** A literal's value if `n` is a static literal (string, number, boolean, null, a template without expressions). */
export function literal(n: Node): { value: unknown } | null {
  n = bare(n);
  if (n.type === 'Literal' && (n.regex || n.bigint !== undefined)) return null;
  if (n.type === 'Literal' || n.type === 'StringLiteral' || n.type === 'NumericLiteral' || n.type === 'BooleanLiteral' || n.type === 'NullLiteral') return { value: n.value };
  if (n.type === 'TemplateLiteral' && n.expressions.length === 0) return { value: n.quasis[0].value.cooked };
  if (n.type === 'Identifier' && n.name === 'undefined') return { value: undefined };
  if (n.type === 'UnaryExpression' && n.operator === '-' && literal(n.argument) && typeof literal(n.argument)!.value === 'number') return { value: -(literal(n.argument)!.value as number) };
  return null;
}

export function tagName(n: Node): string {
  if (n.type === 'JSXIdentifier') return n.name;
  if (n.type === 'JSXMemberExpression') return `${tagName(n.object)}.${n.property.name}`;
  if (n.type === 'JSXNamespacedName') return `${n.namespace.name}:${n.name.name}`;
  return '';
}

export function isComponentTag(n: Node): boolean {
  if (n.type === 'JSXMemberExpression') return true;
  if (n.type !== 'JSXIdentifier') return false;
  const c = n.name[0];
  return c !== undefined && c !== c.toLowerCase();
}

export function attrName(a: Node): { ns: string | null; name: string } {
  if (a.name.type === 'JSXNamespacedName') return { ns: a.name.namespace.name, name: a.name.name.name };
  return { ns: null, name: a.name.name };
}

export const IDENT = /^[A-Za-z_$][\w$]*$/;
export const propKey = (k: string) => (IDENT.test(k) ? k : JSON.stringify(k));
