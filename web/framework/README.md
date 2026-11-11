# fw — FOLO Studio's UI framework

fw replaces Svelte 5 in FOLO Studio. It is the design that tops js-framework-benchmark (Solid / dom-expressions),
written for this app: JSX compiled at build time into cloned DOM templates plus one fine-grained binding per dynamic
part, on a push-pull signal graph. No virtual DOM, no diffing of trees, no per-update allocation in the hot paths.
Its Svelte-facing surface (attribute names, events, `class:`/`style:`/`bind:`/`use:`/transitions, scoped CSS with
Svelte's exact selector rewriting) exists so the ~150 components port one-to-one and render pixel-identically.

- `runtime/` — the runtime, imported as `fw` (`fw/internal` is what compiled code calls).
- `compiler/` — the Vite plugin: JSX transform (oxc-parser + magic-string, exact source maps), scoped CSS
  (postcss + postcss-selector-parser, mirrors Svelte 5's analyzer and transformer), `$icons/<name>` lucide modules.
- `jsx.d.ts`, `icons.d.ts` — JSX types (Svelte's attribute types from svelte/elements, plus fw's directives).
- `testing.ts` — `render()` / `flush()` / `fromHTML()` for vitest + happy-dom.
- `parity/` — the proof page twice (fw and Svelte) and `check.mjs`, the pixel and DOM diff.
- `bench/` — the js-framework-benchmark operations, fw vs Svelte 5 (`run.mjs`), and the runtime's size (`size.mjs`).
- `PORTING.md` — how to port a component, construct by construct. **Read it before porting anything.**

```sh
npx vitest run framework/             # runtime, compiler and CSS-parity tests
npx tsc -p tsconfig.next.json         # types of the fw app and the framework
node framework/parity/check.mjs       # 0 differing pixels, dark and light, page and open menu
node framework/bench/run.mjs          # fw vs Svelte 5, Chrome
node framework/bench/size.mjs         # runtime size, min + gzip
npm run dev  →  http://localhost:5173/next.html   (the fw app; index.html stays the Svelte app)
```

## Design

### Compile time

Each outermost JSX expression becomes a module-level template and a small function. This component

```tsx
<tr class:danger={selected.get() === id}>
  <td class="col-md-1">{id}</td>
  <td class="col-md-4"><a onclick={() => selected.set(id)}>{label.get()}</a></td>
</tr>
```

compiles to (abridged)

```js
const _tmpl$1 = _$template('<tr><td class="col-md-1"></td><td class="col-md-4"><a></a></td></tr>');
(() => {
  const _el$1 = _tmpl$1(), _el$2 = _el$1.firstChild, _el$3 = _el$2.nextSibling, _el$4 = _el$3.firstChild;
  _$toggleClass(_el$1, "danger", () => (selected.get() === id));
  _$insert(_el$2, (id));                       // static: set once
  _el$4.$$click = (() => selected.set(id));    // delegated
  _$insert(_el$4, () => (label.get()));        // one render effect
  return _el$1;
})()
```

- Static HTML (with the file's scope class baked in) is parsed once and cloned with `cloneNode(true)`; dynamic nodes
  are reached by `firstChild`/`nextSibling` walks, never queries. SVG fragments parse inside `<svg>`.
- An expression is reactive only if it contains a call or a property read; identifiers and literals are set once.
- `{a ? b : c}` and `{a && b}` in child position compile to a truthiness memo plus an owning memo: a branch is
  created when the condition flips, as `{#if}` does.
- Components are calls with a props object of getters (`get title() { return expr }`), so `props.title` read where it
  is used is tracked; children are a function (Svelte's snippet), a single `{(x) => ...}` child is passed through.
  A prop expression Svelte would memoize (`$.derived`: it calls something, or is more than an identifier or member
  read) reads a lazy computed (`propMemo`), so the child re-runs only when the value changes.
- Every JS expression stays at its original place in the output (generated code fills the gaps between them), so the
  source map is exact for all embedded code.
- Text follows JSX whitespace rules (see PORTING.md §8 for the Svelte differences). Adjacent static text and
  expressions compile to one text node with one binding (`run`), where Svelte makes one: Chrome shapes per text node.
  An expression that renders a node or a snippet splits its run into one insert per part, once.
- Booleans on attributes follow Svelte: dropped on false only for its boolean attributes, stringified otherwise.

### Reactivity (`runtime/core.ts`)

- **Signals** hold a value and a version. **Computeds** are lazy: recomputed on read, only if a source's version moved
  (checked transitively, so a diamond recomputes once: glitch-free). A computed with no subscribers holds no
  subscriptions, so it is garbage-collected with whatever created it.
- **Effects**: a write pushes a "maybe stale" mark down the graph and queues effects; a flush (microtask, or
  `flush()`) runs render effects (DOM bindings) first, then user effects, parents before children, each re-running only
  if a source actually changed.
- Dependencies are doubly linked nodes shared by the source's subscriber list and the target's source list. A re-run
  walks its previous links and reuses those it reads again (no allocation in steady state; a test asserts it).
- **Owners**: components, effects and roots form a tree; disposing a node disposes its children (in order) then runs
  its cleanups. Context lives on owners (each component gets one, so `setContext` scopes like Svelte's).
- **Memos** (`memo`) are owning computeds: what a branch creates belongs to it, so reading a branch from several
  places, or again, never re-creates it. Functions are either *readers* (memos, list accessors: read in place) or
  *snippets* (called once in an owner of their own).

### DOM (`runtime/dom.ts`, `flow.ts`, `events.ts`)

- `insert` renders text (reusing its text node), nodes, arrays and reactive functions before a marker node (static
  sibling, `<!>` comment) or as the parent's only content (then clearing is one `textContent = ''`).
- Lists are reconciled by node identity: common prefix and suffix skipped, end-for-end swaps handled directly, then
  the longest increasing subsequence of old positions stays and only the rest moves (`moveBefore` keeps focus where
  the browser has it). `For` maps rows by key (an item replaced under the same key updates the row in place),
  `Index` by position.
- Events: Svelte 5's delegated set (click, input, keydown, pointer*, ...) goes through one listener per type on each
  render root and the document, walking the event path with Svelte's propagation algorithm; others use
  `addEventListener`.
- Attributes, properties, `class` (clsx, scope class, `class:`), `style` (string or object, `style:`), spreads and
  `bind:` follow Svelte 5's runtime (`set_attribute`, `set_class`, `set_style`, `set_attributes`, `bind_value`...).
- Transitions: Svelte's engine (keyframes sampled from `css(t, u)`, Web Animations) with local intros/outros on
  `For`/`Show`/`Key`/`Switch` branches.

### Scoped CSS (`compiler/css.ts`)

`css\`...\`` in a `.tsx` is extracted at build time, scoped, and served as the virtual module `<file>.fw.css` that the
file imports after its own imports, as vite-plugin-svelte does, so the bundle has Svelte's cascade order (a file's
rules after its children's; `parity/css-order.mjs` checks the built order against the Svelte app's). Vite's CSS
pipeline does the rest; HMR updates it. The plugin also fails the build on an import that resolves to a file whose
name differs in case (`$lib/ui/Field` → `field.ts` on a case-insensitive disk). The scoping mirrors Svelte 5's
`phases/2-analyze/css` and `3-transform/css`: the first scoped compound of a selector gets `.fw-xxxx` (+0-1-0), later
ones `:where(.fw-xxxx)`, nested rules under a local rule `:where(...)` only, `:global(...)`/`:global {}` and leading or
trailing `:global` unscoped, `:is/:where/:has` arguments scoped like the selector, `:not` arguments only when complex
(and nothing nested inside them), `@keyframes` names prefixed (`-global-` opts out) and `animation(-name)` following.

`compiler/css.test.ts` compiles every `<style>` of every `.svelte` file in `web/src` with the Svelte compiler and
with ours: **171 files, 3,950 selectors, all identical in text and specificity.** By design we keep selectors Svelte
drops as unused (we do not see the elements); they are scoped, so they cannot match outside the file.

The scope class prefix is **`fw-`** (`fw-` + base-36 FNV-1a of the file's path relative to `web/`), e.g. `fw-fyng9x`.

## API

```ts
import { ... } from 'fw';
```

| Area | API |
|---|---|
| State | `signal(v, { equals? })` → `{ get, set, peek }` (`state` alias) · `computed(fn, { equals? })` → `{ get, peek }` · `deep(obj)` (proxy) · `snapshot(x)` |
| Effects | `effect(fn)` (after DOM updates; returns cleanup) · `effect.pre(fn)` · `renderEffect` (internal) · `onMount` · `onCleanup` / `onDestroy` · `untrack` · `isTracking` |
| Scheduling | `flush()` (≙ `flushSync`) · `tick()` · `batch(fn)` |
| Ownership | `createRoot((dispose) => …, parent?)` · `getOwner` · `runWithOwner` · `dispose` · `memo` |
| Context | `setContext` · `getContext` · `hasContext` · `createContext(default)` |
| Components | `Component<P>` · `ComponentProps<C>` · `ElementProps<'button'>` · `defaults(props, {...})` · `splitProps(props, keys)` · `mergeProps(...)` · `bindable(props, key, fallback?)` · `field(obj, key)` · `createUniqueId()` (`propsId`) |
| Control flow | `<Show when keyed? fallback>` · `<For each key? fallback>{(item, i) => …}` · `<Index each fallback>` · `<Switch fallback><Match when>` · `<Key when>` · `<Dynamic component={tagOrComponent} …>` · `<Portal mount?>` |
| DOM | `render(() => <App />, el)` → unmount · `on(target, type, fn)` · `onWindow` · `onDocument` · `clsx` |
| Reactivity utils | `MediaQuery` / `mediaQuery(q)` (`.current`) · `ReactiveMap` · `ReactiveSet` · `createSubscriber` |
| Motion | `fade` · `fly` · `linear` · `cubicOut` · `cubicInOut` · `Tween` |
| Styles | `` css`...` `` (compile-time) |
| Icons | `import X from '$icons/x'` (lucide, from @lucide/svelte's data) · `IconComponent` |
| JSX attributes | `class` · `class:x` · `style` (string or object) · `style:prop` · `on<event>` / `on<event>capture` · `ref` · `use:action` / `use` · `bind:value/checked/group/open/this/clientWidth/clientHeight/offsetWidth/offsetHeight` · `transition:` / `in:` / `out:` · `{...spread}` · `innerHTML` |
| Tests (`framework/testing`) | `render(() => <X />, target?)` → `{ container, unmount }` · `flush` · `tick` · `fromHTML(html)` |

## Performance

Chrome (stable, headless), production builds of the same table in fw (`bench/main-fw.tsx`) and Svelte 5.57
(`bench/Bench.svelte`, the js-framework-benchmark keyed Svelte 5 shape), interleaved runs, median of 15. "Script" is
the framework's work until the DOM is updated; "total" adds the forced style/layout.

| operation (ms) | Svelte 5 total | fw total | Svelte 5 script | fw script |
|---|---:|---:|---:|---:|
| create 1,000 rows | 27.60 | 26.40 | 6.90 | 5.70 |
| replace all 1,000 rows | 26.60 | 23.50 | 8.80 | 5.80 |
| update every 10th row (1k) | 5.60 | 5.20 | 0.40 | 0.30 |
| select row | 0.90 | 0.60 | 0.80 | 0.50 |
| swap rows | 2.00 | 1.90 | 0.50 | 0.50 |
| remove row | 2.00 | 2.00 | 0.60 | 0.60 |
| create 10,000 rows | 223.60 | 213.40 | 41.70 | 32.80 |
| append 1,000 rows (to 1k) | 32.00 | 29.70 | 7.40 | 4.20 |
| clear 1,000 rows | 4.00 | 3.20 | 4.00 | 3.10 |
| clear 10,000 rows | 22.30 | 19.10 | 22.20 | 19.00 |

Small operations, mean of 400 repetitions on 1,000 rows (a single run is about one 0.1 ms timer tick):

| operation (ms) | Svelte 5 script | fw script | Svelte 5 + layout | fw + layout |
|---|---:|---:|---:|---:|
| swap rows | 0.084 | 0.025 | 1.030 | 0.940 |
| select row | 0.099 | 0.040 | 0.111 | 0.052 |
| update every 10th row | 0.130 | 0.095 | 4.428 | 4.035 |
| remove row | 0.050 | 0.022 | 0.765 | 0.746 |

JS heap after GC: empty 1.26 MB (Svelte 1.31), 1,000 rows 2.30 MB (2.71), 10,000 rows 9.73 MB (12.66), after five
create/clear cycles 1.52 MB (1.67): rows release everything.

Runtime size (`bench/size.mjs`, minified + gzip): the core a page needs (signals, effects, templates, `insert`,
attributes, delegated events, components) **4.7 KB**; a typical page (plus `For`, `Show`, classes, props helpers,
spreads) **7.8 KB**; every export (transitions, `Tween`, `deep`, maps, bindings, icons) **12.7 KB**. Tree-shaking keeps
what a bundle does not use out.

Where the time goes, and what keeps it low:
- Creation is a `cloneNode(true)` and a handful of `firstChild`/`nextSibling` reads per row; each dynamic part is one
  effect object (and its accessor closure).
- Updates touch only the bindings whose sources changed (a text node's `data`, one class toggle).
- Re-running an effect reuses its dependency links; reconciling a list allocates one array and, only when rows moved,
  a map and an `Int32Array`.
- Clearing a list that owns its parent is one `textContent = ''`.

## Svelte semantics not matched exactly

- Unused-selector pruning (above). Svelte also scopes only the `:is()` alternatives that match an element; we scope
  all (no selector of the app is affected: the parity test has zero differences).
- `bind:this` is set synchronously at creation (Svelte: after mount); with a signal ref it is cleared synchronously
  when the element goes (Svelte: a microtask later).
- Outros: `For`/`Show`/`Key`/`Switch` run them; a ternary (`{a ? b : c}`) does not, and an outro is not reversed if
  the same keyed row comes back while it runs (it is a new row). Effects of a removed branch stop at once (Svelte keeps
  them paused during the outro).
- `<Icon/>` with a component from a variable is not reactive to the variable: `<Dynamic component={...}>` is.
- Whitespace follows JSX, not Svelte (PORTING.md §8).
- A keyed row's item is an accessor (`item()`), not a variable; unkeyed `{#each}` is `<Index>`.
- Svelte throws on duplicate keys in a keyed `{#each}`; `For` renders duplicates.
