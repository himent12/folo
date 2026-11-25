# fw — FOLO Studio's UI framework

fw is FOLO Studio's UI framework; it replaced Svelte 5. It is the design that tops js-framework-benchmark (Solid / dom-expressions),
written for this app: JSX compiled at build time into cloned DOM templates plus one fine-grained binding per dynamic
part, on a push-pull signal graph. No virtual DOM, no diffing of trees, no per-update allocation in the hot paths.
Its Svelte-facing surface (attribute names, events, `class:`/`style:`/`bind:`/`use:`/transitions, scoped CSS with
Svelte's exact selector rewriting) let the ~150 components port one-to-one: every screen and state of the app rendered
pixel-identically to the Svelte app before that one was removed (CHANGELOG, "Studio on fw").

Related: [Studio's README](../README.md) (the app built on it, and how to add a screen or a component), the
[root README](../../README.md) and [docs/ARCHITECTURE.md](../../docs/ARCHITECTURE.md).
**Read [PORTING.md](PORTING.md) before writing a component**: every construct, with the Svelte 5 one it replaced.

## Where things are

| Path | What it holds |
|---|---|
| `runtime/` | The runtime, imported as `fw` (`runtime/index.ts`, the public exports) |
| `runtime/core.ts`, `links.ts`, `context.ts`, `flushOrder.ts` | Signals, lazy computeds, effects, the owner tree, dependency links, context, the scheduler and tree-order job sorting without per-job path allocations |
| `runtime/dom.ts` | Cloned templates, `insert` with the keyed list reconciler, attribute, property, class and style setters |
| `runtime/events.ts` | Delegated events (one listener per type on each render root and the document) and direct ones |
| `runtime/flow.ts` | `Show`, `For`, `Index`, `Switch`/`Match`, `Key`, `Dynamic`, `Portal` |
| `runtime/component.ts` | Components and props helpers (`mergeProps`, `splitProps`, `defaults`, `bindable`, `field`) |
| `runtime/bind.ts`, `spread.ts`, `text.ts` | `ref`, `use:` and `bind:`; `{...spread}` on an element; text runs |
| `runtime/reactivity.ts` | `deep`, `ReactiveMap`, `ReactiveSet`, `MediaQuery` |
| `runtime/transition.ts`, `tween.ts` | Transitions (`fade`, `fly`, easings) and `Tween` |
| `runtime/icon.ts`, `render.ts`, `names.ts` | Lucide icons as components; `render`; names the compiler and the runtime share |
| `runtime/internal.ts` | `fw/internal`: what compiled JSX calls, not for application code |
| `compiler/index.ts` | The Vite plugin `fw()`: compiles `.tsx`, serves each file's scoped CSS and the `$icons/<name>` modules |
| `compiler/jsx.ts`, `templates.ts`, `ast.ts`, `html.ts` | The JSX transform (oxc-parser + magic-string, exact source maps), template emission/deduplication, and AST/HTML helpers |
| `compiler/css.ts` | Scoped CSS (postcss + postcss-selector-parser, mirrors Svelte 5's analyzer and transformer) |
| `jsx.d.ts`, `icons.d.ts` | JSX types (Svelte 5's element attribute types from `vendor/`, plus fw's directives); `$icons/*` |
| `icons/` | `lucide.json`, the icon data the plugin builds `$icons/<name>` from, and its licence |
| `tools/icons.mjs` | Checks that every imported icon is vendored (`npm run check`); `--add` vendors the missing ones |
| `vendor/` | `svelte-elements.d.ts` and its licence |
| `testing.ts` | `render()` / `flush()` / `tick()` / `fromHTML()` for vitest + happy-dom |
| `bench/` | The js-framework-benchmark operations (`run.mjs`) and the runtime's size (`size.mjs`) |
| `PORTING.md` | Every construct, with the Svelte 5 one it replaced |

Tests sit beside the code: `runtime/*.test.ts(x)` and `compiler/*.test.ts`.

```sh
npx vitest run framework/             # runtime and compiler tests (npm test runs them with the app's)
npx tsc -p tsconfig.json              # types of the app and the framework
node framework/bench/run.mjs          # the benchmark operations, Chrome
node framework/bench/size.mjs         # runtime size, min + gzip
```

## How Studio uses it

- **Vite:** `web/vite.config.ts` adds the plugin (`fw()` from `compiler/index.ts`) and aliases `fw` to
  `runtime/index.ts` and `fw/<name>` to `runtime/<name>.ts`. The plugin compiles every `.tsx` outside `node_modules`
  and imports the file's scoped CSS as the virtual module `<file>.fw.css`.
- **TypeScript:** `web/tsconfig.json` has `"jsx": "preserve"` (the plugin compiles JSX, not tsc), the same `fw`
  paths, and includes `jsx.d.ts` and `icons.d.ts`. The compiler is Node code, type-checked through
  `tsconfig.node.json` and `vite.config.ts`.
- **Tests:** vitest with `environment: 'node'` by default; a DOM test starts with `// @vitest-environment happy-dom`
  and uses `testing.ts`.

## Vendored code

| Files | From | Licence |
|---|---|---|
| `vendor/svelte-elements.d.ts` | svelte@5.57.3 `elements.d.ts` (the one import it had, `Attachment`, is declared in the file) | MIT, `vendor/svelte-LICENSE.md` |
| `icons/lucide.json` | Lucide's icon data, one entry per icon the app imports | ISC, `icons/LICENSE-lucide.txt` |

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
- A *block* is an expression that renders more than text, as far as the file shows: JSX in it, a call of a local
  snippet (`const link = () => <a/>`, every binding of the name in the file being one), a snippet prop
  (`props.children`, or a property the file's types declare `JSX.Element` or a function returning one), or a
  conditional with one in a branch. It is Svelte's `{#if}` / `{@render}` and never joins a text run.
- `{a ? b : c}` and `{a && b}` that render blocks compile to a truthiness memo plus an owning memo: a branch is
  created when the condition flips, as `{#if}` does (a branch that is an expression is a function of its own, so the
  branch's memo tracks only the condition). One that renders text is a text expression, as in Svelte (one binding;
  `a && b` renders nothing for any falsy `a`).
- Components are calls with a props object of getters (`get title() { return expr }`), so `props.title` read where it
  is used is tracked; children are a function (Svelte's snippet), a single `{(x) => ...}` child is passed through.
  A prop expression Svelte would memoize (`$.derived`: it calls something, or is more than an identifier or member
  read) reads a lazy computed (`propMemo`), so the child re-runs only when the value changes.
- Every JS expression stays at its original place in the output (generated code fills the gaps between them), so the
  source map is exact for all embedded code.
- Text follows JSX whitespace rules (see PORTING.md §8 for the Svelte differences). Adjacent static text and
  expressions compile to one text node with one binding (`run`), where Svelte makes one: Chrome shapes per text node.
  A run part that turns out to render more than text (a prop the compiler could not see as a snippet) splits the run
  where Svelte's text sequence would end: the text before it stays in the run's node, the part is rendered after it,
  the text after it is a run of its own. Nothing is evaluated twice: the run's binding becomes the part's own,
  keeping only what the part read (a source read both by the part and by the text before it is the one exception).
  A run in a fragment (no element of its own to put the split's nodes in) is a reader of its nodes, so the insert
  holding the fragment places, moves and removes what a split adds, also a split that happens later.
- Booleans on attributes follow Svelte: dropped on false only for its boolean attributes, stringified otherwise.
- A DOM property given as an expression, even a literal (`checked={true}`, `<option value={1}>`, whose `__value`
  keeps the number), is set as the property, as Svelte does, and so is a property no template attribute sets
  (`indeterminate`, `muted`, `defaultValue`...); other static values are written in the template.

### Reactivity (`runtime/core.ts`)

- **Signals** hold a value and a version. **Computeds** are lazy: recomputed on read, only if a source's version moved
  (checked transitively, so a diamond recomputes once: glitch-free). A computed with no subscribers holds no
  subscriptions, so it is garbage-collected with whatever created it.
- **Effects**: a write pushes a "maybe stale" mark down the graph and queues effects; a flush (microtask, or
  `flush()`) runs in cycles, as Svelte's: the render jobs queued so far (DOM bindings, and the owning computeds below),
  then the user effects queued so far, each in tree order (parents before children, siblings in document order; a
  branch created late stands at its memo's place); what they queue is the next cycle. Each re-runs only if a
  source actually changed. A component's own `effect`s (and `onMount`) are created once it has rendered, as Svelte
  defers a component's top-level `$effect` to its mount: the effects and actions of what it renders run first.
- Dependencies are doubly linked nodes shared by the source's subscriber list and the target's source list. A re-run
  walks its previous links and reuses those it reads again (no allocation in steady state; a test asserts it).
- **Owners**: components, effects and roots form a tree; disposing a node disposes its children (in order) then runs
  its cleanups. Context lives on owners (each component gets one, so `setContext` scopes like Svelte's).
- **Memos** (`memo`) are owning computeds: what a branch creates belongs to it, so reading a branch from several
  places, or again, never re-creates it. A memo (and a list's rows) is created where it is written, as Svelte creates
  a block, so components come into being in document order (and `createUniqueId` numbers them as Svelte does). When
  what a memo (or a list) reads changes, it is re-evaluated in the render phase at its own place in the tree,
  not when the effect that reads it gets its turn: branches are re-created in document order, before the effects
  inside them (which a flip disposes before they run), as Svelte runs blocks in tree order. Functions are either *readers*
  (memos, list accessors: read in place) or *snippets* (called once in a memo of their own, read in an effect of
  their own, so a change inside a branch never re-runs the expression that returned it).

### DOM (`runtime/dom.ts`, `flow.ts`, `events.ts`)

- `insert` renders text (reusing its text node), nodes, arrays and reactive functions before a marker node (static
  sibling, `<!>` comment) or as the parent's only content (then clearing is one `textContent = ''`).
- Lists are reconciled by node identity: common prefix and suffix skipped, end-for-end swaps handled directly, then
  the longest increasing subsequence of old positions stays and only the rest moves (`insertBefore`, as Svelte;
  focus is put back where it was: Chrome's `moveBefore` leaves a moved node's descendants unstyled after a later
  attribute change). `For` maps rows by key (an item replaced under the same key updates the row in place),
  `Index` by position. Both read the list's length and entries (and `For` its keys) tracked, so a `deep()` array
  changed in place (`push`, `list[i] = x`) updates them; creating and disposing rows is untracked.
- Events: Svelte 5's delegated set (click, input, keydown, pointer*, ...) goes through one listener per type on each
  render root and the document, walking the event path with Svelte's propagation algorithm; others use
  `addEventListener`.
- Attributes, properties, `class` (clsx, scope class, `class:`), `style` (string or object, `style:`), spreads and
  `bind:` follow Svelte 5's runtime (`set_attribute`, `set_class`, `set_style`, `set_attributes`, `bind_value`...).
  Styles take Svelte's path exactly: the first time (and whenever the `style` attribute's value changes) the whole
  attribute text is written (`cssText`, directives and style objects appended); later a changed directive or object
  key alone goes through `setProperty` (Chrome serializes an exact half pixel differently on the two paths). A
  `<select>`'s value is selected again when its options change (`init_select`'s MutationObserver).
- Transitions: Svelte's engine (keyframes sampled from `css(t, u)`, Web Animations) with local intros/outros on
  `For`/`Show`/`Key`/`Switch` branches.

### Scoped CSS (`compiler/css.ts`)

`css\`...\`` in a `.tsx` is extracted at build time, scoped, and served as the virtual module `<file>.fw.css` that the
file imports after its own imports, as vite-plugin-svelte does, so the bundle has Svelte's cascade order (a file's
rules after its children's). Vite's CSS
pipeline does the rest; HMR updates it. The plugin also fails the build on an import that resolves to a file whose
name differs in case (`$lib/ui/Field` → `field.ts` on a case-insensitive disk). The scoping mirrors Svelte 5's
`phases/2-analyze/css` and `3-transform/css`: the first scoped compound of a selector gets `.fw-xxxx` (+0-1-0), later
ones `:where(.fw-xxxx)`, nested rules under a local rule `:where(...)` only, `:global(...)`/`:global {}` and leading or
trailing `:global` unscoped, `:is/:where/:has` arguments scoped like the selector, `:not` arguments only when complex
(and nothing nested inside them), `@keyframes` names prefixed (`-global-` opts out) and `animation(-name)` following.

Until Studio left Svelte, a test compiled every `<style>` of every `.svelte` file of the app with the Svelte compiler
and with ours: 171 files, 3,950 selectors, all identical in text and specificity. `compiler/css.test.ts` keeps each
rule of that scoping as Svelte 5 printed it. By design we keep selectors Svelte drops as unused (we do not see the
elements); they are scoped, so they cannot match outside the file.

The scope class prefix is **`fw-`** (`fw-` + base-36 FNV-1a of the file's path relative to `web/`), e.g. `fw-fyng9x`.

## API

```ts
import { ... } from 'fw';
```

| Area | API | Defined in |
|---|---|---|
| State | `signal(v, { equals? })` → `{ get, set, peek }` (`state` alias) · `computed(fn, { equals? })` → `{ get, peek }` · `deep(obj)` (proxy) · `snapshot(x)` | `runtime/core.ts`; `deep`, `snapshot`: `runtime/reactivity.ts` |
| Effects | `effect(fn)` (after DOM updates; `fn` may return a cleanup) · `effect.pre(fn)` · `renderEffect` (internal) · `onMount` · `onCleanup` / `onDestroy` · `untrack` · `isTracking` | `runtime/core.ts` |
| Scheduling | `flush()` (≙ `flushSync`) · `tick()` · `batch(fn)` | `runtime/core.ts` |
| Ownership | `createRoot((dispose) => …, parent?)` · `getOwner` · `runWithOwner` · `dispose` · `memo` | `runtime/core.ts` |
| Context | `setContext` · `getContext` · `hasContext` · `createContext(default)` | `runtime/context.ts` |
| Components | `Component<P>` · `ComponentProps<C>` · `ElementProps<'button'>` · `defaults(props, {...})` · `splitProps(props, keys)` · `mergeProps(...)` · `bindable(props, key, fallback?)` · `field(obj, key)` · `createUniqueId()` (`propsId`) | `runtime/component.ts`; `ElementProps`: `runtime/index.ts` |
| Control flow | `<Show when keyed? fallback>` · `<For each key? fallback>{(item, i) => …}` · `<Index each fallback>` · `<Switch fallback><Match when>` · `<Key when>` · `<Dynamic component={tagOrComponent} …>` · `<Portal mount?>` | `runtime/flow.ts` |
| DOM | `render(() => <App />, el)` → unmount · `on(target, type, fn)` · `onWindow` · `onDocument` · `clsx` | `runtime/render.ts`, `runtime/events.ts`, `runtime/dom.ts` |
| Reactivity utils | `MediaQuery` / `mediaQuery(q)` (`.current`) · `ReactiveMap` · `ReactiveSet` · `createSubscriber` | `runtime/reactivity.ts` |
| Motion | `fade` · `fly` · `linear` · `cubicOut` · `cubicInOut` · `Tween` | `runtime/transition.ts`, `runtime/tween.ts` |
| Styles | `` css`...` `` (compile-time) | `runtime/index.ts` (the marker), `compiler/css.ts` |
| Icons | `import X from '$icons/x'` (lucide, from @lucide/svelte's data) · `IconComponent` | `runtime/icon.ts`, `compiler/index.ts` |
| JSX attributes | `class` · `class:x` · `style` (string or object) · `style:prop` · `on<event>` / `on<event>capture` · `ref` · `use:action` / `use` · `bind:value/checked/group/open/this/clientWidth/clientHeight/offsetWidth/offsetHeight` · `transition:` / `in:` / `out:` · `{...spread}` · `innerHTML` | types `jsx.d.ts`; compiled by `compiler/jsx.ts` |
| Tests (`framework/testing`) | `render(() => <X />, target?)` → `{ container, unmount }` · `flush` · `tick` · `fromHTML(html)` | `testing.ts` |

## Performance

Measured before Studio left Svelte. Chrome (stable, headless), production builds of the same table in fw (`bench/main-fw.tsx`) and Svelte 5.57
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
  all (no selector of the app was affected: the scoping comparison had zero differences).
- `bind:this` is set synchronously at creation (Svelte: after mount); with a signal ref it is cleared synchronously
  when the element goes (Svelte: a microtask later).
- Outros: `For`/`Show`/`Key`/`Switch` run them; a ternary (`{a ? b : c}`) does not, and an outro is not reversed if
  the same keyed row comes back while it runs (it is a new row). Effects of a removed branch stop at once (Svelte keeps
  them paused during the outro).
- `<Icon/>` with a component from a variable is not reactive to the variable: `<Dynamic component={...}>` is.
- Whitespace follows JSX, not Svelte (PORTING.md §8).
- A keyed row's item is an accessor (`item()`), not a variable; unkeyed `{#each}` is `<Index>`.
- Svelte throws on duplicate keys in a keyed `{#each}`; `For` renders duplicates.
