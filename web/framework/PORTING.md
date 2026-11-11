# Porting Svelte 5 → fw

How to port a FOLO Studio component from Svelte 5 (`.svelte`) to fw (`.tsx`), construct by construct. Follow it
mechanically; when something is not covered here, stop and ask rather than invent. The five proof ports are the
reference: `src/lib/ui/{Button,Badge,Segmented,Menu,Dialog,DialogStatus,Spinner,SheetHandle}.tsx`,
`src/lib/charts/Sparkline.tsx`, their tests `*.test.tsx`, and `framework/parity/` (pixel-identical twins).

The rule that explains almost everything below: **a component function runs once.** Nothing in its body re-runs.
Only three places are reactive: JSX expressions (`{...}` children and attribute values), `computed(...)`, and
`effect(...)`. A value is reactive only if it is *read inside one of those* through a call (`sig.get()`, `fn()`) or a
property read (`props.x`, `obj.y`).

## Contents

1. [Files and names](#1-files-and-names)
2. [Script: state, derived, effects](#2-script-state-derived-effects)
3. [Props](#3-props)
4. [Markup](#4-markup)
5. [Blocks: if, each, key, snippets](#5-blocks-if-each-key-snippets)
6. [Directives and special elements](#6-directives-and-special-elements)
7. [Styles](#7-styles)
8. [Whitespace (pixel parity!)](#8-whitespace-pixel-parity)
9. [Modules, classes, stores](#9-modules-classes-stores)
10. [Tests](#10-tests)
11. [Pitfalls checklist](#11-pitfalls-checklist)
12. [Checking your port](#12-checking-your-port)

---

## 1. Files and names

| Svelte | fw |
|---|---|
| `Foo.svelte` | `Foo.tsx` beside it, `export default function Foo(props: FooProps)` |
| `foo.svelte.ts` (runes) | `foo.ts` beside it |
| `foo.ts` that imports `svelte` / `@lucide/svelte` | `foo.fw.ts` beside it (e.g. `ui/field.fw.ts`, `ui/types.fw.ts`) |
| `Foo.svelte.test.ts` | `Foo.test.tsx` |
| `<script module>` exports | top-level `export`s of `Foo.tsx` |
| `import X from './X.svelte'` | `import X from './X'` (resolves `X.tsx`); **`import X from './X.tsx'` when a lowercase twin `x.ts` exists** (`$lib/ui/Field.tsx`, because of `field.ts`) |
| `import Icon from '@lucide/svelte/icons/x'` | `import Icon from '$icons/x'` (same icon data, same `<svg>` output) |
| `import type { IconType } from './types'` | `import type { IconType } from './types.fw'` |
| `import { getField } from './field'` | `import { getField } from './field.fw'` |

**Case.** Windows and macOS disks ignore case, and module resolution tries `.ts` before `.tsx`: `$lib/ui/Field`
would load `field.ts` there (and `Field.tsx` on Linux). Whenever a component has a lowercase twin, import it with
its extension: `import Field from '$lib/ui/Field.tsx'` (`tsconfig.next.json` allows `.tsx` imports). The fw Vite
plugin makes this a build error: an import that resolves to a file whose name differs in case from what it asked for
fails with the fix in the message, so the wrong module is never picked silently.

Keep the old files until the final switch. Never edit a `.svelte` / `.svelte.ts` file: the Svelte app is the baseline
the parity checks diff against. Pure `.ts` modules with no Svelte import (`tooltip.ts`, `focus.ts`, `sheetDrag.ts`,
`format.ts`, `$lib/api/*`) are shared as they are.

Everything comes from `'fw'`: `import { signal, computed, effect, For, Show, css, ... } from 'fw'`.

## 2. Script: state, derived, effects

### `$state` (primitives) and `$state.raw` → `signal`

```svelte
let open = $state(false);
let at = $state<{ top: number } | null>(null);
open = !open;
if (open) ...
```
```tsx
const open = signal(false);          // `state(false)` is the same function, if you prefer the name
const at = signal<{ top: number } | null>(null);
open.set(!open.get());               // or open.set((o) => !o)
if (open.get()) ...
```
`get()` tracks (inside JSX/computed/effect), `peek()` reads without tracking, `set(v)` writes, `set((prev) => next)`
updates. To store a *function* in a signal, wrap it: `s.set(() => fn)`. Equal writes (`Object.is`) are skipped;
`signal(v, { equals: false })` notifies on every write.

### `$state` of an object or array

- Replaced wholesale (`items = [...items, x]`): a `signal` holding immutable values. This is the default.
- **Mutated in place** (`items.push(x)`, `form.name = v`, `rows[i].done = true`): `deep(...)`, a proxy like Svelte's.

```svelte
let form = $state({ name: '', tags: [] as string[] });
form.tags.push('a');
```
```tsx
const form = deep({ name: '', tags: [] as string[] });
form.tags.push('a');                 // reactive: whatever read form.tags updates
```
If the variable itself is also reassigned, hold the proxy in a signal: `const form = signal(deep({...}))`,
`form.set(deep({...}))`, `form.get().tags.push(...)`. `$state.snapshot(x)` → `snapshot(x)`.

### `$derived` / `$derived.by` → `computed`

```svelte
const iconSize = $derived(size === 'lg' ? 16 : 14);
const geo = $derived.by(() => { ...; return { d, a }; });
```
```tsx
const iconSize = computed(() => (p.size === 'lg' ? 16 : 14));
const geo = computed(() => { ...; return { d, a }; });
// read: iconSize.get(), geo.get()
```
Lazy and memoized: recomputed only when read after a source changed. For a one-liner used in one place you may also
write a plain function `const iconSize = () => (p.size === 'lg' ? 16 : 14)` and call it in JSX (no memo; fine when
cheap). A derived that is reassigned locally (Svelte 5.25 writable deriveds) → `signal` + `effect`, ask first.

### `$effect` → `effect`

```svelte
$effect(() => {
  if (!open) return;
  document.addEventListener('pointerdown', away);
  return () => document.removeEventListener('pointerdown', away);
});
```
```tsx
effect(() => {
  if (!open.get()) return;
  document.addEventListener('pointerdown', away);
  return () => document.removeEventListener('pointerdown', away);
});
```
Same timing as Svelte: after the component is in the DOM, and after the DOM updates of each later flush; parents'
effects before children's, siblings in order. The returned function runs before the next run and on teardown.
Teardown order on unmount: children first, then the parent (Svelte's order).

| Svelte | fw |
|---|---|
| `$effect.pre(fn)` | `effect.pre(fn)` (runs before the DOM updates; first run synchronous) |
| `$effect.root(fn)` | `const dispose = createRoot((dispose) => { ...; return dispose; })` |
| `$effect.tracking()` | `isTracking()` |
| `onMount(fn)` | `onMount(fn)` (from 'fw') |
| `onDestroy(fn)` | `onCleanup(fn)` (`onDestroy` is an alias) |
| `untrack(fn)` | `untrack(fn)` |
| `tick()` | `tick()` |
| `flushSync()` | `flush()` |
| `getContext` / `setContext` / `hasContext` | same names from 'fw' (call at component init) |
| `createContext` (Svelte ≥5.40) | `createContext(default)` → `{ get(), set(v) }` |

### `$props.id()` → `createUniqueId()`

Same `c1`, `c2`, ... sequence as Svelte, so ids in the DOM match.

## 3. Props

### Reading props

`props` is an object of getters: **read `props.x` where you use it; never destructure it** (destructuring reads once
and the value never updates).

```svelte
let { tone = 'neutral', children, mono = false } = $props();
<span class="badge {tone}">...</span>
```
```tsx
export default function Badge(props: { tone?: Tone; children: JSX.Element; mono?: boolean }) {
  return <span class={`badge ${props.tone ?? 'neutral'}`} class:mono={props.mono ?? false}>{props.children}</span>;
}
```

### Defaults → `defaults(props, {...})`

```svelte
let { variant = 'secondary', size = 'md', loading = false, disabled, children, class: klass = '', ...rest } = $props();
```
```tsx
const p = defaults(props, { variant: 'secondary', size: 'md', loading: false, class: '' });
const [, rest] = splitProps(p, ['variant', 'size', 'loading', 'disabled', 'children', 'class']);
// use p.variant, p.class (not `klass`), rest
```
A default applies while the prop is `undefined`, as in Svelte. `defaults` evaluates its object once; for a stateful
default (`status = new DialogStatus()`) used once at init write `const status = props.status ?? new DialogStatus()`.

### Rest props → `splitProps`

`splitProps(props, ['a', 'b'])` returns `[{a, b}, rest]`, both live (getters through to `props`). Spread `rest` onto
an element (`<button {...rest}>`) or a component (`<Inner {...rest} />`) as in Svelte; later attributes win.
`mergeProps(a, b, ...)` merges props objects keeping getters (later wins, as Svelte's spread).

### `$bindable` → `bindable(props, 'name', fallback?)`

```svelte
let { open = $bindable(false), value = $bindable() } = $props();
open = false;
```
```tsx
const open = bindable(props, 'open', false);   // a signal-like: open.get(), open.set(false)
const value = bindable(props, 'value');
```
When the parent bound it (`bind:open={sig}`), writes go to the parent's signal; when it passed a plain value, writes
stay local until the parent passes a new value (Svelte's semantics).

### Binding a child's prop: `bind:x={signal}` on a component

```svelte
<Dialog bind:open title="…">  /  <Segmented bind:value={runOn} … />
```
```tsx
<Dialog bind:open={open} title="…">  /  <Segmented bind:value={runOn} … />
```
The right side is anything with `get()`/`set()`: a `signal`, a `bindable(...)` passed down, `field(obj, 'key')` for a
property of a `deep` object or a props object. The shorthand `bind:open` (no value) is not JSX: write the value.

Types: a caller may write `bind:x` instead of `x` for a *required* prop while the component has at most 6 required
props (the "x or bind:x" type doubles with each one, so past that the props stay as declared). A component with more
required props that a caller binds declares the bound prop optional (`value?: T`), which is what a `$bindable` read
with `bindable(props, 'value')` is anyway.

### Prop expressions are memoized as in Svelte

The compiler gives a prop a lazy `computed` when Svelte would give it a `$.derived`: the expression calls something
(`items={rows()}`), or reads state through anything but a plain identifier or member read (`src={err ? null : src}`,
`active={i === index()}`; a template literal counts by its own expressions). The child then sees a new value, and its
effects re-run, only when the value changed. Plain reads (`label={props.label}`) and JSX values are read through as
they are. A memoized prop is evaluated outside any component (like `$derived`): do not create components or effects in
a prop expression (pass a function for that, as for snippets).

### Types

| Svelte | fw |
|---|---|
| `Snippet` | `JSX.Element` |
| `Snippet<[A, B]>` | `(a: A, b: B) => JSX.Element` (see snippets, args are accessors) |
| `Component<P>` | `Component<P>` from 'fw' (= `(props: P) => JSX.Element`) |
| `HTMLButtonAttributes` (svelte/elements) | `ElementProps<'button'>` from 'fw' |
| `ComponentProps<typeof X>` | `ComponentProps<typeof X>` from 'fw' |
| `generics="T extends string"` | `export default function Segmented<T extends string>(props: SegmentedProps<T>)` |
| `typeComponent` / `typeSnippet` | not used in web/src (0 uses); write the function type directly |

## 4. Markup

JSX with Svelte's attribute names (`class`, `for`, `tabindex`, `aria-*`, `stroke-width`), not React's.

| Svelte | fw |
|---|---|
| `{expr}` | `{expr}` |
| `class="a {b}"` | ``class={`a ${b}`}`` |
| `class={[a, { on }]}` (clsx) | `class={[a, { on }]}` (same clsx semantics) |
| `attr="{x}px"` | ``attr={`${x}px`}`` |
| `{href}` shorthand | `href={props.href}` |
| `{onkeydown}` shorthand | `onkeydown={onkeydown}` |
| `tabindex="-1"`, `width="80"` (numbers as strings) | `tabindex={-1}`, `width={80}` (string literals are fine where the type is a string) |
| `<!-- comment -->` | `{/* comment */}` |
| `{@html x}` | `<div innerHTML={x} />` (0 uses today) |
| `{@debug}` | remove |

**Events**: `onclick={handler}` (lowercase, Svelte 5 names), handler types give a typed `currentTarget`. Common
bubbling events (click, input, change, keydown/up, pointer*, mouse*, focusin/out, touch*, dblclick, contextmenu,
beforeinput) are delegated exactly as Svelte delegates them: handler order, `stopPropagation`, disabled elements
behave the same. `onclickcapture` / `onkeydowncapture` work. There are no event modifiers in Svelte 5 code.

**Attribute semantics** follow Svelte 5: `null`/`undefined` removes an attribute; a boolean on an attribute outside
Svelte's boolean list stringifies (`spellcheck={false}` → `spellcheck="false"`, `aria-expanded={false}` →
`"false"`, or the property when the element has one of that name, e.g. `hidden`), while Svelte's boolean attributes
(`disabled`, `checked`, `open`, `readonly`, `required`...) are present when true and absent when false; DOM properties
(`disabled`, `checked`, `value`, `open`, `readonly`...) are set as properties; `value` on `<select>` selects after the
options render. A bare attribute (`<div data-notice>`) is present and empty, as in Svelte.

**Text.** Adjacent static text and expressions are one text node, exactly where Svelte makes one
(`Epoch {a}/{b}`): Chrome shapes each text node on its own, so this keeps glyphs identical. An expression with JSX in
it (`{x ? <b/> : null}`, a list) is a block and splits the text around it, as an `{#if}` does in Svelte. An
expression that turns out to render a node or a snippet (`{props.children}`, `{icon()}`) is still rendered as it is.

**Dynamic or not?** The compiler makes an expression reactive when it contains a call or a property read. A plain
identifier or literal is set once. So `title={label}` with `let label` reassigned later does **not** update: make
`label` a signal (`title={label.get()}`).

## 5. Blocks: if, each, key, snippets

### `{#if}` → a ternary (or `<Show>`)

```svelte
{#if loading}<Spinner />{:else if Icon}<Icon />{:else}none{/if}
```
```tsx
{p.loading ? <Spinner /> : p.icon ? <Dynamic component={p.icon} /> : 'none'}
```
A ternary or `&&` in a JSX child position is compiled like `{#if}`: the branch is created when the condition's
truthiness flips, not on every change. Several nodes in a branch → a fragment `<>...</>`. `{#if a && b}x{/if}` →
`{a && b ? x : null}` (a falsy `&&` left side renders nothing, even `0`).

Use `<Show>` when the branch has `transition:`/`in:`/`out:` (ternaries do not run outros), or to get the narrowed
value: `<Show when={user()} fallback={<i>none</i>}>{(u) => <b>{u().name}</b>}</Show>`.

**A component, row or snippet that *returns* a conditional must wrap it in a fragment**: `return <>{c ? a : b}</>`. A
bare `return c ? a : b` is plain JavaScript, evaluated once.

### `{#each}` → `<For>` (keyed) or `<Index>` (unkeyed)

```svelte
{#each options as o (o.value)}
  {@const on = o.value === value}
  <button class:on onclick={() => select(o.value)}>{o.label}</button>
{:else}
  <p>Nothing</p>
{/each}
```
```tsx
<For each={p.options} key={(o) => o.value} fallback={<p>Nothing</p>}>
  {(o) => {
    const on = () => o().value === value.get();       // {@const} → a function (or computed) in the row
    return <button class:on={on()} onclick={() => select(o().value)}>{o().label}</button>;
  }}
</For>
```
- The row function runs once per row; `o()` is the row's current item (a kept row whose item object was replaced
  under the same key gets the new item, as in Svelte). `{#each list as item, i (key)}` → `(item, i) => ...` with
  `i()` an accessor.
- Without a key, Svelte's `{#each}` is by index: `<Index each={list}>{(item, i) => ...}</Index>` (`item()` accessor,
  `i` a number). `{#each Array(n) as _, i}` → `<Index each={Array.from({ length: n })}>{(_, i) => ...}</Index>`.
- Rows keep their DOM nodes when they move (minimal moves; focus kept). The row function must render JSX: a
  conditional row is `<>{cond ? a : b}</>`.

### `{@const}`

Inside a row or a snippet: a local `const x = () => ...` (accessor; call `x()`), or `computed(...)` if expensive. At
the top of a block it was reactive in Svelte, so it must stay a function here, never a plain `const x = o().y`.

### `{#key value}` → `<Key when={value}>`

```tsx
<Key when={tab.get()}>
  <div class="swap" in:fade={{ duration: motion(160) }}>...</div>
</Key>
```

### `{#if}{:else if}` chains

Nested ternaries are fine; `<Switch fallback={...}><Match when={a}>..</Match><Match when={b}>..</Match></Switch>` reads
better for long chains.

### Snippets and `{@render}` → functions

```svelte
{#snippet option(o: Option, on: boolean)}<button aria-checked={on}>{o.label}</button>{/snippet}
{@render option(o, on)}
{@render children()}   {@render children?.()}   {#if children}...{/if}
```
```tsx
const option = (o: () => Option, on: () => boolean) => <button aria-checked={on()}>{o().label}</button>;
{option(o, on)}
{props.children}        {props.children}          {props.children ? ... : null}
```
- **Snippet parameters that can change are accessors** (`() => T`), called inside: otherwise the snippet would be
  re-created (losing DOM, focus, hover) on every change. Static values can be passed plainly.
- `children` is always rendered as `{props.children}` (never call it). It is `undefined` when the parent gave none.
- A snippet passed to a component (`{#snippet footer()}...{/snippet}` inside `<Dialog>`) becomes a prop holding a
  function: `<Dialog footer={() => <><Button>Cancel</Button></>}>`, or with arguments
  `row={(item) => <td>{item().name}</td>}`. Always pass a function, never `footer={<Button/>}`: a JSX value is
  re-created each time the component reads the prop (an `{#if footer}` check would create it and throw it away). The
  component renders it with `{props.footer}` (no arguments) or `{props.row(item)}`.
- A component's JSX children reach it as `props.children`; a single `{(x) => ...}` child is passed as is (render
  prop, e.g. `For`, `Show`).
- `createRawSnippet` (tests): see [Tests](#10-tests).

## 6. Directives and special elements

| Svelte | fw |
|---|---|
| `class:on={x}` | `class:on={x}` |
| `class:mono` (shorthand) | `class:mono={props.mono}` (JSX `class:mono` alone means `true`) |
| `style:top={v}`, `style:top="1px"` | same (`null`/`undefined` removes it) |
| `style="..."` or `style={...}` with `style:x` | same: the directives are applied over the attribute and win, as in Svelte |
| `style:--c={v}` | `style={{ '--c': v }}` (`--` is not valid in a JSX attribute name; object styles merge per property); with other `style:` directives on the same element, both work together |
| `style:color|important={v}` | `style:color|important` is not JSX: `style={{ color: v }}` + an `!important` rule in css`` |
| `bind:this={el}` (plain `let el`) | `let el!: HTMLElement;` + `ref={(e) => (el = e)}` |
| `bind:this={el}` (`let el = $state()`) | `const el = signal<HTMLElement>()` + `ref={el}` (cleared to `undefined` when the element goes) |
| `bind:value={v}` (input, textarea, select) | `bind:value={v}` (`v` a signal; numbers for `type=number/range`) |
| `bind:checked={v}`, `bind:group={v}` | same (`bind:group` needs the element's `value`) |
| `bind:open={v}` (details/dialog element) | same |
| `bind:clientWidth={w}` (and Height, offsetWidth/Height) | same (`w` a signal) |
| `use:tooltip={o.tip}` | `use:tooltip={o().tip}` (same action, same `update`/`destroy` timing) |
| `use:balanced` (no param) | `use:balanced` |
| several actions | one `use:` per action (`use:tooltip={t} use:fadeScroll`); `use={[[a, p], [b]]}` takes a list evaluated once |
| `transition:fly={{...}}`, `in:fade`, `out:x` | same, with `import { fly, fade, cubicOut } from 'fw'` |
| `<svelte:window onkeydown={f} />` | `onWindow('keydown', f)` in the component body |
| `<svelte:document on…>` | `onDocument('…', f)` |
| `<svelte:element this={as} …>` | `<Dynamic component={as} …>` (a tag name or a component) |
| `<Icon … />` where `Icon` is a prop / variable | `<Dynamic component={Icon} … />` (reacts when the component changes) |
| `<svelte:options namespace="svg" />` | nothing: SVG-only tags (`g`, `path`, `text`...) are created in the SVG namespace wherever they are; wrap an ambiguous root (`a`, `title`) in `<g>` |
| `<svelte:head>`, `<svelte:boundary>` | not used in web/src |

Refs are set synchronously when the element is created (Svelte's `bind:this` is set after mount): code that runs in an
effect sees the element in both.

Actions run after mount (a user effect), as in Svelte; `update(param)` runs when the param expression re-evaluates
(always for an object literal), `destroy` on removal.

Transitions are local, as in Svelte 5: an intro plays when the nearest block (`For` row, `Show`, `Key`, `Switch`) adds
the element after that block's first render; an outro plays when that block removes it, and the nodes stay until it
ends. Durations come from `motion(ms)` as before (0 under reduced motion skips the animation).

## 7. Styles

The `<style>` block moves **verbatim** into a `css` tagged template at the end of the file:

```tsx
import { css } from 'fw';
...
css`
  .badge {
    display: inline-flex;
  }
  :global([data-theme='light']) .seg {
    background: var(--bg-active);
  }
`;
```
- The compiler extracts it, scopes it exactly as Svelte 5 does (same `:global`, nesting, `:is/:where/:has/:not`,
  `@keyframes` and specificity rules; `framework/compiler/css.test.ts` checks every selector of every component of the
  app against the Svelte compiler), and removes the call. One per file, top level, no `${}`.
- Inside the template literal escape three things the way JS needs: `` ` `` → `` \` ``, `${` → `\${`, `\` → `\\`
  (CSS comments with backticks are common). The compiler undoes the escapes, so the CSS is what you wrote.
  `framework/tools/copy-style.cjs` copies a `<style>` block with the escapes done (below).
- Every element the file's JSX creates gets the file's scope class (`fw-` + a hash of its path); Svelte only adds its
  class to elements a selector matches, which never changes what matches.
- Unused selectors are kept (Svelte drops them); they cannot match anything outside the file.
- Cascade order is Svelte's: a file's CSS comes after the CSS of the components it imports, so a parent's rule wins a
  tie of specificity over its child's (`:global(dialog.shortcuts.lg)` over Dialog's `dialog.lg`). Never add a class
  twice to win a tie: `node framework/parity/css-order.mjs` (after `npm run build`) checks the built order against
  the Svelte app's.

#### Copying a style block

```sh
# appends css`...` (escaped) to the end of the .tsx; run once per file
node framework/tools/copy-style.cjs src/lib/ui/Foo.svelte src/lib/ui/Foo.tsx
```

## 8. Whitespace (pixel parity!)

JSX and Svelte keep different whitespace. JSX: whitespace on one line stays as written; **any whitespace containing a
newline disappears**; text lines are joined with one space. Svelte: whitespace between nodes collapses to one space
(newlines included); whitespace at the start and end of an element's (or block's) content is removed; whitespace next
to `{expr}` stays as written.

So the differences that change rendering are:

| Svelte source | Svelte renders | JSX of the same text renders | Write in JSX |
|---|---|---|---|
| `<b>a</b>⏎<i>b</i>` | `a b` (a space) | `ab` | `<b>a</b>{' '}<i>b</i>` (or keep them on one line: `<b>a</b> <i>b</i>`) |
| `{a}⏎{b}` | `a b` | `ab` | `{a}{' '}{b}` |
| `text⏎<b>x</b>` | `text x` | `textx` | `text{' '}<b>x</b>`, or `text <b>x</b>` on one line |
| `<p> x </p>` | `x` | ` x ` | `<p>x</p>` |
| `<p>⏎  x⏎</p>` | `x` | `x` | same |

Rules of thumb:
- Inside flex and grid containers whitespace between items renders nothing either way: no `{' '}` needed (most of
  FOLO's toolbars, buttons, rows).
- In inline / block text (paragraphs, labels, `<Meta>` items, a `<span>` next to a `<b>`), wherever the Svelte
  source has whitespace (including a line break) between two inline things, put `{' '}` there.
- In `<svg>` (outside `<text>`), `<select>`, tables: whitespace-only text is dropped by both.
- When in doubt, compare `textContent` of the two versions in the parity page or a test.

## 9. Modules, classes, stores

### `.svelte.ts` classes with `$state` fields

```ts
class DialogStatus {
  notice = $state<Notice | null>(null);
}
```
```ts
class DialogStatus {
  #notice = signal<Notice | null>(null);
  get notice() { return this.#notice.get(); }
  set notice(v: Notice | null) { this.#notice.set(v); }
}
```
Callers keep `status.notice` / `status.notice = x`. A `$derived` field → a getter over a private `computed`. See
`ui/toast.ts`, `ui/feedback.ts`.

### svelte/reactivity, svelte/motion, svelte/transition, svelte/easing

| Svelte | fw |
|---|---|
| `new MediaQuery(media.phone, false)`, `.current` | same (`MediaQuery` from 'fw'; `.current`, also `.get()`); one `matchMedia` list per instance, as Svelte, so a test's `matchMedia` stub applies to instances created after it |
| `SvelteMap` / `SvelteSet` | `ReactiveMap` / `ReactiveSet` |
| `createSubscriber` | `createSubscriber` |
| `Tween` (`.current`, `.target`, `.set`) | `Tween` (same API) |
| `fade`, `fly`, `cubicOut`, `cubicInOut`, `linear` | same names from 'fw' (same css, same easing) |

### Mounting

`mount(App, { target })` → `render(() => <App />, target)` (returns the unmount function).

## 10. Tests

```ts
// Svelte
import { flushSync, mount, tick, unmount } from 'svelte';
const c = mount(Menu, { target, props: { label: 'More', items } });
flushSync();
unmount(c);
```
```tsx
// fw (Foo.test.tsx, keep `// @vitest-environment happy-dom` on line 1)
import { flush, tick } from 'fw';
import { render, fromHTML } from '../../../framework/testing';
const { container, unmount } = render(() => <Menu label="More" items={items} />);   // or render(..., target)
flush();
unmount();
```
- Keep every assertion as it is. `flushSync()` → `flush()`; `await tick()` stays.
- A props object kept for reuse: `render(() => <Menu {...props} />)`.
- Reactive props from the test: pass a signal read in JSX: `const v = signal('a'); render(() => <X value={v.get()} />); v.set('b'); flush();`
- `createRawSnippet(() => ({ render: () => '<b>x</b>' }))` → `() => fromHTML('<b>x</b>')`; with an argument
  `(why) => ({ render: () => `<i>${why()}</i>` })` → `(why: () => string) => fromHTML(`<i>${why()}</i>`)`.
- `Menu.test.tsx` shows a thin `mount`/`unmount` shim that keeps a long test file's body untouched.

## 11. Pitfalls checklist

1. **Destructuring props** (`const { a } = props`) — reads once. Use `props.a` / `defaults` / `splitProps`.
2. **Reading a signal in the component body** (`const x = s.get()` at the top) — runs once. Read it in JSX, a
   `computed`, or an `effect`.
3. **`{@const}` as a plain const** in a row — must be a function/computed.
4. **Snippet args as values** — pass accessors for anything that changes.
5. **Bare conditional returns** — `return <>{c ? a : b}</>`.
6. **Whitespace** — `{' '}` where Svelte had whitespace between inline content (section 8).
7. **Numbers as strings** — `tabindex={-1}`, not `tabindex="-1"` (types reject the latter).
8. **`class:x` shorthand** — always give the value.
9. **In-place mutation of a `signal` value** (`list.get().push(x)`) does nothing: `list.set([...list.get(), x])`, or
   make it `deep`.
10. **Effects reading what they write** loop (Svelte errors too). Use `untrack`.
11. **Dynamic component variables** — `<Icon/>` where `Icon` comes from props: `<Dynamic component={...}>`.
12. **`style:--x`** — `style={{ '--x': v }}`.
13. **`$effect` that only syncs a value** into another state is better as a `computed`.
14. **Order of effect creation**: write effects in the component body before `return <JSX/>` (Svelte runs the script
    before the template; fw runs effects in creation order).

## 12. Checking your port

- `npx tsc -p tsconfig.next.json` — zero errors.
- `npx vitest run src/path/Foo.test.tsx` — the ported tests pass.
- Pixel parity: add the component (several states) to `framework/parity/Demo.tsx` **and** the same to `Demo.svelte`,
  then `node framework/parity/check.mjs` — must print `0 differing pixels` and `DOM: same` for dark and light.
- `npm run check` (Svelte app still clean) and `npm run build`.
