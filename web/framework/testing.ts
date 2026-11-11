// Test helpers for fw components (vitest + happy-dom), shaped like the Svelte tests' `mount` / `flushSync` /
// `unmount` so their ports are mechanical:
//
//   const { container, unmount } = render(() => <Menu label="More" items={items} />);
//   flush();               // port of flushSync(): effects (and $effect-style `effect`s) have run
//   ...
//   unmount();             // disposes the tree and removes the container
//
// Add `// @vitest-environment happy-dom` at the top of the test file, as the Svelte tests do. `fromHTML` replaces
// `createRawSnippet`.

import { flush, render as mount, tick } from 'fw';

export interface Rendered {
  /** The element the component rendered into (appended to `document.body`). */
  container: HTMLElement;
  /** Disposes everything the component created and removes `container`. */
  unmount(): void;
}

/** Renders `code()` into a fresh `<div>` in `document.body` (or into `target`); render effects have run. */
export function render(code: () => JSX.Element, target?: HTMLElement): Rendered {
  const container = target ?? document.body.appendChild(document.createElement('div'));
  const dispose = mount(code, container);
  return {
    container,
    unmount() {
      dispose();
      if (target === undefined) container.remove();
    },
  };
}

export { flush, tick };

/**
 * Port of `createRawSnippet(() => ({ render: () => html }))`: a snippet (children or a render prop) that renders
 * the HTML string's first element. With arguments: `(why: () => string) => fromHTML(`<b>${why()}</b>`)`.
 */
export function fromHTML(html: string): Node {
  const t = document.createElement('template');
  t.innerHTML = html.trim();
  return t.content.firstChild!;
}
