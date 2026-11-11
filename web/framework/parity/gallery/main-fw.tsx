// The ui gallery, fw side: every `sections/<Name>.tsx` (the twin of `sections/<Name>.svelte`). `?s=<Name>` renders that
// section alone (what gallery.mjs shoots); without it, every section, one after the other, to look at by hand.
import { Dynamic, For, render, type Component } from 'fw';
import '$lib/design/tokens.css';
import '$lib/design/base.css';
import './gallery.css';

const mods = import.meta.glob<{ default: Component<Record<string, never>> }>('./sections/*.tsx', { eager: true });
const sections = Object.entries(mods)
  .map(([path, m]) => ({ name: /([^/]+)\.tsx$/.exec(path)![1]!, component: m.default }))
  .sort((a, b) => a.name.localeCompare(b.name));
(window as unknown as { __gallery: string[] }).__gallery = sections.map((s) => s.name);

const only = new URLSearchParams(location.search).get('s');
const shown = only ? sections.filter((s) => s.name === only) : sections;

render(
  () => (
    <main class="g-gallery">
      <For each={shown}>
        {(s) => (
          <section class="g-section" data-section={s().name}>
            <h2 class="g-title">{s().name}</h2>
            <Dynamic component={s().component} />
          </section>
        )}
      </For>
    </main>
  ),
  document.getElementById('app')!,
);
