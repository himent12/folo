<script lang="ts">
  // The Svelte side of the ui gallery's frame (main-fw.tsx renders the same frame for the fw sections).
  import type { Component } from 'svelte';

  const mods = import.meta.glob<{ default: Component }>('./sections/*.svelte', { eager: true });
  const sections = Object.entries(mods)
    .map(([path, m]) => ({ name: /([^/]+)\.svelte$/.exec(path)![1]!, component: m.default }))
    .sort((a, b) => a.name.localeCompare(b.name));
  (window as unknown as { __gallery: string[] }).__gallery = sections.map((s) => s.name);

  const only = new URLSearchParams(location.search).get('s');
  const shown = only ? sections.filter((s) => s.name === only) : sections;
</script>

<main class="g-gallery">
  {#each shown as s (s.name)}
    <section class="g-section" data-section={s.name}>
      <h2 class="g-title">{s.name}</h2>
      <s.component />
    </section>
  {/each}
</main>
