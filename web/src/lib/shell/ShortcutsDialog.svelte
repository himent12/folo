<script lang="ts">
  import Dialog from '$lib/ui/Dialog.svelte';
  import Kbd from '$lib/ui/Kbd.svelte';
  import { keyPrefs } from '$lib/ui/keyPrefs.svelte';
  import { shortcutList } from './shortcuts';

  let { open = $bindable(false) }: { open?: boolean } = $props();

  const groups = [...new Set(shortcutList.map((s) => s.group))];
</script>

<!-- Every group (General to Datasets): two columns where the dialog is wide, one on phones; the body scrolls with a
     shadow at whichever edge has more (ui/Dialog). -->
<Dialog
  bind:open
  title="Keyboard shortcuts"
  description={keyPrefs.singleKeys
    ? 'Everything in Studio works from the keyboard.'
    : 'Single-key shortcuts are off (Settings). Ctrl or ⌘ K and the arrow keys still work.'}
  size="lg"
>
  <div class="groups">
    {#each groups as g (g)}
      <section>
        <h3 class="overline">{g}</h3>
        <ul>
          {#each shortcutList.filter((s) => s.group === g) as s (s.keys + s.label)}
            <li><span>{s.label}</span><Kbd keys={s.keys} /></li>
          {/each}
        </ul>
      </section>
    {/each}
  </div>
</Dialog>

<style>
  /* Two columns from 2 × 280 px up; a group never splits across them. */
  .groups {
    columns: 280px 2;
    column-gap: var(--space-8);
    /* Each group's own bottom space would add to the dialog's padding after the last one. */
    margin-bottom: calc(-1 * var(--space-5));
  }
  section {
    break-inside: avoid;
    padding-bottom: var(--space-5);
  }
  h3 {
    margin-bottom: var(--space-2);
  }
  ul {
    display: grid;
  }
  li {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-4);
    min-height: 32px;
    border-bottom: 1px solid var(--line-subtle);
    color: var(--text-secondary);
  }
  li:last-child {
    border-bottom: 0;
  }
</style>
