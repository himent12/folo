<script lang="ts">
  import Dialog from '$lib/ui/Dialog.svelte';
  import Kbd from '$lib/ui/Kbd.svelte';
  import { shortcutList } from './shortcuts';

  let { open = $bindable(false) }: { open?: boolean } = $props();

  const groups = [...new Set(shortcutList.map((s) => s.group))];
</script>

<Dialog bind:open title="Keyboard shortcuts" description="Everything in Studio works from the keyboard." size="md">
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
  .groups {
    display: grid;
    gap: var(--space-5);
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
