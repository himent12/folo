<script lang="ts">
  import Dialog from '$lib/ui/Dialog.svelte';
  import Kbd from '$lib/ui/Kbd.svelte';
  import { keyPrefs } from '$lib/ui/keyPrefs.svelte';
  import { isApple } from '$lib/keys';
  import { shortcutList } from './shortcuts';

  let { open = $bindable(false) }: { open?: boolean } = $props();

  /**
   * Off in Settings: the single keys leave their groups for one section of their own, marked as such; a row whose
   * action also has keys that always work (`also`: J K and ↓ ↑) stays in its group with those.
   */
  const working = $derived(
    keyPrefs.singleKeys
      ? shortcutList
      : shortcutList.filter((s) => !s.single || s.also).map((s) => (s.single && s.also ? { ...s, keys: s.also, also: undefined } : s)),
  );
  const off = $derived(keyPrefs.singleKeys ? [] : shortcutList.filter((s) => s.single).map((s) => ({ ...s, also: undefined })));
  const groups = $derived([...new Set(working.map((s) => s.group))]);
  const paletteKeys = isApple ? '⌘ K' : 'Ctrl K';
</script>

<!-- Every group (General to Datasets): two columns where the dialog is wide, one on phones; the body scrolls with a
     shadow at whichever edge has more (ui/Dialog). -->
<Dialog
  bind:open
  title="Keyboard shortcuts"
  description={keyPrefs.singleKeys
    ? 'Everything in Studio works from the keyboard.'
    : `Single-key shortcuts are off in Settings. ${paletteKeys}, the arrow keys and the image inspector's keys still work.`}
  size="lg"
  class="shortcuts"
>
  <div class="groups">
    {#each groups as g (g)}
      <section>
        <h3 class="overline">{g}</h3>
        <ul>
          {#each working.filter((s) => s.group === g) as s (s.keys + s.label)}
            <li>
              <span>{s.label}</span><span class="keys"
                ><Kbd keys={s.keys} />{#if s.also}<span class="or">or</span><Kbd keys={s.also} />{/if}</span
              >
            </li>
          {/each}
        </ul>
      </section>
    {/each}
    {#if off.length}
      <section class="off">
        <h3 class="overline">Disabled in Settings</h3>
        <ul>
          {#each off as s (s.keys + s.label)}
            <li><span>{s.label}<span class="sr-only">, disabled in Settings</span></span><Kbd keys={s.keys} off /></li>
          {/each}
        </ul>
      </section>
    {/if}
  </div>
</Dialog>

<style>
  /* A reference card, not a page: 640 px, two columns of 13 px rows, a group never split across them. */
  :global(dialog.shortcuts.lg) {
    width: min(640px, calc(100vw - 32px));
  }
  .groups {
    columns: 260px 2;
    column-gap: var(--space-6);
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
    font-size: var(--type-base-size);
    line-height: var(--type-base-line);
    color: var(--text-primary);
  }
  .keys {
    display: inline-flex;
    align-items: center;
    gap: var(--space-2);
    flex-shrink: 0;
  }
  .or {
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
  }
  li:last-child {
    border-bottom: 0;
  }
  /* Keys that do nothing now: tertiary labels and outlined caps, a step back from the working ones in both themes. */
  .off li {
    color: var(--text-tertiary);
  }
  @media (max-width: 767px) {
    :global(dialog.shortcuts.lg) {
      width: 100%;
    }
  }
</style>
