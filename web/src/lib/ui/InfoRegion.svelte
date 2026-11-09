<script lang="ts">
  /**
   * The in-place definition of a chart's ⓘ on phones (./infoSlot): rendered by the frame under the chart's heading,
   * empty (and taking no room) until a `chart` InfoButton opens it.
   */
  import type { InfoSlot } from './infoSlot.svelte';

  let { slot }: { slot: InfoSlot } = $props();
</script>

<div class="info-region" id={slot.id} hidden={!slot.content}>
  {#if slot.content?.children}{@render slot.content.children()}{:else if slot.content}{slot.content.text}{/if}
</div>

<style>
  /* A raised well at the frame's inset: the definition reads as part of the chart, pushing the readout down. */
  .info-region {
    margin: 0 var(--info-inset, var(--space-4)) var(--info-after, var(--space-3));
    padding: var(--space-3);
    border-radius: var(--radius-md);
    background: var(--bg-raised);
    font-size: var(--type-base-size);
    line-height: var(--type-base-line);
    color: var(--text-secondary);
  }
  .info-region[hidden] {
    display: none;
  }
  .info-region :global(b),
  .info-region :global(dt) {
    color: var(--text-primary);
    font-weight: var(--weight-medium);
  }
</style>
