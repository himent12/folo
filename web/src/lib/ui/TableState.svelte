<script lang="ts">
  /**
   * An empty or error state in the frame the data would fill (DESIGN_DIRECTION §2.8): one row of a `.table`'s body
   * spanning every column, under the real header row. The block is compact (the message and its actions with the
   * panel's 16 px padding), never held at the loaded list's height; on phones, where the header row is hidden, the
   * frame is the content plus 24 px. The message stays inline at the content's left edge (EmptyState / ErrorState).
   */
  import type { Snippet } from 'svelte';

  let { cols, children }: { cols: number; children: Snippet } = $props();
</script>

<tr class="state">
  <td colspan={cols}>
    <div class="frame">{@render children()}</div>
  </td>
</tr>

<style>
  .state td {
    height: auto;
    padding: 0;
    white-space: normal;
    font-weight: var(--weight-regular);
    color: var(--text-secondary);
    box-shadow: none;
  }
  .state:hover {
    background: none;
  }
  /* One padding for every message in the frame, whatever its own default: 16 px on the panel's edges. */
  .frame {
    padding: var(--space-4);
  }
  .state .frame > :global(*) {
    padding: 0;
  }
  @media (max-width: 767px) {
    .frame {
      padding: var(--space-3) var(--space-4);
    }
  }
</style>
