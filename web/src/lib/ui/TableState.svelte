<script lang="ts">
  /**
   * An empty or error state in the frame the data would fill (DESIGN_DIRECTION §2.8): one row of a `.table`'s body
   * spanning every column, under the real header row, held at the height of a few loaded rows so the panel keeps its
   * shape. The message stays inline at the content's left edge (EmptyState / ErrorState inside).
   */
  import type { Snippet } from 'svelte';

  let { cols, rows = 5, children }: { cols: number; /** Loaded rows the frame stands for. */ rows?: number; children: Snippet } = $props();
</script>

<tr class="state">
  <td colspan={cols}>
    <div class="frame" style:min-height="calc({rows} * var(--size-row))">{@render children()}</div>
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
</style>
