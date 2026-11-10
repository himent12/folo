<script lang="ts">
  /** Adds a run to the comparison: a native picker of the runs not in it yet, dressed as a secondary button. */
  import Plus from '@lucide/svelte/icons/plus';
  import type { Run } from '$lib/api/types';
  import { statusWord } from '$lib/ui/StatusBadge.svelte';

  let { candidates, ids, setIds }: { candidates: Run[]; ids: string[]; setIds: (next: string[]) => void } = $props();
</script>

<!-- The face is a secondary Button's; the native picker lies over all of it, so any point of it opens the list. -->
<label class="add">
  <Plus size={14} strokeWidth={1.75} aria-hidden="true" />
  <span aria-hidden="true">Add run…</span>
  <select
    aria-label="Add a run to the comparison"
    onchange={(e) => {
      const v = (e.currentTarget as HTMLSelectElement).value;
      if (v) setIds([...ids, v]);
    }}
  >
    <option value="">Add run…</option>
    {#each candidates as c (c.id)}<option value={c.id}>{c.name} · {statusWord(c)}</option>{/each}
  </select>
</label>

<style>
  /* Adding a run: a native picker (the platform list is the most accessible) dressed as the standard secondary Button
     (ui/Button): raised fill, default edge, 28 px, the icon and a medium label. */
  .add {
    position: relative;
    display: inline-flex;
    align-items: center;
    align-self: center;
    justify-self: start;
    gap: var(--space-1-5);
    height: var(--control-md);
    padding: 0 var(--space-3);
    border-radius: var(--radius-md);
    border: 1px solid var(--line-default);
    background: var(--bg-raised);
    box-shadow: var(--shadow-sm);
    color: var(--text-primary);
    font-weight: var(--weight-medium);
    transition:
      background-color var(--dur-fast) var(--ease-standard),
      border-color var(--dur-fast) var(--ease-standard);
  }
  .add:hover {
    border-color: var(--line-strong);
    background: color-mix(in srgb, var(--bg-raised), var(--text-primary) 4%);
  }
  .add select {
    position: absolute;
    inset: 0;
    width: 100%;
    opacity: 0;
    cursor: pointer;
  }
  @media (pointer: coarse) {
    .add {
      min-height: 44px;
    }
    /* The native picker lies unseen over the button, but iOS still zooms into a select under 16 px (sourceRules.test.ts). */
    .add select {
      font-size: var(--type-lg-size);
    }
  }
  .add:has(select:focus-visible) {
    outline: 2px solid var(--focus);
    outline-offset: 2px;
  }
  .add select:focus-visible {
    outline: none;
  }
  .add option {
    background: var(--bg-overlay);
  }
</style>
