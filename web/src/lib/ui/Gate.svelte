<script lang="ts">
  /**
   * A control that is disabled for a reason ("Server offline", "Select 2–4 runs to compare"). The reason is a tooltip
   * on hover and the control's accessible description, since a disabled button cannot take focus to show it. Render
   * the control in the snippet with `disabled` while `reason` is set, and pass it the id: `aria-describedby={why}`.
   */
  import type { Snippet } from 'svelte';
  import { tooltip } from './tooltip';

  let { reason, children }: { reason: string | null | undefined; children: Snippet<[string | undefined]> } = $props();

  const uid = $props.id();
</script>

<span class="gate" class:off={!!reason} use:tooltip={reason}>{@render children(reason ? `${uid}-why` : undefined)}</span>
{#if reason}<span class="sr-only" id="{uid}-why">{reason}</span>{/if}

<style>
  .gate {
    display: inline-flex;
  }
  /* A disabled button takes no pointer events in some browsers: the wrapper takes the hover, so the reason shows. */
  .off {
    cursor: not-allowed;
  }
  .off > :global(:disabled) {
    pointer-events: none;
  }
</style>
