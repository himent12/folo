<script lang="ts">
  /** Phones with an image: the classes and the timing as two disclosures under the detections, folded until asked for. */
  import type { Snippet } from 'svelte';
  import ChevronRight from '@lucide/svelte/icons/chevron-right';
  import type { Prediction } from '$lib/api/types';
  import { ms } from '$lib/format';

  let {
    result,
    counts,
    hidden,
    pred,
    where,
    showAllButton,
    classList,
    timingBody,
  }: {
    result: { boxes: Prediction['boxes']; ms: number } | null;
    counts: [number, number][];
    hidden: Set<number>;
    pred: Prediction | null;
    where: string;
    showAllButton: Snippet;
    classList: Snippet;
    timingBody: Snippet;
  } = $props();
</script>

<!-- Phones: classes and timing as two disclosures under the detections, folded until asked for. -->
<div class="discs">
  <details class="disc">
    <summary>
      <ChevronRight size={14} aria-hidden="true" />
      <span class="disc-title">Classes</span>
      {#if result}<span class="disc-meta num">{counts.length} found{hidden.size ? ` · ${hidden.size} hidden` : ''}</span>{/if}
    </summary>
    <div class="disc-body">
      {#if hidden.size}<div class="disc-act">{@render showAllButton()}</div>{/if}
      {@render classList()}
    </div>
  </details>
  <details class="disc">
    <summary>
      <ChevronRight size={14} aria-hidden="true" />
      <span class="disc-title">Timing</span>
      {#if pred}<span class="disc-meta num">{ms(pred.timing.total_ms)} · {where}</span>{/if}
    </summary>
    <div class="disc-body">{@render timingBody()}</div>
  </details>
</div>

<style>
  .discs {
    display: grid;
    gap: var(--space-2);
    order: 3;
  }
  .disc {
    border: 1px solid var(--line-panel);
    border-radius: var(--radius-lg);
    background: var(--bg-panel);
  }
  .disc summary {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    min-height: 44px;
    padding: 0 var(--space-4);
    list-style: none;
    cursor: pointer;
    border-radius: inherit;
  }
  .disc summary::-webkit-details-marker {
    display: none;
  }
  .disc summary :global(svg) {
    flex-shrink: 0;
    color: var(--text-tertiary);
    transition: transform var(--dur-fast) var(--ease-standard);
  }
  .disc[open] summary :global(svg) {
    transform: rotate(90deg);
  }
  .disc summary:focus-visible {
    outline-offset: -2px;
  }
  .disc-title {
    font-weight: var(--weight-medium);
  }
  .disc-meta {
    margin-left: auto;
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
  }
  .disc-body {
    padding: 0 var(--space-4) var(--space-4);
  }
  .disc-act {
    display: flex;
    justify-content: flex-end;
    margin-bottom: var(--space-1);
  }
</style>
