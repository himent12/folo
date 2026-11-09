<script lang="ts">
  /**
   * A stacked bar of class shares in class colours, with 1 px surface gaps between segments, that is one disclosure:
   * "Class balance" (a click, Enter or a tap) opens a compact breakdown, each class's name, box count and share,
   * largest first, the first BREAKDOWN_MAX of a long list and the rest summed. One tab stop per strip, never one per
   * tiny segment, and no native hover titles (round 26: the names and counts were only in them). It is a button, so a
   * caller never puts it inside a link or a label: the card's link sits beside it (a stretched link), with the strip
   * above it.
   */
  import { classInk } from '$lib/design/classInk.svelte';
  import { num } from '$lib/format';
  import InfoButton from '$lib/ui/InfoButton.svelte';

  let { counts, classes, height = 6 }: { counts: number[]; classes: readonly string[]; height?: number } = $props();

  /** The most classes the breakdown lists by name; the rest are one line. */
  const BREAKDOWN_MAX = 10;
  const total = $derived(counts.reduce((a, b) => a + b, 0));
  const share = (c: number) => {
    const p = (c / total) * 100;
    return p > 0 && p < 1 ? '<1%' : `${Math.round(p)}%`;
  };
  const ranked = $derived(
    classes
      .map((name, i) => ({ i, name, n: counts[i] ?? 0 }))
      .filter((c) => c.n > 0)
      .sort((a, b) => b.n - a.n || a.i - b.i),
  );
  const shown = $derived(ranked.slice(0, BREAKDOWN_MAX));
  const rest = $derived(ranked.slice(BREAKDOWN_MAX));
  const restN = $derived(rest.reduce((a, c) => a + c.n, 0));
</script>

{#snippet strip()}
  <span class="strip" style:height="{height}px" aria-hidden="true">
    {#each counts as c, i (i)}
      {#if c > 0}<span style:flex-grow={c} style:background={classInk(i)}></span>{/if}
    {/each}
  </span>
{/snippet}

<!-- A click inside (the strip, the breakdown) never reaches a clickable row around it (Datasets' table). -->
<!-- svelte-ignore a11y_click_events_have_key_events, a11y_no_static_element_interactions -->
<span class="cs" onclick={(e) => e.stopPropagation()}>
  <InfoButton label="Class balance" class="cs-btn" trigger={strip}>
    <span class="bd">
      <span class="bd-h">Class balance <span class="bd-sub num">{num(total)} boxes</span></span>
      {#if total}
        <span class="bd-rows" role="list">
          {#each shown as c (c.i)}
            <span class="bd-row" role="listitem">
              <span class="sw" style:background={classInk(c.i)} aria-hidden="true"></span>
              <span class="bd-name">{c.name}</span>
              <span class="bd-n num">{num(c.n)}</span>
              <span class="bd-p num">{share(c.n)}</span>
            </span>
          {/each}
          {#if rest.length}
            <span class="bd-row bd-rest" role="listitem">
              <span class="sw" aria-hidden="true"></span>
              <span class="bd-name">{rest.length} more classes</span>
              <span class="bd-n num">{num(restN)}</span>
              <span class="bd-p num">{share(restN)}</span>
            </span>
          {/if}
        </span>
      {:else}
        <span>No boxes yet.</span>
      {/if}
    </span>
  </InfoButton>
</span>

<style>
  .cs {
    display: block;
    min-width: 0;
  }
  /* The strip is the trigger: the full width, a taller hit area than its 4–6 px (8 px above and below, 44 px on
     touch), and it thickens under the pointer, the cue that it opens. */
  .cs :global(.info-btn.cs-btn) {
    display: block;
    width: 100%;
    height: auto;
    padding: var(--space-2) 0;
    margin: calc(var(--space-2) * -1) 0;
    text-decoration: none;
    cursor: pointer;
  }
  .strip {
    display: flex;
    gap: 1px;
    width: 100%;
    border-radius: 999px;
    overflow: hidden;
    background: var(--bg-active);
    transition: transform var(--dur-fast) var(--ease-standard);
  }
  :global(.info-btn.cs-btn:hover) .strip,
  :global(.info-btn.cs-btn[aria-expanded='true']) .strip {
    transform: scaleY(1.5);
  }
  .strip > span {
    flex-basis: 0;
    min-width: 2px;
    box-shadow: inset 0 0 0 1px var(--chart-swatch-edge);
  }
  /* The breakdown: a compact table of 24 px rows, name, count and share. */
  .bd {
    display: grid;
    gap: var(--space-2);
    min-width: 200px;
  }
  .bd-h {
    color: var(--text-primary);
    font-weight: var(--weight-medium);
  }
  .bd-sub {
    margin-left: var(--space-1);
    color: var(--text-tertiary);
    font-weight: var(--weight-regular);
  }
  .bd-rows {
    display: grid;
    grid-template-columns: 8px minmax(0, 1fr) auto 40px;
    column-gap: var(--space-2);
  }
  .bd-row {
    display: contents;
  }
  .bd-row > span {
    display: flex;
    align-items: center;
    min-height: 24px;
  }
  .bd-row .sw {
    width: 8px;
    height: 8px;
    min-height: 0;
    align-self: center;
    border-radius: 2px;
    box-shadow: inset 0 0 0 1px var(--chart-swatch-edge);
  }
  .bd-rest .sw {
    box-shadow: inset 0 0 0 1px var(--line-strong);
  }
  .bd-name {
    color: var(--text-primary);
    overflow-wrap: anywhere;
  }
  .bd-n,
  .bd-p {
    justify-content: flex-end;
  }
  .bd-p {
    color: var(--text-tertiary);
  }
</style>
