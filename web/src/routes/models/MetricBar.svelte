<script lang="ts">
  /**
   * A 0..1 metric as a number with a small inline bar (for tables). `bare` draws the bar only. The bar is a neutral
   * meter; `best` only strengthens it, the "best" marker itself is drawn by the caller.
   */
  import { metric } from '$lib/format';

  let {
    value,
    best = false,
    tone = 'accent',
    bare = false,
  }: { value: number | null | undefined; best?: boolean; tone?: 'accent' | 'quiet'; bare?: boolean } = $props();

  const v = $derived(value === null || value === undefined || value < 0 ? null : Math.min(1, value));
</script>

<span class="mb {tone}" class:best class:bare>
  <span class="track" aria-hidden="true"><span class="fill" style:transform="scaleX({v ?? 0})"></span></span>
  {#if !bare}<span class="val num">{metric(v)}</span>{/if}
</span>

<style>
  .mb {
    display: inline-flex;
    align-items: center;
    justify-content: flex-end;
    gap: var(--space-2);
  }
  .bare {
    display: flex;
    width: 100%;
  }
  .track {
    position: relative;
    width: 44px;
    height: 4px;
    border-radius: var(--radius-full);
    background: var(--bg-active);
    overflow: hidden;
  }
  .bare .track {
    width: 100%;
  }
  .fill {
    position: absolute;
    inset: 0;
    transform-origin: left;
    border-radius: inherit;
    background: var(--text-tertiary);
  }
  .accent .fill {
    background: var(--text-secondary);
  }
  .best .fill {
    background: var(--text-primary);
  }
  .val {
    min-width: 38px;
    text-align: right;
  }
  .accent .val {
    font-weight: var(--weight-medium);
  }
  .quiet .val {
    color: var(--text-secondary);
  }
  .best .val {
    color: var(--text-primary);
  }
</style>
