<script lang="ts">
  /** A labelled metric: small label, big tabular value, optional unit and a secondary line. */
  import type { Snippet } from 'svelte';
  import Skeleton from './Skeleton.svelte';

  let {
    label,
    value,
    unit,
    sub,
    size = 'md',
    loading = false,
  }: {
    label: string;
    value: string;
    unit?: string;
    sub?: Snippet | string;
    size?: 'sm' | 'md' | 'lg';
    /** The label is real; the value (and sub, when given) are placeholders in their exact line boxes. */
    loading?: boolean;
  } = $props();
</script>

<div class="stat {size}">
  <span class="label">{label}</span>
  {#if loading}
    <span class="value ph"><Skeleton w={size === 'lg' ? '96px' : '64px'} h={size === 'lg' ? '28px' : size === 'sm' ? '12px' : '16px'} /></span>
    {#if sub}<span class="sub ph"><Skeleton w="72px" h="10px" /></span>{/if}
  {:else}
    <!-- A value is a metric, always in ink: status hue goes on a mark beside its words, never on the number (§2.6). -->
    <span class="value">{value}{#if unit}<span class="unit">{unit}</span>{/if}</span>
    {#if sub}
      <span class="sub">{#if typeof sub === 'string'}{sub}{:else}{@render sub()}{/if}</span>
    {/if}
  {/if}
</div>

<style>
  /* Rows pack to the top: a stat without a sub line, stretched beside one with it, keeps its value on the shared
     baseline instead of spreading its rows over the extra height. */
  .stat {
    display: grid;
    align-content: start;
    gap: 2px;
    min-width: 0;
  }
  .label {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-tertiary);
    white-space: nowrap;
  }
  /* Inline flow, not flex: a unit or a superscript never changes the line box, so every stat's value sits on the same
     baseline whatever it contains. */
  .value {
    display: block;
    /* A fixed row: glyphs from a fallback font (superscripts like ⁻³) cannot grow the line box. */
    height: var(--type-lg-line);
    font-size: var(--type-lg-size);
    line-height: var(--type-lg-line);
    letter-spacing: var(--type-lg-tracking);
    font-weight: var(--weight-medium);
    font-variant-numeric: tabular-nums;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .sm .value {
    height: var(--type-md-line);
    font-size: var(--type-md-size);
    line-height: var(--type-md-line);
  }
  .lg .value {
    height: var(--type-3xl-line);
    font-size: var(--type-3xl-size);
    line-height: var(--type-3xl-line);
    letter-spacing: var(--type-3xl-tracking);
    font-weight: var(--weight-semibold);
  }
  .ph {
    display: flex;
    align-items: center;
  }
  .sub.ph {
    height: var(--type-sm-line);
  }
  .unit {
    /* line-height 1 keeps a smaller unit from growing the line box, so values with and without units share a baseline. */
    line-height: 1;
    margin-left: 4px;
    font-size: 0.7em;
    font-weight: var(--weight-regular);
    color: var(--text-tertiary);
    letter-spacing: 0;
  }
  .lg .unit {
    font-size: 0.5em;
  }
  .sub {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
    font-variant-numeric: tabular-nums;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
</style>
