<script lang="ts">
  /** A labelled metric: small label, big tabular value, optional unit and a secondary line. */
  import type { Snippet } from 'svelte';

  let {
    label,
    value,
    unit,
    sub,
    size = 'md',
    tone,
  }: {
    label: string;
    value: string;
    unit?: string;
    sub?: Snippet | string;
    size?: 'sm' | 'md' | 'lg';
    tone?: 'accent' | 'danger';
  } = $props();
</script>

<div class="stat {size}">
  <span class="label">{label}</span>
  <span class="value {tone ?? ''}">{value}{#if unit}<span class="unit">{unit}</span>{/if}</span>
  {#if sub}
    <span class="sub">{#if typeof sub === 'string'}{sub}{:else}{@render sub()}{/if}</span>
  {/if}
</div>

<style>
  .stat {
    display: grid;
    gap: 2px;
    min-width: 0;
  }
  .label {
    font-size: var(--type-xs-size);
    line-height: var(--type-xs-line);
    color: var(--text-tertiary);
    white-space: nowrap;
  }
  .value {
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
    font-size: var(--type-md-size);
    line-height: var(--type-md-line);
  }
  .lg .value {
    font-size: var(--type-3xl-size);
    line-height: var(--type-3xl-line);
    letter-spacing: var(--type-3xl-tracking);
    font-weight: var(--weight-semibold);
  }
  .value.accent {
    color: var(--accent-text);
  }
  .value.danger {
    color: var(--danger-text);
  }
  .unit {
    margin-left: 3px;
    font-size: 0.7em;
    font-weight: var(--weight-regular);
    color: var(--text-tertiary);
    letter-spacing: 0;
  }
  .lg .unit {
    font-size: 0.5em;
  }
  .sub {
    font-size: var(--type-xs-size);
    line-height: var(--type-xs-line);
    color: var(--text-secondary);
    font-variant-numeric: tabular-nums;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
</style>
