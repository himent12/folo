<script lang="ts" generics="T extends string">
  /** A styled native <select>: the platform picker is the most accessible and fastest option on every device. */
  import ChevronDown from '@lucide/svelte/icons/chevrons-up-down';

  interface Option {
    value: T;
    label: string;
    disabled?: boolean;
  }

  let {
    value = $bindable(),
    options,
    id,
    size = 'md',
    label,
    onchange,
    disabled = false,
  }: {
    value: T;
    options: Option[];
    id?: string;
    size?: 'sm' | 'md';
    /** Accessible name when there is no visible <label for>. */
    label?: string;
    onchange?: (v: T) => void;
    disabled?: boolean;
  } = $props();
</script>

<div class="select {size}">
  <select {id} bind:value aria-label={label} {disabled} onchange={() => onchange?.(value)}>
    {#each options as o (o.value)}
      <option value={o.value} disabled={o.disabled}>{o.label}</option>
    {/each}
  </select>
  <span class="chev" aria-hidden="true"><ChevronDown size={14} strokeWidth={1.75} /></span>
</div>

<style>
  .select {
    position: relative;
    display: flex;
    min-width: 0;
  }
  select {
    -webkit-appearance: none;
    appearance: none;
    width: 100%;
    height: var(--control-md);
    padding: 0 28px 0 12px;
    border-radius: var(--radius-md);
    background: var(--bg-raised);
    border: 1px solid var(--line-default);
    color: var(--text-primary);
    cursor: pointer;
    transition: border-color var(--dur-fast) var(--ease-standard);
    text-overflow: ellipsis;
  }
  .sm select {
    height: var(--control-sm);
    padding: 0 28px 0 8px;
    border-radius: var(--radius-sm);
    font-size: var(--type-sm-size);
  }
  select:hover {
    border-color: var(--line-strong);
  }
  select:focus-visible {
    outline: none;
    border-color: var(--focus);
    box-shadow: 0 0 0 3px color-mix(in srgb, var(--focus) 18%, transparent);
  }
  select:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  option {
    background: var(--bg-overlay);
    color: var(--text-primary);
  }
  .chev {
    position: absolute;
    right: 8px;
    top: 50%;
    transform: translateY(-50%);
    display: flex;
    color: var(--text-tertiary);
    pointer-events: none;
  }
  .sm .chev {
    right: 6px;
  }
  /* Touch: 44 px controls. */
  @media (pointer: coarse) {
    select,
    .sm select {
      height: 44px;
    }
  }
</style>
