<script lang="ts" generics="T extends string">
  /** A styled native <select>: the platform picker is the most accessible and fastest option on every device. */
  import ChevronDown from '@lucide/svelte/icons/chevrons-up-down';

  interface Option {
    value: T;
    label: string;
    disabled?: boolean;
    /** Consecutive options with the same group are listed under it (`<optgroup>`); none, at the top level. */
    group?: string;
    /** What the option is, as its tooltip (a native option holds no markup). */
    title?: string;
  }

  let {
    value = $bindable(),
    options,
    id,
    size = 'md',
    label,
    onchange,
    disabled = false,
    invalid = false,
    describedby,
  }: {
    value: T;
    options: Option[];
    id?: string;
    size?: 'sm' | 'md';
    /** Accessible name when there is no visible <label for>. */
    label?: string;
    onchange?: (v: T) => void;
    disabled?: boolean;
    /** The field shows an error: announced, and tied to its message by `describedby` (Field's `<for>-error`). */
    invalid?: boolean;
    describedby?: string;
  } = $props();

  /** Runs of options: the ungrouped ones as they are, each group's under its label. */
  const runs = $derived(
    options.reduce<{ group?: string; options: Option[] }[]>((out, o) => {
      const last = out.at(-1);
      if (last && last.group === o.group) last.options.push(o);
      else out.push({ group: o.group, options: [o] });
      return out;
    }, []),
  );
</script>

{#snippet option(o: Option)}<option value={o.value} disabled={o.disabled} title={o.title}>{o.label}</option>{/snippet}

<div class="select {size}">
  <select {id} bind:value aria-label={label} {disabled} aria-invalid={invalid || undefined} aria-describedby={describedby} onchange={() => onchange?.(value)}>
    {#each runs as r, i (i)}
      {#if r.group}
        <optgroup label={r.group}>{#each r.options as o (o.value)}{@render option(o)}{/each}</optgroup>
      {:else}
        {#each r.options as o (o.value)}{@render option(o)}{/each}
      {/if}
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
    border: 1px solid var(--line-control);
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
    border-color: var(--line-control-hover);
  }
  /* Focus: the global :focus-visible ring (base.css), never a halo or a lime edge of its own (§2.9). */
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
