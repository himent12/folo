<script lang="ts">
  import type { HTMLInputAttributes } from 'svelte/elements';
  import type { IconType } from './types';

  interface Props extends Omit<HTMLInputAttributes, 'value' | 'size'> {
    value?: string | number | null;
    icon?: IconType;
    /** Unit/suffix shown inside the right edge. */
    suffix?: string;
    invalid?: boolean;
    size?: 'sm' | 'md';
    el?: HTMLInputElement;
  }

  let {
    value = $bindable(),
    icon: Icon,
    suffix,
    invalid = false,
    size = 'md',
    el = $bindable(),
    class: klass = '',
    ...rest
  }: Props = $props();
</script>

<div class="input {size} {klass}" class:invalid class:has-icon={!!Icon}>
  {#if Icon}<span class="icon"><Icon size={14} strokeWidth={1.75} aria-hidden="true" /></span>{/if}
  <input bind:this={el} bind:value aria-invalid={invalid || undefined} {...rest} />
  {#if suffix}<span class="suffix">{suffix}</span>{/if}
</div>

<style>
  .input {
    position: relative;
    display: flex;
    align-items: center;
    height: var(--control-md);
    border-radius: var(--radius-md);
    background: var(--bg-inset);
    border: 1px solid var(--line-default);
    transition:
      border-color var(--dur-fast) var(--ease-standard),
      box-shadow var(--dur-fast) var(--ease-standard);
    min-width: 0;
  }
  :global([data-theme='light']) .input {
    background: var(--bg-panel);
  }
  .sm {
    height: var(--control-sm);
    border-radius: var(--radius-sm);
  }
  .input:hover {
    border-color: var(--line-strong);
  }
  .input:focus-within {
    border-color: var(--focus);
    box-shadow: 0 0 0 3px color-mix(in srgb, var(--focus) 18%, transparent);
  }
  .invalid,
  .invalid:hover {
    border-color: var(--danger-solid);
  }
  input {
    flex: 1;
    min-width: 0;
    height: 100%;
    padding: 0 10px;
    background: transparent;
    border: 0;
    outline: none;
    color: var(--text-primary);
    font-variant-numeric: tabular-nums;
  }
  .sm input {
    padding: 0 var(--space-2);
    font-size: var(--type-sm-size);
  }
  .has-icon input {
    padding-left: 30px;
  }
  input::placeholder {
    color: var(--text-tertiary);
  }
  input:focus-visible {
    outline: none;
  }
  input[type='number'] {
    -moz-appearance: textfield;
    appearance: textfield;
  }
  input[type='number']::-webkit-inner-spin-button,
  input[type='number']::-webkit-outer-spin-button {
    -webkit-appearance: none;
    margin: 0;
  }
  .icon {
    position: absolute;
    left: 10px;
    display: flex;
    color: var(--text-tertiary);
    pointer-events: none;
  }
  .suffix {
    padding-right: 10px;
    color: var(--text-tertiary);
    font-size: var(--type-sm-size);
    white-space: nowrap;
  }
</style>
