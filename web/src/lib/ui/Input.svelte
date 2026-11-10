<script lang="ts">
  import type { HTMLInputAttributes } from 'svelte/elements';
  import { describedBy, getField } from './field';
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
    'aria-describedby': ownDescribedby,
    ...rest
  }: Props = $props();

  // An enclosing Field's hint or error describes the input and its error marks it, with no wiring by the caller.
  const field = getField();
</script>

<!-- aria-invalid on the input styles the frame (base.css): a danger edge, a danger focus ring. -->
<div class="input {size} {klass}" class:has-icon={!!Icon}>
  {#if Icon}<span class="icon"><Icon size={14} strokeWidth={1.75} aria-hidden="true" /></span>{/if}
  <input
    bind:this={el}
    bind:value
    aria-invalid={invalid || field?.invalid || undefined}
    aria-describedby={describedBy(ownDescribedby, field?.describedby)}
    {...rest}
  />
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
    /* The edge is what marks the field: line.control is >= 3:1 on every surface (WCAG 1.4.11). */
    border: 1px solid var(--line-control);
    transition: border-color var(--dur-fast) var(--ease-standard);
    min-width: 0;
  }
  :global([data-theme='light']) .input {
    background: var(--bg-panel);
  }
  .sm {
    height: var(--control-sm);
    border-radius: var(--radius-sm);
  }
  .input:hover:not(:has(> input:disabled)) {
    border-color: var(--line-control-hover);
  }
  /* Off (a list toolbar over an empty or failed list): in place, its edge kept, its ink and placeholder disabled. */
  .input:has(> input:disabled) {
    cursor: not-allowed;
  }
  input:disabled,
  input:disabled::placeholder {
    color: var(--text-disabled);
    cursor: not-allowed;
  }
  /* Focus is the one global ring (base.css :focus-visible), drawn on the field's frame while its input has it: never
     a halo or a lime edge of its own (DESIGN_DIRECTION §2.9). An invalid field's ring is danger (base.css). */
  .input:has(> input:focus-visible) {
    outline: var(--size-focus-ring) solid var(--focus);
    outline-offset: var(--size-focus-offset);
  }
  input {
    flex: 1;
    min-width: 0;
    height: 100%;
    padding: 0 12px;
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
    padding-left: 28px;
  }
  input::placeholder {
    color: var(--text-tertiary);
  }
  /* The browser's clear button of a search field: neutral ink, not the platform's blue. */
  input::-webkit-search-cancel-button {
    filter: grayscale(1);
    opacity: 0.6;
    cursor: pointer;
  }
  input::-webkit-search-cancel-button:hover {
    opacity: 1;
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
    left: 8px;
    display: flex;
    color: var(--text-tertiary);
    pointer-events: none;
  }
  .suffix {
    padding-right: 12px;
    color: var(--text-tertiary);
    font-size: var(--type-sm-size);
    white-space: nowrap;
  }
  /* Touch: 44 px controls, and the typed text at 16 px (type.lg), under which iOS zooms the page into the field on
     focus (sourceRules.test.ts). The suffix and the labels around the field keep their sizes. */
  @media (pointer: coarse) {
    .input,
    .sm {
      height: 44px;
    }
    input,
    .sm input {
      font-size: var(--type-lg-size);
    }
  }
</style>
