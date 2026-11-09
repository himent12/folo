<script lang="ts">
  /**
   * A multi-line text field (a labeling job's instructions): ui/Input's frame, edge, ring and touch text for several
   * lines. An enclosing Field's hint or error describes it and its error marks it (ui/field).
   */
  import type { HTMLTextareaAttributes } from 'svelte/elements';
  import { describedBy, getField } from './field';

  interface Props extends Omit<HTMLTextareaAttributes, 'value'> {
    value?: string;
    invalid?: boolean;
  }

  let { value = $bindable(''), invalid = false, class: klass = '', 'aria-describedby': ownDescribedby, ...rest }: Props = $props();

  const field = getField();
</script>

<textarea
  class="ta {klass}"
  bind:value
  aria-invalid={invalid || field?.invalid || undefined}
  aria-describedby={describedBy(ownDescribedby, field?.describedby)}
  {...rest}
></textarea>

<style>
  .ta {
    display: block;
    width: 100%;
    min-width: 0;
    padding: var(--space-2) 12px;
    border-radius: var(--radius-md);
    background: var(--bg-inset);
    /* The edge is what marks the field: line.control is >= 3:1 on every surface (WCAG 1.4.11). */
    border: 1px solid var(--line-control);
    color: var(--text-primary);
    line-height: var(--type-base-line);
    resize: vertical;
    transition: border-color var(--dur-fast) var(--ease-standard);
  }
  :global([data-theme='light']) .ta {
    background: var(--bg-panel);
  }
  .ta:hover:not(:disabled, :read-only) {
    border-color: var(--line-control-hover);
  }
  /* Read-only (an agent run has them): the text stays readable, on the panel surface, without an edge to type in. */
  .ta:read-only:not(:disabled) {
    background: transparent;
    border-color: var(--line-subtle);
    color: var(--text-secondary);
    resize: none;
  }
  .ta:disabled {
    color: var(--text-disabled);
    cursor: not-allowed;
  }
  .ta::placeholder {
    color: var(--text-tertiary);
  }
  /* Focus is the one global ring (base.css :focus-visible), never a halo or a lime edge of its own. */
  .ta:focus-visible {
    outline: var(--size-focus-ring) solid var(--focus);
    outline-offset: var(--size-focus-offset);
  }
  /* Touch: the typed text at 16 px (type.lg), under which iOS zooms the page into the field (touchText.test.ts). */
  @media (pointer: coarse) {
    .ta {
      font-size: var(--type-lg-size);
    }
  }
</style>
