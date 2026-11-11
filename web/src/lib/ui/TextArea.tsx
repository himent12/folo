/**
 * A multi-line text field (a labeling job's instructions): ui/Input's frame, edge, ring and touch text for several
 * lines. An enclosing Field's hint or error describes it and its error marks it (ui/field).
 */
import { bindable, css, defaults, splitProps, type ElementProps } from 'fw';
import { describedBy, getField } from './field';

export interface TextAreaProps extends Omit<ElementProps<'textarea'>, 'value'> {
  value?: string;
  invalid?: boolean;
}

export default function TextArea(props: TextAreaProps) {
  const p = defaults(props, { invalid: false, class: '' });
  const value = bindable(props, 'value', '');
  const [, rest] = splitProps(props, ['value', 'invalid', 'class', 'aria-describedby']);
  const field = getField();

  return (
    <textarea
      class={`ta ${p.class}`}
      bind:value={value}
      aria-invalid={p.invalid || field?.invalid || undefined}
      aria-describedby={describedBy(p['aria-describedby'], field?.describedby)}
      {...rest}
    ></textarea>
  );
}

css`
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
  /* Touch: the typed text at 16 px (type.lg), under which iOS zooms the page into the field (sourceRules.test.ts). */
  @media (pointer: coarse) {
    .ta {
      font-size: var(--type-lg-size);
    }
  }
`;
