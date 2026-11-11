// fw port of ToggleButton.svelte.
/**
 * A small on / off button for a view option of the content under it (an image grid's Boxes): `aria-pressed`, and a
 * neutral pressed look, a stronger edge and primary text (lime is for the primary action, focus and live state).
 */
import { css, Dynamic } from 'fw';
import type { IconType } from './types.fw';

export interface ToggleButtonProps {
  pressed: boolean;
  label: string;
  icon: IconType;
  onchange: (on: boolean) => void;
}

export default function ToggleButton(props: ToggleButtonProps) {
  return (
    <button type="button" class="toggle" aria-pressed={props.pressed} onclick={() => props.onchange(!props.pressed)}>
      <Dynamic component={props.icon} size={14} strokeWidth={1.75} aria-hidden="true" />
      {props.label}
    </button>
  );
}

css`
  .toggle {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    height: var(--control-sm);
    padding: 0 var(--space-2);
    border-radius: var(--radius-sm);
    border: 1px solid var(--line-default);
    color: var(--text-secondary);
    font-size: var(--type-sm-size);
    font-weight: var(--weight-medium);
    white-space: nowrap;
    transition:
      border-color var(--dur-fast) var(--ease-standard),
      color var(--dur-fast) var(--ease-standard);
  }
  .toggle:hover {
    color: var(--text-primary);
    border-color: var(--line-strong);
  }
  .toggle[aria-pressed='true'] {
    border-color: var(--text-tertiary);
    color: var(--text-primary);
  }
  @media (pointer: coarse) {
    .toggle {
      height: 44px;
    }
  }
`;
