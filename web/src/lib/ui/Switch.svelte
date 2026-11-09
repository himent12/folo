<script lang="ts">
  /**
   * A labelled on/off switch. Its name is the label alone; its own hint, and an enclosing Field's message, are its
   * accessible description (`ui/field`), so a hint is never folded into the name nor left unannounced.
   */
  import { describedBy, getField } from './field';

  const uid = $props.id();
  let {
    checked = $bindable(false),
    label,
    hint,
    onchange,
    disabled = false,
    id,
    describedby,
  }: {
    checked?: boolean;
    label: string;
    hint?: string;
    onchange?: (v: boolean) => void;
    /** Off and not operable; say why next to it and pass that element's id as `describedby`. */
    disabled?: boolean;
    id?: string;
    describedby?: string;
  } = $props();

  const field = getField();
  const base = $derived(id ?? `switch-${uid}`);
</script>

<label class="switch" class:disabled>
  <span class="text">
    <span class="label" id="{base}-label">{label}</span>
    {#if hint}<span class="hint" id="{base}-desc">{hint}</span>{/if}
  </span>
  <input
    type="checkbox"
    role="switch"
    {id}
    {disabled}
    aria-labelledby="{base}-label"
    aria-describedby={describedBy(describedby, hint && `${base}-desc`, field?.describedby)}
    bind:checked
    onchange={() => onchange?.(checked)}
  />
  <span class="track" aria-hidden="true"><span class="thumb"></span></span>
</label>

<style>
  .switch {
    display: flex;
    align-items: center;
    gap: var(--space-3);
    cursor: pointer;
    position: relative;
  }
  .text {
    flex: 1;
    display: grid;
    gap: 2px;
  }
  .label {
    color: var(--text-primary);
  }
  /* Helper text keeps a readable measure (DESIGN_DIRECTION §2.3: ≤ 60 characters a line). */
  .hint {
    max-inline-size: var(--measure);
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
  }
  input {
    position: absolute;
    opacity: 0;
    width: 1px;
    height: 1px;
  }
  .track {
    flex-shrink: 0;
    position: relative;
    width: 30px;
    height: 18px;
    border-radius: 999px;
    background: var(--bg-active);
    /* Off, the track's edge is what shows the control: line.control (>= 3:1, WCAG 1.4.11). */
    box-shadow: inset 0 0 0 1px var(--line-control);
    transition: background-color var(--dur-fast) var(--ease-standard);
  }
  .thumb {
    position: absolute;
    top: 2px;
    left: 2px;
    width: 14px;
    height: 14px;
    border-radius: 50%;
    background: var(--text-secondary);
    transition:
      transform var(--dur-base) var(--ease-out),
      background-color var(--dur-fast) var(--ease-standard);
  }
  /* On: the ink track with an inverse knob, as the checkboxes are ink (DESIGN_DIRECTION §2.6). Lime is for "now" and
     the one action, never a control's state, so a switch never competes with Start training. */
  input:checked + .track {
    background: var(--text-primary);
    box-shadow: none;
  }
  input:checked + .track .thumb {
    transform: translateX(12px);
    background: var(--text-inverse);
  }
  /* Disabled: the label and thumb in text.disabled, the edge kept so the control still reads as a switch; the hint stays
     readable (it may be the reason). */
  .disabled {
    cursor: not-allowed;
  }
  .disabled .label {
    color: var(--text-disabled);
  }
  .disabled .thumb {
    background: var(--text-disabled);
  }
  input:focus-visible + .track {
    outline: 2px solid var(--focus);
    outline-offset: 2px;
  }
</style>
