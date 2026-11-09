<script lang="ts">
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
</script>

<label class="switch" class:disabled>
  <span class="text">
    <span class="label">{label}</span>
    {#if hint}<span class="hint">{hint}</span>{/if}
  </span>
  <input type="checkbox" role="switch" {id} {disabled} aria-describedby={describedby} bind:checked onchange={() => onchange?.(checked)} />
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
  .hint {
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
  input:checked + .track {
    background: var(--accent-line);
    box-shadow: none;
  }
  input:checked + .track .thumb {
    transform: translateX(12px);
    background: var(--accent-on);
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
