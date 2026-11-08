<script lang="ts">
  /** A labelled range slider with a live value readout. Native <input type=range>, so keyboard and AT work. */
  const uid = $props.id();
  let {
    label,
    value = $bindable(),
    min = 0,
    max = 1,
    step = 0.01,
    format = (v: number) => v.toFixed(2),
    hint,
    oninput,
    inline = false,
    id = `slider-${uid}`,
  }: {
    label: string;
    value: number;
    min?: number;
    max?: number;
    step?: number;
    format?: (v: number) => string;
    hint?: string;
    oninput?: (v: number) => void;
    /** Label, track and value on one row (compact toolbars). */
    inline?: boolean;
    id?: string;
  } = $props();

  const pct = $derived(((value - min) / (max - min || 1)) * 100);
</script>

<div class="slider" class:inline>
  <div class="top">
    <label for={id}>{label}</label>
    <output for={id} class="val">{format(value)}</output>
  </div>
  <input
    {id}
    type="range"
    {min}
    {max}
    {step}
    bind:value
    oninput={() => oninput?.(value)}
    style:--pct="{pct}%"
    aria-describedby={hint ? `${id}-hint` : undefined}
  />
  {#if hint}<p class="hint" id="{id}-hint">{hint}</p>{/if}
</div>

<style>
  .slider {
    display: grid;
    gap: 6px;
  }
  .inline {
    grid-template-columns: auto minmax(0, 1fr) auto;
    align-items: center;
    gap: var(--space-3);
  }
  .inline .top {
    display: contents;
  }
  .inline .val {
    order: 2;
    min-width: 4ch;
    text-align: right;
  }
  .inline .hint {
    display: none;
  }
  .top {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
  }
  label {
    color: var(--text-secondary);
    font-weight: var(--weight-medium);
  }
  .val {
    font-variant-numeric: tabular-nums;
    font-size: var(--type-sm-size);
    font-variant-numeric: tabular-nums;
    color: var(--text-primary);
  }
  .hint {
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
  }
  input {
    -webkit-appearance: none;
    appearance: none;
    width: 100%;
    height: 16px;
    background: transparent;
    margin: 0;
    cursor: pointer;
  }
  input::-webkit-slider-runnable-track {
    height: 4px;
    border-radius: 999px;
    background: linear-gradient(
      to right,
      var(--text-secondary) 0,
      var(--text-secondary) var(--pct),
      var(--line-control) var(--pct),
      var(--line-control) 100%
    );
  }
  input::-moz-range-track {
    height: 4px;
    border-radius: 999px;
    /* The unfilled rail marks the control's extent: line.control (>= 3:1, WCAG 1.4.11). */
    background: var(--line-control);
  }
  input::-moz-range-progress {
    height: 4px;
    border-radius: 999px;
    background: var(--text-secondary);
  }
  input::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 14px;
    height: 14px;
    margin-top: -5px; /* grid-ok: centres the 14 px thumb on the 4 px track */
    border-radius: 50%;
    background: var(--text-primary);
    border: 3px solid var(--bg-panel);
    box-shadow: 0 0 0 1px var(--line-strong), var(--shadow-sm);
    transition: transform var(--dur-fast) var(--ease-standard);
  }
  input::-moz-range-thumb {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--text-primary);
    border: 3px solid var(--bg-panel);
    box-shadow: 0 0 0 1px var(--line-strong);
  }
  input:hover::-webkit-slider-thumb,
  input:active::-webkit-slider-thumb {
    transform: scale(1.15);
  }
  input:focus-visible {
    outline: none;
  }
  @media (pointer: coarse) {
    input {
      height: 44px;
    }
    input::-webkit-slider-thumb {
      width: 20px;
      height: 20px;
      margin-top: -8px;
    }
  }
  input:focus-visible::-webkit-slider-thumb {
    box-shadow: 0 0 0 2px var(--focus);
  }
  input:focus-visible::-moz-range-thumb {
    box-shadow: 0 0 0 2px var(--focus);
  }
</style>
