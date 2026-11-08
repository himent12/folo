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
    id?: string;
  } = $props();

  const pct = $derived(((value - min) / (max - min || 1)) * 100);
</script>

<div class="slider">
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
    font-family: var(--font-mono);
    font-size: var(--type-sm-size);
    font-variant-numeric: tabular-nums;
    color: var(--text-primary);
  }
  .hint {
    font-size: var(--type-xs-size);
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
      var(--accent-line) 0,
      var(--accent-line) var(--pct),
      var(--bg-active) var(--pct),
      var(--bg-active) 100%
    );
  }
  input::-moz-range-track {
    height: 4px;
    border-radius: 999px;
    background: var(--bg-active);
  }
  input::-moz-range-progress {
    height: 4px;
    border-radius: 999px;
    background: var(--accent-line);
  }
  input::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 14px;
    height: 14px;
    margin-top: -5px;
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
  input:focus-visible::-webkit-slider-thumb {
    box-shadow: 0 0 0 2px var(--focus);
  }
  input:focus-visible::-moz-range-thumb {
    box-shadow: 0 0 0 2px var(--focus);
  }
</style>
