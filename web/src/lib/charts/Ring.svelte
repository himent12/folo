<script lang="ts">
  /** A small utilisation ring (0..1) with the value in the middle. */
  let {
    value,
    size = 44,
    stroke = 4,
    color = 'var(--accent-line)',
    label,
  }: { value: number | null; size?: number; stroke?: number; color?: string; label: string } = $props();

  const r = $derived((size - stroke) / 2);
  const c = $derived(2 * Math.PI * r);
  const v = $derived(value === null ? 0 : Math.max(0, Math.min(1, value)));
</script>

<div class="ring" style:width="{size}px" style:height="{size}px" role="img" aria-label="{label}: {value === null ? 'unknown' : `${Math.round(v * 100)}%`}">
  <svg width={size} height={size} viewBox="0 0 {size} {size}" aria-hidden="true">
    <circle class="track" cx={size / 2} cy={size / 2} {r} stroke-width={stroke} />
    <circle
      class="val"
      cx={size / 2}
      cy={size / 2}
      {r}
      stroke-width={stroke}
      style:stroke={color}
      stroke-dasharray="{c * v} {c}"
      transform="rotate(-90 {size / 2} {size / 2})"
    />
  </svg>
  <span class="num">{value === null ? '—' : Math.round(v * 100)}</span>
</div>

<style>
  .ring {
    position: relative;
    display: grid;
    place-items: center;
    flex-shrink: 0;
  }
  svg {
    position: absolute;
    inset: 0;
  }
  .track {
    fill: none;
    stroke: var(--bg-active);
  }
  .val {
    fill: none;
    stroke-linecap: round;
    transition: stroke-dasharray var(--dur-chart) var(--ease-standard);
  }
  .num {
    font-family: var(--font-mono);
    font-size: var(--type-xs-size);
    font-variant-numeric: tabular-nums;
    color: var(--text-primary);
  }
</style>
