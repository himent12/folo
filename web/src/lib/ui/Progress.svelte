<script lang="ts">
  /** A thin determinate progress bar. `value` is 0..1. */
  type Tone = 'accent' | 'success' | 'danger' | 'warning' | 'neutral';
  let { value, tone = 'accent', size = 'sm', label }: { value: number; tone?: Tone; size?: 'xs' | 'sm' | 'md'; label?: string } =
    $props();
  const v = $derived(Math.max(0, Math.min(1, Number.isFinite(value) ? value : 0)));
</script>

<div
  class="track {size} {tone}"
  role="progressbar"
  aria-valuemin={0}
  aria-valuemax={100}
  aria-valuenow={Math.round(v * 100)}
  aria-label={label}
>
  <div class="fill" style:transform="scaleX({v})"></div>
</div>

<style>
  .track {
    position: relative;
    width: 100%;
    height: 4px;
    border-radius: var(--radius-full);
    background: var(--bg-active);
    overflow: hidden;
  }
  .xs {
    height: 2px;
  }
  .md {
    height: 6px;
  }
  .fill {
    position: absolute;
    inset: 0;
    transform-origin: left;
    border-radius: inherit;
    background: var(--accent-solid);
    transition: transform var(--dur-slow) var(--ease-standard);
  }
  :global([data-theme='light']) .accent .fill {
    background: var(--accent-line);
  }
  .success .fill {
    background: var(--success-solid);
  }
  .danger .fill {
    background: var(--danger-solid);
  }
  .warning .fill {
    background: var(--warning-solid);
  }
  .neutral .fill {
    background: var(--text-tertiary);
  }
</style>
