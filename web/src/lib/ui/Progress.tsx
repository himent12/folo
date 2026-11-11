/** A thin determinate progress bar. `value` is 0..1. */
import { computed, css, defaults } from 'fw';

type Tone = 'accent' | 'success' | 'danger' | 'warning' | 'neutral';

export interface ProgressProps {
  value: number;
  tone?: Tone;
  size?: 'xs' | 'sm' | 'md';
  label?: string;
}

export default function Progress(props: ProgressProps) {
  const p = defaults(props, { tone: 'accent', size: 'sm' });
  const v = computed(() => Math.max(0, Math.min(1, Number.isFinite(p.value) ? p.value : 0)));
  return (
    <div
      class={`track ${p.size} ${p.tone}`}
      role="progressbar"
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={Math.floor(v.get() * 100 + 1e-9)}
      aria-label={p.label}
    >
      <div class="fill" style:transform={`scaleX(${v.get()})`}></div>
    </div>
  );
}

css`
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
`;
