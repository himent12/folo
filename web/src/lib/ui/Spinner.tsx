// The small spinning ring; with a `label` it is a status that screen readers announce.
import { css, defaults } from 'fw';

export default function Spinner(props: { size?: number; label?: string }) {
  const p = defaults(props, { size: 14 });
  return (
    <svg
      class="spinner"
      width={p.size}
      height={p.size}
      viewBox="0 0 16 16"
      role={p.label ? 'status' : undefined}
      aria-label={p.label}
      aria-hidden={p.label ? undefined : 'true'}
    >
      <circle cx="8" cy="8" r="6.25" fill="none" stroke="currentColor" stroke-opacity="0.2" stroke-width="1.5" />
      <path d="M8 1.75a6.25 6.25 0 0 1 6.25 6.25" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" />
    </svg>
  );
}

css`
  .spinner {
    animation: spin 700ms linear infinite;
    flex-shrink: 0;
  }
  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }
`;
