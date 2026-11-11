import { css, defaults } from 'fw';

export interface LogoProps {
  size?: number;
  wordmark?: boolean;
  product?: string;
}

export default function Logo(props: LogoProps) {
  const p = defaults(props, { size: 20, wordmark: true, product: 'Studio' });
  return (
    <span class="logo">
      <svg width={p.size} height={p.size} viewBox="0 0 24 24" fill="none" aria-hidden="true">
        <path
          d="M3 8.5V5a2 2 0 0 1 2-2h3.5M15.5 3H19a2 2 0 0 1 2 2v3.5M21 15.5V19a2 2 0 0 1-2 2h-3.5M8.5 21H5a2 2 0 0 1-2-2v-3.5"
          stroke="currentColor"
          stroke-width="2"
          stroke-linecap="round"
        />
        <circle class="dot" cx="12" cy="12" r="3.25" />
      </svg>
      {p.wordmark ? (
        <span class="word">
          <span class="folo">FOLO</span> <span class="product">{p.product}</span>
        </span>
      ) : null}
    </span>
  );
}

css`
  .logo {
    display: inline-flex;
    align-items: center;
    gap: var(--space-2);
    color: var(--text-primary);
  }
  .dot {
    fill: var(--accent-solid);
  }
  :global([data-theme='light']) .dot {
    fill: var(--accent-line);
  }
  .word {
    font-size: var(--type-md-size);
    line-height: 1;
    white-space: nowrap;
  }
  .folo {
    font-weight: var(--weight-semibold);
    letter-spacing: -0.02em;
  }
  .product {
    color: var(--text-tertiary);
    font-weight: var(--weight-regular);
  }
`;
