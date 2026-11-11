import { css, defaults } from 'fw';

export interface LogoProps {
  size?: number;
  wordmark?: boolean;
  product?: string;
}

/**
 * The FOLO logo (design/logo.svg, design/logo-wordmark.svg). With `wordmark`, "folo" drawn with its last o as the
 * accent dot, then the product name; without, the mark alone: the f and the dot, `size` px square.
 */
export default function Logo(props: LogoProps) {
  const p = defaults(props, { size: 20, wordmark: true, product: 'Studio' });
  return (
    <span class="logo" class:lockup={p.wordmark}>
      {p.wordmark ? (
        <svg width={(p.size * 0.8 * 207) / 93} height={p.size * 0.8} viewBox="0 0 207 93" fill="none" aria-hidden="true">
          <path d="M16 91V29A20 20 0 0 1 36 9h8" stroke="currentColor" stroke-width="16" />
          <path d="M0 38h38" stroke="currentColor" stroke-width="14" />
          <circle cx="74" cy="61" r="23" stroke="currentColor" stroke-width="16" />
          <path d="M125 1v90" stroke="currentColor" stroke-width="16" />
          <circle class="dot" cx="176" cy="61" r="31" />
        </svg>
      ) : (
        <svg width={p.size} height={p.size} viewBox="0 0 24 24" fill="none" aria-hidden="true">
          <path d="M4.75 21V9.25A4.75 4.75 0 0 1 9.5 4.5h2" stroke="currentColor" stroke-width="4" />
          <path d="M.5 10.625h8" stroke="currentColor" stroke-width="3.5" />
          <circle class="dot" cx="17.25" cy="14.75" r="6.375" />
        </svg>
      )}
      {p.wordmark ? <span class="product">{p.product}</span> : null}
    </span>
  );
}

css`
  .logo {
    display: inline-flex;
    align-items: center;
    color: var(--text-primary);
  }
  /* The product name sits on the wordmark's baseline (the SVG's bottom edge, less the round letters' overshoot). */
  .lockup {
    align-items: baseline;
    gap: var(--space-1-5);
  }
  .dot {
    fill: var(--accent-solid);
  }
  .product {
    color: var(--text-tertiary);
    font-size: var(--type-md-size);
    font-weight: var(--weight-regular);
    line-height: 1;
    white-space: nowrap;
  }
`;
