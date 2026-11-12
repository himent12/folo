/**
 * Why Start is off, as a way to the field that needs changing (the field may be far away). The mark in front carries
 * the danger hue; the link itself is ink, since danger never means navigation.
 */
import ArrowRight from '$icons/arrow-right';
import CircleAlert from '$icons/circle-alert';
import { css } from 'fw';
import type { Jump } from './jump';

export interface WhyLineProps {
  id: string;
  cls?: string;
  /** The first problem, or null while Start is on (or the form is still filling in). */
  why: { text: string; key: Jump } | null;
  jump: (key: Jump) => void;
}

export default function WhyLine(p: WhyLineProps) {
  return (
    <>
      {p.why ? (
        <p class={`why ${p.cls ?? ''}`} id={p.id} role={p.id === 'start-why' ? 'status' : undefined}>
          <CircleAlert size={14} strokeWidth={2} aria-hidden="true" />
          <button
            type="button"
            class="why-go"
            onclick={() => {
              const w = p.why;
              if (w) p.jump(w.key);
            }}
          >
            {p.why?.text}
            <ArrowRight size={12} strokeWidth={2} aria-hidden="true" />
          </button>
        </p>
      ) : null}
    </>
  );
}

css`
  .why {
    display: flex;
    align-items: flex-start;
    gap: var(--space-2);
    margin-top: var(--space-2);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-primary);
  }
  .why > :global(svg) {
    flex-shrink: 0;
    margin-top: 2px;
    color: var(--danger-solid);
  }
  /* The reason Start is off leads to the field: an ink link, underlined, the arrow in tertiary. */
  .why-go {
    display: inline;
    min-width: 0;
    color: inherit;
    text-align: left;
    text-decoration: underline;
    text-decoration-color: color-mix(in srgb, currentColor 40%, transparent);
    text-underline-offset: 3px;
    border-radius: var(--radius-xs);
  }
  .why-go:hover {
    text-decoration-color: currentColor;
  }
  .why-go :global(svg) {
    display: inline;
    margin-left: var(--space-1);
    vertical-align: -1px;
    color: var(--text-tertiary);
  }
  @media (pointer: coarse) {
    .why {
      align-items: center;
    }
    .why > :global(svg) {
      margin-top: 0;
    }
    .why-go {
      display: inline-block;
      min-height: 44px;
      padding-block: var(--space-3);
    }
  }
  @media (max-width: 1023px) {
    /* The reason sits on the dock's left, where the estimate would be. */
    .dock-why {
      flex: 1;
      min-width: 0;
      margin-top: 0;
    }
  }
`;
