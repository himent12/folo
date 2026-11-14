import { createUniqueId, css, defaults, Dynamic } from 'fw';
import InfoRegion from './InfoRegion';
import { provideInfoSlot } from './infoSlot';

export interface PanelProps {
  title?: string;
  /** Small text after the title (a count, a unit, a hint). */
  meta?: string;
  actions?: JSX.Element;
  children?: JSX.Element;
  /** Inner padding of the body. */
  pad?: 'none' | 'sm' | 'md';
  /**
   * Kept for callers: every header now has its hairline (design/UNIFIED.md §2, Panel), so this no longer changes the
   * look.
   */
  divided?: boolean;
  class?: string;
  id?: string;
  /** Element for the title, so pages keep a sane heading outline. */
  as?: 'h2' | 'h3';
}

export default function Panel(props: PanelProps) {
  const p = defaults(props, { pad: 'md', divided: false, class: '', as: 'h2' });
  // A chart's ⓘ in the header opens its definition here on phones, under the heading (ui/infoSlot).
  const uid = createUniqueId();
  const info = provideInfoSlot(`${uid}-info`);

  return (
    <section class={`panel pad-${p.pad} ${p.class}`} aria-labelledby={p.title && p.id ? `${p.id}-title` : undefined} id={p.id}>
      {p.title || p.actions ? (
        <>
          <header>
            {p.title ? (
              <Dynamic component={p.as} class="title" id={p.id ? `${p.id}-title` : undefined}>
                {p.title}
                {p.meta ? <span class="meta">{p.meta}</span> : null}
              </Dynamic>
            ) : null}
            {p.actions ? <div class="actions">{p.actions}</div> : null}
          </header>
          {p.actions ? <InfoRegion slot={info} /> : null}
        </>
      ) : null}
      {p.children ? <div class="body">{p.children}</div> : null}
    </section>
  );
}

css`
  .panel {
    background: var(--bg-panel);
    border: 1px solid var(--line-panel);
    border-radius: var(--radius-lg);
    min-width: 0;
    display: flex;
    flex-direction: column;
  }
  :global([data-theme='dark']) .panel {
    box-shadow: inset 0 1px 0 #ffffff06;
  }
  /* The header band (design/UNIFIED.md §2, Panel): 40 px, the title in 13 / 500, a qualifier after it in 12 px
     text.tertiary, controls at the right, and a hairline under it; the body under it has 16 px all round. */
  header {
    display: flex;
    align-items: center;
    gap: var(--space-3);
    min-height: var(--control-xl);
    padding: var(--space-1) var(--space-3) var(--space-1) var(--space-4);
    border-bottom: 1px solid var(--line-subtle);
  }
  .title {
    flex: 1;
    min-width: 0;
    display: flex;
    align-items: baseline;
    gap: var(--space-2);
    font-size: var(--type-base-size);
    line-height: var(--type-base-line);
    letter-spacing: var(--type-base-tracking);
    font-weight: var(--weight-medium);
    color: var(--text-primary);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .meta {
    font-size: var(--type-sm-size);
    letter-spacing: 0;
    color: var(--text-tertiary);
    font-weight: var(--weight-regular);
    font-variant-numeric: tabular-nums;
  }
  .actions {
    display: flex;
    align-items: center;
    gap: var(--space-1);
    flex-shrink: 0;
  }
  /* Phones: controls wrap under the title instead of truncating it. */
  @media (max-width: 767px) {
    header {
      flex-wrap: wrap;
      row-gap: var(--space-2);
    }
    .title {
      flex: 1 0 auto;
      max-width: 100%;
    }
    .actions {
      flex-wrap: wrap;
      margin-left: auto;
    }
  }
  .body {
    flex: 1;
    min-height: 0;
    display: flex;
    flex-direction: column;
  }
  .pad-md .body {
    padding: 0 var(--space-4) var(--space-4);
  }
  .pad-md header ~ .body,
  .pad-md > .body:first-child {
    padding-top: var(--space-4);
  }
  .pad-sm .body {
    padding: 0 var(--space-3) var(--space-3);
  }
  .pad-sm header ~ .body,
  .pad-sm > .body:first-child {
    padding-top: var(--space-3);
  }
`;
