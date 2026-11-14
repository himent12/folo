/**
 * Standard page frame: title row (title, optional meta line and actions) above the content.
 *
 * The header (design/UNIFIED.md §2): the title in 20 / 600; on a detail page its status badge on the same line
 * (`badge`); under it one line of facts in 12 px text.secondary separated by " · " (`sub`: what the page holds right
 * now, "9 runs · 2 active · 1 failed", not a sentence describing the page); the actions at the right, at most one lime
 * primary, 32 px high.
 *
 * One rule for the actions on every page: from 768 px up they sit on the title row, right-aligned, whatever the
 * length of the lead and meta lines (which run the full width under and above it), and on one row: they never wrap. Below 768 px a page passes its
 * secondary actions as `more`: they are buttons on wider screens and one ⋯ menu on phones, beside `actions` (the
 * primary, which keeps a short word there).
 *
 * Phones, one row (DESIGN_DIRECTION §10 / §12): a page that passes `subPhone` (a short meta line, one line at most)
 * gets it in place of `sub`, with the primary and the ⋯ menu on its right, so the first content sits 16 px under it
 * instead of under a second row of full-width buttons.
 *
 * Snippets (`sub`, `subPhone`, `actions`, `lead`) are functions returning JSX: `sub={() => <>…</>}`.
 */
import { computed, css, defaults, For } from 'fw';
import Button from '$lib/ui/Button';
import Menu, { type MenuItem } from '$lib/ui/Menu';

/** Whether a page has rendered yet in this session: only the first fades in (see `.enter`). */
let entered = false;

export interface PageProps {
  title: string;
  /** A meta line under the title (snippet for rich content). */
  sub?: JSX.Element;
  /** Phones: a short meta line (one line, ellipsis as a guard) in place of `sub`; the header becomes one row. */
  subPhone?: JSX.Element;
  /** The page's own actions (its primary last). */
  actions?: JSX.Element;
  /** Secondary actions: secondary buttons before `actions` from 768 px up, one ⋯ menu after them on phones. */
  more?: MenuItem[];
  /** The ⋯ menu's accessible name, e.g. "More model actions". */
  moreLabel?: string;
  /** Something before the title. Kept for callers; a detail page's status badge goes on the title line (`badge`). */
  lead?: JSX.Element;
  /** A status badge (ui/StatusBadge) on the title's line, after the name: a detail page's header. */
  badge?: JSX.Element;
  children: JSX.Element;
  wide?: boolean;
  /** A form page (column and rail): the same content edge as every page, the column's measure set by the page. */
  narrow?: boolean;
  /** Phones: keep the actions beside the title block instead of wrapping them onto a row of their own. */
  inlineActions?: boolean;
}

export default function Page(props: PageProps) {
  const p = defaults(props, { more: [] as MenuItem[], moreLabel: 'More actions', wide: false, narrow: false, inlineActions: false });

  const hasActions = computed(() => !!p.actions || p.more.length > 0);
  /** Only the title: on phones the top bar shows it, so the header takes no room and content starts at the gutter. */
  const bare = computed(() => !p.sub && !p.subPhone && !hasActions.get() && !p.lead && !p.badge);
  // The app's first page fades in as Studio appears; moving between pages after that is immediate (a fade on every
  // route change decorates a move the person already made, and makes a list feel slow to arrive).
  const enter = !entered;
  entered = true;

  return (
    <div class="page" class:wide={p.wide} class:narrow={p.narrow} class:enter={enter}>
      <header class:inline={p.inlineActions} class:one-row={!!p.subPhone} class:bare={bare.get()}>
        <div class="titles">
          {p.lead ? <div class="lead">{p.lead}</div> : null}
          <div class="title-line">
            {/* Route changes move focus here (App), so the new page is announced by its name. */}
            <h1 tabindex={-1} data-route-focus>
              {p.title}
            </h1>
            {p.badge ? <span class="badge-slot">{p.badge}</span> : null}
          </div>
          {p.sub ? (
            <div class="sub" class:wide-only={!!p.subPhone}>
              {p.sub}
            </div>
          ) : null}
          {p.subPhone ? <div class="sub phone-only">{p.subPhone}</div> : null}
        </div>
        {hasActions.get() ? (
          <div class="actions">
            {p.more.length ? (
              <For each={p.more} key={(it) => it.label}>
                {(it) => (
                  <span class="more-wide">
                    <Button icon={it().icon} href={it().href} onclick={it().onclick} variant={it().danger ? 'danger' : 'secondary'}>
                      {it().label}
                    </Button>
                  </span>
                )}
              </For>
            ) : null}
            {p.actions}
            {p.more.length ? (
              <span class="more-phone">
                <Menu label={p.moreLabel} items={p.more} />
              </span>
            ) : null}
          </div>
        ) : null}
      </header>
      {p.children}
    </div>
  );
}

css`
  /* The content column is \`size.content_max\` (1200, DESIGN_DIRECTION 2) wide at most; the side gutters are outside it. */
  .page {
    width: 100%;
    max-width: calc(var(--size-content-max) + 2 * var(--space-6));
    margin: 0 auto;
    padding: var(--space-6) var(--space-6) var(--space-12);
    display: flex;
    flex-direction: column;
    gap: var(--space-5);
  }
  .page.enter {
    animation: in var(--dur-slow) var(--ease-out);
  }
  .page.wide {
    max-width: none;
  }
  /* Title row: the title on the left, the actions on its right edge (centred on the title's line); the lead and meta
     lines span the whole width, so a long meta line never pushes the actions down or to the left. Dense placement puts
     the actions back up beside the title, after the meta line in the source. The title keeps at least 40 % (240 px).
     **The actions never wrap onto a second row, at any tier** (DESIGN_DIRECTION §10): a page whose actions would not
     fit beside its title at 768 px moves the extra (a hint, a secondary action) into its content or its ⋯ menu, never
     a second row that leaves the title floating between them (the shoot's QA lists a wrapped header). */
  header {
    display: grid;
    grid-template-columns: minmax(min(40%, 240px), 1fr) auto;
    grid-auto-flow: row dense;
    align-items: center;
    column-gap: var(--space-4);
    row-gap: var(--space-1);
  }
  .titles {
    display: contents;
  }
  .lead,
  .sub {
    grid-column: 1 / -1;
  }
  .title-line {
    grid-column: 1;
    display: flex;
    align-items: center;
    gap: var(--space-2);
    min-width: 0;
  }
  .badge-slot {
    display: inline-flex;
    flex-shrink: 0;
  }
  .lead {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    margin-bottom: 2px;
  }
  h1 {
    font-size: var(--type-xl-size);
    line-height: var(--type-xl-line);
    letter-spacing: var(--type-xl-tracking);
    font-weight: var(--type-xl-weight);
    overflow-wrap: anywhere;
  }
  /* Route changes focus the title (App, shell/routeFocus) so a screen reader announces the page; it is not a control,
     so it wears no ring, also after a keyboard navigation (G R, Esc, the palette), where the browser would draw one. */
  h1:focus-visible {
    outline: none;
  }
  .sub {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: var(--space-1) var(--space-3);
    min-width: 0;
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
  }
  .actions {
    grid-column: 2;
    display: flex;
    align-items: center;
    justify-content: flex-end;
    gap: var(--space-2);
    flex-wrap: nowrap;
    min-width: 0;
  }
  /* A page header's buttons are 32 high (design/UNIFIED.md §2, Buttons), where a pointer is precise; touch keeps its
     44 px targets. */
  @media (min-width: 768px) and (pointer: fine) {
    .actions :global(.btn:not(.sm)) {
      --h: var(--control-lg);
    }
  }
  /* An actions snippet with nothing to show this time (e.g. no create action over an empty list) takes no room. */
  .actions:empty {
    display: none;
  }
  .more-wide {
    display: contents;
  }
  .more-phone,
  .phone-only {
    display: none;
  }
  @keyframes in {
    from {
      opacity: 0;
    }
  }
  @media (max-width: 767px) {
    .page {
      padding: var(--space-4) var(--space-4) var(--space-8);
      gap: var(--space-4);
    }
    /* Phones: the title block, then the actions on a row of their own (or beside it with inlineActions); secondary
       actions fold into the ⋯ menu. */
    header {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: var(--space-4);
      flex-wrap: wrap;
    }
    .titles {
      display: grid;
      gap: var(--space-1);
      min-width: 0;
    }
    .actions {
      justify-content: flex-start;
    }
    .more-wide {
      display: none;
    }
    .more-phone {
      display: contents;
    }
    header.inline {
      flex-wrap: nowrap;
    }
    header.inline .actions {
      flex-wrap: nowrap;
      flex-shrink: 0;
    }
    /* A lead longer than the room beside the actions (a cloud run's "Uploading data" and its time at 320 px) wraps
       whole items onto a second line rather than spill past them. */
    .lead {
      flex-wrap: wrap;
      row-gap: var(--space-1);
    }
    /* One row: the short meta line on the left (one line), the primary and the ⋯ menu on the right, centred on it. */
    header.one-row {
      flex-wrap: nowrap;
      align-items: center;
    }
    header.one-row .titles {
      flex: 1;
    }
    header.one-row .actions {
      flex-wrap: nowrap;
      flex-shrink: 0;
    }
    .wide-only {
      display: none;
    }
    .sub.phone-only {
      display: block;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    /* A title-only header leaves no empty slot (and no gap after it): the first content sits 16 px under the bar. */
    header.bare,
    header.bare .titles {
      display: contents;
    }
    /* The top bar shows the title on phones; the heading stays for assistive tech and the outline. */
    h1 {
      position: absolute;
      width: 1px;
      height: 1px;
      overflow: hidden;
      clip-path: inset(50%);
      white-space: nowrap;
    }
  }
`;
