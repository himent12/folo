<script lang="ts" module>
  /** Whether a page has rendered yet in this session: only the first fades in (see `.enter`). */
  let entered = false;
</script>

<script lang="ts">
  /**
   * Standard page frame: title row (title, optional meta line and actions) above the content.
   *
   * One rule for the actions on every page: from 768 px up they sit on the title row, right-aligned, whatever the
   * length of the lead and meta lines (which run the full width under and above it). Below 768 px a page passes its
   * secondary actions as `more`: they are buttons on wider screens and one ⋯ menu on phones, beside `actions` (the
   * primary, which keeps a short word there).
   *
   * Phones, one row (DESIGN_DIRECTION §10 / §12): a page that passes `subPhone` (a short meta line, one line at most)
   * gets it in place of `sub`, with the primary and the ⋯ menu on its right, so the first content sits 16 px under it
   * instead of under a second row of full-width buttons.
   */
  import type { Snippet } from 'svelte';
  import Button from '$lib/ui/Button.svelte';
  import Menu, { type MenuItem } from '$lib/ui/Menu.svelte';

  let {
    title,
    sub,
    subPhone,
    actions,
    more = [],
    moreLabel = 'More actions',
    lead,
    children,
    wide = false,
    narrow = false,
    inlineActions = false,
  }: {
    title: string;
    /** A meta line under the title (snippet for rich content). */
    sub?: Snippet;
    /** Phones: a short meta line (one line, ellipsis as a guard) in place of `sub`; the header becomes one row. */
    subPhone?: Snippet;
    /** The page's own actions (its primary last). */
    actions?: Snippet;
    /** Secondary actions: secondary buttons before `actions` from 768 px up, one ⋯ menu after them on phones. */
    more?: MenuItem[];
    /** The ⋯ menu's accessible name, e.g. "More model actions". */
    moreLabel?: string;
    /** Something before the title, e.g. a status badge row. */
    lead?: Snippet;
    children: Snippet;
    wide?: boolean;
    /** A form page (column and rail): the same content edge as every page, the column's measure set by the page. */
    narrow?: boolean;
    /** Phones: keep the actions beside the title block instead of wrapping them onto a row of their own. */
    inlineActions?: boolean;
  } = $props();

  const hasActions = $derived(!!actions || more.length > 0);
  /** Only the title: on phones the top bar shows it, so the header takes no room and content starts at the gutter. */
  const bare = $derived(!sub && !subPhone && !hasActions && !lead);
  // The app's first page fades in as Studio appears; moving between pages after that is immediate (a fade on every
  // route change decorates a move the person already made, and makes a list feel slow to arrive).
  const enter = !entered;
  entered = true;
</script>

<div class="page" class:wide class:narrow class:enter>
  <header class:inline={inlineActions} class:one-row={!!subPhone} class:bare>
    <div class="titles">
      {#if lead}<div class="lead">{@render lead()}</div>{/if}
      <!-- Route changes move focus here (App), so the new page is announced by its name. -->
      <h1 tabindex="-1" data-route-focus>{title}</h1>
      {#if sub}<div class="sub" class:wide-only={!!subPhone}>{@render sub()}</div>{/if}
      {#if subPhone}<div class="sub phone-only">{@render subPhone()}</div>{/if}
    </div>
    {#if hasActions}
      <div class="actions">
        {#if more.length}
          {#each more as it (it.label)}
            <span class="more-wide">
              <Button icon={it.icon} href={it.href} onclick={it.onclick} variant={it.danger ? 'danger' : 'secondary'}>{it.label}</Button>
            </span>
          {/each}
        {/if}
        {@render actions?.()}
        {#if more.length}<span class="more-phone"><Menu label={moreLabel} items={more} /></span>{/if}
      </div>
    {/if}
  </header>
  {@render children()}
</div>

<style>
  .page {
    width: 100%;
    max-width: var(--size-content-max);
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
     the actions back up beside the title, after the meta line in the source. The title keeps at least 40 % (240 px),
     and a long action row wraps right-aligned instead of squeezing it. */
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
  h1 {
    grid-column: 1;
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
  .sub {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: var(--space-1) var(--space-3);
    min-width: 0;
    color: var(--text-secondary);
  }
  .actions {
    grid-column: 2;
    display: flex;
    align-items: center;
    justify-content: flex-end;
    gap: var(--space-2);
    flex-wrap: wrap;
    min-width: 0;
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
</style>
