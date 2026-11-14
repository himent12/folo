import ChevronLeft from '$icons/chevron-left';
import ChevronRight from '$icons/chevron-right';
import Monitor from '$icons/monitor';
import Moon from '$icons/moon';
import Search from '$icons/search';
import Settings from '$icons/settings';
import Sun from '$icons/sun';
import { computed, css, Index } from 'fw';
import { prefs } from '$lib/prefs';
import { href } from '$lib/router';
import Button from '$lib/ui/Button';
import Kbd from '$lib/ui/Kbd';
import Logo from './Logo';
import OfflineNotice from './OfflineNotice';
import { page } from './crumbs';

export default function Topbar(props: { onsearch: () => void }) {
  /** On phones the breadcrumb collapses to its last item; a back button leads to the parent. */
  const parent = computed(() => (page.crumbs.length >= 2 ? page.crumbs[page.crumbs.length - 2] : undefined));
  const themeIcon = computed(() => (prefs.theme === 'dark' ? Moon : prefs.theme === 'light' ? Sun : Monitor));
  const themeLabel = computed(() => `Theme: ${prefs.theme} (click to change)`);

  return (
    <header class="topbar" data-tip-chrome>
      {parent.get() ? (
        <a href={href(parent.get()?.path ?? '/')} class="mobile-back" aria-label={`Back to ${parent.get()?.label}`}>
          <ChevronLeft size={20} strokeWidth={1.75} />
        </a>
      ) : (
        <a href={href('/')} class="mobile-logo" aria-label="FOLO Studio home">
          <Logo wordmark={false} />
        </a>
      )}
      {/* Breadcrumbs on a nested page only (design/UNIFIED.md §1): a top-level page's own title says where you are, so
          the bar's left stays empty there. Phones: the bar carries the page title (the H1 is visually hidden). */}
      <ol class="crumbs" class:top={!parent.get()} aria-label="Breadcrumb">
        <Index each={page.crumbs}>
          {(c, i) => {
            const last = () => i === page.crumbs.length - 1;
            return (
              <>
                <li class:last={last()}>
                  {c().path && !last() ? <a href={href(c().path ?? '')}>{c().label}</a> : <span aria-current={last() ? 'page' : undefined}>{c().label}</span>}
                </li>
                {!last() ? (
                  <li class="sep" aria-hidden="true">
                    <ChevronRight size={12} strokeWidth={1.75} />
                  </li>
                ) : null}
              </>
            );
          }}
        </Index>
      </ol>

      <OfflineNotice />

      <div class="actions">
        <button class="search" onclick={() => props.onsearch()} aria-label="Search and commands" aria-keyshortcuts="Control+K Meta+K">
          <Search size={14} strokeWidth={1.75} aria-hidden="true" />
          <span class="search-label">Search or jump to…</span>
          <span class="search-keys">
            <Kbd keys="Mod+K" subtle />
          </span>
        </button>
        <Button variant="ghost" icon={themeIcon.get()} aria-label={themeLabel.get()} title={themeLabel.get()} onclick={() => prefs.cycleTheme()} />
        <span class="mobile-only">
          <Button variant="ghost" icon={Settings} href={href('/settings')} aria-label="Settings" />
        </span>
      </div>
    </header>
  );
}

css`
  .topbar {
    position: sticky;
    top: 0;
    z-index: var(--z-sticky);
    height: var(--size-topbar);
    display: flex;
    align-items: center;
    gap: var(--space-4);
    padding: 0 var(--space-4) 0 var(--space-6);
    background: color-mix(in srgb, var(--bg-app) 86%, transparent);
    backdrop-filter: blur(12px) saturate(1.4);
    border-bottom: 1px solid var(--line-subtle);
  }
  .mobile-logo,
  .mobile-back,
  .mobile-only {
    display: none;
  }
  /* Crumbs as Desktop's: 13 px, the parents in text.tertiary, the current page text.primary 500, 12 px chevrons. */
  .crumbs {
    display: flex;
    align-items: center;
    gap: var(--space-1-5);
    min-width: 0;
    flex: 1;
  }
  .crumbs li {
    display: flex;
    align-items: center;
    min-width: 0;
    color: var(--text-tertiary);
  }
  .crumbs li a {
    color: var(--text-tertiary);
    border-radius: var(--radius-sm);
    padding: 2px 4px;
    margin: -2px -4px;
    transition: color var(--dur-fast) var(--ease-standard);
  }
  .crumbs li a:hover {
    color: var(--text-primary);
  }
  .crumbs li.last {
    color: var(--text-primary);
    font-weight: var(--weight-medium);
  }
  /* A top-level page: no trail from 768 px up (the list keeps its place, so the actions stay at the right). */
  @media (min-width: 768px) {
    .crumbs.top li {
      display: none;
    }
  }
  .crumbs li.last span {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .sep {
    flex-shrink: 0;
    color: var(--text-disabled);
  }
  .crumbs li:not(.last):not(.sep) {
    flex-shrink: 0;
  }
  .actions {
    display: flex;
    align-items: center;
    gap: var(--space-2);
  }
  .search {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    height: var(--control-md);
    width: 280px;
    padding: 0 var(--space-1) 0 var(--space-2);
    border-radius: var(--radius-md);
    border: 1px solid var(--line-default);
    background: var(--bg-raised);
    color: var(--text-tertiary);
    font-size: var(--type-sm-size);
    transition:
      border-color var(--dur-fast) var(--ease-standard),
      color var(--dur-fast) var(--ease-standard);
  }
  .search:hover {
    border-color: var(--line-strong);
    color: var(--text-secondary);
  }
  .search-label {
    flex: 1;
    text-align: left;
  }

  @media (max-width: 1023px) {
    .search {
      width: var(--control-md);
      padding: 0;
      justify-content: center;
      border-color: transparent;
      background: none;
    }
    .search:hover {
      background: var(--bg-hover);
    }
    .search-label,
    .search-keys {
      display: none;
    }
  }
  /* Touch, where it is an icon: a 44 px target like the theme and settings buttons beside it (shell F17). */
  @media (max-width: 1023px) and (pointer: coarse) {
    .search {
      width: 44px;
      height: 44px;
    }
  }
  @media (max-width: 767px) {
    .topbar {
      padding: 0 var(--space-2) 0 var(--space-4);
      gap: var(--space-3);
    }
    .mobile-logo,
    .mobile-only {
      display: inline-flex;
    }
    .mobile-back {
      display: grid;
      place-items: center;
      width: 44px;
      height: 44px;
      margin-left: calc(-1 * var(--space-3));
      margin-right: calc(-1 * var(--space-2));
      border-radius: var(--radius-md);
      color: var(--text-secondary);
    }
    .mobile-back:hover {
      background: var(--bg-hover);
      color: var(--text-primary);
    }
    /* The bar is the page title: the current page only, its parent is the back button. */
    .crumbs li {
      display: none;
    }
    .crumbs li.last {
      display: flex;
      font-size: var(--type-lg-size);
      line-height: 20px;
      font-weight: var(--weight-semibold);
      letter-spacing: -0.005em;
    }
  }
`;
