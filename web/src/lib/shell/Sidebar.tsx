import PanelLeftClose from '$icons/panel-left-close';
import PanelLeftOpen from '$icons/panel-left-open';
import { computed, css, Dynamic, effect, For, signal, untrack } from 'fw';
import { router, section, href } from '$lib/router';
import Kbd from '$lib/ui/Kbd';
import { keyPrefs } from '$lib/ui/keyPrefs';
import { reducedMotion } from '$lib/ui/motion';
import { tooltip } from '$lib/ui/tooltip';
import Logo from './Logo';
import { mainNav, settingsNav, type NavItem } from './nav';
import { activeRuns, runsBadgeText } from './runList';
import { sidebar } from './sidebarState';

export interface SidebarProps {
  /** The rail's footer (the connection status), given whether the rail is collapsed. */
  footer?: (collapsed: () => boolean) => JSX.Element;
  /** The live run's row (pinned above Settings), given whether the rail is collapsed. */
  live?: (collapsed: () => boolean) => JSX.Element;
}

export default function Sidebar(props: SidebarProps) {
  const current = computed(() => section(router.route.name));
  // Collapse and expand move the page once: the rail takes its new width at once (so charts re-measure one time, not
  // every frame of a width transition), and the navigation clips open or closed over the page (a clip-path animation,
  // composited). Closing keeps the wide look until its clip has run, then collapses.
  const slim = computed(() => sidebar.collapsed);
  const anim = signal<'open' | 'close' | null>(null);
  let seen: boolean | undefined;
  effect(() => {
    const c = slim.get();
    if (seen !== undefined && c !== seen) anim.set(reducedMotion() ? null : c ? 'close' : 'open');
    seen = c;
  });
  const collapsed = computed(() => slim.get() && anim.get() !== 'close');
  const isCollapsed = () => collapsed.get();
  // A snippet is called once (again only if the prop itself changes): what it reads is read where it is used.
  const snippet = (f: ((collapsed: () => boolean) => JSX.Element) | undefined) => (f ? untrack(() => f(isCollapsed)) : null);

  // The Runs badge (design/UNIFIED.md §1): how many runs train or wait, at the row's right end; the icon rail has no
  // room for it, so there a lime dot on the icon says one is training and the tooltip says the counts.
  const badge = computed(() => {
    const a = activeRuns();
    return a.training + a.queued;
  });
  const item = (n: () => NavItem) => {
    const active = () => current.get() === n().id;
    const runs = () => n().id === 'runs';
    const named = () => (runs() && badge.get() > 0 ? `${n().label}, ${runsBadgeText()}` : n().label);
    return (
      <li>
        <a
          href={href(n().path)}
          class="item"
          class:active={active()}
          aria-current={active() ? 'page' : undefined}
          aria-label={collapsed.get() || (runs() && badge.get() > 0) ? named() : undefined}
          use:tooltip={collapsed.get() ? { text: named(), keys: keyPrefs.singleKeys ? n().keys : undefined, side: 'right' } : null}
        >
          <span class="icon">
            <Dynamic component={n().icon} size={16} strokeWidth={1.75} aria-hidden="true" />
            {runs() && collapsed.get() && activeRuns().live ? <span class="live-dot" aria-hidden="true"></span> : null}
          </span>
          <span class="label">{n().label}</span>
          {keyPrefs.singleKeys ? (
            <span class="keys">
              <Kbd keys={n().keys} subtle />
            </span>
          ) : null}
          {runs() && badge.get() > 0 ? (
            <span class="badge num" class:held={activeRuns().held} aria-hidden="true">
              {badge.get()}
            </span>
          ) : null}
        </a>
      </li>
    );
  };

  // The rail is the shell's column: it runs the page's full height (its fill and edge reach the bottom of a long page,
  // full-page captures included), and the navigation sticks to the viewport inside it.
  return (
    <div
      class="shell-rail"
      class:slim={slim.get()}
      class:collapsed={collapsed.get()}
      class:anim-open={anim.get() === 'open'}
      class:anim-close={anim.get() === 'close'}
    >
      <nav class="sidebar" aria-label="Main" onanimationend={(e) => e.target === e.currentTarget && anim.set(null)}>
        <div class="head">
          <a href={href('/')} class="brand" aria-label="FOLO Studio home">
            <Logo wordmark={!collapsed.get()} />
          </a>
        </div>

        <ul class="nav">
          <For each={mainNav} key={(n) => n.id}>
            {(n) => item(n)}
          </For>
        </ul>

        <div class="spacer"></div>

        {/* The live run: one row pinned above Settings. */}
        {props.live ? <div class="live">{snippet(props.live)}</div> : null}

        <ul class="nav">{item(() => settingsNav)}</ul>
        <div class="foot">
          {props.footer ? snippet(props.footer) : null}
          <button
            class="collapse"
            onclick={() => sidebar.toggle()}
            aria-label={collapsed.get() ? 'Expand sidebar' : 'Collapse sidebar'}
            use:tooltip={{
              text: collapsed.get() ? 'Expand sidebar' : 'Collapse sidebar',
              keys: keyPrefs.singleKeys ? '[' : undefined,
              side: collapsed.get() ? 'right' : 'top',
            }}
          >
            {collapsed.get() ? <PanelLeftOpen size={16} strokeWidth={1.75} /> : <PanelLeftClose size={16} strokeWidth={1.75} />}
          </button>
        </div>
      </nav>
    </div>
  );
}

css`
  .shell-rail {
    width: var(--size-sidebar);
    flex-shrink: 0;
    background: var(--bg-app);
    border-right: 1px solid var(--line-subtle);
    z-index: var(--z-sidebar);
  }
  .shell-rail.slim {
    width: var(--size-sidebar-collapsed);
  }
  /* Opening: the wide navigation is revealed from the icon column's edge. Closing: it stays wide over the page while its
     clip closes to the icon column, then takes the collapsed look. */
  .anim-open .sidebar {
    animation: rail-open var(--dur-slow) var(--ease-standard);
  }
  .anim-close .sidebar {
    width: var(--size-sidebar);
    background: var(--bg-app);
    border-right: 1px solid var(--line-subtle);
    animation: rail-close var(--dur-slow) var(--ease-standard) forwards;
  }
  @keyframes rail-open {
    from {
      clip-path: inset(0 calc(100% - var(--size-sidebar-collapsed)) 0 0);
    }
    to {
      clip-path: inset(0);
    }
  }
  @keyframes rail-close {
    from {
      clip-path: inset(0);
    }
    to {
      clip-path: inset(0 calc(100% - var(--size-sidebar-collapsed)) 0 0);
    }
  }
  .sidebar {
    position: sticky;
    top: 0;
    height: 100dvh;
    display: flex;
    flex-direction: column;
    padding: 0 var(--space-2) var(--space-2);
  }
  .head {
    height: var(--size-topbar);
    display: flex;
    align-items: center;
    padding: 0 var(--space-2);
    margin-bottom: var(--space-2);
  }
  .collapsed .head {
    padding: 0 0 0 8px;
  }
  .brand {
    display: inline-flex;
    padding: var(--space-1);
    margin: calc(-1 * var(--space-1));
  }
  .nav {
    display: grid;
    gap: 1px;
  }
  /* Nav rows (design/UNIFIED.md §1): 32 high, 14 px regular in text.secondary; the current row text.primary 500 on
     bg.active. */
  .item {
    position: relative;
    display: flex;
    align-items: center;
    gap: var(--space-2);
    height: var(--control-lg);
    padding: 0 var(--space-2);
    border-radius: var(--radius-md);
    color: var(--text-secondary);
    font-size: var(--type-md-size);
    line-height: var(--type-md-line);
    letter-spacing: var(--type-md-tracking);
    white-space: nowrap;
    overflow: hidden;
    transition:
      background-color var(--dur-fast) var(--ease-standard),
      color var(--dur-fast) var(--ease-standard);
  }
  .item:hover {
    background: var(--bg-hover);
    color: var(--text-primary);
  }
  .item.active {
    background: var(--bg-active);
    color: var(--text-primary);
    font-weight: var(--weight-medium);
  }
  .icon {
    position: relative;
    display: inline-flex;
    flex-shrink: 0;
  }
  /* The icon rail's live mark: a lime dot on the Runs icon while a run trains. */
  .live-dot {
    position: absolute;
    top: -2px;
    right: -4px;
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--accent-line);
    box-shadow: 0 0 0 2px var(--bg-app);
  }
  /* The Runs badge: lime text on accent.muted, 20 high; neutral while its counts are the last known (the list stopped
     refreshing), as the live row's rail is. */
  .badge {
    display: inline-grid;
    place-items: center;
    flex-shrink: 0;
    min-width: var(--control-xs);
    height: var(--control-xs);
    padding: 0 var(--space-1-5);
    border-radius: var(--radius-full);
    background: var(--accent-muted);
    color: var(--accent-text);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    font-weight: var(--weight-medium);
    letter-spacing: 0;
  }
  .label {
    flex: 1;
    min-width: 0;
  }
  .keys {
    opacity: 0;
    transition: opacity var(--dur-fast) var(--ease-standard);
  }
  .item:hover .keys {
    opacity: 1;
  }
  .badge.held {
    background: var(--bg-active);
    color: var(--text-secondary);
  }
  .collapsed .label,
  .collapsed .keys,
  .collapsed .badge {
    display: none;
  }
  .live {
    margin-bottom: 1px;
  }
  .spacer {
    flex: 1;
  }
  .foot {
    display: flex;
    align-items: center;
    gap: var(--space-1);
    margin-top: var(--space-2);
    padding-top: var(--space-2);
    border-top: 1px solid var(--line-subtle);
  }
  .collapsed .foot {
    flex-direction: column;
  }
  .collapse {
    display: grid;
    place-items: center;
    width: var(--control-md);
    height: var(--control-md);
    border-radius: var(--radius-md);
    color: var(--text-tertiary);
    margin-left: auto;
    flex-shrink: 0;
  }
  .collapsed .collapse {
    margin: 0 auto;
  }
  .collapse:hover {
    background: var(--bg-hover);
    color: var(--text-primary);
  }
`;
