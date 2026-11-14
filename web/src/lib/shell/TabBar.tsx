// The phone tab bar: one tab per section of `tabNav` (./nav).
import { computed, css, Dynamic, For } from 'fw';
import { router, section, href } from '$lib/router';
import { tabNav, tabOf } from './nav';
import { activeRuns, runsBadgeText } from './runList';

export default function TabBar() {
  // A section that lives under a tab (Benchmarks under Models) makes that tab current.
  const current = computed(() => tabOf(section(router.route.name)));

  return (
    <nav class="tabbar" aria-label="Main" data-tip-chrome>
      <For each={tabNav} key={(n) => n.id}>
        {(n) => {
          const active = () => current.get() === n().id;
          // The Runs badge has no room here: a lime dot on the icon while a run trains (the sidebar's live row), and
          // the counts in the tab's name (design/UNIFIED.md §1).
          const runs = () => n().id === 'runs' && runsBadgeText() !== '';
          return (
            <a href={href(n().path)} class:active={active()} aria-current={active() ? 'page' : undefined} aria-label={runs() ? `${n().label}, ${runsBadgeText()}` : undefined}>
              <span class="icon">
                <Dynamic component={n().icon} size={20} strokeWidth={1.75} aria-hidden="true" />
                {n().id === 'runs' && activeRuns().live ? <span class="live-dot" aria-hidden="true"></span> : null}
              </span>
              <span>{n().label}</span>
            </a>
          );
        }}
      </For>
    </nav>
  );
}

css`
  .tabbar {
    display: none;
  }
  @media (max-width: 767px) {
    .tabbar {
      position: fixed;
      left: 0;
      right: 0;
      bottom: 0;
      z-index: var(--z-sidebar);
      display: grid;
      grid-template-columns: repeat(5, 1fr);
      height: calc(var(--tabbar-h) + env(safe-area-inset-bottom));
      padding: var(--space-1) var(--space-1) calc(var(--space-1) + env(safe-area-inset-bottom));
      background: color-mix(in srgb, var(--bg-app) 88%, transparent);
      backdrop-filter: blur(16px) saturate(1.4);
      border-top: 1px solid var(--line-subtle);
    }
    /* A page showing a bottom dock (shell/dock.ts) takes the bottom edge: one fixed bar, not two. */
    :global(:root[data-dock]) .tabbar {
      display: none;
    }
    a {
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 2px;
      padding: var(--space-1-5) 0 var(--space-1);
      border-radius: var(--radius-md);
      color: var(--text-tertiary);
      font-size: var(--type-sm-size);
      line-height: var(--type-sm-line);
      font-weight: var(--weight-medium);
    }
    .icon {
      position: relative;
      display: inline-flex;
    }
    .live-dot {
      position: absolute;
      top: -1px;
      right: -4px;
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--accent-line);
      box-shadow: 0 0 0 2px var(--bg-app);
    }
    /* The current section: text.primary on bg.active, like the sidebar; lime is for action, focus and live only. */
    a.active {
      background: var(--bg-active);
      color: var(--text-primary);
    }
  }
`;
