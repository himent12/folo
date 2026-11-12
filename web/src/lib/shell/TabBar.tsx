import { computed, css, Dynamic, For } from 'fw';
import { router, section, href } from '$lib/router';
import { tabNav, tabOf } from './nav';

export default function TabBar() {
  // A section that lives under a tab (Benchmarks under Models) makes that tab current.
  const current = computed(() => tabOf(section(router.route.name)));

  return (
    <nav class="tabbar" aria-label="Main" data-tip-chrome>
      <For each={tabNav} key={(n) => n.id}>
        {(n) => {
          const active = () => current.get() === n().id;
          return (
            <a href={href(n().path)} class:active={active()} aria-current={active() ? 'page' : undefined}>
              <Dynamic component={n().icon} size={20} strokeWidth={1.75} aria-hidden="true" />
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
    /* The current section: text.primary on bg.active, like the sidebar; lime is for action, focus and live only. */
    a.active {
      background: var(--bg-active);
      color: var(--text-primary);
    }
  }
`;
