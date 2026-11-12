// The shell (sidebar, top bar, tab bar, palette, shortcuts, toasts, run watch) and the screen of
// the current route. Each screen is a chunk of its own (./routes.ts): the shell renders at once, a screen when its
// chunk is there. Moving to a screen whose chunk has not loaded yet keeps the screen on display until it has, then
// swaps in one step, as `{#key}` does; chunks are fetched ahead (a link hovered, focused or touched, the G of a
// shortcut, the palette opening, idle time after startup), so that wait is rarely there at all. The screen kept on
// display keeps its route too (`router.holdWhile`): it never reads the route of the screen to come. A chunk that fails
// to load shows an error with Retry (ScreenError), and the screen once any later try brings it in.
import { computed, css, Dynamic, effect, Key, onCleanup, onMount, signal, tick, untrack, type Component } from 'fw';
import { hub } from '$lib/api/index';
import { prefs } from '$lib/prefs';
import { router, type Route } from '$lib/router';
import CommandPalette from '$lib/shell/CommandPalette';
import ConnectionStatus from '$lib/shell/ConnectionStatus';
import { installEntryMemory } from '$lib/shell/entryMemory';
// The extension: runWatch.ts (its logic) differs from this file's name only in case.
import RunWatch from '$lib/shell/RunWatch.tsx';
import { holdRouteFocus, isNewPage } from '$lib/shell/routeFocus';
import { installShortcuts } from '$lib/shell/shortcuts';
import ShortcutsDialog from '$lib/shell/ShortcutsDialog';
import Sidebar from '$lib/shell/Sidebar';
import SidebarLive from '$lib/shell/SidebarLive';
import TabBar from '$lib/shell/TabBar';
import Topbar from '$lib/shell/Topbar';
import Toaster from '$lib/ui/Toaster';
import { screens as appScreens, type Screens } from './routes';
import NotFound from './routes/NotFound';
import ScreenError from './ScreenError';

type Screen = { path: string | null; comp: Component<Record<string, never>> | null };

const NONE: Screen = { path: null, comp: null };
/** After the first screen has shown, the other chunks load in idle time (once). */
const IDLE_PRELOAD_MS = 1000;


export interface AppProps {
  /** The screens' chunks (tests pass their own); default: the app's (./routes.ts). */
  screens?: Screens;
}

export default function App(props: AppProps) {
  const screens = untrack(() => props.screens) ?? appScreens;
  const paletteOpen = signal(false);
  const shortcutsOpen = signal(false);

  effect(() => {
    document.documentElement.dataset.theme = prefs.resolvedTheme;
  });

  onMount(() => {
    void hub.connect();
    const forget = installEntryMemory(() => screen.peek().path);
    const uninstall = installShortcuts({
      palette: () => paletteOpen.set(!paletteOpen.peek()),
      shortcuts: () => shortcutsOpen.set(true),
    });
    // Fetch a screen's chunk ahead: a link to it hovered, focused or touched; every screen on a "go to" G.
    const intent = (e: Event) => screens.preloadHref((e.target as Element | null)?.closest?.('a[href^="#/"]')?.getAttribute('href'));
    const onkey = (e: KeyboardEvent) => {
      if (e.key === 'g' || e.key === 'G') screens.preloadAll();
    };
    document.addEventListener('pointerover', intent, { passive: true });
    document.addEventListener('focusin', intent);
    document.addEventListener('touchstart', intent, { passive: true });
    document.addEventListener('keydown', onkey);
    return () => {
      forget();
      uninstall();
      document.removeEventListener('pointerover', intent);
      document.removeEventListener('focusin', intent);
      document.removeEventListener('touchstart', intent);
      document.removeEventListener('keydown', onkey);
    };
  });

  // The palette may lead anywhere.
  effect(() => {
    if (paletteOpen.get()) screens.preloadAll();
  });

  /** Whether the URL's route `to` waits for its screen's chunk (neither loaded nor failed yet). */
  const waiting = (to: Route) => to.name !== 'not-found' && screens.screenOf(to.name) === undefined && !screens.failure(to.name);
  /** A route whose chunk could not be loaded: the error, with Retry. */
  const failed: Component<Record<string, never>> = () => <ScreenError screens={screens} />;

  /** What `main` shows: the screen of the route on display once its chunk is there, else the one shown until then. */
  let last: Screen = NONE;
  // Until then the route on display (`router.route`, what screens read) stays the one of the screen on display; with
  // none on display yet (startup), it follows the URL.
  onCleanup(router.holdWhile((to) => last.path !== null && waiting(to)));
  const screen = computed(
    () => {
      // The URL's route: its chunk is fetched (a failed one is tried again by Retry, or by any other request).
      const to = router.location;
      if (to.name !== 'not-found' && waiting(to)) void screens.preload(to.name).catch(() => {});
      const r = router.route;
      if (r.name === 'not-found') return (last = { path: r.path, comp: NotFound });
      const comp = screens.screenOf(r.name);
      if (comp !== undefined) return (last = { path: r.path, comp });
      if (screens.failure(r.name)) return (last = { path: r.path, comp: failed });
      return last;
    },
    { equals: (a, b) => a.path === b.path && a.comp === b.comp },
  );

  /**
   * Whether the first screen has been created. The shortcuts dialog (it takes a `createUniqueId`) waits for it: the
   * legacy app creates the screen before it, in document order, so every id a screen and the dialog take comes in the
   * same sequence here, whenever the screen's chunk arrives. The dialog is closed until asked for, so nothing shows late.
   */
  const started = computed(() => screen.get().path !== null);

  // Once the first screen is up, the others load in idle time.
  effect(() => {
    if (screen.get().path === null) return;
    const t = setTimeout(() => {
      const idle = (globalThis as { requestIdleCallback?: (f: () => void, o?: { timeout: number }) => number }).requestIdleCallback;
      if (idle) idle(() => screens.preloadAll(), { timeout: 5000 });
      else screens.preloadAll();
    }, IDLE_PRELOAD_MS);
    return () => clearTimeout(t);
  });

  // A new page takes focus (its heading) once it has rendered; a query change on the same page leaves focus alone.
  let mainEl!: HTMLElement;
  let shownPath = router.route.path;
  let shownComp: Screen['comp'] | null = null;
  let releaseFocus: (() => void) | null = null;
  effect(() => {
    const { path, comp } = screen.get();
    // The screen that replaces its own load error (Retry) takes focus too when the error had it: the button it was on
    // is gone, so focus would fall to the body.
    // Checked again in the frame: focus the user moves out of `main` meanwhile stays where they put it.
    const focusFree = () => {
      const at = document.activeElement;
      return at === null || at === document.body || !!mainEl?.contains(at);
    };
    const newPage = path !== null && isNewPage(shownPath, path);
    const recovered = shownComp === failed && comp !== failed && focusFree();
    shownComp = comp;
    if (path === null || (!newPage && !recovered)) return;
    shownPath = path;
    releaseFocus?.();
    releaseFocus = null;
    void tick().then(() =>
      requestAnimationFrame(() => {
        // `mainEl` is gone if the app was torn down before the frame.
        if (mainEl?.isConnected && router.route.path === path && (newPage || focusFree())) releaseFocus = holdRouteFocus(mainEl);
      }),
    );
  });

  return (
    <>
      <a class="skip" href="#content">
        Skip to content
      </a>
      <div class="app">
        <Sidebar live={(collapsed) => <SidebarLive collapsed={collapsed()} />} footer={(collapsed) => <ConnectionStatus collapsed={collapsed()} />} />
        <div class="main">
          <Topbar onsearch={() => paletteOpen.set(true)} />
          <main id="content" tabindex={-1} ref={(e) => (mainEl = e)}>
            <Key when={screen.get()}>{() => <Dynamic component={untrack(() => screen.get().comp)} />}</Key>
          </main>
        </div>
        <TabBar />
      </div>
      <CommandPalette bind:open={paletteOpen} onshortcuts={() => shortcutsOpen.set(true)} />
      {started.get() ? <ShortcutsDialog bind:open={shortcutsOpen} /> : null}
      <Toaster />
      <RunWatch />
    </>
  );
}

css`
  .app {
    display: flex;
    min-height: 100dvh;
  }
  .main {
    flex: 1;
    min-width: 0;
    display: flex;
    flex-direction: column;
  }
  main {
    flex: 1;
    min-width: 0;
    outline: none;
  }
  .skip {
    position: fixed;
    top: 8px;
    left: 8px;
    z-index: var(--z-toast);
    padding: 6px 12px;
    border-radius: var(--radius-md);
    background: var(--accent-solid);
    color: var(--accent-on);
    font-weight: var(--weight-medium);
    transform: translateY(-200%);
  }
  .skip:focus-visible {
    transform: none;
  }
  @media (max-width: 767px) {
    /* The sidebar's own column only (pages have rails of their own, which phones show as sections). */
    .app :global(.shell-rail) {
      display: none;
    }
    /* Clear of the tab bar (56 px + 8), or of nothing but 8 px while a page's dock replaces it (shell/dock.ts). */
    main {
      padding-bottom: calc(var(--tabbar-h) + var(--space-2) + env(safe-area-inset-bottom));
    }
  }
`;
