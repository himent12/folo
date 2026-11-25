// The shell (sidebar, top bar, tab bar, palette, shortcuts, toasts, run watch) and the screen of
// the current route. Each screen is a chunk of its own (./routes.ts): the shell renders at once, a screen when its
// chunk is there. Moving to a screen whose chunk has not loaded yet keeps the screen on display until it has, then
// swaps in one step, as `{#key}` does; chunks are fetched ahead (a link hovered, focused or touched, the G of a
// shortcut, the palette opening, idle time after startup), so that wait is rarely there at all. The screen kept on
// display keeps its route too (`router.holdWhile`): it never reads the route of the screen to come. A chunk that fails
// to load shows an error with Retry (ScreenError), and the screen once any later try brings it in.
import { computed, createRoot, css, Dynamic, effect, getOwner, Key, onCleanup, onMount, signal, tick, untrack, type Component } from 'fw';
import { hub } from '$lib/api/index';
import { applyTheme } from '$lib/design/applyTheme';
import { retryFailedStyles } from '$lib/failedStyles';
import { prefs } from '$lib/prefs';
import { router, type Route } from '$lib/router';
import CommandPalette from '$lib/shell/CommandPalette';
import ConnectionStatus from '$lib/shell/ConnectionStatus';
import { installEntryMemory } from '$lib/shell/entryMemory';
import GChordIndicator from '$lib/shell/GChordIndicator';
// The extension: runWatch.ts (its logic) differs from this file's name only in case.
import RunWatch from '$lib/shell/RunWatch.tsx';
import { holdRouteFocus, isNewPage } from '$lib/shell/routeFocus';
import { installShortcuts } from '$lib/shell/shortcuts';
import Sidebar from '$lib/shell/Sidebar';
import SidebarLive from '$lib/shell/SidebarLive';
import TabBar from '$lib/shell/TabBar';
import Topbar from '$lib/shell/Topbar';
import Toaster from '$lib/ui/Toaster';
import { toast } from '$lib/ui/toast';
import { screens as appScreens, type Screens } from './routes';
import NotFound from './routes/NotFound';
import ScreenError, { ScreenCrash } from './ScreenError';

type Screen = { path: string | null; comp: Component<Record<string, never>> | null };
type ShortcutsDialogModule = { default: Component<{ open?: boolean }> };

const NONE: Screen = { path: null, comp: null };
/** After the first screen has shown, the other chunks load gently in staggered idle batches (once). */
const IDLE_PRELOAD_MS = 1000;

export interface AppProps {
  /** The screens' chunks (tests pass their own); default: the app's (./routes.ts). */
  screens?: Screens;
  /** Lazy loader for ShortcutsDialog (tests may override); default: dynamic import. */
  loadShortcuts?: () => Promise<ShortcutsDialogModule>;
}

export default function App(props: AppProps) {
  const screens = untrack(() => props.screens) ?? appScreens;
  const paletteOpen = signal(false);
  const shortcutsOpen = signal(false);
  const shortcutsComp = signal<Component<{ open?: boolean }> | null>(null);
  let mounted = true;
  onCleanup(() => {
    mounted = false;
  });

  // The theme (design/THEMES.md): `data-theme` the mode, `data-theme-id` the built-in palette's block in tokens.css,
  // and a custom theme's own colours as inline properties over it, the last theme's cleared first.
  effect(() => applyTheme(document.documentElement, prefs.activeTheme));

  onMount(() => {
    void hub.connect();
    const forget = installEntryMemory(() => screen.peek().path);
    const uninstall = installShortcuts({
      palette: () => {
        const opening = !paletteOpen.peek();
        if (opening && shortcutsOpen.peek() && !shortcutsComp.peek()) shortcutsOpen.set(false);
        paletteOpen.set(opening);
      },
      shortcuts: () => shortcutsOpen.set(true),
      // Only preload on a real G-chord (guarded in installShortcuts against typing, modifiers, open dialogs, and singleKeys off).
      gChord: () => screens.preloadAll(),
      // Escape cancels a pending lazy ShortcutsDialog load instead of navigating back on a detail page.
      escape: () => {
        if (shortcutsOpen.peek() && !shortcutsComp.peek()) {
          shortcutsOpen.set(false);
          return true;
        }
        return false;
      },
    });
    // Fetch a screen's chunk ahead: a link to it hovered, focused or touched.
    const intent = (e: Event) => screens.preloadHref((e.target as Element | null)?.closest?.('a[href^="#/"]')?.getAttribute('href'));
    document.addEventListener('pointerover', intent, { passive: true });
    document.addEventListener('focusin', intent);
    document.addEventListener('touchstart', intent, { passive: true });
    return () => {
      forget();
      uninstall();
      document.removeEventListener('pointerover', intent);
      document.removeEventListener('focusin', intent);
      document.removeEventListener('touchstart', intent);
    };
  });

  // Lazy-load ShortcutsDialog on first open so it stays out of the initial entry bundle.
  let shortcutsLoad: Promise<void> | null = null;
  effect(() => {
    if (!shortcutsOpen.get() || shortcutsComp.peek() || shortcutsLoad) return;
    const load = untrack(() => props.loadShortcuts) ?? (() => (void retryFailedStyles(), import('$lib/shell/ShortcutsDialog')));
    shortcutsLoad = load().then(
      (m) => {
        shortcutsLoad = null;
        if (!mounted) return;
        toast.clear('shortcuts');
        shortcutsComp.set(() => m.default);
      },
      (e: unknown) => {
        shortcutsLoad = null;
        if (!mounted || !shortcutsOpen.peek()) return;
        shortcutsOpen.set(false);
        toast.error(e, 'Could not load keyboard shortcuts', { key: 'shortcuts' });
      },
    );
  });

  // The palette may lead anywhere.
  effect(() => {
    if (paletteOpen.get()) screens.preloadAll();
  });

  /** Whether the URL's route `to` waits for its screen's chunk (neither loaded nor failed yet). */
  const waiting = (to: Route) => to.name !== 'not-found' && screens.screenOf(to.name) === undefined && !screens.failure(to.name);
  /** A route whose chunk could not be loaded: the error, with Retry. */
  const failed: Component<Record<string, never>> = () => <ScreenError screens={screens} />;
  /**
   * The screen, or the error it threw while it was created (ScreenCrash), so `main` never keeps the one before; what it
   * built until then is disposed, so none of its effects run.
   */
  const show = (Comp: Screen['comp']) => {
    if (!Comp) return undefined;
    // In a scope of its own: what the screen built before it threw (a child that polls, a listener) is torn down with it.
    let drop = () => {};
    try {
      return createRoot((dispose) => ((drop = dispose), <Comp />), getOwner());
    } catch (error) {
      drop();
      console.error(error);
      return <ScreenCrash error={error} />;
    }
  };

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

  /** Whether the first screen has been created (so startup idle preloading can begin once). */
  const started = computed(() => screen.get().path !== null);

  // Once the first screen is up, stagger the remaining chunks in small idle batches (skipped under Save-Data).
  effect(() => {
    if (!started.get()) return;
    if ((navigator as { connection?: { saveData?: boolean } }).connection?.saveData) return;
    let stop: (() => void) | undefined;
    const t = setTimeout(() => {
      stop = screens.preloadStaggered();
    }, IDLE_PRELOAD_MS);
    return () => {
      clearTimeout(t);
      stop?.();
    };
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
            <Key when={screen.get()}>{() => show(untrack(() => screen.get().comp))}</Key>
          </main>
        </div>
        <TabBar />
      </div>
      <CommandPalette bind:open={paletteOpen} onshortcuts={() => shortcutsOpen.set(true)} />
      {shortcutsComp.get() ? <Dynamic component={shortcutsComp.get()!} bind:open={shortcutsOpen} /> : null}
      <GChordIndicator />
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
