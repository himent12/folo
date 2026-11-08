<script lang="ts">
  import { onMount, type Component } from 'svelte';
  import { hub } from '$lib/api/index.svelte';
  import { prefs } from '$lib/prefs.svelte';
  import { router, type RouteName } from '$lib/router.svelte';
  import CommandPalette from '$lib/shell/CommandPalette.svelte';
  import ConnectionStatus from '$lib/shell/ConnectionStatus.svelte';
  import ShortcutsDialog from '$lib/shell/ShortcutsDialog.svelte';
  import Sidebar from '$lib/shell/Sidebar.svelte';
  import SidebarLive from '$lib/shell/SidebarLive.svelte';
  import TabBar from '$lib/shell/TabBar.svelte';
  import Topbar from '$lib/shell/Topbar.svelte';
  import { installShortcuts } from '$lib/shell/shortcuts';
  import Toaster from '$lib/ui/Toaster.svelte';
  import Dataset from './routes/Dataset.svelte';
  import Datasets from './routes/Datasets.svelte';
  import Model from './routes/Model.svelte';
  import Models from './routes/Models.svelte';
  import NotFound from './routes/NotFound.svelte';
  import Overview from './routes/Overview.svelte';
  import Run from './routes/Run.svelte';
  import Runs from './routes/Runs.svelte';
  import RunNew from './routes/RunNew.svelte';
  import RunCompare from './routes/RunCompare.svelte';
  import Playground from './routes/Playground.svelte';
  import Settings from './routes/Settings.svelte';

  const screens: Record<Exclude<RouteName, 'not-found'>, Component> = {
    overview: Overview,
    runs: Runs,
    run: Run,
    'run-new': RunNew,
    'run-compare': RunCompare,
    datasets: Datasets,
    dataset: Dataset,
    models: Models,
    model: Model,
    playground: Playground,
    settings: Settings,
  };

  let paletteOpen = $state(false);
  let shortcutsOpen = $state(false);

  $effect(() => {
    document.documentElement.dataset.theme = prefs.resolvedTheme;
  });

  onMount(() => {
    void hub.connect();
    return installShortcuts({
      palette: () => (paletteOpen = !paletteOpen),
      shortcuts: () => (shortcutsOpen = true),
    });
  });

  const Screen = $derived(router.route.name === 'not-found' ? null : screens[router.route.name]);
</script>

<a class="skip" href="#content">Skip to content</a>
<div class="app">
  <Sidebar>
    {#snippet live(collapsed)}<SidebarLive {collapsed} />{/snippet}
    {#snippet footer(collapsed)}<ConnectionStatus {collapsed} />{/snippet}
  </Sidebar>
  <div class="main">
    <Topbar onsearch={() => (paletteOpen = true)} />
    <main id="content" tabindex="-1">
      {#key router.route.path}
        {#if Screen}
          <Screen />
        {:else}
          <NotFound />
        {/if}
      {/key}
    </main>
  </div>
  <TabBar />
</div>
<CommandPalette bind:open={paletteOpen} onshortcuts={() => (shortcutsOpen = true)} />
<ShortcutsDialog bind:open={shortcutsOpen} />
<Toaster />

<style>
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
    padding: 6px 10px;
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
    .app :global(.sidebar) {
      display: none;
    }
    main {
      padding-bottom: calc(64px + env(safe-area-inset-bottom));
    }
  }
</style>
