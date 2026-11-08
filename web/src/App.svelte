<script lang="ts">
  import { router } from '$lib/router.svelte';
  import { prefs } from '$lib/prefs.svelte';
  import Sidebar from '$lib/shell/Sidebar.svelte';
  import Topbar from '$lib/shell/Topbar.svelte';
  import TabBar from '$lib/shell/TabBar.svelte';
  import Placeholder from './routes/Placeholder.svelte';
  import NotFound from './routes/NotFound.svelte';

  const titles: Record<string, string> = {
    overview: 'Overview',
    datasets: 'Datasets',
    dataset: 'Dataset',
    runs: 'Runs',
    'run-new': 'New run',
    'run-compare': 'Compare runs',
    run: 'Run',
    models: 'Models',
    model: 'Model',
    playground: 'Playground',
    settings: 'Settings',
  };

  $effect(() => {
    document.documentElement.dataset.theme = prefs.resolvedTheme;
  });

  let paletteOpen = $state(false);
</script>

<div class="app">
  <Sidebar />
  <div class="main">
    <Topbar onsearch={() => (paletteOpen = true)} />
    <main id="content">
      {#key router.route.path}
        {#if router.route.name === 'not-found'}
          <NotFound />
        {:else}
          <Placeholder title={titles[router.route.name] ?? ''} />
        {/if}
      {/key}
    </main>
  </div>
  <TabBar />
</div>

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
