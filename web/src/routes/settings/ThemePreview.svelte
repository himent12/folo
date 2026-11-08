<script lang="ts">
  /**
   * A tiny Studio window drawn in one theme's colours, whatever the current theme is. The app's CSS variables only
   * describe the active theme, so the colours come straight from the design tokens (design/tokens.json). Drawn in the
   * theme's ink only: a preview shows surfaces and text, not data colours.
   */
  import tokens from '../../../../design/tokens.json';

  let { theme }: { theme: 'dark' | 'light' } = $props();
  const c = $derived(tokens.color[theme]);
</script>

<span
  class="win"
  aria-hidden="true"
  style:--app={c.bg.app}
  style:--panel={c.bg.panel}
  style:--line={c.line.default}
  style:--t1={c.text.primary}
  style:--t3={c.text.tertiary}
>
  <span class="side">
    <span class="logo"></span>
    <span class="nav on"></span>
    <span class="nav"></span>
    <span class="nav"></span>
  </span>
  <span class="main">
    <span class="title"></span>
    <span class="card">
      <svg viewBox="0 0 60 24" preserveAspectRatio="none">
        <polyline points="0,22 10,15 20,11 30,8 40,6 50,5 60,4" fill="none" stroke="var(--t1)" stroke-width="1.5" vector-effect="non-scaling-stroke" />
        <polyline points="0,23 10,19 20,16 30,14 40,13 50,12 60,11" fill="none" stroke="var(--t3)" stroke-width="1.5" vector-effect="non-scaling-stroke" />
      </svg>
    </span>
  </span>
</span>

<style>
  .win {
    display: flex;
    width: 100%;
    height: 100%;
    background: var(--app);
    overflow: hidden;
  }
  .side {
    display: flex;
    flex-direction: column;
    gap: 4px;
    width: 30%;
    padding: 7px 6px;
    background: var(--panel);
    border-right: 1px solid var(--line);
  }
  .logo {
    width: 10px;
    height: 10px;
    margin-bottom: 3px;
    border-radius: 2px;
    border: 1.5px solid var(--t1);
  }
  .nav {
    height: 4px;
    width: 80%;
    border-radius: 2px;
    background: var(--t3);
    opacity: 0.5;
  }
  .nav.on {
    background: var(--t1);
    opacity: 0.9;
  }
  .main {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 6px;
    padding: 8px;
  }
  .title {
    width: 40%;
    height: 5px;
    border-radius: 2px;
    background: var(--t1);
  }
  .card {
    flex: 1;
    display: flex;
    padding: 6px;
    border-radius: 3px;
    background: var(--panel);
    border: 1px solid var(--line);
  }
  svg {
    width: 100%;
    height: 100%;
    max-width: none;
  }
</style>
