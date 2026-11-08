<script lang="ts">
  import Check from '@lucide/svelte/icons/check';
  import Monitor from '@lucide/svelte/icons/monitor';
  import Moon from '@lucide/svelte/icons/moon';
  import PlugZap from '@lucide/svelte/icons/plug-zap';
  import Sun from '@lucide/svelte/icons/sun';
  import pkg from '../../package.json';
  import { HttpApi } from '$lib/api/client';
  import { hub } from '$lib/api/index.svelte';
  import { bytes } from '$lib/format';
  import { prefs, type ThemePref, type Units } from '$lib/prefs.svelte';
  import Page from '$lib/shell/Page.svelte';
  import { setCrumbs } from '$lib/shell/crumbs.svelte';
  import Button from '$lib/ui/Button.svelte';
  import Field from '$lib/ui/Field.svelte';
  import Kbd from '$lib/ui/Kbd.svelte';
  import Input from '$lib/ui/Input.svelte';
  import Panel from '$lib/ui/Panel.svelte';
  import Segmented from '$lib/ui/Segmented.svelte';
  import { toast } from '$lib/ui/toast.svelte';
  import ThemePreview from './settings/ThemePreview.svelte';

  setCrumbs({ label: 'Settings' });

  const sections = [
    { id: 'connection', label: 'Connection' },
    { id: 'appearance', label: 'Appearance' },
    { id: 'units', label: 'Units' },
    { id: 'shortcuts', label: 'Shortcuts' },
    { id: 'about', label: 'About' },
  ];

  // Scroll-spy from geometry: the current section is the last one whose top has passed just below the top bar
  // (the first while nothing has; the last once the page is scrolled to the bottom).
  let current = $state<string>('connection');
  $effect(() => {
    let raf = 0;
    const update = () => {
      raf = 0;
      const line = 56 + 24;
      const els = sections.map((x) => document.getElementById(x.id)).filter((e): e is HTMLElement => !!e);
      let id = els[0]?.id ?? 'connection';
      for (const e of els) if (e.getBoundingClientRect().top <= line) id = e.id;
      if (window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 2) id = els[els.length - 1]?.id ?? id;
      current = id;
    };
    const onscroll = () => {
      if (!raf) raf = requestAnimationFrame(update);
    };
    update();
    window.addEventListener('scroll', onscroll, { passive: true });
    window.addEventListener('resize', onscroll);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener('scroll', onscroll);
      window.removeEventListener('resize', onscroll);
    };
  });

  // ---- connection ----
  const origin = typeof location === 'undefined' ? '' : location.origin;
  let url = $state(prefs.serverUrl);
  let testing = $state(false);
  let test = $state<{ ok: true; version: string; mock: boolean; ms: number } | { ok: false; message: string } | null>(null);
  const forced = typeof location !== 'undefined' && new URLSearchParams(location.search).get('mock') === '1';

  const normalized = $derived(url.trim().replace(/\/+$/, ''));
  const invalid = $derived(normalized !== '' && !/^https?:\/\/[^\s/]+/i.test(normalized));
  const dirty = $derived(normalized !== prefs.serverUrl);

  async function runTest() {
    if (invalid) return;
    testing = true;
    test = null;
    const t0 = performance.now();
    try {
      const h = await new HttpApi(normalized).health(AbortSignal.timeout(3000));
      test = { ok: true, version: h.version, mock: h.mock, ms: performance.now() - t0 };
    } catch (e) {
      const timeout = e instanceof DOMException && (e.name === 'TimeoutError' || e.name === 'AbortError');
      test = { ok: false, message: timeout ? 'No answer within 3 s.' : e instanceof Error ? e.message : String(e) };
    } finally {
      testing = false;
    }
  }

  async function save() {
    if (invalid) return;
    prefs.serverUrl = normalized;
    url = prefs.serverUrl;
    await hub.connect();
    if (hub.mode === 'live') toast.success('Connected', `folo serve ${hub.version} at ${hub.serverUrl}`);
    else if (forced) toast.info('Saved', 'Demo data stays on while the address has ?mock=1.');
    else toast.error(new Error(`Could not reach ${hub.serverUrl}; showing demo data.`), 'Saved, but not connected');
  }

  // ---- appearance / units ----
  const themes: { value: ThemePref; label: string; icon: typeof Moon }[] = [
    { value: 'dark', label: 'Dark', icon: Moon },
    { value: 'light', label: 'Light', icon: Sun },
    { value: 'system', label: 'System', icon: Monitor },
  ];
  let units = $state<Units>(prefs.units);
  const sample = 5_600_000_000;

  // ---- shortcuts ----
  /** Opens the shortcuts dialog the way the shell does: the `?` key. */
  function showShortcuts() {
    window.dispatchEvent(new KeyboardEvent('keydown', { key: '?', bubbles: true }));
  }
</script>

<Page title="Settings" narrow>
  {#snippet sub()}<span>Preferences are stored in this browser.</span>{/snippet}

  <div class="layout">
    <div class="col">
      <Panel title="Connection" id="connection" divided>
        <div class="stack">
          <div class="status" data-mode={hub.mode}>
            <span class="led" aria-hidden="true"></span>
            <div class="status-text">
              {#if hub.mode === 'live'}
                <p class="s-title">Live <span class="mono faint">folo serve {hub.version}</span></p>
                <p class="s-desc">Connected to <span class="mono">{hub.serverUrl}</span>.</p>
              {:else if hub.mode === 'mock'}
                <p class="s-title">Demo data</p>
                <p class="s-desc">
                  {#if hub.reason === 'forced'}
                    The address has <span class="mono">?mock=1</span>, so Studio uses its built-in demo server.
                  {:else}
                    Could not reach <span class="mono">{hub.serverUrl}</span>, so Studio shows built-in demo data. Start
                    <span class="mono">folo serve</span> or point Studio at it below.
                  {/if}
                </p>
              {:else}
                <p class="s-title">Connecting…</p>
                <p class="s-desc">Looking for <span class="mono">{hub.serverUrl}</span>.</p>
              {/if}
            </div>
          </div>

          <Field
            label="Server URL"
            for="server-url"
            hint={origin
              ? `Empty means this page's own address: requests go to ${origin}/api. Enter a URL only to reach a folo serve elsewhere.`
              : "Empty means this page's own address. Enter a URL only to reach a folo serve elsewhere."}
            error={invalid ? 'Use a full address like http://127.0.0.1:7878.' : null}
          >
            <div class="url-row">
              <Input
                id="server-url"
                bind:value={url}
                placeholder="Same as this page"
                spellcheck={false}
                autocomplete="off"
                {invalid}
                class="mono-input"
                onkeydown={(e: KeyboardEvent) => {
                  if (e.key === 'Enter') void runTest();
                }}
              />
              <Button icon={PlugZap} loading={testing} disabled={invalid} onclick={runTest}>Test</Button>
            </div>
          </Field>

          {#if test}
            <div class="test" class:ok={test.ok} role="status">
              {#if test.ok}
                <Check size={14} strokeWidth={2.25} aria-hidden="true" />
                <span>Reachable: <span class="mono">folo serve {test.version}</span>{test.mock ? ' (mock mode)' : ''}</span>
                <span class="num faint">{Math.round(test.ms)} ms</span>
              {:else}
                <span class="x" aria-hidden="true">×</span>
                <span>{test.message}</span>
              {/if}
            </div>
          {/if}

          <div class="row-actions">
            <!-- The primary action only once there is something to save; reconnecting as is stays secondary. -->
            <Button variant={dirty ? 'primary' : 'secondary'} disabled={invalid} onclick={save}>{dirty ? 'Save and reconnect' : 'Reconnect'}</Button>
          </div>
        </div>
      </Panel>

      <Panel title="Appearance" id="appearance" divided>
        <div class="themes" role="radiogroup" aria-label="Theme">
          {#each themes as t (t.value)}
            {@const on = prefs.theme === t.value}
            <label class="theme" class:on>
              <input type="radio" name="theme" value={t.value} checked={on} onchange={() => (prefs.theme = t.value)} />
              <span class="preview">
                {#if t.value === 'system'}
                  <span class="split dark-half"><ThemePreview theme="dark" /></span>
                  <span class="split light-half"><ThemePreview theme="light" /></span>
                {:else}
                  <ThemePreview theme={t.value} />
                {/if}
              </span>
              <span class="theme-label">
                <t.icon size={14} strokeWidth={1.75} aria-hidden="true" />
                {t.label}
              </span>
            </label>
          {/each}
        </div>
        <p class="note">
          {#if prefs.theme === 'system'}Follows your operating system, currently {prefs.resolvedTheme}.{:else}Cycle themes from the top bar or the command palette.{/if}
        </p>
      </Panel>

      <Panel title="Units" id="units" divided>
        <div class="setting">
          <div class="setting-text">
            <p>File and memory sizes</p>
            <p class="note">Binary units match what operating systems and GPU drivers report.</p>
          </div>
          <Segmented
            label="Size units"
            bind:value={units}
            onchange={(v) => (prefs.units = v)}
            options={[
              { value: 'binary', label: 'Binary', meta: bytes(sample, 'binary') },
              { value: 'metric', label: 'Metric', meta: bytes(sample, 'metric') },
            ]}
          />
        </div>
      </Panel>

      <Panel title="Keyboard shortcuts" id="shortcuts" divided>
        <div class="setting">
          <div class="setting-text">
            <p>Press <Kbd keys="?" /> anywhere for all shortcuts</p>
            <p class="note">Navigation, the command palette and actions on the current screen.</p>
          </div>
          <Button onclick={showShortcuts}>Show shortcuts</Button>
        </div>
      </Panel>

      <Panel title="About" id="about" divided>
        <dl class="about">
          <dt>FOLO Studio</dt>
          <dd class="num">{pkg.version}</dd>
          <dt>Server</dt>
          <dd class="num">{hub.mode === 'live' ? `folo serve ${hub.version}` : hub.mode === 'mock' ? `built-in demo ${hub.version}` : '—'}</dd>
          <dt>HTTP API</dt>
          <dd class="mono">docs/API.md</dd>
          <dt>Architecture</dt>
          <dd class="mono">docs/ARCHITECTURE.md</dd>
          <dt>Design system</dt>
          <dd class="mono">design/README.md</dd>
          <dt>Command line</dt>
          <dd class="mono">folo --help</dd>
        </dl>
      </Panel>
    </div>
    <nav class="toc" aria-label="Settings sections">
      <p class="toc-title">On this page</p>
      {#each sections as s (s.id)}<a
          href="#/settings"
          class:on={current === s.id}
          aria-current={current === s.id ? 'location' : undefined}
          onclick={(e) => {
            e.preventDefault();
            current = s.id;
            document.getElementById(s.id)?.scrollIntoView({ block: 'start' });
          }}>{s.label}</a
        >{/each}
    </nav>
  </div>
</Page>

<style>
  /* The form column (720) and a sticky rail of section anchors on its right, centred as one 920 px measure. */
  .layout {
    display: grid;
    grid-template-columns: minmax(0, 720px) 168px;
    gap: var(--space-8);
    align-items: start;
  }
  .toc {
    position: sticky;
    top: calc(var(--size-topbar) + var(--space-6));
    display: grid;
    gap: 2px;
    padding-left: var(--space-3);
    border-left: 1px solid var(--line-subtle);
  }
  .toc-title {
    padding: 0 var(--space-2) var(--space-1);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-tertiary);
  }
  .toc a {
    display: flex;
    align-items: center;
    height: 28px;
    padding: 0 var(--space-2);
    border-radius: var(--radius-md);
    color: var(--text-secondary);
    transition:
      background-color var(--dur-fast) var(--ease-standard),
      color var(--dur-fast) var(--ease-standard);
  }
  .toc a.on {
    background: var(--bg-active);
    color: var(--text-primary);
  }
  .toc a:hover {
    background: var(--bg-hover);
    color: var(--text-primary);
  }
  .col {
    display: grid;
    gap: var(--space-4);
    min-width: 0;
  }
  .col :global(section.panel) {
    scroll-margin-top: calc(var(--size-topbar) + var(--space-4));
  }
  .stack {
    display: grid;
    gap: var(--space-4);
  }
  .status {
    display: flex;
    gap: var(--space-3);
    padding: var(--space-3);
    border-radius: var(--radius-md);
    background: var(--bg-inset);
    border: 1px solid var(--line-subtle);
  }
  :global([data-theme='light']) .status {
    background: var(--bg-raised);
  }
  .led {
    flex-shrink: 0;
    width: 8px;
    height: 8px;
    margin-top: 6px;
    border-radius: 50%;
    background: var(--text-tertiary);
  }
  [data-mode='live'] .led {
    background: var(--success-solid);
    box-shadow: 0 0 0 3px var(--success-muted);
  }
  [data-mode='mock'] .led {
    background: var(--warning-solid);
    box-shadow: 0 0 0 3px var(--warning-muted);
  }
  .status-text {
    display: grid;
    gap: 2px;
    min-width: 0;
  }
  .s-title {
    font-weight: var(--weight-medium);
    display: flex;
    gap: var(--space-2);
    align-items: baseline;
  }
  .s-desc {
    color: var(--text-secondary);
    overflow-wrap: anywhere;
  }
  .url-row {
    display: flex;
    gap: var(--space-2);
  }
  .url-row :global(.input) {
    flex: 1;
  }
  /* The typed URL is an identifier (mono); the placeholder is a sentence, so it stays in Geist. */
  .url-row :global(.mono-input input) {
    font-family: var(--font-mono);
    font-size: var(--type-sm-size);
  }
  .url-row :global(.mono-input input::placeholder) {
    font-family: var(--font-sans);
    font-size: var(--type-base-size);
  }
  .test {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    font-size: var(--type-sm-size);
    color: var(--danger-text);
  }
  .test.ok {
    color: var(--success-text);
  }
  .test.ok > span:not(.faint) {
    color: var(--text-primary);
  }
  .test .faint {
    margin-left: auto;
  }
  .x {
    font-weight: var(--weight-semibold);
  }
  .row-actions {
    display: flex;
    justify-content: flex-end;
    padding-top: var(--space-3);
    border-top: 1px solid var(--line-subtle);
  }
  .themes {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: var(--space-3);
  }
  .theme {
    position: relative;
    display: grid;
    gap: var(--space-2);
    cursor: pointer;
  }
  .theme input {
    position: absolute;
    opacity: 0;
    width: 1px;
    height: 1px;
  }
  .preview {
    position: relative;
    display: block;
    aspect-ratio: 16 / 10;
    border-radius: var(--radius-md);
    overflow: hidden;
    box-shadow: 0 0 0 1px var(--line-default);
    transition: box-shadow var(--dur-fast) var(--ease-standard);
  }
  .theme:hover .preview {
    box-shadow: 0 0 0 1px var(--line-strong);
  }
  /* Selection is neutral (lime is for the primary action, focus and live state). */
  .theme.on .preview {
    box-shadow:
      0 0 0 1px var(--bg-panel),
      0 0 0 3px var(--text-secondary);
  }
  .theme input:focus-visible + .preview {
    outline: var(--size-focus-ring) solid var(--focus);
    outline-offset: 4px;
  }
  .split {
    position: absolute;
    inset: 0;
    display: flex;
  }
  .light-half {
    clip-path: polygon(100% 0, 100% 100%, 0 100%);
  }
  .theme-label {
    display: flex;
    align-items: center;
    gap: var(--space-1-5);
    color: var(--text-secondary);
    font-weight: var(--weight-medium);
  }
  .on .theme-label {
    color: var(--text-primary);
  }
  .note {
    margin-top: var(--space-3);
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
  }
  .setting {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-4);
    flex-wrap: wrap;
  }
  .setting-text .note {
    margin-top: 2px;
  }
  .about {
    display: grid;
    grid-template-columns: auto minmax(0, 1fr);
    gap: var(--space-2) var(--space-6);
  }
  .about dt {
    color: var(--text-tertiary);
  }
  .about dd {
    text-align: right;
    color: var(--text-secondary);
  }
  @media (max-width: 1023px) {
    .layout {
      grid-template-columns: minmax(0, 1fr);
    }
    .toc {
      display: none;
    }
  }
  @media (max-width: 767px) {
    .themes {
      gap: var(--space-2);
    }
    .row-actions :global(.btn) {
      width: 100%;
    }
  }
</style>
