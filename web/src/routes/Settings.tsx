import Check from '$icons/check';
import ChevronRight from '$icons/chevron-right';
import CircleCheck from '$icons/circle-check';
import CircleMinus from '$icons/circle-minus';
import CircleSlash from '$icons/circle-slash';
import Monitor from '$icons/monitor';
import Moon from '$icons/moon';
import Sun from '$icons/sun';
import { computed, css, Dynamic, effect, For, Index, signal } from 'fw';
import pkg from '../../package.json';
import { hub } from '$lib/api/index';
import { Resource } from '$lib/api/resource';
import { isApple } from '$lib/keys';
import { bytes } from '$lib/format';
import { prefs, type ThemePref, type Units } from '$lib/prefs';
import Page from '$lib/shell/Page';
import { setCrumbs } from '$lib/shell/crumbs';
import { showShortcuts } from '$lib/shell/shortcuts';
import { fadeScroll } from '$lib/ui/fadeScroll';
import Button from '$lib/ui/Button';
import DriverReport from '$lib/ui/DriverReport';
import ErrorState from '$lib/ui/ErrorState';
import Kbd from '$lib/ui/Kbd';
import Meta from '$lib/ui/Meta';
import { keyPrefs } from '$lib/ui/keyPrefs';
import Panel from '$lib/ui/Panel';
import Remedy from '$lib/ui/Remedy';
import Segmented from '$lib/ui/Segmented';
import Switch from '$lib/ui/Switch';
import type { IconType } from '$lib/ui/types';
import { capabilityRows, deviceSummary, driverReport, type CapState } from './settings/capabilities';
import ThemePreview from './settings/ThemePreview';
import Throughput from './settings/Throughput';
import Cloud from './settings/Cloud';
import { hasFeature } from '$lib/throughput';
import { hasCloud } from '$lib/cloud';
import { href, router } from '$lib/router';
import { SCROLL_KEYS, sectionAt } from './settings/spy';
import ConnectionPanel from './settings/ConnectionPanel';

const capMarks: Record<CapState, IconType> = { on: CircleCheck, off: CircleMinus, none: CircleSlash };
/** "Ctrl K" or "⌘ K": the one this platform uses. */
const paletteKeys = isApple ? '⌘ K' : 'Ctrl K';
/** A note in parts, identifiers (in backticks) set as code. */
const parts = (t: string) => t.split('`').map((text, i) => ({ text, code: i % 2 === 1 }));

// ---- appearance / units ----
const themes: { value: ThemePref; label: string; icon: IconType }[] = [
  { value: 'dark', label: 'Dark', icon: Moon },
  { value: 'light', label: 'Light', icon: Sun },
  { value: 'system', label: 'System', icon: Monitor },
];
const sample = 5_600_000_000;

export default function Settings() {
  setCrumbs({ label: 'Settings' });

  // What the server can use (f16, tensor cores, mixed precision); servers older than `capabilities` show no panel. It
  // sits low on the page: a real server answers once its GPU is up, and the panel must not push the forms down.
  const sys = new Resource((s) => hub.api.system(s));
  const caps = computed(() => sys.data?.capabilities);
  /** `/api/system` failed with nothing loaded: System says so (with Retry) instead of being missing (shell F11c). */
  const sysFailed = computed(() => sys.error !== undefined && !sys.data);
  /** FOLO's own GPU buffer pool (`gpu_pool`): absent until the server has used the GPU, and from older servers. */
  const pool = computed(() => sys.data?.gpu_pool);
  const capRows = computed(() => {
    const c = caps.get();
    return c ? capabilityRows(c, sys.data?.gpus ?? []) : [];
  });

  /** Cloud compute: where work runs, beside Connection (servers with the `cloud` feature). */
  const cloud = computed(() => hasCloud(sys.data));
  const sections = computed(() => [
    { id: 'connection', label: 'Connection' },
    ...(cloud.get() ? [{ id: 'cloud', label: 'Cloud' }] : []),
    { id: 'appearance', label: 'Appearance' },
    { id: 'units', label: 'Units' },
    { id: 'shortcuts', label: 'Keyboard shortcuts' },
    ...(caps.get() || sysFailed.get() ? [{ id: 'system', label: 'System' }] : []),
    { id: 'about', label: 'About' },
  ]);
  /** The server's own reasons as plain text, for an issue or a terminal. */
  const report = computed(() => {
    const c = caps.get();
    const d = sys.data;
    return c && d ? driverReport(capRows.get(), c.device, deviceSummary(c, d.gpus, d.cpu.name)) : '';
  });

  // Scroll-spy from geometry (./settings/spy): the current section is the one under the sticky chrome (the top bar,
  // and below 1024 px the strip under it), never the last one because the page has bottomed out. A section chosen by a
  // click or by the URL (`#/settings?section=system`) stays current until the person scrolls the page themselves, so a
  // short last section that cannot reach the chrome still reads as the one they asked for.
  const current = signal<string>('connection');
  /** The section a click or the URL chose; released by the person's own scroll (wheel, touch, keys, scrollbar). */
  let chosen: string | null = null;
  /** Below 1024 px: the sticky jump strip under the top bar (the rail's place on wider screens). */
  const jumpEl = signal<HTMLElement>();
  let spy = () => {};
  effect(() => {
    let raf = 0;
    const update = () => {
      raf = 0;
      if (chosen) {
        current.set(chosen);
        return;
      }
      const jump = jumpEl.peek();
      const strip = jump && jump.offsetParent ? jump.getBoundingClientRect().bottom : 48;
      const tops = sections.peek().flatMap((x) => {
        const el = document.getElementById(x.id);
        return el ? [{ id: x.id, top: el.getBoundingClientRect().top }] : [];
      });
      current.set(sectionAt(tops, strip + 24) ?? 'connection');
    };
    // Svelte's spy read `sections` and `jumpEl` on its first run: a change to either sets it up again.
    void sections.get();
    void jumpEl.get();
    spy = update;
    const onscroll = () => {
      if (!raf) raf = requestAnimationFrame(update);
    };
    const release = () => {
      if (chosen === null) return;
      chosen = null;
      onscroll();
    };
    const onkey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      if (SCROLL_KEYS.has(e.key) && !t?.closest('input, textarea, select, [role=slider], [role=radiogroup]')) release();
    };
    // The page's own scrollbar: a press on the root element, not on content.
    const onpointer = (e: PointerEvent) => e.target === document.documentElement && release();
    update();
    window.addEventListener('scroll', onscroll, { passive: true });
    window.addEventListener('resize', onscroll);
    window.addEventListener('wheel', release, { passive: true });
    window.addEventListener('touchmove', release, { passive: true });
    window.addEventListener('keydown', onkey);
    window.addEventListener('pointerdown', onpointer);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener('scroll', onscroll);
      window.removeEventListener('resize', onscroll);
      window.removeEventListener('wheel', release);
      window.removeEventListener('touchmove', release);
      window.removeEventListener('keydown', onkey);
      window.removeEventListener('pointerdown', onpointer);
    };
  });

  // The jump strip keeps the current section's chip in view, whole and clear of the faded edges, as the page scrolls.
  effect(() => {
    void current.get();
    const jump = jumpEl.get();
    const on = jump?.querySelector<HTMLElement>('a.on');
    if (!jump || !on || !jump.offsetParent) return;
    const left = on.offsetLeft - jump.offsetLeft;
    const fade = 48;
    if (left - fade < jump.scrollLeft || left + on.offsetWidth + fade > jump.scrollLeft + jump.clientWidth)
      jump.scrollTo({ left: Math.max(0, left - (jump.clientWidth - on.offsetWidth) / 2), behavior: 'instant' });
  });

  /** Goes to a section: its heading clear of the sticky chrome (scroll-margin-top), and focus on its heading, so the
   *  keyboard and a screen reader continue from there. It stays the current one until the person scrolls. The focus
   *  shows its ring only when a key led here (a jump link taken with Enter): opened from the URL or by a click, Chrome
   *  would otherwise ring the heading as if tabbed to (round 26: a lime ring around the whole System panel). */
  function goTo(id: string, keyboard = false) {
    chosen = id;
    current.set(id);
    const el = document.getElementById(id);
    if (!el) return;
    el.scrollIntoView({ block: 'start' });
    const target = document.getElementById(`${id}-title`) ?? el;
    if (!target.hasAttribute('tabindex')) target.setAttribute('tabindex', '-1');
    target.focus({ preventScroll: true, ...(keyboard ? {} : { focusVisible: false }) });
    spy();
  }

  // `#/settings?section=system`: the page opens on that section (once it exists: System waits for the server).
  const asked = computed(() => (router.route.name === 'settings' ? router.route.query.get('section') : null));
  let opened: string | null = null;
  effect(() => {
    const a = asked.get();
    // Back to the bare page: the next `?section=` opens its section again, the one opened last included.
    if (!a) opened = null;
    if (!a || a === opened || !sections.get().some((s) => s.id === a)) return;
    opened = a;
    requestAnimationFrame(() => goTo(a));
  });

  const units = signal<Units>(prefs.units);

  /** A section link, in the rail and in the strip: the same anchor, the same behaviour. */
  const link = (s: () => { id: string; label: string }) => (
    <a
      href={href(`/settings?section=${s().id}`)}
      class:on={current.get() === s().id}
      aria-current={current.get() === s().id ? 'location' : undefined}
      onclick={(e) => {
        e.preventDefault();
        // A click from the keyboard (Enter) has no pointer detail.
        goTo(s().id, e.detail === 0);
      }}
    >
      {s().label}
    </a>
  );

  return (
    <Page title="Settings" narrow sub={() => <span>Preferences are stored in this browser.</span>}>
      {/* Below 1024 px the rail goes: its sections as one sticky strip under the top bar, so System and About are a tap
          away from the top of a long page. It scrolls sideways where the links do not fit (a phone): the clipped edge
          fades and carries a chevron, so the sections past it read as more to scroll to, never as a cut label. */}
      <div class="jump-bar">
        <nav class="jump" aria-label="Settings sections" data-scrolls use:fadeScroll ref={jumpEl}>
          <For each={sections.get()} key={(s) => s.id}>
            {(s) => link(s)}
          </For>
        </nav>
        <span class="jump-more" aria-hidden="true">
          <ChevronRight size={16} strokeWidth={1.75} />
        </span>
      </div>

      <div class="layout">
        <div class="col">
          <ConnectionPanel />

          {cloud.get() ? <Cloud /> : null}

          <Panel title="Appearance" id="appearance" divided>
            <div class="themes" role="radiogroup" aria-label="Theme">
              <For each={themes} key={(t) => t.value}>
                {(t) => {
                  const on = () => prefs.theme === t().value;
                  return (
                    <label class="theme" class:on={on()}>
                      <input
                        type="radio"
                        name="theme"
                        value={t().value}
                        checked={on()}
                        aria-label={on() ? `${t().label}, current` : t().label}
                        onchange={() => (prefs.theme = t().value)}
                      />
                      <span class="preview">
                        {t().value === 'system' ? (
                          /* Both themes, each a whole window of its own: dark on the left, light on the right, a 1 px seam. */
                          <span class="split">
                            <ThemePreview theme="dark" half />
                            <span class="seam"></span>
                            <ThemePreview theme="light" half />
                          </span>
                        ) : (
                          <ThemePreview theme={t().value as 'dark' | 'light'} />
                        )}
                      </span>
                      {/* The selection said in words too ("Current"), not only by the ring: under the card's own name,
                          inside its column, so it never reads as marking the card beside it. */}
                      <span class="theme-label">
                        <span class="theme-name">
                          <Dynamic component={t().icon} size={14} strokeWidth={1.75} aria-hidden="true" />
                          {t().label}
                        </span>
                        {on() ? (
                          <span class="current">
                            <Check size={12} strokeWidth={2.25} aria-hidden="true" />
                            Current
                          </span>
                        ) : null}
                      </span>
                    </label>
                  );
                }}
              </For>
            </div>
            <p class="note">
              {prefs.theme === 'system' ? <>Follows your operating system, currently {prefs.resolvedTheme}.</> : 'Cycle themes from the top bar or the command palette.'}
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
            <div class="stack">
              {/* Touch-only devices have no keyboard to look shortcuts up for (the switch stays, for a paired one). */}
              <div class="setting has-keys">
                <div class="setting-text">
                  <p>
                    {keyPrefs.singleKeys ? (
                      <>
                        Press <Kbd keys="?" /> anywhere for all shortcuts
                      </>
                    ) : (
                      'All shortcuts'
                    )}
                  </p>
                  <p class="note">Navigation, the command palette and actions on the current screen.</p>
                </div>
                <Button onclick={showShortcuts}>Show shortcuts</Button>
              </div>
              <div class="sep has-keys"></div>
              <Switch
                label="Single-key shortcuts"
                hint={`N, G then a letter, [, ? and J / K / X in lists. Turn off if you use speech input or press keys by accident; ${paletteKeys} and the arrow keys keep working.`}
                checked={keyPrefs.singleKeys}
                onchange={(v) => (keyPrefs.singleKeys = v)}
              />
            </div>
          </Panel>

          {caps.get() && sys.data ? (
            /* What runs on `device: auto` can use: each feature's state as a mark and a word, and in a sentence why not. */
            <Panel title="System" id="system" divided>
              <div class="stack">
                <div class="setting fact">
                  <div class="setting-text">
                    <p>Default device</p>
                    <p class="note">{deviceSummary(caps.get()!, sys.data!.gpus, sys.data!.cpu.name)}</p>
                  </div>
                  <span class="mono device">{caps.get()?.device}</span>
                </div>
                {pool.get() ? (
                  /* What FOLO itself holds on the GPU (in the one size format): in use, kept for reuse, and the peak. */
                  <div class="setting fact">
                    <div class="setting-text">
                      <p>FOLO's GPU memory</p>
                      <p class="pool num">
                        <Meta>
                          <span>{bytes(pool.get()!.live_bytes, prefs.units)} in use</span>
                          <span>{bytes(pool.get()!.cached_bytes, prefs.units)} cached</span>
                          <span>peak {bytes(pool.get()!.peak_live_bytes, prefs.units)}</span>
                        </Meta>
                      </p>
                      <p class="note">The peak is since the last run started. Cached memory goes back to the driver when a run ends, and after 30 s without predictions.</p>
                    </div>
                    <span class="mono device">{pool.get()?.device}</span>
                  </div>
                ) : null}
                <ul class="caps" aria-label="Capabilities">
                  <For each={capRows.get()} key={(c) => c.id}>
                    {(c) => (
                      <li class="cap">
                        <div class="setting-text">
                          <p>{c().label}</p>
                          <p class="note">
                            <Index each={parts(c().note)}>{(p) => <>{p().code ? <code>{p().text}</code> : p().text}</>}</Index>
                          </p>
                        </div>
                        <span class="cap-state" data-state={c().state}>
                          <Dynamic component={capMarks[c().state]} size={14} strokeWidth={1.75} aria-hidden="true" />
                          {c().word}
                        </span>
                        {/* The fix under the row, across its full width: the command is one line that never wraps. */}
                        {c().fix ? (
                          <div class="cap-fix">
                            <Remedy cause={c().fix!.say} shell={c().fix!.shell} command={c().fix!.command} />
                          </div>
                        ) : null}
                      </li>
                    )}
                  </For>
                </ul>
                {capRows.get().some((c) => c.raw) ? (
                  /* The server's words, verbatim, for an issue or a terminal: folded, and copied whole. */
                  <DriverReport report={report.get()} />
                ) : null}
                {hasFeature(sys.data, 'throughput') ? (
                  /* What this server has measured of its own speed (servers before #130 list no such feature). */
                  <Throughput benchmark={hasFeature(sys.data, 'benchmark')} />
                ) : null}
              </div>
            </Panel>
          ) : sysFailed.get() ? (
            <Panel title="System" id="system" divided>
              <ErrorState error={sys.error} retry={() => sys.refresh()} compact hint="The System and Cloud sections show once it loads." />
            </Panel>
          ) : null}

          <Panel title="About" id="about" divided>
            <dl class="about">
              <dt>FOLO Studio</dt>
              <dd class="num">{pkg.version}</dd>
              <dt>Server</dt>
              <dd class="num">
                {hub.mode === 'live' ? (
                  <>
                    <span class="mono">folo serve</span> {hub.version}
                  </>
                ) : hub.mode === 'mock' ? (
                  <>Built-in demo {hub.version}</>
                ) : (
                  '—'
                )}
              </dd>
              <dt>Documentation</dt>
              <dd>Guides in FOLO's repository</dd>
              <dt>Command line</dt>
              <dd class="mono">folo --help</dd>
            </dl>
          </Panel>
        </div>
        <nav class="toc" aria-label="Settings sections">
          <p class="toc-title">On this page</p>
          <For each={sections.get()} key={(s) => s.id}>
            {(s) => link(s)}
          </For>
          {/* The rail ends on purpose (as a list ends on its .list-foot): the version of this Studio, so six links never
              read as an orphaned list at the top of a tall column. */}
          <p class="toc-foot num">FOLO Studio {pkg.version}</p>
        </nav>
      </div>
    </Page>
  );
}

css`
  /* Form page (DESIGN_DIRECTION 15): the page grid's 12 columns (16 px gutters) of the content column, the form on
     the first 8 from the shared content edge and the sticky rail of section anchors on the last 4, as every page's
     8 + 4 rail. The two fill the content width: no gap between them and no empty margin after them. */
  .layout {
    display: grid;
    grid-template-columns: repeat(12, minmax(0, 1fr));
    column-gap: var(--space-4);
    align-items: start;
  }
  .layout > .col {
    grid-column: span 8;
  }
  .layout > .toc {
    grid-column: span 4;
  }

  /* The rail sticks 16 px under the top bar (top 64), as every page's rail (DESIGN_DIRECTION §2.2). */
  .toc {
    position: sticky;
    top: calc(var(--size-topbar) + var(--space-4));
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
  .toc-foot {
    display: grid;
    gap: 2px;
    margin-top: var(--space-3);
    padding: var(--space-3) var(--space-2) 0;
    border-top: 1px solid var(--line-subtle);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-tertiary);
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
  .sep {
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
    align-content: start;
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
  /* System: the dark window and the light one side by side, split down the middle by a line.default seam (round 25's
     diagonal black-and-white cut read as neither theme). */
  .split {
    position: absolute;
    inset: 0;
    display: grid;
    grid-template-columns: minmax(0, 1fr) 1px minmax(0, 1fr);
  }
  .seam {
    background: var(--line-default);
  }
  /* Every card reserves the "Current" line under its name, so the three names sit at one height whichever is chosen
     (round 24: only the chosen card's label grew a line). */
  .theme-label {
    display: grid;
    grid-template-rows: auto var(--type-sm-line);
    justify-items: start;
  }
  .theme-name {
    display: flex;
    align-items: center;
    gap: var(--space-1-5);
    color: var(--text-secondary);
    font-weight: var(--weight-medium);
  }
  .on .theme-name {
    color: var(--text-primary);
  }
  /* Under the name, starting where the name's text does (past its 14 px icon and the 6 px gap). */
  .current {
    display: inline-flex;
    align-items: center;
    gap: var(--space-1);
    padding-left: 20px;
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
  }
  /* Helper text keeps a readable measure (DESIGN_DIRECTION §2.3: ≤ 60 characters a line). */
  .note {
    max-inline-size: var(--measure);
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
  /* The text gives way (wrapping its note) before the control drops under it: the control keeps the row's right edge
     until the text would be under 240 px (phones). */
  .setting > .setting-text {
    flex: 1 1 240px;
    min-width: 0;
  }
  .setting-text .note {
    margin-top: 2px;
  }
  /* A value that states a fact (System's "gpu:0", a capability's "✓ Available") sits on its row's first line, beside
     the name it answers, never centred on a four-line description; a control (a switch, a button) stays centred. */
  .setting.fact {
    align-items: baseline;
  }
  .cap .setting-text {
    min-width: 0;
  }
  .device {
    font-size: var(--type-sm-size);
    color: var(--text-secondary);
  }
  /* Figures people read: secondary ink (§2.3, tertiary is for skimming). */
  .pool {
    margin-top: 2px;
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
  }
  .caps {
    display: grid;
  }
  /* A capability: its name and sentence, the state on the right, and its fix (when there is one) across both. */
  .cap {
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto;
    align-items: baseline;
    gap: var(--space-3) var(--space-4);
    padding: var(--space-3) 0;
    border-top: 1px solid var(--line-subtle);
  }
  .cap-fix {
    grid-column: 1 / -1;
    min-width: 0;
  }
  .cap:last-child {
    padding-bottom: 0;
  }
  /* The word sets the baseline the row aligns to (the icon is centred on it), so "Available" sits on the name's line. */
  .cap-state {
    flex-shrink: 0;
    display: inline-flex;
    align-items: baseline;
    gap: var(--space-1-5);
    color: var(--text-secondary);
  }
  .cap-state > :global(svg) {
    align-self: center;
  }
  .cap-state[data-state='on'] {
    color: var(--text-primary);
  }
  .cap-state[data-state='on'] :global(svg) {
    color: var(--success-text);
  }
  .cap-state:not([data-state='on']) :global(svg) {
    color: var(--text-tertiary);
  }
  /* A command stays whole (never "--tensor-" / "cores"). */
  .note code {
    white-space: nowrap;
    font-family: var(--font-mono);
    color: var(--text-secondary);
  }
  /* Touch-only: no "Show shortcuts" (the palette drops its entry too). */
  @media (hover: none) and (pointer: coarse) {
    .has-keys {
      display: none;
    }
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
  .jump-bar {
    display: none;
  }
  @media (max-width: 1023px) {
    .layout {
      grid-template-columns: minmax(0, 1fr);
    }
    .layout > .col {
      grid-column: auto;
    }
    .toc {
      display: none;
    }
    /* A full-bleed strip on the page's background, just under the top bar: the sections as links, the current one
       neutral (bg.active, as the rail's), a hairline under it once content scrolls beneath. */
    .jump-bar {
      position: sticky;
      top: var(--size-topbar);
      z-index: var(--z-sticky);
      display: block;
      /* Full bleed: out over the page's side gutters (24 px; 16 on phones). */
      margin: 0 calc(-1 * var(--space-6));
      background: var(--bg-app);
      border-bottom: 1px solid var(--line-subtle);
    }
    .jump {
      display: flex;
      align-items: center;
      gap: var(--space-1);
      height: 44px;
      padding: 0 var(--space-5);
      overflow-x: auto;
      scrollbar-width: none;
    }
    .jump::-webkit-scrollbar {
      display: none;
    }
    /* A scroller says so (DESIGN_DIRECTION §2.12): the edge with more behind it fades over 48 px, wide enough that the
       cut chip reads as passing under the edge, and the end carries a chevron on clear ground (the fade ends before it). */
    .jump:global([data-fade='end']) {
      mask-image: linear-gradient(to right, #000 calc(100% - 80px), transparent calc(100% - 32px));
    }
    .jump:global([data-fade='start']) {
      mask-image: linear-gradient(to left, #000 calc(100% - 48px), transparent);
    }
    .jump:global([data-fade='both']) {
      mask-image: linear-gradient(to right, transparent, #000 48px, #000 calc(100% - 80px), transparent calc(100% - 32px));
    }
    .jump-more {
      position: absolute;
      top: 0;
      right: var(--space-2);
      bottom: 0;
      display: none;
      align-items: center;
      color: var(--text-tertiary);
      pointer-events: none;
    }
    .jump:global([data-fade='end']) + .jump-more,
    .jump:global([data-fade='both']) + .jump-more {
      display: flex;
    }
    .jump a {
      flex-shrink: 0;
      display: flex;
      align-items: center;
      height: 32px;
      padding: 0 var(--space-3);
      border-radius: var(--radius-md);
      color: var(--text-secondary);
      white-space: nowrap;
    }
    .jump a.on {
      background: var(--bg-active);
      color: var(--text-primary);
    }
    @media (hover: hover) {
      .jump a:hover {
        background: var(--bg-hover);
        color: var(--text-primary);
      }
    }
    /* A section's heading lands clear of the top bar and the strip. */
    .col :global(section.panel) {
      scroll-margin-top: calc(var(--size-topbar) + 44px + var(--space-4));
    }
  }
  @media (max-width: 767px) {
    .jump-bar {
      margin: 0 calc(-1 * var(--space-4));
    }
    .jump {
      padding: 0 var(--space-3);
    }
  }
  /* Touch: 44 px targets, so the strip is 52 px. */
  @media (max-width: 1023px) and (pointer: coarse) {
    .jump {
      height: 52px;
    }
    .jump a {
      height: 44px;
    }
    .col :global(section.panel) {
      scroll-margin-top: calc(var(--size-topbar) + 52px + var(--space-4));
    }
  }
  @media (max-width: 767px) {
    .themes {
      gap: var(--space-2);
    }
  }
`;
