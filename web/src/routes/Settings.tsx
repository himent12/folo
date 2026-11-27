// Settings (`#/settings`): appearance and units, the connection to the server, what the server can use, cloud
// connections and measured throughput. The parts are in ./settings/.
import ChevronRight from '$icons/chevron-right';
import CircleCheck from '$icons/circle-check';
import CircleMinus from '$icons/circle-minus';
import CircleSlash from '$icons/circle-slash';
import { computed, css, Dynamic, effect, For, Index, onCleanup, signal } from 'fw';
import pkg from '../../package.json';
import { hub } from '$lib/api/index';
import { Resource } from '$lib/api/resource';
import { isApple } from '$lib/keys';
import { bytes } from '$lib/format';
import { prefs, type Units } from '$lib/prefs';
import Page from '$lib/shell/Page';
import { setCrumbs } from '$lib/shell/crumbs';
import { returnedToPosition } from '$lib/shell/entryMemory';
import { landOn } from '$lib/shell/routeFocus';
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
import Row from './settings/Row';
import Throughput from './settings/Throughput';
import Cloud from './settings/Cloud';
import { hasFeature } from '$lib/throughput';
import { hasCloud } from '$lib/cloud';
import { href, router } from '$lib/router';
import { SCROLL_KEYS, sectionAt } from './settings/spy';
import ConnectionPanel from './settings/ConnectionPanel';
import Appearance from './settings/Appearance';
import DiagnosticsPanel from './settings/DiagnosticsPanel';
import { isSystemInfo } from './settings/diagnostics';

const capMarks: Record<CapState, IconType> = { on: CircleCheck, off: CircleMinus, none: CircleSlash };
/** "Ctrl K" or "⌘ K": the one this platform uses. */
const paletteKeys = isApple ? '⌘ K' : 'Ctrl K';
/** A note in parts, identifiers (in backticks) set as code. */
const parts = (t: string) => t.split('`').map((text, i) => ({ text, code: i % 2 === 1 }));

// ---- units ----
const sample = 5_600_000_000;
/** How long a section opened from the URL is held at the top while the page above it loads. */
const HOLD_MS = 3000;

export default function Settings() {
  setCrumbs({ label: 'Settings' });

  // What the server can use (f16, tensor cores, mixed precision); servers older than `capabilities` show no panel. It
  // sits low on the page: a real server answers once its GPU is up, and the panel must not push the forms down.
  const initialPingMs = signal<number | null>(null);
  const initialPingSeq = signal(0);
  const initialPingEpoch = signal<number | null>(null);
  let prevHubEpoch = hub.epoch;
  let prevConnecting = hub.mode === 'connecting';
  effect(() => {
    const ep = hub.epoch;
    const connecting = hub.mode === 'connecting';
    if (ep === prevHubEpoch && connecting === prevConnecting) return;
    prevHubEpoch = ep;
    prevConnecting = connecting;
    initialPingMs.set(null);
    if (ep !== initialPingEpoch.peek() || connecting) initialPingEpoch.set(null);
    initialPingSeq.set(initialPingSeq.peek() + 1);
  });
  const sys = new Resource(async (s) => {
    const ep = hub.epoch;
    const t0 = performance.now();
    try {
      const data = await hub.api.system(s);
      if (!isSystemInfo(data)) throw new Error('Invalid /api/system response');
      if (!s.aborted && hub.epoch === ep && hub.mode !== 'connecting') {
        initialPingMs.set(performance.now() - t0);
        initialPingEpoch.set(ep);
        initialPingSeq.set(initialPingSeq.peek() + 1);
      }
      return data;
    } catch (e) {
      if (!s.aborted && hub.epoch === ep && hub.mode !== 'connecting') {
        initialPingMs.set(null);
        initialPingSeq.set(initialPingSeq.peek() + 1);
      }
      throw e;
    }
  });
  const caps = computed(() => sys.data?.capabilities);
  /** `/api/system` failed with nothing loaded while connected: System says so (with Retry) instead of being missing (shell F11c). */
  const sysFailed = computed(() => hub.mode !== 'connecting' && sys.error !== undefined && !sys.data);
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
    ...(hasFeature(sys.data, 'throughput') ? [{ id: 'throughput', label: 'Throughput' }] : []),
    { id: 'diagnostics', label: 'Diagnostics' },
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
  /** Below 1024 px: the sticky jump strip under the top bar. Wider screens have no section navigation (design/UNIFIED.md §4). */
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

  // `#/settings?section=system`: the page opens on that section (once it exists: System waits for the server). Its
  // heading is the page's landing (shell/routeFocus), so the route focus of a page opened from elsewhere (the
  // palette's "Units" on Runs) keeps to it instead of taking the page back to its top (v-shell N6).
  const asked = computed(() => (router.route.name === 'settings' ? router.route.query.get('section') : null));
  let opened: string | null = null;
  let jumping = 0;
  effect(() => {
    const a = asked.get();
    // Back to the bare page: the next `?section=` opens its section again, the one opened last included.
    if (!a) opened = null;
    if (!a || a === opened || !sections.get().some((s) => s.id === a)) return;
    opened = a;
    // Back, Forward or a reload to where the person had scrolled: that position wins (shell/entryMemory), the section
    // is jumped to and held only when it is navigated to anew. A hold of the entry before lets go at its next resize.
    if (returnedToPosition()) {
      cancelAnimationFrame(jumping);
      chosen = null;
      return;
    }
    const heading = document.getElementById(`${a}-title`) ?? document.getElementById(a);
    if (heading) {
      if (!heading.hasAttribute('tabindex')) heading.setAttribute('tabindex', '-1');
      landOn(heading);
    }
    jumping = requestAnimationFrame(() => (goTo(a), hold(a)));
  });
  /**
   * Keeps a section the URL opened at the top while what is above it loads and moves it down (the Cloud panel and its
   * connections answer after the page has shown), until the person scrolls (`chosen` released) or HOLD_MS pass.
   */
  let unhold = () => {};
  function hold(id: string) {
    unhold();
    const el = document.getElementById(id);
    const col = el?.parentElement;
    if (!el || !col || typeof ResizeObserver === 'undefined') return;
    const ro = new ResizeObserver(() => {
      if (chosen === id) el.scrollIntoView({ block: 'start' });
      else unhold();
    });
    ro.observe(col);
    const t = setTimeout(() => unhold(), HOLD_MS);
    unhold = () => {
      ro.disconnect();
      clearTimeout(t);
      unhold = () => {};
    };
  }
  onCleanup(() => {
    // A page left within the frame never jumps (nor holds the next page's section).
    cancelAnimationFrame(jumping);
    unhold();
  });

  // Command palette picking the section already named in `?section=` after the user scrolled away: `router.go` treats a
  // same-URL push as a no-op, so the palette dispatches `folo:settings-section` to request the jump again.
  effect(() => {
    const onSection = (e: Event) => {
      const id = (e as CustomEvent<unknown>).detail;
      if (typeof id !== 'string' || !sections.peek().some((s) => s.id === id)) return;
      cancelAnimationFrame(jumping);
      unhold();
      opened = id;
      jumping = requestAnimationFrame(() => (goTo(id), hold(id)));
    };
    window.addEventListener('folo:settings-section', onSection);
    return () => window.removeEventListener('folo:settings-section', onSection);
  });

  const units = signal<Units>(prefs.units);
  // Changed in another tab too (prefs follows the storage event): the control shows what the sizes use.
  effect(() => units.set(prefs.units));

  /** Explicit activation of a jump link: scroll and focus immediately, drop any pending URL hold, and sync `?section=`
   *  in place so sharing the URL opens the chosen section while passive scroll stays free of history churn. */
  function jumpTo(id: string, keyboard: boolean) {
    // A navigation away whose screen chunk is still loading holds `router.route` on Settings while `router.location` is
    // already the destination: never replace its history entry or cancel the navigation.
    if (router.route.name !== 'settings' || router.location.name !== 'settings') return;
    cancelAnimationFrame(jumping);
    jumping = 0;
    unhold();
    opened = id;
    goTo(id, keyboard);
    if (router.route.query.get('section') !== id) {
      const next = new URLSearchParams(router.route.query);
      next.set('section', id);
      router.go(`/settings?${next.toString()}`, { replace: true });
    }
  }

  /** A section link in the strip (below 1024 px). */
  const link = (s: () => { id: string; label: string }) => (
    <a
      href={href(`/settings?section=${s().id}`)}
      class:on={current.get() === s().id}
      aria-current={current.get() === s().id ? 'location' : undefined}
      onclick={(e) => {
        if (e.metaKey || e.ctrlKey || e.shiftKey || e.altKey || e.button !== 0) return;
        e.preventDefault();
        // A click from the keyboard (Enter) has no pointer detail.
        jumpTo(s().id, e.detail === 0);
      }}
    >
      {s().label}
    </a>
  );

  return (
    <Page title="Settings" narrow>
      {/* Below 1024 px the sections are one sticky strip under the top bar, so System and About are a tap away from the
          top of a long page. It scrolls sideways where the links do not fit (a phone): the clipped edge fades and
          carries a chevron, so the sections past it read as more to scroll to, never as a cut label. */}
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

      {/* One column of at most 880 px, centred with the page header (design/UNIFIED.md §4): each section a panel, each
          setting a row. */}
      <div class="col settings-col">
        <ConnectionPanel />

        {cloud.get() ? <Cloud /> : null}

        <Appearance />

        <Panel title="Units" meta="saved in this browser" id="units" divided pad="none">
          <Row
            name="File and memory sizes"
            desc="Binary units match what operating systems and GPU drivers report."
            control={
              <Segmented
                label="Size units"
                bind:value={units}
                onchange={(v) => (prefs.units = v)}
                options={[
                  { value: 'binary', label: 'Binary', meta: bytes(sample, 'binary') },
                  { value: 'metric', label: 'Metric', meta: bytes(sample, 'metric') },
                ]}
              />
            }
          />
        </Panel>

        <Panel title="Keyboard shortcuts" meta="saved in this browser" id="shortcuts" divided pad="none">
          {/* Touch-only devices have no keyboard to look shortcuts up for (the switch stays, for a paired one). */}
          <Row
            class="has-keys"
            name={
              keyPrefs.singleKeys ? (
                <>
                  Press <Kbd keys="?" /> anywhere for all shortcuts
                </>
              ) : (
                'All shortcuts'
              )
            }
            desc="Navigation, the command palette and actions on the current screen."
            control={<Button onclick={showShortcuts}>Show shortcuts</Button>}
          />
          <div class="s-row switch-row">
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
          <Panel title="System" id="system" divided pad="none">
            <Row
              fact
              name="Default device"
              desc={deviceSummary(caps.get()!, sys.data!.gpus, sys.data!.cpu.name)}
              control={<span class="mono device">{caps.get()?.device}</span>}
            />
            {pool.get() ? (
              /* What FOLO itself holds on the GPU (in the one size format): in use, kept for reuse, and the peak. */
              <Row
                fact
                class="pool-row"
                name="FOLO's GPU memory"
                desc={
                  <>
                    <span class="pool num">
                      <Meta>
                        <span>{bytes(pool.get()!.live_bytes, prefs.units)} in use</span>
                        <span>{bytes(pool.get()!.cached_bytes, prefs.units)} cached</span>
                        <span>peak {bytes(pool.get()!.peak_live_bytes, prefs.units)}</span>
                      </Meta>
                    </span>
                    <span class="pool-note">The peak is since the last run started. Cached memory goes back to the driver when a run ends, and after 30 s without predictions.</span>
                  </>
                }
                control={<span class="mono device">{pool.get()?.device}</span>}
              />
            ) : null}
            <ul class="caps s-rows" aria-label="Capabilities">
              <For each={capRows.get()} key={(c) => c.id}>
                {(c) => (
                  <li class="cap">
                    <Row
                      fact
                      name={c().label}
                      desc={<Index each={parts(c().note)}>{(p) => <>{p().code ? <code>{p().text}</code> : p().text}</>}</Index>}
                      control={
                        <span class="cap-state" data-state={c().state}>
                          <Dynamic component={capMarks[c().state]} size={14} strokeWidth={1.75} aria-hidden="true" />
                          {c().word}
                        </span>
                      }
                      below={
                        /* The fix under the row, across its full width: the command is one line that never wraps. */
                        c().fix ? (
                          <div class="cap-fix">
                            <Remedy cause={c().fix!.say} shell={c().fix!.shell} command={c().fix!.command} />
                          </div>
                        ) : undefined
                      }
                    />
                  </li>
                )}
              </For>
            </ul>
            {capRows.get().some((c) => c.raw) ? (
              /* The server's words, verbatim, for an issue or a terminal: folded, and copied whole. */
              <div class="s-row report-row">
                <DriverReport report={report.get()} />
              </div>
            ) : null}
          </Panel>
        ) : sysFailed.get() ? (
          <Panel title="System" id="system" divided>
            <ErrorState error={sys.error} retry={() => sys.refresh()} compact hint="The System and Cloud sections show once it loads." />
          </Panel>
        ) : null}

        {hasFeature(sys.data, 'throughput') ? (
          /* What this server has measured of its own speed (servers before #130 list no such feature). */
          <Throughput benchmark={hasFeature(sys.data, 'benchmark')} />
        ) : null}

        <DiagnosticsPanel
          system={initialPingEpoch.get() === hub.epoch ? sys.data : undefined}
          systemError={sys.error ?? sys.stale}
          initialPingMs={initialPingMs.get()}
          initialPingSeq={initialPingSeq.get()}
          studioVersion={pkg.version}
        />

        <Panel title="About" id="about" divided pad="none">
          <dl class="about">
            <div class="s-row fact-row">
              <dt>FOLO Studio</dt>
              <dd class="num">{pkg.version}</dd>
            </div>
            <div class="s-row fact-row">
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
            </div>
            <div class="s-row fact-row">
              <dt>Documentation</dt>
              <dd>Guides in FOLO's repository</dd>
            </div>
            <div class="s-row fact-row">
              <dt>Command line</dt>
              <dd class="mono">folo --help</dd>
            </div>
          </dl>
        </Panel>
      </div>
    </Page>
  );
}

css`
  /* The column and the page header with it: at most 880 px (Desktop's settings COLUMN_MAX), centred in the page. The
     page's own side padding stays outside it. */
  :global(.page:has(.settings-col)) {
    max-width: calc(880px + 2 * var(--space-6));
  }
  .col {
    display: grid;
    gap: var(--space-4);
    min-width: 0;
  }
  /* Root html already reserves --size-topbar + --space-3 via scroll-padding-top; add the remaining 4 px to total --space-4. */
  .col :global(section.panel) {
    scroll-margin-top: var(--space-1);
  }
  /* The switch is its own row: its label and hint on the left, the switch on the right edge. */
  .switch-row :global(.switch) {
    flex: 1;
    min-width: 0;
  }
  /* Its hint is the row's description: one line where the column allows, as every row's. */
  .switch-row :global(.switch .hint) {
    max-inline-size: none;
  }
  .report-row {
    display: block;
    padding-top: var(--space-3);
    padding-bottom: var(--space-3);
  }
  .device {
    font-size: var(--type-sm-size);
    color: var(--text-secondary);
  }
  /* Figures people read: secondary ink (§2.3, tertiary is for skimming). */
  .pool {
    display: block;
    color: var(--text-secondary);
  }
  .pool-note {
    display: block;
  }
  .caps {
    display: grid;
  }
  .cap-fix {
    min-width: 0;
    padding-bottom: var(--space-2);
  }
  /* The word sets the baseline the row aligns to (the icon is centred on it), so "Available" sits on the name's line. */
  .cap-state {
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
  .cap :global(code) {
    white-space: nowrap;
    font-family: var(--font-mono);
    color: var(--text-secondary);
  }
  /* Touch-only: no "Show shortcuts" (the palette drops its entry too). */
  @media (hover: none) and (pointer: coarse) {
    .col :global(.has-keys) {
      display: none;
    }
  }
  /* A fact of About: its name on the left, its value on the right, a row like every setting's. */
  .about dt {
    color: var(--text-secondary);
  }
  .about dd {
    text-align: right;
    color: var(--text-primary);
  }
  .jump-bar {
    display: none;
  }
  @media (max-width: 1023px) {
    /* A full-bleed strip on the page's background, just under the top bar: the sections as underline tabs, a
       hairline under it once content scrolls beneath. */
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
      align-items: stretch;
      gap: var(--space-4);
      height: 44px;
      padding: 0 var(--space-6);
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
      position: relative;
      flex-shrink: 0;
      display: flex;
      align-items: center;
      height: 100%;
      color: var(--text-secondary);
      font-weight: var(--weight-medium);
      white-space: nowrap;
      transition: color var(--dur-fast) var(--ease-standard);
    }
    /* A page's sections are underline tabs (design/UNIFIED.md §2, ui/Tabs): the current one in ink with a 2 px line on
       the bar's hairline, no fill. */
    .jump a::after {
      content: '';
      position: absolute;
      left: 0;
      right: 0;
      bottom: 0;
      height: 2px;
      border-radius: 2px 2px 0 0;
      background: var(--text-primary);
      transform: scaleX(0);
      transition: transform var(--dur-base) var(--ease-out);
    }
    .jump a.on {
      color: var(--text-primary);
    }
    .jump a.on::after {
      transform: scaleX(1);
    }
    @media (hover: hover) {
      .jump a:hover {
        color: var(--text-primary);
      }
    }
    /* A section's heading lands clear of the top bar (html scroll-padding-top) and the 44 px jump strip. */
    .col :global(section.panel) {
      scroll-margin-top: calc(44px + var(--space-1));
    }
  }
  @media (max-width: 767px) {
    .jump-bar {
      margin: 0 calc(-1 * var(--space-4));
    }
    .jump {
      padding: 0 var(--space-4);
    }
  }
  /* Touch: 44 px targets, so the strip is 52 px. */
  @media (max-width: 1023px) and (pointer: coarse) {
    .jump {
      height: 52px;
    }
    .col :global(section.panel) {
      scroll-margin-top: calc(52px + var(--space-1));
    }
  }
`;
