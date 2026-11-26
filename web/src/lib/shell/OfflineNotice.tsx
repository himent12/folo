/**
 * The app-wide word that the server stopped answering (`hub.offline`), or answers as something that is not a usable
 * FOLO server or stopped letting Studio in (`hub.refused`, said instead), while Studio probes it: a pill in the top bar,
 * between the page's trail and the bar's actions, with an inline Retry action and (on wider screens, where the top bar
 * has no dedicated Settings button) a Settings link to Connection. The bar's height is fixed, so nothing on the page
 * moves when it comes or goes, and nothing of the page is ever under it (v-shell N5). Phones show "Offline" + Retry
 * (next to the phone top bar's Settings icon). It goes when the server answers again; the sidebar footer then says
 * Connected.
 */
import TriangleAlert from '$icons/triangle-alert';
import WifiOff from '$icons/wifi-off';
import { computed, css, effect, onCleanup, signal } from 'fw';
import { hub } from '$lib/api/index';
import { href, router } from '$lib/router';

export default function OfflineNotice() {
  const retrying = signal(false);
  /** Latched refusal wording across the synchronous clear at the start of `hub.connect()`, so copy never flickers. */
  const lastRefused = signal<string | null>(null);
  const onSettings = computed(() => router.route.name === 'settings');
  const demoErr = computed(() => hub.demoError !== null);
  // Prioritize slow demo load over `hub.offline` so a stalled retry (or `?mock=1&scenario=error`) still offers Reload.
  const demoSlow = computed(() => !demoErr.get() && hub.demoSlow);
  const demoReload = computed(() => demoErr.get() || demoSlow.get());
  const visible = computed(() => hub.offline || demoReload.get() || retrying.get());
  const refused = computed(() => hub.refused ?? (retrying.get() ? lastRefused.get() : null));
  let slotEl: HTMLDivElement | undefined;
  let retryBtn: HTMLButtonElement | undefined;
  let reloadBtn: HTMLButtonElement | undefined;
  let reloadFocused = false;
  let alive = true;
  onCleanup(() => {
    alive = false;
  });

  const focusFallback = () =>
    slotEl?.closest('header')?.querySelector<HTMLElement>(':scope > .actions button, :scope > .actions a[href]') ?? document.getElementById('content');

  // If the demo chunk finishes while the top-bar Reload button holds focus, hand focus to a stable control instead of body.
  effect(() => {
    if (!demoReload.get() && reloadFocused) {
      reloadFocused = false;
      const active = document.activeElement;
      if (!active || active === document.body || active === reloadBtn) focusFallback()?.focus();
    }
  });

  async function retry() {
    if (retrying.peek()) return;
    lastRefused.set(hub.refused);
    retrying.set(true);
    try {
      await hub.connect();
    } catch {
      /* hub.connect handles connection errors internally */
    } finally {
      if (!alive) return;
      const hadFocus = !!retryBtn && document.activeElement === retryBtn;
      const fallback = hadFocus ? focusFallback() : null;
      retrying.set(false);
      lastRefused.set(null);
      // If recovery hid the notice while Retry held keyboard focus, hand focus to a stable control instead of body.
      if (hadFocus && !hub.offline) fallback?.focus();
    }
  }

  return (
    <div class="slot" ref={(el) => (slotEl = el)}>
      <div class="notice" class:idle={!visible.get()} class:bad={demoErr.get()}>
        <span class="msg" role="status">
          {visible.get() ? (
            demoErr.get() ? (
              <>
                <TriangleAlert size={14} strokeWidth={2} aria-hidden="true" />
                <span class="long">Demo data failed to load.</span>
                <span class="short">Demo failed</span>
              </>
            ) : demoSlow.get() ? (
              <>
                <TriangleAlert size={14} strokeWidth={2} aria-hidden="true" />
                <span class="long">Demo data is taking a while to load.</span>
                <span class="short">Loading demo…</span>
              </>
            ) : (
              <>
                <WifiOff size={14} strokeWidth={2} aria-hidden="true" />
                <span class="long">{refused.get() ? `${refused.get()} Retrying…` : "Can't reach folo serve. Retrying…"}</span>
                <span class="short">{refused.get() ? 'Not connected' : 'Offline'}</span>
              </>
            )
          ) : null}
        </span>
        {visible.get() ? (
          <span class="actions">
            {demoReload.get() ? (
              <button
                ref={(el) => (reloadBtn = el)}
                type="button"
                class="action"
                onfocus={() => (reloadFocused = true)}
                onblur={() => (reloadFocused = false)}
                onclick={() => location.reload()}
              >
                Reload
              </button>
            ) : (
              <>
                <button
                  ref={(el) => (retryBtn = el)}
                  type="button"
                  class="action"
                  aria-disabled={retrying.get() || undefined}
                  aria-busy={retrying.get() || undefined}
                  onclick={() => void retry()}
                >
                  {retrying.get() ? 'Retrying…' : 'Retry'}
                </button>
                {!onSettings.get() ? (
                  <a class="action settings-link" href={href('/settings?section=connection')}>
                    Settings
                  </a>
                ) : null}
              </>
            )}
          </span>
        ) : null}
      </div>
    </div>
  );
}

css`
  .slot {
    display: flex;
    min-width: 0;
    flex-shrink: 1;
  }
  .notice {
    display: inline-flex;
    align-items: center;
    gap: var(--space-2);
    min-width: 0;
    height: var(--control-sm);
    padding: 0 var(--space-1-5) 0 var(--space-3);
    border-radius: var(--radius-full);
    background: var(--bg-overlay);
    border: 1px solid var(--line-strong);
    color: var(--text-secondary);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    white-space: nowrap;
  }
  /* Keep the status live region mounted while online with zero visual footprint. */
  .notice.idle {
    width: 0;
    height: 0;
    padding: 0;
    border: 0;
    overflow: hidden;
  }
  .msg {
    display: inline-flex;
    align-items: center;
    gap: var(--space-2);
    min-width: 0;
  }
  .msg :global(svg) {
    flex-shrink: 0;
    color: var(--warning-text);
  }
  .notice.bad .msg :global(svg) {
    color: var(--danger-text);
  }
  .long,
  .short {
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .short {
    display: none;
  }
  .actions {
    display: inline-flex;
    align-items: center;
    gap: var(--space-1);
    flex-shrink: 0;
    padding-left: var(--space-2);
    border-left: 1px solid var(--line-subtle);
  }
  .action {
    position: relative;
    display: inline-flex;
    align-items: center;
    height: var(--control-xs);
    padding: 0 var(--space-1-5);
    border-radius: var(--radius-full);
    color: var(--text-primary);
    font-size: var(--type-sm-size);
    font-weight: var(--weight-medium);
    line-height: 1;
    white-space: nowrap;
    transition:
      background-color var(--dur-fast) var(--ease-standard),
      color var(--dur-fast) var(--ease-standard);
  }
  @media (hover: hover) {
    .action:hover:not([aria-disabled='true']) {
      background: var(--bg-hover);
    }
    a.action:hover {
      text-decoration: underline;
      text-underline-offset: 3px;
      text-decoration-color: var(--line-strong);
    }
  }
  .action:active:not([aria-disabled='true']) {
    background: var(--bg-active);
  }
  .action[aria-disabled='true'] {
    color: var(--text-tertiary);
    cursor: not-allowed;
  }
  /* Touch: expand hit targets to 44 x 44 px without growing the pill past the fixed top bar. */
  @media (pointer: coarse) {
    .notice:not(.idle) {
      height: var(--control-md);
    }
    .actions {
      gap: var(--space-2);
    }
    .action {
      height: var(--control-sm);
      padding: 0 var(--space-2);
    }
    .action::after {
      content: '';
      position: absolute;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      min-width: 44px;
      min-height: 44px;
      width: 100%;
      height: 100%;
    }
  }
  @media (max-width: 767px) {
    .notice:not(.idle) {
      gap: var(--space-1-5);
      padding: 0 var(--space-1-5) 0 var(--space-2);
    }
    .msg {
      gap: var(--space-1-5);
    }
    .actions {
      padding-left: var(--space-1-5);
    }
    /* The phone top bar already renders a dedicated 44 px Settings button beside this pill. */
    .settings-link {
      display: none;
    }
    .long {
      display: none;
    }
    .short {
      display: inline;
    }
  }
`;
