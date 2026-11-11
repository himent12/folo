/**
 * The app-wide word that the server stopped answering (`hub.offline`), while Studio probes it: a pill hung under the
 * top bar, staying there as the page scrolls (on phones, above the bottom bar). Its slot has no height and the pill lets
 * presses through, so nothing on the page moves or is blocked when it comes or goes. It goes when the server answers
 * again; the sidebar footer then says Connected. Not inside the top bar: its backdrop-filter would make the bar the
 * containing block of the phone's fixed pill.
 */
import WifiOff from '$icons/wifi-off';
import { css } from 'fw';
import { hub } from '$lib/api/index';

export default function OfflineNotice() {
  return (
    <div class="slot" role="status">
      {hub.offline ? (
        <p class="notice">
          <WifiOff size={14} strokeWidth={2} aria-hidden="true" />
          <span>Can't reach folo serve. Retrying…</span>
        </p>
      ) : null}
    </div>
  );
}

css`
  .slot {
    position: sticky;
    top: var(--size-topbar);
    z-index: var(--z-sticky);
    height: 0;
    display: flex;
    align-items: flex-start;
    justify-content: center;
    pointer-events: none;
  }
  .notice {
    display: inline-flex;
    flex-shrink: 0;
    align-items: center;
    gap: var(--space-2);
    max-width: calc(100% - 2 * var(--space-4));
    margin: var(--space-2) 0 0;
    padding: 6px 12px;
    border-radius: var(--radius-full);
    background: var(--bg-overlay);
    border: 1px solid var(--line-strong);
    box-shadow: var(--shadow-md);
    color: var(--text-secondary);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
  }
  .notice :global(svg) {
    flex-shrink: 0;
    color: var(--warning-text);
  }
  /* Phones: the top of the page is its title and primary action, so the word sits above the bottom bar, where toasts
     come up. */
  @media (max-width: 767px) {
    .slot {
      position: fixed;
      top: auto;
      left: 0;
      right: 0;
      bottom: calc(var(--tabbar-h) + var(--dock-h) + var(--space-2) + env(safe-area-inset-bottom));
      height: auto;
    }
    .notice {
      margin: 0;
    }
  }
`;
