/**
 * The app-wide word that the server stopped answering (`hub.offline`), or answered a connect as something that is not
 * a usable FOLO server (`hub.refused`, said instead), while Studio probes it: a pill in the top bar,
 * between the page's trail and the bar's actions. The bar's height is fixed, so nothing on the page moves when it comes
 * or goes, and nothing of the page is ever under it (v-shell N5). Phones say "Offline" (the sidebar's Settings page has
 * the rest). It goes when the server answers again; the sidebar footer then says Connected.
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
          <span class="long">{hub.refused ? `${hub.refused} Retrying…` : "Can't reach folo serve. Retrying…"}</span>
          <span class="short">{hub.refused ? 'Not connected' : 'Offline'}</span>
        </p>
      ) : null}
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
    padding: 0 var(--space-3);
    border-radius: var(--radius-full);
    background: var(--bg-overlay);
    border: 1px solid var(--line-strong);
    color: var(--text-secondary);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    white-space: nowrap;
  }
  .notice :global(svg) {
    flex-shrink: 0;
    color: var(--warning-text);
  }
  .long {
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .short {
    display: none;
  }
  @media (max-width: 767px) {
    .long {
      display: none;
    }
    .short {
      display: inline;
    }
  }
`;
