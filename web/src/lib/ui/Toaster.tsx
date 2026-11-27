// Toasts never take focus: they announce through the live region, and a click on their buttons keeps focus where it
// was (mousedown is not allowed to move it). A keyboard user can still Tab in; while focus or the pointer is in the
// stack nothing auto-dismisses, and closing the toast that holds focus hands it back to where the user came from, so
// it never falls to <body> (round 17: the Playground lost focus after a toast closed).
import CircleCheck from '$icons/circle-check';
import CircleAlert from '$icons/circle-alert';
import Info from '$icons/info';
import X from '$icons/x';
import { css, cubicOut, Dynamic, effect, fly, For, onCleanup } from 'fw';
import { toast } from './toast';
import { motion } from './motion';
import Meta from './Meta';

const icons = { success: CircleCheck, error: CircleAlert, info: Info };

export default function Toaster() {
  let el!: HTMLElement;
  let hovered = false;
  let focused = false;
  let from: HTMLElement | null = null;
  const hold = () => toast.hold(hovered || focused);

  // A focused toast removed programmatically (toast.clear / key replacement) never fires focusout on DOM removal, and
  // while its fly outro runs the dying <li> still holds document.activeElement: release hold and return focus as soon
  // as the focused toast leaves toast.items.
  effect(() => {
    const items = toast.items;
    if (!focused) return;
    const active = document.activeElement;
    const host = active instanceof Element && el?.contains(active) ? active.closest<HTMLElement>('[data-toast]') : null;
    const activeId = host ? Number(host.dataset.toast) : NaN;
    if (host && items.some((t) => t.id === activeId)) return;
    const inDyingToast = host !== null;
    focused = false;
    hold();
    if (from?.isConnected && (inDyingToast || !active || active === document.body)) {
      from.focus({ preventScroll: true });
    }
    from = null;
  });

  // Publish the active toast stack's height as --toast-h (same pattern as shell/dock.ts's --dock-h) so mobile `main`
  // scroll clearance and `scroll-padding-bottom` keep the bottom-most control tappable above a visible toast.
  let ro: ResizeObserver | null = null;
  let mounted = true;
  const syncToastHeight = () => {
    if (typeof document === 'undefined') return;
    if (!mounted || !toast.items.length || !el) {
      document.documentElement.style.removeProperty('--toast-h');
      return;
    }
    const h = el.offsetHeight;
    if (h > 0) document.documentElement.style.setProperty('--toast-h', `${h + 16}px`);
    else document.documentElement.style.removeProperty('--toast-h');
  };

  effect(() => {
    const count = toast.items.length;
    if (count === 0) {
      document.documentElement.style.removeProperty('--toast-h');
      return;
    }
    syncToastHeight();
    queueMicrotask(syncToastHeight);
  });

  const bindHost = (node: HTMLElement) => {
    el = node;
    ro?.disconnect();
    ro = typeof ResizeObserver === 'undefined' ? null : new ResizeObserver(syncToastHeight);
    ro?.observe(node);
    syncToastHeight();
  };

  onCleanup(() => {
    mounted = false;
    ro?.disconnect();
    ro = null;
    if (typeof document !== 'undefined') document.documentElement.style.removeProperty('--toast-h');
    from = null;
    if (hovered || focused) {
      hovered = false;
      focused = false;
      hold();
    }
  });

  function onfocusin(e: FocusEvent) {
    const prev = e.relatedTarget;
    if (prev instanceof HTMLElement && !el.contains(prev)) from = prev;
    focused = true;
    hold();
  }
  function onfocusout(e: FocusEvent) {
    if (e.relatedTarget instanceof Node && el.contains(e.relatedTarget)) return;
    focused = false;
    hold();
  }

  function close(id: number, run?: () => void) {
    run?.();
    // Only the toast that holds focus hands it back: closing another (by mouse) leaves focus, and the hold, where they
    // are. The action may also have moved focus itself (e.g. opened a page): then it is not ours to place.
    const inside = !!el.querySelector(`[data-toast="${id}"]`)?.contains(document.activeElement);
    toast.dismiss(id);
    if (!inside) return;
    focused = false;
    hold();
    if (from?.isConnected) from.focus({ preventScroll: true });
    from = null;
  }

  const keepFocus = (e: MouseEvent) => e.preventDefault();

  return (
    <section
      class="toaster"
      aria-label="Notifications"
      ref={bindHost}
      onpointerenter={() => ((hovered = true), hold())}
      onpointerleave={() => ((hovered = false), hold())}
      onfocusin={onfocusin}
      onfocusout={onfocusout}
    >
      {/* No live region on the list: each toast is its own (role status, or alert for errors), so it is announced once. */}
      <ol>
        <For each={toast.items} key={(t) => t.id}>
          {(t) => (
            <li
              class={`toast ${t().tone}`}
              data-toast={t().id}
              role={t().tone === 'error' ? 'alert' : 'status'}
              transition:fly={{ y: 8, duration: motion(160), easing: cubicOut }}
            >
              <span class="icon">
                <Dynamic component={icons[t().tone]} size={16} strokeWidth={1.75} />
              </span>
              {/* Line 1 the title beside the action and close; line 2 the detail under all three, the toast's full
                  width, so "Finished training · best mAP50-95 0.535 at epoch 60" stays one line on a phone instead of
                  four beside "Open run". Facts are a ui/Meta line: they wrap whole, never "at / epoch 60". */}
              <p class="title">{t().title}</p>
              {t().detail ? (
                <p class="detail">
                  {typeof t().detail === 'string' ? (
                    t().detail
                  ) : (
                    <Meta>
                      <For each={t().detail as string[]} key={(d) => d}>
                        {(d) => <span>{d()}</span>}
                      </For>
                    </Meta>
                  )}
                </p>
              ) : null}
              {t().action ? (
                <button type="button" class="action" onmousedown={keepFocus} onclick={() => close(t().id, t().action!.run)}>
                  {t().action!.label}
                </button>
              ) : null}
              <button type="button" class="close" aria-label="Dismiss" onmousedown={keepFocus} onclick={() => close(t().id)}>
                <X size={14} />
              </button>
            </li>
          )}
        </For>
      </ol>
    </section>
  );
}

css`
  .toaster {
    position: fixed;
    right: var(--space-4);
    bottom: var(--space-4);
    z-index: var(--z-toast);
    /* Wide enough for the title on one line and the detail on the next (round 24's 380 px toast wrapped a one-line
       result into three beside its action): the text column is at least 240 px beside an action, 360 px without. */
    width: min(440px, calc(100vw - 32px));
    pointer-events: none;
  }
  ol {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
  }
  /* Two rows: the icon, the title, the action and close on the first (centred on one another); the detail on the
     second, from the title's left edge to the toast's padding. */
  .toast {
    pointer-events: auto;
    display: grid;
    grid-template-columns: auto minmax(0, 1fr) auto auto;
    align-items: center;
    column-gap: 12px;
    row-gap: 2px;
    padding: var(--space-3) var(--space-2) var(--space-3) var(--space-3);
    border-radius: var(--radius-lg);
    background: var(--bg-overlay);
    box-shadow: var(--shadow-lg);
  }
  /* No action: three columns, so the close button keeps its place with no empty gap before it. */
  .toast:not(:has(.action)) {
    grid-template-columns: auto minmax(0, 1fr) auto;
  }
  .icon {
    grid-row: 1;
    display: flex;
    color: var(--text-secondary);
  }
  .success .icon {
    color: var(--success-text);
  }
  .error .icon {
    color: var(--danger-text);
  }
  .info .icon {
    color: var(--info-text);
  }
  /* Title on line 1 (13/500): a run's name, which may break at its hyphens but never mid-word. */
  .title {
    grid-row: 1;
    grid-column: 2;
    min-width: 0;
    font-weight: var(--weight-medium);
  }
  /* The detail on line 2 at 12 px secondary, under the title and running on under the action. */
  .detail {
    grid-row: 2;
    grid-column: 2 / -1;
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
    text-wrap: pretty;
    overflow-wrap: anywhere;
  }
  /* A toast's action is a secondary button in ink ("Open run"): lime is the page's one primary, and an action is not
     a status, so it takes neither the accent nor the toast's hue (DESIGN_DIRECTION §2.6). */
  .action {
    grid-row: 1;
    grid-column: 3;
    flex-shrink: 0;
    height: var(--control-sm);
    padding: 0 var(--space-2);
    border-radius: var(--radius-sm);
    border: 1px solid var(--line-default);
    background: var(--bg-raised);
    font-weight: var(--weight-medium);
    color: var(--text-primary);
  }
  .action:hover {
    border-color: var(--line-strong);
    background: color-mix(in srgb, var(--bg-raised), var(--text-primary) 4%);
  }
  .close {
    grid-row: 1;
    grid-column: -2;
    position: relative;
    display: grid;
    place-items: center;
    width: var(--control-sm);
    height: var(--control-sm);
    border-radius: var(--radius-sm);
    color: var(--text-tertiary);
  }
  /* At least a 44 × 44 px target around the small button (a real 44 px box on touch, below). */
  .close::after {
    content: '';
    position: absolute;
    inset: min(0px, calc((var(--control-sm) - 44px) / 2));
  }
  .close:hover {
    background: var(--bg-hover);
    color: var(--text-primary);
  }
  /* Touch: both buttons are real 44 px boxes side by side (no invisible targets overlapping), pulled into the toast's
     padding so the card keeps its height. */
  @media (pointer: coarse) {
    .action,
    .close {
      height: 44px;
      margin: calc(-1 * var(--space-2)) 0;
    }
    .action {
      padding: 0 var(--space-3);
    }
    .close {
      width: 44px;
    }
    .close::after {
      content: none;
    }
  }
  @media (max-width: 767px) {
    .toaster {
      right: var(--space-4);
      left: var(--space-4);
      width: auto;
      bottom: calc(var(--tabbar-h) + var(--dock-h) + var(--space-4) + env(safe-area-inset-bottom));
    }
  }
`;
