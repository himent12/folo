/**
 * A dialog's status region: the result of an action taken in the dialog (a copied path, a download the demo cannot
 * make, a failed stop), said where the person is instead of in the toaster behind the modal. Two live regions that
 * are always in the DOM (`status`, and `alert` for errors), so the first result is announced too; the result stays
 * until the next one, Dismiss or the dialog closing. Nothing here takes focus; Dismiss, reached by Tab, hands focus
 * back to where it came from (as a toast's close does), never to <body>.
 */
import CircleCheck from '$icons/circle-check';
import CircleAlert from '$icons/circle-alert';
import Info from '$icons/info';
import X from '$icons/x';
import { css, Dynamic, Key } from 'fw';
import Button from './Button';
import type { DialogStatus } from './feedback';

export default function DialogStatusRegion(props: { status: DialogStatus; class?: string }) {
  const icons = { success: CircleCheck, error: CircleAlert, info: Info };
  let el!: HTMLElement;
  let from: HTMLElement | null = null;

  function onfocusin(e: FocusEvent) {
    if (e.relatedTarget instanceof HTMLElement && !el.contains(e.relatedTarget)) from = e.relatedTarget;
  }

  function dismiss() {
    const inside = el.contains(document.activeElement);
    props.status.clear();
    if (!inside) return;
    const back = from?.isConnected ? from : el.closest('dialog')?.querySelector<HTMLElement>('button:not([disabled]), [tabindex="0"]');
    back?.focus({ preventScroll: true });
    from = null;
  }

  const notice = () => {
    const n = () => props.status.notice!;
    return (
      <Key when={n().id}>
        <div class={`notice ${n().tone}`} data-notice>
          <span class="icon">
            <Dynamic component={icons[n().tone]} size={16} strokeWidth={1.75} aria-hidden="true" />
          </span>
          <div class="text">
            <p class="title">{n().title}</p>
            {n().detail ? <p class="detail">{n().detail}</p> : null}
          </div>
          <Button variant="ghost" size="sm" icon={X} aria-label="Dismiss" onclick={dismiss} />
        </div>
      </Key>
    );
  };

  return (
    <div class={`dialog-status ${props.class ?? ''}`} class:on={!!props.status.notice} ref={(e) => (el = e)} onfocusin={onfocusin}>
      <div role="status">{props.status.notice && props.status.notice.tone !== 'error' ? notice() : null}</div>
      <div role="alert">{props.status.notice?.tone === 'error' ? notice() : null}</div>
    </div>
  );
}

css`
  .notice {
    display: flex;
    align-items: flex-start;
    gap: var(--space-2);
    padding: var(--space-2) var(--space-1) var(--space-2) var(--space-3);
    border: 1px solid var(--line-panel);
    border-radius: var(--radius-md);
    background: var(--bg-panel);
    font-size: var(--type-base-size);
    line-height: var(--type-base-line);
  }
  .icon {
    display: flex;
    padding-top: 4px;
    color: var(--text-secondary);
  }
  .success .icon {
    color: var(--success-text);
  }
  /* The error shows through its icon; the title stays ink (DESIGN_DIRECTION §2.6). */
  .error .icon {
    color: var(--danger-text);
  }
  .info .icon {
    color: var(--info-text);
  }
  .text {
    flex: 1;
    min-width: 0;
    display: grid;
    gap: 2px;
    padding-top: 2px;
  }
  .title {
    font-weight: var(--weight-medium);
  }
  .detail {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
    overflow-wrap: anywhere;
  }
`;
