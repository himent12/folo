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
import { css, Dynamic, Show } from 'fw';
import Button from './Button';
import CodeText from './CodeText';
import type { DialogStatus, Notice } from './feedback';

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

  // Each notice is drawn from the value it had when shown (Show keyed: a new notice is a new one), never by reading
  // `status.notice` again inside it: once a notice had replaced one of another tone (a copied path, then a failed
  // copy), clearing it (Dismiss, the dialog closing) ran the notice's own bindings before its branch was removed, and
  // they read null ("Cannot read properties of null (reading 'tone')").
  const notice = (n: Notice) => (
    <div class={`notice ${n.tone}`} data-notice>
      <span class="icon">
        <Dynamic component={icons[n.tone]} size={16} strokeWidth={1.75} aria-hidden="true" />
      </span>
      <div class="text">
        <p class="title">{n.title}</p>
        {/* A detail may name a command in backticks, as the server writes one: set in mono (ui/CodeText). */}
        {n.detail ? (
          <p class="detail">
            <CodeText text={n.detail} />
          </p>
        ) : null}
      </div>
      <Button variant="ghost" size="sm" icon={X} aria-label="Dismiss" onclick={dismiss} />
    </div>
  );
  const shown = (error: boolean) => {
    const n = props.status.notice;
    return n && (n.tone === 'error') === error ? n : null;
  };

  return (
    <div class={`dialog-status ${props.class ?? ''}`} class:on={!!props.status.notice} ref={(e) => (el = e)} onfocusin={onfocusin}>
      <div role="status">
        <Show when={shown(false)} keyed>
          {(n) => notice(n())}
        </Show>
      </div>
      <div role="alert">
        <Show when={shown(true)} keyed>
          {(n) => notice(n())}
        </Show>
      </div>
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
