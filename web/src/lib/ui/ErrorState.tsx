// fw port of ErrorState.svelte.
import TriangleAlert from '$icons/triangle-alert';
import RotateCw from '$icons/rotate-cw';
import Settings from '$icons/settings';
import { computed, css } from 'fw';
import { ApiError } from '$lib/api/client';
import { hub } from '$lib/api/index';
import { sentence } from '$lib/format';
import { isOom, oomDetail } from '$lib/oom';
import { href } from '$lib/router';
import Button from './Button';
import CodeText from './CodeText';
import { isOffline } from './offline';

export interface ErrorStateProps {
  error: unknown;
  retry?: () => void;
  title?: string;
  compact?: boolean;
  actions?: JSX.Element;
  /** What to do, in this screen's terms, instead of the server's hint (which then goes with the technical details). */
  hint?: string;
}

export default function ErrorState(props: ErrorStateProps) {
  // Connection failures look the same everywhere: what was tried, and both ways out (retry, settings).
  const offline = computed(() => isOffline(props.error));
  const heading = computed(() => (offline.get() ? "Can't reach the FOLO server" : (props.title ?? 'Could not load this')));
  // Out of GPU memory says so plainly; the driver's report goes with the technical details.
  const oom = computed(() => !offline.get() && isOom(props.error));
  const raw = computed(() => (props.error instanceof Error ? props.error.message : String(props.error)));
  const message = computed(() =>
    offline.get()
      ? `Nothing answered at ${hub.serverUrl}.`
      : oom.get()
        ? 'The GPU ran out of memory.'
        : props.error instanceof ApiError
          ? sentence(raw.get())
          : raw.get(),
  );
  const serverHint = computed(() => (props.error instanceof ApiError && props.error.hint ? sentence(props.error.hint) : undefined));
  const hint = computed(() => (offline.get() ? 'Start `folo serve`, or check the server URL.' : (props.hint ?? serverHint.get())));
  const code = computed(() =>
    props.error instanceof ApiError ? `${props.error.code}${props.error.status ? ` · HTTP ${props.error.status}` : ''}` : undefined,
  );
  const detail = computed(() => (oom.get() ? oomDetail(raw.get()) : null));
  /** The server's own hint, kept with the details when the screen says what applies here. */
  const hiddenHint = computed(() => {
    const s = serverHint.get();
    return !offline.get() && props.hint && s ? s.replaceAll('`', '') : null;
  });

  return (
    <div class="err" class:compact={props.compact ?? false} role="alert">
      <div class="icon">
        <TriangleAlert size={16} strokeWidth={1.75} />
      </div>
      <div class="text">
        <p class="title">{heading.get()}</p>
        {/* Identifiers in backticks, as the server writes them, set in mono and never broken inside (ui/CodeText). */}
        <p class="msg">
          <CodeText text={message.get()} />
        </p>
        {hint.get() ? (
          <p class="hint">
            <CodeText text={hint.get()!} />
          </p>
        ) : null}
        {code.get() || detail.get() || hiddenHint.get() ? (
          <details class="code">
            <summary>Technical details</summary>
            {detail.get() ? <span class="line">{detail.get()}</span> : null}
            {hiddenHint.get() ? <span class="line">{`Server hint: ${hiddenHint.get()}`}</span> : null}
            {code.get() ? <span class="line">{code.get()}</span> : null}
          </details>
        ) : null}
        {/* The ways out sit under the cause they answer, not at the far edge of the panel: one row of buttons of one
            height, centred on each other (Connection settings a ghost button beside Retry, never a text link off its
            baseline); on phones the one remedy layout (ui/Button's .cta-stack), each a full-width 44 px button, the
            primary first. */}
        {props.retry || offline.get() || props.actions ? (
          <div class="acts cta-stack">
            {props.actions}
            {props.retry ? (
              <Button size="sm" icon={RotateCw} onclick={props.retry}>
                Retry
              </Button>
            ) : null}
            {offline.get() ? (
              <Button size="sm" variant="ghost" icon={Settings} href={href('/settings')}>
                Connection settings
              </Button>
            ) : null}
          </div>
        ) : null}
      </div>
    </div>
  );
}

css`
  /* In place of the data, inside the panel that would hold it: no second box, the red is in the icon and title. */
  .err {
    display: flex;
    align-items: flex-start;
    gap: var(--space-3);
    padding: var(--space-4);
  }
  .compact {
    padding: var(--space-3) 0;
  }
  /* Frameless in a page's content slot (a detail page whose object could not be loaded or does not exist), as
     EmptyState there: the page's content edge is its edge, never a card around the message. */
  :global(.page) > .err {
    padding: var(--space-1) 0;
  }
  /* Inside a padded panel the panel's own padding is the edge (as for EmptyState), so a titled panel holding the error
     is its header, the message and 16 px, not 28. */
  :global(.pad-md > .body) > .err.compact,
  :global(.pad-sm > .body) > .err.compact {
    padding: var(--space-1) 0 0;
  }
  .icon {
    color: var(--danger-text);
    padding-top: 2px;
  }
  .text {
    flex: 1;
    min-width: 0;
    display: grid;
    gap: 2px;
  }
  /* The title in ink: the danger shows through the icon (DESIGN_DIRECTION §2.6, titles and links are ink). */
  .title {
    font-weight: var(--weight-medium);
    color: var(--text-primary);
  }
  .msg {
    color: var(--text-secondary);
  }
  .hint {
    color: var(--text-tertiary);
  }
  .code {
    margin-top: 2px;
    font-family: var(--font-mono);
    color: var(--text-tertiary);
    font-size: var(--type-sm-size);
  }
  .code .line {
    display: block;
    overflow-wrap: anywhere;
  }
  .code summary {
    display: flex;
    align-items: center;
    gap: var(--space-1);
    cursor: pointer;
    font-family: var(--font-sans);
    width: max-content;
    list-style: none;
  }
  .code summary::-webkit-details-marker {
    display: none;
  }
  /* The system disclosure chevron (as in Config), not the native triangle. */
  .code summary::before {
    content: '';
    width: 5px;
    height: 5px;
    margin: 0 2px;
    border-right: 1.5px solid currentColor;
    border-bottom: 1.5px solid currentColor;
    transform: rotate(-45deg);
    transition: transform var(--dur-fast) var(--ease-standard);
  }
  .code[open] summary::before {
    transform: rotate(45deg);
  }
  /* Touch: the disclosure is a 44 px target, as Outcome's driver report is. */
  @media (pointer: coarse) {
    .code summary {
      min-height: 44px;
    }
  }
  .acts {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: var(--space-2);
    margin-top: var(--space-2);
  }
`;
