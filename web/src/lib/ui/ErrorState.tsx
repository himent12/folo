// How a screen or a panel shows an error: what went wrong, what to do and Retry; a 404 for something deleted since
// says it is gone and links to its list.
import Crosshair from '$icons/crosshair';
import TriangleAlert from '$icons/triangle-alert';
import RotateCw from '$icons/rotate-cw';
import Settings from '$icons/settings';
import { computed, css, onMount, signal } from 'fw';
import { ApiError } from '$lib/api/client';
import { hub } from '$lib/api/index';
import { sentence } from '$lib/format';
import { isOom, oomDetail } from '$lib/oom';
import { href } from '$lib/router';
import Button from './Button';
import CodeText from './CodeText';
import { isOffline } from './offline';

/**
 * The things a link can point at that may have been deleted since, by the server's `404 <what>_not_found` code: what
 * to call it, and where the ones that exist are listed. Any screen whose object (or a part of it) is gone says so the
 * same way, whatever it passes: "Run not found · This run was deleted or the link is wrong." and a button to the list,
 * never Retry (it cannot help) nor the server's hint (written for agents: "list_runs (MCP) or GET /api/runs …").
 */
const GONE: Record<string, { noun: string; list?: { label: string; path: string } }> = {
  run_not_found: { noun: 'run', list: { label: 'All runs', path: '/runs' } },
  model_not_found: { noun: 'model', list: { label: 'All models', path: '/models' } },
  dataset_not_found: { noun: 'dataset', list: { label: 'All datasets', path: '/datasets' } },
  label_job_not_found: { noun: 'labeling job', list: { label: 'All labeling jobs', path: '/datasets/label' } },
  bench_not_found: { noun: 'benchmark', list: { label: 'All benchmarks', path: '/benchmarks' } },
  bench_entry_not_found: { noun: 'entry', list: { label: 'All benchmarks', path: '/benchmarks' } },
  bench_loop_not_found: { noun: 'loop', list: { label: 'All benchmarks', path: '/benchmarks' } },
  image_not_found: { noun: 'image' },
  bench_image_not_found: { noun: 'image' },
  label_image_not_found: { noun: 'image' },
};

/** What is gone when `error` is a 404 for a deleted thing (GONE), else null. */
export function goneOf(error: unknown): (typeof GONE)[string] | null {
  return error instanceof ApiError && error.status === 404 ? (GONE[error.code] ?? null) : null;
}

/** The collections of docs/API.md, as the error title names them ("Could not load runs") and one of them by id. */
const LISTS: Record<string, [all: string, one: string]> = {
  runs: ['runs', 'this run'],
  models: ['models', 'this model'],
  datasets: ['datasets', 'this dataset'],
  jobs: ['labeling jobs', 'this labeling job'],
  benchmarks: ['benchmarks', 'this benchmark'],
  entries: ['entries', 'this entry'],
  loops: ['loops', 'this loop'],
  images: ['images', 'this image'],
  connections: ['cloud connections', 'this cloud connection'],
  instances: ['cloud instances', 'this cloud instance'],
  providers: ['cloud providers', 'this cloud provider'],
  offers: ['GPU offers', 'this GPU offer'],
  agents: ['labeling agents', 'this labeling agent'],
};
/** A part of a thing (`/runs/{id}/metrics`) or a page of its own (`/system`), as the title names it. */
const PARTS: Record<string, string> = {
  metrics: "the run's metrics",
  samples: 'the validation samples',
  logs: 'the log',
  eval: 'the evaluation',
  issues: 'the dataset check',
  system: 'the system information',
  throughput: 'the throughput table',
  cli: 'the command',
  catalog: 'the conditions',
  result: 'the result',
  estimate: 'the estimate',
};

/**
 * What a failed read was for, from its request (`GET /api/runs/run_1/metrics` → "the run's metrics"), so an error
 * whose screen passes no title still names what could not be loaded. Null for anything but a read of the API.
 */
export function readOf(request: string | undefined): string | null {
  const m = /^GET \/api\/(.+)$/.exec(request ?? '');
  if (!m) return null;
  const parts = m[1]!.split('/');
  const last = parts[parts.length - 1]!;
  const before = parts[parts.length - 2];
  if (LISTS[last]) return LISTS[last][0];
  if (before && LISTS[before]) return LISTS[before][1];
  return PARTS[last] ?? `the ${last.replace(/[-_]/g, ' ')}`;
}

/** A server hint written for agents and API clients (an endpoint, an MCP tool): kept with the details, not shown. */
const forAgents = (hint: string) => /\b(?:GET|POST|PUT|PATCH|DELETE) \/api\/|\(MCP\)|docs\/API\.md/.test(hint);

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
  const gone = computed(() => goneOf(props.error));
  const heading = computed(() => {
    const g = gone.get();
    if (offline.get()) return "Can't reach the FOLO server";
    if (g) return `${sentence(g.noun).replace(/\.$/, '')} not found`;
    if (props.title) return props.title;
    // No title from the screen: what the request was for, else what kind of failure it was (never "this").
    const what = props.error instanceof ApiError ? readOf(props.error.request) : null;
    if (what) return `Could not load ${what}`;
    return props.error instanceof ApiError && props.error.status >= 500 ? 'The server ran into a problem' : 'Something went wrong';
  });
  // Out of GPU memory says so plainly; the driver's report goes with the technical details.
  const oom = computed(() => !offline.get() && isOom(props.error));
  const raw = computed(() => (props.error instanceof Error ? props.error.message : String(props.error)));
  const message = computed(() =>
    offline.get()
      ? `Nothing answered at ${hub.serverUrl}.`
      : gone.get()
        ? `This ${gone.get()!.noun} was deleted or the link is wrong.`
        : oom.get()
        ? 'The GPU ran out of memory.'
        : props.error instanceof ApiError
          ? sentence(raw.get())
          : raw.get(),
  );
  const serverHint = computed(() => (props.error instanceof ApiError && props.error.hint ? sentence(props.error.hint) : undefined));
  const hint = computed(() => {
    if (offline.get()) return 'Start `folo serve`, or check the server URL.';
    if (gone.get()) return undefined;
    const s = serverHint.get();
    return props.hint ?? (s && !forAgents(s) ? s : undefined);
  });
  const code = computed(() =>
    props.error instanceof ApiError ? `${props.error.code}${props.error.status ? ` · HTTP ${props.error.status}` : ''}` : undefined,
  );
  const request = computed(() => (props.error instanceof ApiError ? props.error.request : undefined));
  const detail = computed(() => (oom.get() ? oomDetail(raw.get()) : null));
  /**
   * The server's own hint, kept with the details when the screen says what applies here or when it is written for
   * agents; a deleted thing's hint is only ever the list endpoint, so it goes.
   */
  const hiddenHint = computed(() => {
    const s = serverHint.get();
    return !offline.get() && !gone.get() && s && (props.hint || forAgents(s)) ? s.replaceAll('`', '') : null;
  });
  const retry = computed(() => (gone.get() ? undefined : props.retry));
  // The way to the list, unless the page's header already has that very link (the detail pages' "All models").
  let root!: HTMLElement;
  const headerHasList = signal(false);
  onMount(() => {
    const path = gone.peek()?.list?.path;
    headerHasList.set(!!path && !!root.closest('.page')?.querySelector(`:scope > header a[href="${href(path)}"]`));
  });
  const list = computed(() => (headerHasList.get() ? undefined : gone.get()?.list));

  return (
    <div class="err" class:compact={props.compact ?? false} class:gone={!!gone.get()} role={gone.get() ? 'status' : 'alert'} ref={(e) => (root = e)}>
      {/* A deleted thing is not a failure: Not found's crosshair in ink, as the unknown-URL page has. */}
      <div class="icon">{gone.get() ? <Crosshair size={16} strokeWidth={1.75} /> : <TriangleAlert size={16} strokeWidth={1.75} />}</div>
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
        {code.get() || request.get() || detail.get() || hiddenHint.get() ? (
          <details class="code">
            <summary>Technical details</summary>
            {detail.get() ? <span class="line">{detail.get()}</span> : null}
            {hiddenHint.get() ? <span class="line">{`Server hint: ${hiddenHint.get()}`}</span> : null}
            {code.get() ? <span class="line">{code.get()}</span> : null}
            {request.get() ? <span class="line">{request.get()}</span> : null}
          </details>
        ) : null}
        {/* The ways out sit under the cause they answer, not at the far edge of the panel: one row of buttons of one
            height, centred on each other (Connection settings a ghost button beside Retry, never a text link off its
            baseline); on phones the one remedy layout (ui/Button's .cta-stack), each a full-width 44 px button, the
            primary first. */}
        {retry.get() || offline.get() || props.actions || list.get() ? (
          <div class="acts cta-stack">
            {props.actions}
            {list.get() ? (
              <Button size="sm" href={href(list.get()!.path)}>
                {list.get()!.label}
              </Button>
            ) : null}
            {retry.get() ? (
              <Button size="sm" icon={RotateCw} onclick={retry.get()}>
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
  .gone .icon {
    color: var(--text-tertiary);
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
