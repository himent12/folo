// fw port of UploadStatus.svelte.
/**
 * An addition of images, said where the images are: its progress while files are sent ("Uploading 34 of 120…"), then
 * how it went ("118 added · 2 skipped"), the files left out with their reasons one click away; or why it failed (when
 * it failed part way, both: what was added, then how many files were not sent and why).
 */
import X from '$icons/x';
import { computed, createUniqueId, css, effect, Index, signal } from 'fw';
import { UploadIncomplete } from '$lib/api/upload';
import { num } from '$lib/format';
import Button from '$lib/ui/Button';
import ErrorState from '$lib/ui/ErrorState';
import Meta from '$lib/ui/Meta';
import Progress from '$lib/ui/Progress';
import Spinner from '$lib/ui/Spinner';
import type { Uploads } from './uploads';

export default function UploadStatus(props: { uploads: Uploads }) {
  const uid = createUniqueId();
  const open = signal(false);
  const result = computed(() => props.uploads.result);
  /** An upload that stopped part way: what it added shows, then why the rest was not sent. */
  const incomplete = computed(() => (props.uploads.error instanceof UploadIncomplete ? props.uploads.error : null));
  effect(() => {
    void result.get();
    open.set(false);
  });

  const dismiss = () => (
    <Button size="sm" onclick={() => props.uploads.dismiss()}>
      Dismiss
    </Button>
  );

  return (
    <>
      {props.uploads.progress ? (
        <div class="status" role="status">
          <p class="line num">
            <Spinner size={12} /> Uploading {num(props.uploads.progress!.sent)} of {num(props.uploads.progress!.total)}…
          </p>
          <Progress
            value={props.uploads.progress!.total ? props.uploads.progress!.sent / props.uploads.progress!.total : 0}
            tone="accent"
            label={`${props.uploads.progress!.sent} of ${props.uploads.progress!.total} files uploaded`}
          />
        </div>
      ) : props.uploads.error && !incomplete.get() ? (
        <ErrorState error={props.uploads.error} title="Could not upload the images" compact actions={dismiss} />
      ) : result.get() ? (
        <>
          <div class="status" role="status">
            <div class="head">
              <p class="line num">
                <Meta>
                  <span>{num(result.get()!.added.length)} added</span>
                  {result.get()!.skipped.length ? <span>{num(result.get()!.skipped.length)} skipped</span> : null}
                </Meta>
              </p>
              {result.get()!.skipped.length ? (
                <button type="button" class="more" aria-expanded={open.get()} aria-controls={`${uid}-skipped`} onclick={() => open.set(!open.get())}>
                  {open.get() ? 'Hide skipped' : 'Show skipped'}
                </button>
              ) : null}
              {!incomplete.get() ? (
                <span class="close">
                  <Button variant="ghost" size="sm" icon={X} aria-label="Dismiss" onclick={() => props.uploads.dismiss()} />
                </span>
              ) : null}
            </div>
            {result.get()!.skipped.length ? (
              <ul class="skipped scroll-shade" id={`${uid}-skipped`} hidden={!open.get()}>
                <Index each={result.get()!.skipped}>
                  {(s) => (
                    <li>
                      <span class="mono f">{s().file}</span>
                      <span class="r">{s().reason}</span>
                    </li>
                  )}
                </Index>
              </ul>
            ) : null}
          </div>
          {incomplete.get() ? (
            <ErrorState
              error={incomplete.get()!.cause}
              title={`${num(incomplete.get()!.unsent)} ${incomplete.get()!.unsent === 1 ? 'file was' : 'files were'} not sent`}
              compact
              actions={dismiss}
            />
          ) : null}
        </>
      ) : null}
    </>
  );
}

css`
  .status {
    display: grid;
    gap: var(--space-2);
    padding: var(--space-2) var(--space-3);
    border: 1px solid var(--line-panel);
    border-radius: var(--radius-lg);
    background: var(--bg-panel);
  }
  .head {
    display: flex;
    align-items: center;
    gap: var(--space-3);
  }
  .line {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    font-size: var(--type-sm-size);
    color: var(--text-secondary);
  }
  .close {
    margin-left: auto;
  }
  .more {
    font-size: var(--type-sm-size);
    color: var(--text-primary);
    text-decoration: underline;
    text-decoration-color: var(--line-strong);
    text-underline-offset: 3px;
  }
  @media (pointer: coarse) {
    .more {
      min-height: 44px;
    }
  }
  .skipped {
    display: grid;
    gap: var(--space-1);
    max-height: 160px;
    overflow-y: auto;
    font-size: var(--type-sm-size);
  }
  .skipped[hidden] {
    display: none;
  }
  .skipped li {
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(0, 2fr);
    gap: var(--space-3);
  }
  .f {
    overflow-wrap: anywhere;
  }
  .r {
    color: var(--text-secondary);
  }
`;
