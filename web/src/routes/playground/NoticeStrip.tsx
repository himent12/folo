/**
 * Inline notice strip shown between the Playground image toolbar and stage when inference needs a local `.folo` file
 * or fails (with recovery actions such as switching to browser inference, picking a smaller model, or retrying).
 */
import FileUp from '$icons/file-up';
import TriangleAlert from '$icons/triangle-alert';
import { bindable, css, Key, type Signal } from 'fw';
import type { Model } from '$lib/api/types';
import type { Backend } from '$lib/inference/browser';
import Button from '$lib/ui/Button';
import ErrorState from '$lib/ui/ErrorState';

export interface NoticeStripProps {
  notice: 'file' | 'error' | null;
  noticeEl?: HTMLElement | undefined;
  badFile: { name: string; why: string | null } | null;
  model: Model | null;
  chooseWeights: () => void;
  runOn?: 'server' | Backend;
  error: unknown;
  retryTick?: number;
  oom: boolean;
  browserOff: string | null;
  smaller: Model | null;
  localBackend: Backend;
  chooseModel: (id: string) => void;
}

export default function NoticeStrip(props: NoticeStripProps) {
  const noticeEl = bindable(props, 'noticeEl') as Signal<HTMLElement | undefined>;
  const runOn = bindable(props, 'runOn', 'server');
  const retryTick = bindable(props, 'retryTick', 0);

  return (
    <>
      {props.notice ? (
        <Key when={props.notice}>
          <div class="notice cta-row" ref={noticeEl}>
            {props.notice === 'file' ? (
              <div class="need-file" class:bad={!!props.badFile} role="alert">
                {props.badFile ? <TriangleAlert size={16} strokeWidth={1.75} aria-hidden="true" /> : <FileUp size={16} strokeWidth={1.75} aria-hidden="true" />}
                <div class="nf-body">
                  {props.badFile ? (
                    <>
                      <p class="n-title">
                        {props.badFile.why ? `${props.badFile.name} could not be loaded` : `${props.badFile.name} is not a .folo weights file`}
                      </p>
                      <p class="nf-text">
                        {props.badFile?.why ? <>{props.badFile?.why} </> : null}Choose {props.model?.name}'s <span class="mono">weights/best.folo</span> from its run's folder, or run it on the server.
                      </p>
                    </>
                  ) : (
                    <>
                      <p class="n-title">This server cannot send model files yet</p>
                      <p class="nf-text">
                        To run {props.model?.name} in the browser, choose its <span class="mono">.folo</span> file (<span class="mono">weights/best.folo</span> in the run's folder), or run it on the server.
                      </p>
                    </>
                  )}
                  <div class="nf-actions cta-stack">
                    <Button size="sm" variant="primary" icon={FileUp} onclick={props.chooseWeights}>
                      Choose .folo file
                    </Button>
                    <Button size="sm" onclick={() => runOn.set('server')}>
                      Run on the server
                    </Button>
                  </div>
                </div>
              </div>
            ) : (
              <ErrorState
                error={props.error}
                title="Prediction failed"
                retry={() => retryTick.set((t) => t + 1)}
                compact
                hint={
                  !props.oom
                    ? undefined
                    : props.browserOff
                      ? props.smaller
                        ? 'Use a smaller model.'
                        : undefined
                      : props.smaller
                        ? 'Run it in this browser, or use a smaller model.'
                        : 'Run it in this browser instead.'
                }
                actions={() => (
                  <>
                    {props.oom && !props.browserOff ? (
                      <>
                        <Button size="sm" variant="primary" onclick={() => runOn.set(props.localBackend)}>
                          Run in this browser
                        </Button>
                        {props.smaller ? (
                          <Button
                            size="sm"
                            title={`${props.smaller?.name} is ${props.smaller?.arch}: a smaller model of the same classes`}
                            onclick={() => props.smaller && props.chooseModel(props.smaller.id)}
                          >
                            Use {props.smaller?.name}
                          </Button>
                        ) : null}
                      </>
                    ) : null}
                  </>
                )}
              />
            )}
          </div>
        </Key>
      ) : null}
    </>
  );
}

css`
  .notice {
    padding: 0 var(--space-4);
    border-bottom: 1px solid var(--line-subtle);
  }
  .need-file {
    display: flex;
    align-items: flex-start;
    gap: var(--space-3);
    padding: var(--space-3) 0;
  }
  .need-file > :global(svg) {
    flex-shrink: 0;
    margin-top: 2px;
    color: var(--text-tertiary);
  }
  .need-file.bad > :global(svg) {
    color: var(--danger-text);
  }
  .nf-body {
    display: grid;
    gap: 2px;
    min-width: 0;
  }
  .n-title {
    font-weight: var(--weight-medium);
  }
  .n-title:focus,
  .notice :global(.title:focus) {
    outline: none;
  }
  .nf-text {
    color: var(--text-secondary);
  }
  .nf-actions {
    display: flex;
    flex-wrap: wrap;
    gap: var(--space-2);
    margin-top: var(--space-2);
  }
`;
