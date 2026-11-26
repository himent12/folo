/**
 * The Stop confirm of a training run, in one wording on the Run page and the Runs list: what stops, and what is kept
 * (the last finished epoch's checkpoint, none before the first ends; a cloud run's instance is released). A failure
 * shows in the dialog, which stays open for Retry; a success closes it and toasts.
 */
import CircleStop from '$icons/circle-stop';
import { onCleanup, signal } from 'fw';
import { hub } from '$lib/api/index';
import { runsChanged } from '$lib/api/runs';
import type { Run } from '$lib/api/types';
import Button from '$lib/ui/Button';
import Dialog from '$lib/ui/Dialog';
import { DialogStatus } from '$lib/ui/feedback';
import { toast } from '$lib/ui/toast';

/** What Stop does to run `r` with `done` epochs finished. */
export function stopDescription(r: Pick<Run, 'status' | 'cloud'>, done: number): string {
  if (r.cloud && r.status === 'queued') return 'Its instance is released now; nothing has trained yet.';
  // An epoch whose last batch is done when the stop comes still validates and saves (folo-vision train/run.rs).
  if (done === 0) return `Training stops after the current step. No epoch has finished yet: unless one finishes first, nothing is saved${r.cloud ? ', and the instance is released' : ''}.`;
  return r.cloud
    ? 'Training stops after the current step. The last checkpoint is copied here, saved as a model, and the instance released.'
    : 'Training stops after the current step. The last checkpoint is kept and saved as a model.';
}

/**
 * The toast once the server took the stop of `r` (as it now is) with `done` epochs finished. Until the run has stopped
 * it says what is certain: an epoch whose last batch is done still validates and saves, and a stop after the run's
 * last batch changes nothing (it completes with all its results, docs/API.md), so "nothing is saved" is not known yet.
 * A run that has ended by then (the stream said so before the answer came) is told as it ended. One the page still
 * shows queued or running (resumed elsewhere since, or its `stopping` not streamed yet) gets no toast: the page says
 * what the stream reports.
 */
export function stoppedToast(r: Pick<Run, 'name' | 'cloud' | 'epoch' | 'status'>, done: number): void {
  if (r.status === 'queued' || r.status === 'running') return;
  // page-level: both callers have closed their confirm first.
  if (r.status === 'completed') return void toast.info(`${r.name} completed`, 'It finished before the stop took effect, with all its results.');
  if (r.status === 'failed') return void toast.info(`${r.name} failed`, 'Its page has the error and the log.');
  const stopped = r.status === 'stopped';
  // Only `done` is known finished: a stopped run's `epoch` may be the one its last step was in, never finished. One
  // that finished while the stop was out is not known here, so a stopped run with none known claims nothing.
  toast.info(
    `${stopped ? 'Stopped' : 'Stopping'} ${r.name}`,
    r.cloud && r.epoch === 0
      ? 'Its instance is released; nothing had trained yet.'
      : done
        ? `The checkpoint from epoch ${done} is kept.`
        : stopped
          ? undefined
          : 'No epoch had finished yet. Unless one finishes before it stops, nothing is saved.',
  );
}

/** Epochs a run in the list has finished: the one in progress is not (the list has no epoch history). */
export const finishedEpochs = (r: Pick<Run, 'status' | 'epoch'>) => (r.status === 'running' || r.status === 'stopping' ? Math.max(0, r.epoch - 1) : r.epoch);

export interface StopRunProps {
  /** The run to stop; null: closed. */
  run: Run | null;
  onclose: () => void;
  onstopped?: (run: Run) => void;
}

export default function StopRun(props: StopRunProps) {
  const busy = signal(false);
  const status = new DialogStatus();
  let mounted = true;
  onCleanup(() => (mounted = false));

  async function stop() {
    const cur = props.run;
    if (!cur || busy.peek()) return;
    const targetId = cur.id;
    busy.set(true);
    try {
      const r = await hub.api.stopRun(targetId);
      runsChanged();
      if (mounted && props.run?.id === targetId) props.onclose();
      // page-level: the confirm closed on the line above (or the page left while stopping).
      stoppedToast(r, finishedEpochs(cur));
      if (mounted) props.onstopped?.(r);
    } catch (e) {
      if (mounted && props.run?.id === targetId) status.error(e, 'Could not stop the run');
      else toast.error(e, 'Could not stop the run');
    } finally {
      busy.set(false);
    }
  }

  return (
    <Dialog
      open={!!props.run}
      onclose={props.onclose}
      onbeforeclose={() => !busy.peek()}
      title={props.run ? `Stop ${props.run.name}?` : ''}
      description={props.run ? stopDescription(props.run, finishedEpochs(props.run)) : undefined}
      size="sm"
      status={status}
      footer={() => (
        <>
          <Button disabled={busy.get()} onclick={() => !busy.peek() && props.onclose()}>
            Keep training
          </Button>{' '}
          <Button variant="destructive" icon={CircleStop} loading={busy.get()} onclick={stop}>
            Stop run
          </Button>
        </>
      )}
    />
  );
}
