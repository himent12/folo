/**
 * The Delete run confirm, one on the Run page and the Runs list: it names what goes (the run, its history, its weights
 * and the models saved from them, ./deleteWords), and its button says so. It counts them from the models list fetched
 * when it opens, never from the page's: that one can predate the run's end, when its weights became models. A failure
 * shows in the dialog, which stays open for Retry; a success closes it and toasts.
 */
import Trash from '$icons/trash-2';
import { computed, effect, signal } from 'fw';
import { hub } from '$lib/api/index';
import { modelsChanged } from '$lib/api/models';
import { runsChanged } from '$lib/api/runs';
import type { Model, Run } from '$lib/api/types';
import { prefs } from '$lib/prefs';
import Button from '$lib/ui/Button';
import Dialog from '$lib/ui/Dialog';
import { DialogStatus } from '$lib/ui/feedback';
import { toast } from '$lib/ui/toast';
import { deleteWords, runModels } from './deleteWords';

export interface DeleteRunProps {
  /** The run to delete; null: closed. */
  run: Pick<Run, 'id' | 'name'> | null;
  onclose: () => void;
  ondeleted?: (run: Pick<Run, 'id' | 'name'>) => void;
}

export default function DeleteRun(props: DeleteRunProps) {
  const busy = signal(false);
  const status = new DialogStatus();
  /** The models list fetched for this opening; undefined until it arrives (or when it fails). */
  const models = signal<readonly Model[] | undefined>(undefined);
  const runId = computed(() => props.run?.id ?? null);
  effect(() => {
    const id = runId.get();
    models.set(undefined);
    if (!id) return;
    const ctrl = new AbortController();
    hub.api.models(ctrl.signal).then(
      (m) => {
        if (!ctrl.signal.aborted) models.set(m);
      },
      () => {},
    );
    return () => ctrl.abort();
  });
  const words = computed(() => (props.run ? deleteWords(props.run, models.get(), prefs.units) : null));

  async function remove() {
    const r = props.run;
    if (!r) return;
    const known = models.peek();
    const gone = known ? runModels(r, known).length : undefined;
    busy.set(true);
    try {
      await hub.api.deleteRun(r.id);
      runsChanged();
      props.onclose();
      // page-level: the confirm closed on the line above.
      toast.success(`Deleted ${r.name}`, gone ? `Its weights and ${gone === 1 ? 'model are' : `${gone} models are`} gone too.` : undefined);
      modelsChanged();
      props.ondeleted?.(r);
    } catch (e) {
      status.error(e, 'Could not delete the run');
    } finally {
      busy.set(false);
    }
  }

  return (
    <Dialog
      open={!!props.run}
      onclose={props.onclose}
      title={props.run ? `Delete ${props.run.name}?` : ''}
      description={words.get()?.description}
      size="sm"
      status={status}
      footer={() => (
        <>
          <Button onclick={props.onclose}>Cancel</Button>{' '}
          <Button variant="destructive" icon={Trash} loading={busy.get()} onclick={remove}>
            {words.get()?.button ?? 'Delete run'}
          </Button>
        </>
      )}
    />
  );
}
