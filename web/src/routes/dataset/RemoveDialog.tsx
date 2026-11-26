/**
 * "Remove from FOLO": unregisters a dataset (its files stay on disk), saying first what that means for the runs that
 * trained on it: they keep their metrics and models, their sample images go with the dataset (until the folder is
 * registered again, which gives it its id back), and a run training or queued on it now carries on. A dataset FOLO
 * wrote into its own data dir (a generated sample set: a folder people rarely open, up to a gigabyte) can go with its
 * files, while no run on the server is queued or training.
 */
import TriangleAlert from '$icons/triangle-alert';
import Trash from '$icons/trash-2';
import { bindable, computed, css, effect, onCleanup, signal, untrack } from 'fw';
import { hub } from '$lib/api/index';
import { Resource } from '$lib/api/resource';
import type { Dataset, Run } from '$lib/api/types';
import { router } from '$lib/router';
import Button from '$lib/ui/Button';
import Dialog from '$lib/ui/Dialog';
import { DialogStatus } from '$lib/ui/feedback';
import Switch from '$lib/ui/Switch';
import { toast } from '$lib/ui/toast';
import { count } from '$lib/format';
import { datasetsChanged } from '../datasets/changes';

export interface RemoveDialogProps {
  open?: boolean;
  dataset: Pick<Dataset, 'id' | 'name' | 'path'> & Partial<Pick<Dataset, 'splits'>>;
}

/** A path with `/` separators and no trailing one, compared ignoring case when it is a Windows one. */
const norm = (p: string) => {
  const s = p.replace(/\\/g, '/').replace(/\/+$/, '');
  return /^[A-Za-z]:/.test(s) ? s.toLowerCase() : s;
};

/** True when `path` is inside `<dataDir>/datasets/`, where the server writes the datasets it makes. */
export function writtenByServer(path: string, dataDir: string | null | undefined): boolean {
  return !!dataDir && norm(path).startsWith(`${norm(dataDir)}/datasets/`);
}

const ACTIVE: ReadonlySet<Run['status']> = new Set(['queued', 'running', 'stopping']);

/** "1 run" / "3 runs". */

/** What removing a dataset means for the runs that used it, in a sentence or two; null when no run used it. */
export function runsImpact(used: readonly Pick<Run, 'name' | 'status'>[]): { lost: string; active: string | null } | null {
  if (!used.length) return null;
  const active = used.filter((r) => ACTIVE.has(r.status));
  return {
    lost: `${used.length === 1 ? '1 run trained' : `${count(used.length, 'run')} trained`} on it. ${used.length === 1 ? 'It keeps its' : 'They keep their'} metrics and models, but ${used.length === 1 ? 'its' : 'their'} sample images will not show until the folder is registered again.`,
    active: active.length
      ? `${active.length === 1 ? `${active[0]!.name} is` : `${count(active.length, 'run')} are`} training or queued on it now: ${active.length === 1 ? 'it carries' : 'they carry'} on.`
      : null,
  };
}

export default function RemoveDialog(props: RemoveDialogProps) {
  const open = bindable(props, 'open', false);
  const deleting = signal(false);
  let mounted = true;
  onCleanup(() => (mounted = false));
  /** A failed removal says so in the dialog, which stays open for another try (the toaster sits behind the modal). */
  const status = new DialogStatus();
  /** The server's runs, read each time the dialog opens (a run may have started since). */
  const runs = new Resource(async (s) => (open.get() ? hub.api.runs(s) : null));
  /** The runs that trained on the dataset. */
  const used = computed(() => runs.data?.filter((r) => r.config.dataset_id === props.dataset.id) ?? null);
  const impact = computed(() => (used.get() ? runsImpact(used.get()!) : null));
  /** The server's data dir, to tell a dataset it wrote (its files can go too) from a folder of the person's own. */
  const system = new Resource(async (s) => (open.get() ? hub.api.system(s) : null));
  const ours = computed(() => writtenByServer(props.dataset.path, system.data?.data_dir));
  /** A run on any dataset is queued or training: the server deletes files only while none is (docs/API.md). */
  const busy = computed(() => !!runs.data?.some((r) => ACTIVE.has(r.status)));
  /** Its files can go with it: the server wrote them, and no run is queued or training (the server checks both too). */
  const canDeleteFiles = computed(() => ours.get() && !!runs.data && !busy.get());
  const withFiles = signal(false);
  // Each time it opens, the files stay unless asked again.
  effect(() => {
    if (open.get()) untrack(() => withFiles.set(false));
  });
  const files = () => withFiles.get() && canDeleteFiles.get();
  const images = () => (props.dataset.splits ? count(props.dataset.splits.train + props.dataset.splits.val + props.dataset.splits.test, 'generated image') : 'generated images');

  async function remove() {
    if (deleting.peek()) return;
    const d = props.dataset;
    const epoch = hub.epoch;
    const startPath = router.location.path;
    deleting.set(true);
    try {
      const gone = files();
      await hub.api.deleteDataset(d.id, { files: gone });
      if (hub.epoch !== epoch) return;
      datasetsChanged();
      if (mounted) open.set(false);
      toast.success('Dataset removed', gone ? `${d.name} and its ${images()} were deleted.` : `${d.name} is no longer registered. Files on disk were not touched.`);
      if (mounted && router.location.path === startPath) router.go('/datasets', { replace: true });
    } catch (e) {
      if (hub.epoch !== epoch) return;
      if (mounted && open.peek()) status.error(e, 'Could not remove the dataset');
      else toast.error(e, 'Could not remove the dataset');
    } finally {
      deleting.set(false);
    }
  }

  return (
    <Dialog
      bind:open={open}
      onbeforeclose={() => !deleting.peek()}
      title={`Remove ${props.dataset.name} from FOLO?`}
      size="sm"
      status={status}
      footer={() => (
        <>
          <Button disabled={deleting.get()} onclick={() => !deleting.peek() && open.set(false)}>
            Cancel
          </Button>
          <Button variant="destructive" icon={Trash} loading={deleting.get()} onclick={remove}>
            {files() ? 'Remove and delete files' : 'Remove from FOLO'}
          </Button>
        </>
      )}
    >
      <div class="confirm">
        {files() ? (
          <p>
            <b>{props.dataset.name}</b> will be unregistered, and its folder <span class="mono">{props.dataset.path}</span> deleted.
          </p>
        ) : (
          <p>
            <b>{props.dataset.name}</b> will be unregistered. The files in <span class="mono">{props.dataset.path}</span> stay on disk.
          </p>
        )}
        {impact.get() ? <p>{impact.get()!.lost}</p> : null}
        {canDeleteFiles.get() ? (
          <div class="files">
            <Switch bind:checked={withFiles} label={`Also delete the ${images()}`} hint="FOLO made them in its own data folder. This cannot be undone." />
          </div>
        ) : ours.get() && busy.get() ? (
          <p>Its {images()} can be deleted too once no run is queued or training.</p>
        ) : null}
        {impact.get()?.active ? (
          <p class="active">
            <TriangleAlert size={14} strokeWidth={1.75} aria-hidden="true" />
            <span>{impact.get()!.active}</span>
          </p>
        ) : null}
      </div>
    </Dialog>
  );
}

css`
  .confirm {
    display: grid;
    gap: var(--space-2);
    color: var(--text-secondary);
  }
  .confirm b {
    color: var(--text-primary);
    font-weight: var(--weight-medium);
  }
  .confirm .mono {
    color: var(--text-primary);
    overflow-wrap: anywhere;
  }
  .files {
    padding-top: var(--space-1);
  }
  .active {
    display: flex;
    align-items: flex-start;
    gap: var(--space-2);
    color: var(--text-primary);
  }
  .active :global(svg) {
    flex-shrink: 0;
    margin-top: 2px;
    color: var(--warning-text);
  }
`;
