/**
 * "Create dataset": the job's labeled images written as a YOLO dataset that trains like any other. Says plainly what
 * goes in and what is left out; the name follows the Datasets rules (one click to the nearest valid one).
 */
import Database from '$icons/database';
import { bindable, computed, createUniqueId, css, effect, signal, untrack } from 'fw';
import { hub } from '$lib/api/index';
import type { Dataset, LabelJob } from '$lib/api/types';
import { num } from '$lib/format';
import Button from '$lib/ui/Button';
import Dialog from '$lib/ui/Dialog';
import Field from '$lib/ui/Field.tsx';
import { DialogStatus } from '$lib/ui/feedback';
import Gate from '$lib/ui/Gate';
import Input from '$lib/ui/Input';
import Segmented from '$lib/ui/Segmented';
import { NAME_RE, suggestName } from '../datasets/names';
import { images } from './words';

export interface DatasetDialogProps {
  open?: boolean;
  job: LabelJob;
  oncreated: (ds: Dataset) => void;
}

export default function DatasetDialog(props: DatasetDialogProps) {
  const open = bindable(props, 'open', false);
  const uid = createUniqueId();
  const status = new DialogStatus();
  const name = signal('');
  const val = signal<'0.1' | '0.2' | '0.3'>('0.2');
  const busy = signal(false);
  const touched = signal(false);

  // Every opening (or another job) starts afresh; a change of the job while it is open (a run's progress) keeps what was
  // typed, and a submit in flight.
  const jobId = computed(() => props.job.id);
  effect(() => {
    if (!open.get()) return;
    void jobId.get();
    untrack(() => {
      const job = props.job;
      name.set(NAME_RE.test(job.name) ? job.name : (suggestName(job.name) ?? ''));
      val.set('0.2');
      touched.set(false);
      busy.set(false);
    });
  });

  const nameError = computed(() => {
    const n = name.get();
    return !n.trim() ? 'Required.' : NAME_RE.test(n.trim()) ? null : 'Letters, digits, ".", "_" and "-" only (max 64).';
  });
  const fixed = computed(() => (nameError.get() && name.get().trim() ? suggestName(name.get()) : null));
  const labeled = computed(() => props.job.counts.labeled);
  const left = computed(() => props.job.counts.pending + props.job.counts.failed);
  const valImages = computed(() => Math.min(Math.max(0, labeled.get() - 1), Math.round(labeled.get() * Number(val.get()))));
  const reason = computed(() => (!name.get().trim() ? 'Enter a name' : touched.get() && nameError.get() ? 'Fix the name first' : null));

  async function submit(e: SubmitEvent) {
    e.preventDefault();
    touched.set(true);
    if (nameError.peek() || busy.peek()) return;
    busy.set(true);
    status.clear();
    try {
      const ds = await hub.api.createLabelDataset(props.job.id, { name: name.peek().trim(), val_fraction: Number(val.peek()) });
      open.set(false);
      props.oncreated(ds);
    } catch (err) {
      status.error(err, 'Could not create the dataset');
    } finally {
      busy.set(false);
    }
  }

  return (
    <Dialog
      bind:open={open}
      title="Create dataset"
      description="The labeled images become a YOLO dataset you can train on. The job stays, so you can keep labeling."
      status={status}
      footer={() => (
        <>
          {!name.get().trim() ? (
            <p class="why" data-gate-why>
              Enter a name
            </p>
          ) : null}
          <Button variant="ghost" onclick={() => open.set(false)}>
            Cancel
          </Button>
          <Gate reason={reason.get()}>
            {(why) => (
              <Button
                variant="primary"
                type="submit"
                form={`${uid}-form`}
                icon={Database}
                loading={busy.get()}
                disabled={!!why()}
                aria-describedby={why()}
              >
                Create dataset
              </Button>
            )}
          </Gate>
        </>
      )}
    >
      <form id={`${uid}-form`} class="form" onsubmit={submit} novalidate>
        <Field
          label="Name"
          for={`${uid}-name`}
          error={touched.get() ? nameError.get() : null}
          fix={touched.get() && fixed.get() ? { label: `Try ${fixed.get()}`, apply: () => name.set(fixed.peek()!) } : null}
        >
          <Input id={`${uid}-name`} bind:value={name} autocomplete="off" spellcheck={false} invalid={touched.get() && !!nameError.get()} />
        </Field>
        <Field label="Validation share" hint={`About ${images(labeled.get() - valImages.get())} to train on and ${images(valImages.get())} to check the model with.`}>
          <Segmented
            label="Validation share"
            bind:value={val}
            options={[
              { value: '0.1', label: '10%' },
              { value: '0.2', label: '20%' },
              { value: '0.3', label: '30%' },
            ]}
          />
        </Field>
        <p class="facts">
          {images(labeled.get())} with labels, {num(props.job.counts.boxes)} {props.job.counts.boxes === 1 ? 'box' : 'boxes'}.
          {left.get() ? (
            <>
              {' '}
              {images(left.get())} without labels {left.get() === 1 ? 'is' : 'are'} left out.
            </>
          ) : null}
        </p>
      </form>
    </Dialog>
  );
}

css`
  .form {
    display: grid;
    gap: var(--space-4);
  }
  .form :global(.seg) {
    justify-self: start;
  }
  .facts {
    color: var(--text-secondary);
  }
  .why {
    margin-right: auto;
    align-self: center;
    font-size: var(--type-sm-size);
    color: var(--text-secondary);
  }
  @media (max-width: 767px) {
    .why {
      flex-basis: 100%;
    }
  }
`;
