/**
 * "New benchmark": a name, what the camera sees and what makes it hard (an agent designing the plan reads it), where
 * the images come from (a dataset's split, or a labeling job's labeled images), and how many at most. It is made
 * with the default plan (every condition at severities 1, 3 and 5), which can change on its page until the first
 * entry. A quiet note says why its images belong out of training, and a warning when the split is a training split.
 * Validation, gating and feedback as the labeling job dialog's.
 */
import Info from '$icons/info';
import Plus from '$icons/plus';
import TriangleAlert from '$icons/triangle-alert';
import { bindable, computed, createUniqueId, css, effect, signal, untrack } from 'fw';
import { hub } from '$lib/api/index';
import { Resource } from '$lib/api/resource';
import type { Split } from '$lib/api/types';
import { benchNameError, descriptionError, IMAGES_DEFAULT, IMAGES_MAX } from '$lib/bench/rules';
import { num } from '$lib/format';
import { router } from '$lib/router';
import Button from '$lib/ui/Button';
import Dialog from '$lib/ui/Dialog';
import ErrorState from '$lib/ui/ErrorState';
import Field from '$lib/ui/Field.tsx';
import Gate from '$lib/ui/Gate';
import Input from '$lib/ui/Input';
import Segmented from '$lib/ui/Segmented';
import Select from '$lib/ui/Select';
import TextArea from '$lib/ui/TextArea';
import { toast } from '$lib/ui/toast';
import { imagesText, unreviewedText } from './words';

const FIX: Record<string, string> = { name: 'Fix the name first', description: 'Shorten the description first', max: 'Fix the image limit first' };

/**
 * The labeled images of job `id` that are not reviewed, read 1000 at a time; null when the count of labeled images
 * changed between pages (an image labeled, removed or relabeled meanwhile shifts the offsets: a page could be skipped).
 */
async function countUnreviewed(id: string, s: AbortSignal): Promise<number | null> {
  let n = 0;
  let total: number | null = null;
  for (let offset = 0; ; ) {
    const page = await hub.api.labelImages(id, { status: 'labeled', offset, limit: 1000 }, s);
    if (total !== null && page.total !== total) return null;
    total = page.total;
    n += page.images.filter((i) => !i.reviewed).length;
    offset += page.images.length;
    if (!page.images.length || offset >= page.total) return offset >= page.total ? n : null;
  }
}

export default function NewBenchDialog(props: { open?: boolean }) {
  const open = bindable(props, 'open', false);

  const uid = createUniqueId();
  const datasets = new Resource((s) => hub.api.datasets(s));
  const jobs = new Resource((s) => hub.api.labelJobs(s));

  const name = signal('');
  const description = signal('');
  const kind = signal<'dataset' | 'label_job'>('dataset');
  const datasetId = signal('');
  const split = signal<Split>('val');
  const jobId = signal('');
  const maxText = signal(String(IMAGES_DEFAULT));
  const touched = signal(false);
  const busy = signal(false);
  const error = signal<unknown>(undefined);

  // Every opening starts blank, on the first dataset with validation images (else the first dataset) and its
  // validation split: the split a benchmark is honest on.
  effect(() => {
    if (!open.get()) return;
    untrack(() => {
      name.set('');
      description.set('');
      kind.set('dataset');
      maxText.set(String(IMAGES_DEFAULT));
      touched.set(false);
      busy.set(false);
      error.set(undefined);
      datasetId.set('');
      jobId.set('');
    });
  });
  effect(() => {
    const list = datasets.data;
    if (!open.get() || !list?.length || datasetId.get()) return;
    untrack(() => {
      const d = list.find((x) => x.splits.val > 0) ?? list[0]!;
      datasetId.set(d.id);
      split.set(d.splits.val > 0 ? 'val' : d.splits.test > 0 ? 'test' : 'train');
    });
  });
  effect(() => {
    const list = jobs.data;
    if (!open.get() || !list?.length || jobId.get()) return;
    untrack(() => jobId.set((list.find((j) => j.counts.labeled > 0) ?? list[0]!).id));
  });

  const dataset = computed(() => datasets.data?.find((d) => d.id === datasetId.get()));
  const job = computed(() => jobs.data?.find((j) => j.id === jobId.get()));
  // How many of the job's labeled images no person has checked (`LabelImage.reviewed`): the boxes on them become the
  // benchmark's truth as an agent drew them. Read a page of 1000 at a time, with the job it counts (a Resource keeps
  // its last data while it reloads, so a count shows only for its own job); null for a dataset or a count that cannot
  // be trusted (the job's labeled images changed under the paging three times running).
  const unreviewed = new Resource(async (s): Promise<{ job: string; n: number } | null> => {
    const id = jobId.get();
    if (kind.get() !== 'label_job' || !id) return null;
    for (let attempt = 0; attempt < 3; attempt++) {
      const n = await countUnreviewed(id, s);
      if (n !== null) return { job: id, n };
    }
    return null;
  });
  const unreviewedHere = computed(() => {
    const u = unreviewed.data;
    return u && !unreviewed.loading && !unreviewed.error && u.job === jobId.get() ? u.n : 0;
  });
  /** Images the source holds (a split's, or a job's labeled ones). */
  const sourceImages = computed(() => (kind.get() === 'dataset' ? (dataset.get()?.splits[split.get()] ?? 0) : (job.get()?.counts.labeled ?? 0)));
  const max = computed(() => Number(maxText.get()));
  const taken = computed(() => {
    const m = max.get();
    return Number.isInteger(m) && m >= 1 ? Math.min(m, sourceImages.get()) : sourceImages.get();
  });

  const errs = computed(() => {
    const m = max.get();
    return {
      name: benchNameError(name.get()),
      description: descriptionError(description.get()),
      max: !maxText.get().trim() || !Number.isInteger(m) || m < 1 || m > IMAGES_MAX ? `A whole number from 1 to ${num(IMAGES_MAX)}.` : null,
    };
  });
  const show = (e: string | null) => (touched.get() ? e : null);
  /** A number field says a value out of range as soon as it is typed: the hint that reads fine for it would mislead. */
  const maxError = computed(() => (touched.get() || maxText.get().trim() ? errs.get().max : null));
  const sourceWhy = computed(() =>
    kind.get() === 'dataset'
      ? !datasets.data
        ? 'Loading datasets'
        : !dataset.get()
          ? 'Register a dataset first'
          : sourceImages.get() === 0
            ? `Pick a split with images`
            : null
      : jobs.error
        ? 'This server does not label images'
        : !jobs.data
          ? 'Loading labeling jobs'
          : !job.get()
            ? 'Make a labeling job first'
            : sourceImages.get() === 0
              ? 'Label some of its images first'
              : null,
  );
  const missing = computed(() => (!name.get().trim() ? 'Enter a name' : sourceWhy.get()));
  const shownError = computed(() => (touched.get() ? Object.entries(errs.get()).find(([, e]) => e)?.[0] : maxError.get() ? 'max' : undefined));
  const reason = computed(() => {
    const se = shownError.get();
    return missing.get() ?? (se ? FIX[se] : null);
  });

  const splitOptions = computed(() => {
    const ds = dataset.get();
    return (['val', 'test', 'train'] as const).map((s) => ({
      value: s,
      label: s === 'val' ? 'Validation' : s === 'test' ? 'Test' : 'Training',
      meta: ds ? num(ds.splits[s]) : undefined,
      disabled: !!ds && ds.splits[s] === 0,
      tip: ds && ds.splits[s] === 0 ? `${ds.name} has no ${s} images` : undefined,
    }));
  });
  const jobOptions = computed(() =>
    (jobs.data ?? []).map((j) => ({ value: j.id, label: `${j.name} · ${num(j.counts.labeled)} labeled`, short: j.name, disabled: j.counts.labeled === 0 })),
  );
  const sampleHint = computed(() => {
    const src = sourceImages.get();
    const t = taken.get();
    return !src ? `At most ${num(IMAGES_MAX)}.` : t < src ? `A seeded sample of ${num(t)} of its ${imagesText(src)}.` : `It takes all ${imagesText(src)}.`;
  });

  async function submit(e: SubmitEvent) {
    e.preventDefault();
    touched.set(true);
    error.set(undefined);
    if (Object.values(errs.peek()).some((x) => x) || reason.peek() || busy.peek()) return;
    busy.set(true);
    try {
      const created = await hub.api.createBench({
        name: name.peek().trim(),
        description: description.peek().trim(),
        source: kind.peek() === 'dataset' ? { kind: 'dataset', dataset_id: datasetId.peek(), split: split.peek() } : { kind: 'label_job', job_id: jobId.peek() },
        max_images: max.peek(),
      });
      // The dialog closes first: what was made is said over the benchmark's page it opens.
      open.set(false);
      toast.success('Benchmark created', `${created.name} · ${imagesText(created.images)} · the default plan`);
      router.go(`/benchmarks/${created.id}`);
    } catch (err) {
      error.set(err);
    } finally {
      busy.set(false);
    }
  }

  return (
    <Dialog
      bind:open={open}
      title="New benchmark"
      description="Freeze a set of labeled images to grade models on. It starts with the default plan of conditions, which you can change until the first entry."
      footer={() => (
        <>
          {missing.get() ? (
            <p class="why" data-gate-why>
              {missing.get()}
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
                icon={Plus}
                loading={busy.get()}
                disabled={!!why()}
                aria-describedby={why()}
              >
                Create benchmark
              </Button>
            )}
          </Gate>
        </>
      )}
    >
      <form id={`${uid}-form`} class="form" onsubmit={submit} novalidate>
        <Field label="Name" for={`${uid}-name`} error={show(errs.get().name)}>
          <Input id={`${uid}-name`} bind:value={name} placeholder="e.g. Pond fish" autocomplete="off" invalid={!!show(errs.get().name)} />
        </Field>
        <Field
          label="What the camera sees"
          for={`${uid}-desc`}
          error={show(errs.get().description)}
          hint="Where the camera is and what makes the job hard. An agent reads it when it designs the plan, so the plan fits the camera."
        >
          <TextArea
            id={`${uid}-desc`}
            bind:value={description}
            rows={3}
            placeholder="A camera over a garden pond, looking down through murky green water at koi and goldfish."
            invalid={!!show(errs.get().description)}
          />
        </Field>
        <Field label="Images from">
          <Segmented
            label="Images from"
            bind:value={kind}
            options={[
              { value: 'dataset', label: 'A dataset split' },
              { value: 'label_job', label: 'A labeling job', disabled: !!jobs.error, tip: jobs.error ? 'This server does not label images' : undefined },
            ]}
          />
        </Field>
        {kind.get() === 'dataset' ? (
          <>
            <div class="pair">
              <Field label="Dataset" for={`${uid}-ds`}>
                <Select
                  id={`${uid}-ds`}
                  bind:value={datasetId}
                  disabled={!datasets.data?.length}
                  options={(datasets.data ?? []).map((d) => ({ value: d.id, label: d.name }))}
                />
              </Field>
              <Field
                label="Split"
                hint={
                  dataset.get()
                    ? `${imagesText(sourceImages.get())} in the ${split.get() === 'val' ? 'validation' : split.get() === 'train' ? 'training' : 'test'} split of ${dataset.get()!.name}.`
                    : undefined
                }
              >
                <Segmented label="Split" bind:value={split} options={splitOptions.get()} />
              </Field>
            </div>
            {datasets.error && !datasets.data ? (
              <ErrorState error={datasets.error} title="Could not load datasets" retry={() => datasets.refresh()} compact />
            ) : null}
          </>
        ) : (
          <Field
            label="Labeling job"
            for={`${uid}-job`}
            hint={job.get() ? `Its ${imagesText(job.get()!.counts.labeled)} with labels; the others are left out.` : undefined}
          >
            <Select id={`${uid}-job`} bind:value={jobId} disabled={!jobs.data?.length} options={jobOptions.get()} />
          </Field>
        )}
        <Field label="At most" for={`${uid}-max`} error={maxError.get()} hint={sampleHint.get()}>
          <Input id={`${uid}-max`} bind:value={maxText} inputmode="numeric" suffix="images" class="max" invalid={!!maxError.get()} />
        </Field>
        {kind.get() === 'dataset' && split.get() === 'train' ? (
          <p class="note warn">
            <TriangleAlert size={14} strokeWidth={1.75} aria-hidden="true" />
            <span>A model trained on {dataset.get()?.name ?? 'this dataset'}'s training split has seen these images: on them, a benchmark measures memory, not sight. Pick the validation or test split.</span>
          </p>
        ) : kind.get() === 'label_job' && job.get() && unreviewedHere.get() ? (
          <p class="note warn" data-unreviewed>
            <TriangleAlert size={14} strokeWidth={1.75} aria-hidden="true" />
            <span>
              {unreviewedText(unreviewedHere.get(), job.get()!.counts.labeled)}
            </span>
          </p>
        ) : kind.get() === 'label_job' ? (
          <p class="note">
            <Info size={14} strokeWidth={1.75} aria-hidden="true" />
            <span>Keep these images out of training: a model trained on a dataset made from this job has seen them.</span>
          </p>
        ) : (
          <p class="note">
            <Info size={14} strokeWidth={1.75} aria-hidden="true" />
            <span>Keep these images out of training. A benchmark of a validation or test split stays honest while runs train on the training split.</span>
          </p>
        )}
        {error.get() ? <ErrorState error={error.get()} title="Could not create the benchmark" compact /> : null}
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
  /* Dataset and split side by side: the split's three choices keep their width, the dataset takes the rest (the
     split's hint wraps under its choices rather than widening their column). */
  .pair {
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto;
    gap: var(--space-4);
    align-items: start;
  }
  .pair :global(.field:last-child .msg) {
    width: 0;
    min-width: 100%;
  }
  .form :global(.max) {
    max-width: 160px;
  }
  .note {
    display: grid;
    grid-template-columns: 14px minmax(0, 1fr);
    gap: var(--space-2);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
  }
  .note :global(svg) {
    margin-top: 1px;
    color: var(--text-tertiary);
  }
  .note.warn :global(svg) {
    color: var(--warning-solid);
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
    .pair {
      grid-template-columns: minmax(0, 1fr);
    }
  }
`;
