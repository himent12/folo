/**
 * "New benchmark": a name, what the camera sees and what makes it hard (an agent designing the plan reads it), where
 * the images come from (a dataset's split, or a labeling job's labeled images), and how many at most. It is made
 * with the default plan (every condition at severities 1, 3 and 5), which can change on its page until the first
 * entry. A quiet note says why its images belong out of training, and a warning when the split is a training split.
 * Validation, gating and feedback as the labeling job dialog's.
 */
import Info from '$icons/info';
import Plus from '$icons/plus';
import Sparkles from '$icons/sparkles';
import TriangleAlert from '$icons/triangle-alert';
import { bindable, computed, createUniqueId, css, effect, onCleanup, signal, untrack } from 'fw';
import { hub } from '$lib/api/index';
import { Resource } from '$lib/api/resource';
import type { CreateBench, Dataset, LabelJob, Split, SystemInfo } from '$lib/api/types';
import {
  BENCH_GENERATED_FEATURE,
  benchNameError,
  descriptionError,
  GENERATED_CLASSES_DEFAULT,
  GENERATED_CLASSES_MAX,
  GENERATED_IMAGES_DEFAULT,
  GENERATED_IMGSZ_DEFAULT,
  generatedShapeCollision,
  IMAGES_DEFAULT,
  IMAGES_MAX,
} from '$lib/bench/rules';
import { num } from '$lib/format';
import { LABEL_FEATURE } from '$lib/label';
import { router } from '$lib/router';
import { parseSeed, seedArrows, seedError } from '$lib/seed';
import Button from '$lib/ui/Button';
import ClassChips, { addClasses, classesLeftOut } from '$lib/ui/ClassChips';
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

const FIX: Record<string, string> = {
  name: 'Fix the name first',
  description: 'Shorten the description first',
  max: 'Fix the image limit first',
  genImages: 'Fix the scene count first',
  genClasses: 'Add at least one class',
  genSeed: 'Fix the seed first',
};

const GEN_SIZES = ['320', '480', '640', '1024'] as const;
type GenSize = (typeof GEN_SIZES)[number];

/**
 * The labeled images of job `id` that are not reviewed, read 1000 at a time; null when the count of labeled images
 * changed between pages (an image labeled, removed or relabeled meanwhile shifts the offsets: a page could be skipped).
 */
async function countUnreviewed(id: string, s: AbortSignal): Promise<{ n: number; labeled: number } | null> {
  let n = 0;
  let total: number | null = null;
  for (let offset = 0; ; ) {
    const page = await hub.api.labelImages(id, { status: 'labeled', offset, limit: 1000 }, s);
    if (total !== null && page.total !== total) return null;
    total = page.total;
    n += page.images.filter((i) => !i.reviewed).length;
    offset += page.images.length;
    if (!page.images.length || offset >= page.total) return offset >= page.total ? { n, labeled: page.total } : null;
  }
}

export default function NewBenchDialog(props: { open?: boolean }) {
  const open = bindable(props, 'open', false);

  const uid = createUniqueId();
  // Memoized boolean so `hub.features` flipping to `null` during an outage does not trigger unnecessary refetches.
  const noLabel = computed(() => hub.lacks(LABEL_FEATURE));
  // Defer source lists until the dialog actually opens, and skip labeling jobs when the server lacks agent labeling.
  const datasets = new Resource<Dataset[] | undefined>((s) => (open.get() ? hub.api.datasets(s) : Promise.resolve(undefined)));
  const jobs = new Resource<LabelJob[] | undefined>((s) => (open.get() && !noLabel.get() ? hub.api.labelJobs(s) : Promise.resolve(undefined)));
  const system = new Resource<SystemInfo | undefined>((s) => (open.get() && hub.features === null ? hub.api.system(s) : Promise.resolve(undefined)));
  const labelOff = computed(() => noLabel.get() || !!jobs.error);
  const generatedOff = computed(() => {
    if (hub.features !== null) return !hub.features.includes(BENCH_GENERATED_FEATURE);
    return !system.data?.features?.includes(BENCH_GENERATED_FEATURE);
  });

  const name = signal('');
  const description = signal('');
  const kind = signal<'dataset' | 'label_job' | 'generated'>('dataset');
  const datasetId = signal('');
  const split = signal<Split>('val');
  const jobId = signal('');
  const maxText = signal(String(IMAGES_DEFAULT));
  const genImagesText = signal(String(GENERATED_IMAGES_DEFAULT));
  const genClasses = signal<string[]>([...GENERATED_CLASSES_DEFAULT]);
  const genClassDraft = signal('');
  const genLeftOut = signal<string[]>([]);
  const genImgsz = signal<GenSize>(String(GENERATED_IMGSZ_DEFAULT) as GenSize);
  const genSeed = signal('0');
  const varyAspect = signal(false);
  const touched = signal(false);
  const busy = signal(false);
  const error = signal<unknown>(undefined);

  let alive = true;
  let submitSeq = 0;
  onCleanup(() => {
    alive = false;
    submitSeq++;
  });

  // Every opening starts blank, on the first dataset with validation images (else the first dataset) and its
  // validation split: the split a benchmark is honest on.
  effect(() => {
    if (!open.get()) return;
    untrack(() => {
      submitSeq++;
      name.set('');
      description.set('');
      kind.set('dataset');
      maxText.set(String(IMAGES_DEFAULT));
      genImagesText.set(String(GENERATED_IMAGES_DEFAULT));
      genClasses.set([...GENERATED_CLASSES_DEFAULT]);
      genClassDraft.set('');
      genLeftOut.set([]);
      genImgsz.set(String(GENERATED_IMGSZ_DEFAULT) as GenSize);
      genSeed.set('0');
      varyAspect.set(false);
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
  const unreviewed = new Resource(async (s): Promise<{ job: string; n: number; labeled: number } | null> => {
    const id = jobId.get();
    if (kind.get() !== 'label_job' || !id) return null;
    for (let attempt = 0; attempt < 3; attempt++) {
      const c = await countUnreviewed(id, s);
      if (c !== null) return { job: id, ...c };
    }
    return null;
  });
  const unreviewedHere = computed(() => {
    const u = unreviewed.data;
    // With the labeled count of the same read: the job's list may be older (an agent labeled more since).
    return u && !unreviewed.loading && !unreviewed.error && u.job === jobId.get() && u.n ? u : null;
  });
  /** Images the source holds (a split's, or a job's labeled ones). */
  const sourceImages = computed(() => (kind.get() === 'dataset' ? (dataset.get()?.splits[split.get()] ?? 0) : kind.get() === 'label_job' ? (job.get()?.counts.labeled ?? 0) : 0));
  const max = computed(() => Number(maxText.get()));
  const genImages = computed(() => Number(genImagesText.get()));
  const taken = computed(() => {
    const m = max.get();
    return Number.isInteger(m) && m >= 1 ? Math.min(m, sourceImages.get()) : sourceImages.get();
  });

  const pendingLeftOut = computed(() => {
    const draft = genClassDraft.get();
    return draft.trim() ? classesLeftOut(genClasses.get(), draft, GENERATED_CLASSES_MAX) : [];
  });
  // Clear a previous left-out notice once the user removes a chip to make room or switches away from generated scenes.
  effect(() => {
    if (kind.get() !== 'generated' || genClasses.get().length < GENERATED_CLASSES_MAX) {
      untrack(() => {
        if (genLeftOut.peek().length) genLeftOut.set([]);
      });
    }
  });

  const errs = computed(() => {
    const k = kind.get();
    if (k === 'generated') {
      const gi = genImages.get();
      const pending = genClassDraft.get().trim();
      const effectiveClasses = pending ? addClasses(genClasses.get(), pending, GENERATED_CLASSES_MAX) : genClasses.get();
      const dropped = pendingLeftOut.get().length ? pendingLeftOut.get() : genLeftOut.get();
      const shapeDup = generatedShapeCollision(effectiveClasses);
      const genClassesErr = dropped.length
        ? `At most ${GENERATED_CLASSES_MAX} classes: ${dropped.join(', ')} did not fit.`
        : effectiveClasses.length === 0
          ? 'Add at least one class.'
          : shapeDup
            ? `Two classes claim the built-in "${shapeDup}" shape.`
            : null;
      return {
        name: benchNameError(name.get()),
        description: descriptionError(description.get()),
        max: null,
        genImages: !genImagesText.get().trim() || !Number.isInteger(gi) || gi < 1 || gi > IMAGES_MAX ? `A whole number from 1 to ${num(IMAGES_MAX)}.` : null,
        genClasses: genClassesErr,
        genSeed: seedError(genSeed.get()),
      };
    }
    const m = max.get();
    return {
      name: benchNameError(name.get()),
      description: descriptionError(description.get()),
      max: !maxText.get().trim() || !Number.isInteger(m) || m < 1 || m > IMAGES_MAX ? `A whole number from 1 to ${num(IMAGES_MAX)}.` : null,
      genImages: null,
      genClasses: null,
      genSeed: null,
    };
  });
  const show = (e: string | null) => (touched.get() ? e : null);
  /** A number field says a value out of range as soon as it is typed: the hint that reads fine for it would mislead. */
  const maxError = computed(() => (touched.get() || maxText.get().trim() ? errs.get().max : null));
  const genImagesError = computed(() => (touched.get() || genImagesText.get().trim() ? errs.get().genImages : null));
  const genSeedError = computed(() => (touched.get() || genSeed.get().trim() ? errs.get().genSeed : null));
  /** Surface dropped classes (>8) or shape collisions immediately, not only after Submit. */
  const genClassesError = computed(() =>
    touched.get() || pendingLeftOut.get().length || genLeftOut.get().length || generatedShapeCollision(genClasses.get()) ? errs.get().genClasses : null,
  );
  const sourceWhy = computed(() =>
    kind.get() === 'generated'
      ? generatedOff.get()
        ? 'This server does not support generated benchmarks'
        : null
      : kind.get() === 'dataset'
        ? !datasets.data
          ? 'Loading datasets'
          : !dataset.get()
            ? 'Register a dataset first'
            : sourceImages.get() === 0
              ? `Pick a split with images`
              : null
        : labelOff.get()
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
  const shownError = computed(() =>
    touched.get()
      ? Object.entries(errs.get()).find(([, e]) => e)?.[0]
      : kind.get() === 'generated'
        ? genClassesError.get()
          ? 'genClasses'
          : genImagesError.get()
            ? 'genImages'
            : genSeedError.get()
              ? 'genSeed'
              : undefined
        : maxError.get()
          ? 'max'
          : undefined,
  );
  const connecting = computed(() => hub.mode === 'connecting');
  const reason = computed(() => {
    if (connecting.get()) return 'Connecting to server';
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
    if (kind.peek() === 'generated' && genClassDraft.peek().trim()) {
      const leftOut = classesLeftOut(genClasses.peek(), genClassDraft.peek(), GENERATED_CLASSES_MAX);
      if (leftOut.length) {
        genLeftOut.set(leftOut);
        touched.set(true);
        return;
      }
      genClasses.set(addClasses(genClasses.peek(), genClassDraft.peek(), GENERATED_CLASSES_MAX));
      genClassDraft.set('');
      genLeftOut.set([]);
    }
    touched.set(true);
    error.set(undefined);
    if (Object.values(errs.peek()).some((x) => x) || reason.peek() || busy.peek() || connecting.peek()) return;
    const seq = ++submitSeq;
    const epoch = hub.epoch;
    const startPath = router.location.path;
    const k = kind.peek();
    const body: CreateBench =
      k === 'generated'
        ? {
            name: name.peek().trim(),
            description: description.peek().trim(),
            source: {
              kind: 'generated',
              images: genImages.peek(),
              classes: genClasses.peek(),
              imgsz: Number(genImgsz.peek()),
              seed: parseSeed(genSeed.peek()) ?? 0,
              vary_aspect: varyAspect.peek(),
            },
          }
        : {
            name: name.peek().trim(),
            description: description.peek().trim(),
            source: k === 'dataset' ? { kind: 'dataset', dataset_id: datasetId.peek(), split: split.peek() } : { kind: 'label_job', job_id: jobId.peek() },
            max_images: max.peek(),
          };
    busy.set(true);
    try {
      const created = await hub.api.createBench(body);
      if (hub.epoch !== epoch || connecting.peek()) return;
      // The dialog closes first: what was made is said over the benchmark's page it opens.
      if (alive && seq === submitSeq) open.set(false);
      toast.success('Benchmark created', `${created.name} · ${imagesText(created.images)} · the default plan`);
      if (alive && seq === submitSeq && router.location.path === startPath) router.go(`/benchmarks/${created.id}`);
    } catch (err) {
      if (!alive || seq !== submitSeq || hub.epoch !== epoch || connecting.peek()) return;
      error.set(err);
    } finally {
      if (alive && seq === submitSeq) busy.set(false);
    }
  }

  return (
    <Dialog
      bind:open={open}
      onbeforeclose={() => !busy.peek()}
      title="New benchmark"
      description="Freeze a set of labeled images to grade models on. It starts with the default plan of conditions, which you can change until the first entry."
      footer={() => (
        <>
          {missing.get() ? (
            <p class="why" data-gate-why>
              {missing.get()}
            </p>
          ) : null}
          <Button variant="ghost" disabled={busy.get()} onclick={() => !busy.peek() && open.set(false)}>
            Cancel
          </Button>
          <Gate reason={reason.get()}>
            {(why) => (
              <Button
                variant="primary"
                type="submit"
                form={`${uid}-form`}
                icon={kind.get() === 'generated' ? Sparkles : Plus}
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
              { value: 'label_job', label: 'A labeling job', disabled: labelOff.get(), tip: labelOff.get() ? 'This server does not label images' : undefined },
              { value: 'generated', label: 'Generated scenes', disabled: generatedOff.get(), tip: generatedOff.get() ? 'This server does not support generated benchmarks' : undefined },
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
        ) : kind.get() === 'label_job' ? (
          <Field
            label="Labeling job"
            for={`${uid}-job`}
            hint={job.get() ? `Its ${imagesText(job.get()!.counts.labeled)} with labels; the others are left out.` : undefined}
          >
            <Select id={`${uid}-job`} bind:value={jobId} disabled={!jobs.data?.length} options={jobOptions.get()} />
          </Field>
        ) : (
          <>
            <Field
              label="Classes"
              for={`${uid}-gcls`}
              error={genClassesError.get()}
              hint={
                genClasses.get().length >= GENERATED_CLASSES_MAX
                  ? `${GENERATED_CLASSES_MAX} of ${GENERATED_CLASSES_MAX}: one shape each.`
                  : `Press Enter or comma to add. Up to ${GENERATED_CLASSES_MAX} distinct classes.`
              }
            >
              <ClassChips
                id={`${uid}-gcls`}
                bind:classes={genClasses}
                bind:draft={genClassDraft}
                max={GENERATED_CLASSES_MAX}
                placeholder="circle, square, …"
                invalid={!!genClassesError.get()}
                onleftout={(names) => genLeftOut.set(names)}
              />
            </Field>
            <div class="gen-row">
              <Field label="Scenes" for={`${uid}-gimg`} error={genImagesError.get()} hint={`1 to ${num(IMAGES_MAX)} scenes.`}>
                <Input id={`${uid}-gimg`} bind:value={genImagesText} inputmode="numeric" suffix="images" invalid={!!genImagesError.get()} />
              </Field>
              <Field label="Seed" for={`${uid}-gseed`} error={genSeedError.get()} hint="Same seed produces byte-identical scenes.">
                <Input id={`${uid}-gseed`} inputmode="numeric" bind:value={genSeed} onkeydown={seedArrows(genSeed)} invalid={!!genSeedError.get()} />
              </Field>
            </div>
            <Field label="Image size" hint={varyAspect.get() ? `Long side ${genImgsz.get()} px with mixed aspect ratios.` : `Square images, ${genImgsz.get()} × ${genImgsz.get()} px.`}>
              <Segmented
                label="Image size"
                bind:value={genImgsz}
                options={GEN_SIZES.map((s) => ({ value: s, label: s }))}
              />
            </Field>
            <label class="aspect-toggle">
              <input type="checkbox" bind:checked={varyAspect} />
              <span>Mix aspect ratios (1:1, 4:3, 3:4, 16:9, 3:2)</span>
            </label>
          </>
        )}
        {kind.get() !== 'generated' ? (
          <Field label="At most" for={`${uid}-max`} error={maxError.get()} hint={sampleHint.get()}>
            <Input id={`${uid}-max`} bind:value={maxText} inputmode="numeric" suffix="images" class="max" invalid={!!maxError.get()} />
          </Field>
        ) : null}
        {kind.get() === 'generated' ? (
          <p class="note">
            <Info size={14} strokeWidth={1.75} aria-hidden="true" />
            <span>Renders deterministic synthetic shape scenes directly into this benchmark without creating a dataset.</span>
          </p>
        ) : kind.get() === 'dataset' && split.get() === 'train' ? (
          <p class="note warn">
            <TriangleAlert size={14} strokeWidth={1.75} aria-hidden="true" />
            <span>A model trained on {dataset.get()?.name ?? 'this dataset'}'s training split has seen these images: on them, a benchmark measures memory, not sight. Pick the validation or test split.</span>
          </p>
        ) : kind.get() === 'label_job' && job.get() && unreviewedHere.get() ? (
          <p class="note warn" data-unreviewed>
            <TriangleAlert size={14} strokeWidth={1.75} aria-hidden="true" />
            <span>
              {unreviewedText(unreviewedHere.get()!.n, unreviewedHere.get()!.labeled)}
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
  .gen-row {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: var(--space-4);
    align-items: start;
  }
  .aspect-toggle {
    display: inline-flex;
    align-items: center;
    gap: var(--space-2);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
    cursor: pointer;
    justify-self: start;
  }
  .aspect-toggle input {
    accent-color: var(--accent-solid);
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
    .pair,
    .gen-row {
      grid-template-columns: minmax(0, 1fr);
    }
  }
`;
