<script lang="ts">
  /**
   * "New benchmark": a name, what the camera sees and what makes it hard (an agent designing the plan reads it), where
   * the images come from (a dataset's split, or a labeling job's labeled images), and how many at most. It is made
   * with the default plan (every condition at severities 1, 3 and 5), which can change on its page until the first
   * entry. A quiet note says why its images belong out of training, and a warning when the split is a training split.
   * Validation, gating and feedback as the labeling job dialog's.
   */
  import { untrack } from 'svelte';
  import Info from '@lucide/svelte/icons/info';
  import Plus from '@lucide/svelte/icons/plus';
  import TriangleAlert from '@lucide/svelte/icons/triangle-alert';
  import { hub } from '$lib/api/index.svelte';
  import { Resource } from '$lib/api/resource.svelte';
  import type { Split } from '$lib/api/types';
  import { benchNameError, descriptionError, IMAGES_DEFAULT, IMAGES_MAX } from '$lib/bench/rules';
  import { num } from '$lib/format';
  import { router } from '$lib/router.svelte';
  import Button from '$lib/ui/Button.svelte';
  import Dialog from '$lib/ui/Dialog.svelte';
  import ErrorState from '$lib/ui/ErrorState.svelte';
  import Field from '$lib/ui/Field.svelte';
  import Gate from '$lib/ui/Gate.svelte';
  import Input from '$lib/ui/Input.svelte';
  import Segmented from '$lib/ui/Segmented.svelte';
  import Select from '$lib/ui/Select.svelte';
  import TextArea from '$lib/ui/TextArea.svelte';
  import { toast } from '$lib/ui/toast.svelte';
  import { imagesText } from './words';

  let { open = $bindable(false) }: { open?: boolean } = $props();

  const uid = $props.id();
  const datasets = new Resource((s) => hub.api.datasets(s));
  const jobs = new Resource((s) => hub.api.labelJobs(s));

  let name = $state('');
  let description = $state('');
  let kind = $state<'dataset' | 'label_job'>('dataset');
  let datasetId = $state('');
  let split = $state<Split>('val');
  let jobId = $state('');
  let maxText = $state(String(IMAGES_DEFAULT));
  let touched = $state(false);
  let busy = $state(false);
  let error = $state.raw<unknown>(undefined);

  // Every opening starts blank, on the first dataset with validation images (else the first dataset) and its
  // validation split: the split a benchmark is honest on.
  $effect(() => {
    if (!open) return;
    untrack(() => {
      name = '';
      description = '';
      kind = 'dataset';
      maxText = String(IMAGES_DEFAULT);
      touched = false;
      busy = false;
      error = undefined;
      datasetId = '';
      jobId = '';
    });
  });
  $effect(() => {
    const list = datasets.data;
    if (!open || !list?.length || datasetId) return;
    untrack(() => {
      const d = list.find((x) => x.splits.val > 0) ?? list[0]!;
      datasetId = d.id;
      split = d.splits.val > 0 ? 'val' : d.splits.test > 0 ? 'test' : 'train';
    });
  });
  $effect(() => {
    const list = jobs.data;
    if (!open || !list?.length || jobId) return;
    untrack(() => (jobId = (list.find((j) => j.counts.labeled > 0) ?? list[0]!).id));
  });

  const dataset = $derived(datasets.data?.find((d) => d.id === datasetId));
  const job = $derived(jobs.data?.find((j) => j.id === jobId));
  /** Images the source holds (a split's, or a job's labeled ones). */
  const sourceImages = $derived(kind === 'dataset' ? (dataset?.splits[split] ?? 0) : (job?.counts.labeled ?? 0));
  const max = $derived(Number(maxText));
  const taken = $derived(Number.isInteger(max) && max >= 1 ? Math.min(max, sourceImages) : sourceImages);

  const errs = $derived({
    name: benchNameError(name),
    description: descriptionError(description),
    max: !maxText.trim() || !Number.isInteger(max) || max < 1 || max > IMAGES_MAX ? `A whole number from 1 to ${num(IMAGES_MAX)}.` : null,
  });
  const show = (e: string | null) => (touched ? e : null);
  const sourceWhy = $derived(
    kind === 'dataset'
      ? !datasets.data
        ? 'Loading datasets'
        : !dataset
          ? 'Register a dataset first'
          : sourceImages === 0
            ? `Pick a split with images`
            : null
      : jobs.error
        ? 'This server does not label images'
        : !jobs.data
          ? 'Loading labeling jobs'
          : !job
            ? 'Make a labeling job first'
            : sourceImages === 0
              ? 'Label some of its images first'
              : null,
  );
  const missing = $derived(!name.trim() ? 'Enter a name' : sourceWhy);
  const FIX: Record<string, string> = { name: 'Fix the name first', description: 'Shorten the description first', max: 'Fix the image limit first' };
  const shownError = $derived(touched ? Object.entries(errs).find(([, e]) => e)?.[0] : undefined);
  const reason = $derived(missing ?? (shownError ? FIX[shownError] : null));

  const splitOptions = $derived(
    (['val', 'test', 'train'] as const).map((s) => ({
      value: s,
      label: s === 'val' ? 'Validation' : s === 'test' ? 'Test' : 'Training',
      meta: dataset ? num(dataset.splits[s]) : undefined,
      disabled: !!dataset && dataset.splits[s] === 0,
      tip: dataset && dataset.splits[s] === 0 ? `${dataset.name} has no ${s} images` : undefined,
    })),
  );
  const jobOptions = $derived((jobs.data ?? []).map((j) => ({ value: j.id, label: `${j.name} · ${num(j.counts.labeled)} labeled`, short: j.name, disabled: j.counts.labeled === 0 })));
  const sampleHint = $derived(
    !sourceImages
      ? `At most ${num(IMAGES_MAX)}.`
      : taken < sourceImages
        ? `A seeded sample of ${num(taken)} of its ${imagesText(sourceImages)}.`
        : `It takes all ${imagesText(sourceImages)}.`,
  );

  async function submit(e: SubmitEvent) {
    e.preventDefault();
    touched = true;
    error = undefined;
    if (Object.values(errs).some((x) => x) || reason || busy) return;
    busy = true;
    try {
      const created = await hub.api.createBench({
        name: name.trim(),
        description: description.trim(),
        source: kind === 'dataset' ? { kind: 'dataset', dataset_id: datasetId, split } : { kind: 'label_job', job_id: jobId },
        max_images: max,
      });
      // The dialog closes first: what was made is said over the benchmark's page it opens.
      open = false;
      toast.success('Benchmark created', `${created.name} · ${imagesText(created.images)} · the default plan`);
      router.go(`/benchmarks/${created.id}`);
    } catch (err) {
      error = err;
    } finally {
      busy = false;
    }
  }
</script>

<Dialog bind:open title="New benchmark" description="Freeze a set of labeled images to grade models on. It starts with the default plan of conditions, which you can change until the first entry.">
  <form id="{uid}-form" class="form" onsubmit={submit} novalidate>
    <Field label="Name" for="{uid}-name" error={show(errs.name)}>
      <Input id="{uid}-name" bind:value={name} placeholder="e.g. Pond fish" autocomplete="off" invalid={!!show(errs.name)} />
    </Field>
    <Field
      label="What the camera sees"
      for="{uid}-desc"
      error={show(errs.description)}
      hint="Where the camera is and what makes the job hard. An agent reads it when it designs the plan, so the plan fits the camera."
    >
      <TextArea id="{uid}-desc" bind:value={description} rows={3} placeholder="A camera over a garden pond, looking down through murky green water at koi and goldfish." invalid={!!show(errs.description)} />
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
    {#if kind === 'dataset'}
      <div class="pair">
        <Field label="Dataset" for="{uid}-ds">
          <Select id="{uid}-ds" bind:value={datasetId} disabled={!datasets.data?.length} options={(datasets.data ?? []).map((d) => ({ value: d.id, label: d.name }))} />
        </Field>
        <Field label="Split" hint={dataset ? `${imagesText(sourceImages)} in the ${split === 'val' ? 'validation' : split === 'train' ? 'training' : 'test'} split of ${dataset.name}.` : undefined}>
          <Segmented label="Split" bind:value={split} options={splitOptions} />
        </Field>
      </div>
      {#if datasets.error && !datasets.data}<ErrorState error={datasets.error} title="Could not load datasets" retry={() => datasets.refresh()} compact />{/if}
    {:else}
      <Field label="Labeling job" for="{uid}-job" hint={job ? `Its ${imagesText(job.counts.labeled)} with labels, boxes as reviewed; the others are left out.` : undefined}>
        <Select id="{uid}-job" bind:value={jobId} disabled={!jobs.data?.length} options={jobOptions} />
      </Field>
    {/if}
    <Field label="At most" for="{uid}-max" error={show(errs.max)} hint={sampleHint}>
      <Input id="{uid}-max" bind:value={maxText} inputmode="numeric" suffix="images" class="max" invalid={!!show(errs.max)} />
    </Field>
    {#if kind === 'dataset' && split === 'train'}
      <p class="note warn"><TriangleAlert size={14} strokeWidth={1.75} aria-hidden="true" /><span>A model trained on {dataset?.name ?? 'this dataset'}'s training split has seen these images: on them, a benchmark measures memory, not sight. Pick the validation or test split.</span></p>
    {:else}
      <p class="note"><Info size={14} strokeWidth={1.75} aria-hidden="true" /><span>Keep these images out of training. A benchmark of a validation or test split stays honest while runs train on the training split.</span></p>
    {/if}
    {#if error}<ErrorState {error} title="Could not create the benchmark" compact />{/if}
  </form>
  {#snippet footer()}
    {#if missing}<p class="why" data-gate-why>{missing}</p>{/if}
    <Button variant="ghost" onclick={() => (open = false)}>Cancel</Button>
    <Gate {reason}>
      {#snippet children(why)}
        <Button variant="primary" type="submit" form="{uid}-form" icon={Plus} loading={busy} disabled={!!why} aria-describedby={why}>Create benchmark</Button>
      {/snippet}
    </Gate>
  {/snippet}
</Dialog>

<style>
  .form {
    display: grid;
    gap: var(--space-4);
  }
  .form :global(.seg) {
    justify-self: start;
  }
  /* Dataset and split side by side: the split's three choices keep their width, the dataset takes the rest. */
  .pair {
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto;
    gap: var(--space-4);
    align-items: start;
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
</style>
