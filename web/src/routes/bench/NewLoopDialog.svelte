<script lang="ts">
  /**
   * "New loop" (`POST /api/bench/benchmarks/{id}/loops`, docs/BENCH.md, Self-improvement loops): does a model's own
   * labeling train a better model? The form: the first teacher (a model whose classes are the benchmark's, same names in
   * the same order; the others listed, off, with the reason), the pool it labels (a dataset split, `train` by default, or
   * a labeling job's images; its own labels are never read and images also in the benchmark are left out), how many
   * generations, how the students train (architecture, the teacher's by default; epochs; image size; starting
   * weights), and the confidence from which the teacher's labels are kept (by default the threshold its benchmark entry
   * found). Before starting, what will happen and what it costs, in a sentence. Opened from the Loops tab and from an
   * entry's "start a loop" (`teacher` preset).
   */
  import Repeat from '@lucide/svelte/icons/repeat';
  import { untrack } from 'svelte';
  import { hub } from '$lib/api/index.svelte';
  import type { Arch, Bench, BenchEntry, BenchLoop, Dataset, LabelJob, Model, Split, StartBenchLoop } from '$lib/api/types';
  import { GENERATIONS_MAX } from '$lib/bench/rules';
  import { conf, num } from '$lib/format';
  import { FROM_SCRATCH } from '$lib/lineage';
  import Button from '$lib/ui/Button.svelte';
  import Dialog from '$lib/ui/Dialog.svelte';
  import ErrorState from '$lib/ui/ErrorState.svelte';
  import Field from '$lib/ui/Field.svelte';
  import Gate from '$lib/ui/Gate.svelte';
  import Input from '$lib/ui/Input.svelte';
  import Segmented from '$lib/ui/Segmented.svelte';
  import Select from '$lib/ui/Select.svelte';
  import Switch from '$lib/ui/Switch.svelte';
  import { loopPlanText, teacherEligibility } from './words';

  let {
    open = $bindable(false),
    bench,
    models,
    datasets,
    jobs,
    entries,
    teacher: preset = null,
    onstarted,
  }: {
    open?: boolean;
    bench: Bench;
    models: readonly Model[] | undefined;
    datasets: readonly Dataset[] | undefined;
    jobs: readonly LabelJob[] | undefined;
    /** The benchmark's entries: a teacher's own entry gives the default threshold, and is reused when complete. */
    entries: readonly BenchEntry[];
    /** The model to start with (an entry's "start a loop"). */
    teacher?: string | null;
    onstarted: (loop: BenchLoop) => void;
  } = $props();

  const uid = $props.id();
  const ARCHS: Arch[] = ['folo-n', 'folo-s', 'folo-m', 'folo-l', 'folo-x'];
  const SPLIT_WORD: Record<Split, string> = { train: 'Train', val: 'Validation', test: 'Test' };

  let teacherId = $state('');
  let poolKind = $state<'dataset' | 'label_job'>('dataset');
  let datasetId = $state('');
  let split = $state<Split>('train');
  let jobId = $state('');
  let generations = $state('1');
  let arch = $state<string>('');
  let epochs = $state('30');
  let imgsz = $state('');
  let weights = $state('none');
  let ownThreshold = $state(true);
  let minConf = $state('0.50');
  let name = $state('');
  let touched = $state(false);
  let busy = $state(false);
  let error = $state.raw<unknown>(undefined);

  const eligible = (m: Model) => teacherEligibility(m, bench).ok;
  const ranked = $derived([...(models ?? [])].sort((a, b) => Number(eligible(b)) - Number(eligible(a)) || a.name.localeCompare(b.name)));
  const teacherOptions = $derived(
    ranked.map((m) => {
      const e = teacherEligibility(m, bench);
      return { value: m.id, label: e.ok ? m.name : `${m.name} · ${e.why}`, short: m.name, group: e.ok ? 'The benchmark’s classes, in its order' : 'Not eligible', disabled: !e.ok };
    }),
  );
  const teacher = $derived(models?.find((m) => m.id === teacherId));

  // Every opening starts fresh: the preset teacher, else the first eligible one; the pool's dataset the teacher's own
  // classes' first dataset with training images.
  $effect(() => {
    if (!open) return;
    untrack(() => {
      touched = false;
      busy = false;
      error = undefined;
      teacherId = preset ?? '';
      generations = '1';
      epochs = '30';
      weights = 'none';
      ownThreshold = true;
      minConf = '0.50';
      name = '';
      arch = '';
      imgsz = '';
      poolKind = 'dataset';
      split = 'train';
      datasetId = '';
      jobId = '';
    });
  });
  $effect(() => {
    if (!open || teacherId || !ranked.length) return;
    const first = ranked.find(eligible);
    if (first) untrack(() => (teacherId = first.id));
  });
  // The student follows the teacher until someone picks its size.
  $effect(() => {
    const t = teacher;
    if (!t) return;
    untrack(() => {
      if (!arch) arch = t.arch;
      if (!imgsz) imgsz = String(t.imgsz);
    });
  });
  const sameClasses = (classes: readonly string[]) => classes.length === bench.classes.length && classes.every((c, i) => c.toLowerCase() === bench.classes[i]!.toLowerCase());
  const datasetOptions = $derived(
    [...(datasets ?? [])]
      .sort((a, b) => Number(sameClasses(b.classes)) - Number(sameClasses(a.classes)) || a.name.localeCompare(b.name))
      .map((d) => ({ value: d.id, label: d.name, group: sameClasses(d.classes) ? 'The benchmark’s classes' : 'Other datasets (their labels are not read anyway)' })),
  );
  $effect(() => {
    if (!open || datasetId || !datasetOptions.length) return;
    const src = bench.source.kind === 'dataset' ? bench.source.dataset_id : null;
    const pick = datasetOptions.find((o) => o.value === src) ?? datasetOptions[0]!;
    untrack(() => (datasetId = pick.value));
  });
  $effect(() => {
    if (!open || jobId || !jobs?.length) return;
    untrack(() => (jobId = jobs[0]!.id));
  });
  const dataset = $derived(datasets?.find((d) => d.id === datasetId));
  const job = $derived(jobs?.find((j) => j.id === jobId));
  const poolImages = $derived(poolKind === 'dataset' ? (dataset ? dataset.splits[split] : null) : job ? job.counts.images : null);
  /** The benchmark's own source split is in it: the pool loses those images. */
  const overlaps = $derived(
    poolKind === 'dataset' ? bench.source.kind === 'dataset' && bench.source.dataset_id === datasetId && (bench.source.split ?? 'val') === split : bench.source.kind === 'label_job' && bench.source.job_id === jobId,
  );

  // The teacher's own complete entry on this benchmark (reused by the loop) and the threshold it found.
  const teacherEntry = $derived(
    entries.find((e) => e.contestant.kind === 'model' && e.contestant.model_id === teacherId && e.status === 'done' && e.fingerprint === bench.fingerprint && !e.score?.partial),
  );
  let found = $state<{ entry: string; threshold: number | null } | null>(null);
  $effect(() => {
    const e = teacherEntry;
    found = null;
    if (!e) return;
    let gone = false;
    hub.api.benchResult(bench.id, e.id).then(
      (r) => !gone && (found = { entry: e.id, threshold: r.self_labeling.threshold }),
      () => {},
    );
    return () => (gone = true);
  });
  const thresholdText = $derived(
    !teacherEntry
      ? 'decided when its entry is done (it runs first)'
      : !found
        ? 'reading its entry…'
        : found.threshold !== null
          ? `${conf(found.threshold)}, where its entry’s labels are 90% right`
          : '0.50: its entry never gets 90% of its labels right',
  );

  const bigger = $derived(teacher ? ARCHS[ARCHS.indexOf(teacher.arch) + 1] : undefined);
  const archOptions = $derived(
    ARCHS.map((a) => ({ value: a, label: a === teacher?.arch ? `${a} · the teacher’s size` : a === bigger ? `${a} · the next bigger model` : a })),
  );
  const weightOptions = $derived([
    { value: 'none', label: FROM_SCRATCH },
    ...(models ?? []).filter((m) => m.arch === arch && sameClasses(m.classes)).map((m) => ({ value: m.id, label: m.id === teacherId ? `${m.name} · the teacher’s weights` : m.name })),
  ]);
  $effect(() => {
    if (weights !== 'none' && !weightOptions.some((o) => o.value === weights)) untrack(() => (weights = 'none'));
  });

  // ---------- validation ----------
  const whole = (t: string, lo: number, hi: number) => {
    const n = Number(t);
    return t.trim() !== '' && Number.isInteger(n) && n >= lo && n <= hi;
  };
  const errs = $derived({
    epochs: whole(epochs, 1, 1000) ? null : 'A whole number from 1 to 1,000.',
    imgsz: whole(imgsz, 32, 4096) && Number(imgsz) % 32 === 0 ? null : 'A multiple of 32 from 32 to 4,096.',
    conf: ownThreshold || (Number.isFinite(Number(minConf)) && minConf.trim() !== '' && Number(minConf) >= 0 && Number(minConf) <= 1) ? null : 'From 0 to 1.',
    name: name.trim().length > 100 ? 'At most 100 characters.' : null,
  });
  const show = (e: string | null) => (touched ? e : null);
  const missingWhy = $derived(
    !models
      ? 'Loading models'
      : !models.some(eligible)
        ? `No model of this server has ${bench.name}’s classes in its order`
        : !teacher
          ? 'Pick a teacher'
          : !eligible(teacher)
            ? 'That model cannot teach on this benchmark'
            : poolKind === 'dataset'
              ? !dataset
                ? 'Pick a dataset'
                : !poolImages
                  ? `Its ${SPLIT_WORD[split].toLowerCase()} split has no images`
                  : null
              : !job
                ? jobs && !jobs.length
                  ? 'No labeling job on this server'
                  : 'Pick a labeling job'
                : !job.counts.images
                  ? 'That job has no images'
                  : null,
  );
  const shownErr = $derived(touched ? Object.values(errs).find((e) => e) : undefined);
  const reason = $derived(missingWhy ?? (shownErr ? 'Fix the fields first' : null));
  const summary = $derived(
    loopPlanText({ generations: Number(generations), epochs: whole(epochs, 1, 100000) ? Number(epochs) : 30, poolImages: poolImages ?? null, teacherEntry: !!teacherEntry }),
  );

  async function submit(e: SubmitEvent) {
    e.preventDefault();
    touched = true;
    error = undefined;
    if (Object.values(errs).some((x) => x) || reason || busy || !teacher) return;
    const body: StartBenchLoop = {
      teacher: teacher.id,
      pool: poolKind === 'dataset' ? { kind: 'dataset', dataset_id: datasetId, split } : { kind: 'label_job', job_id: jobId },
      generations: Number(generations),
      ...(ownThreshold ? {} : { min_confidence: Number(minConf) }),
      student: { arch: arch as Arch, epochs: Number(epochs), imgsz: Number(imgsz), ...(weights !== 'none' ? { weights } : {}) },
      ...(name.trim() ? { name: name.trim() } : {}),
    };
    busy = true;
    try {
      const loop = await hub.api.startBenchLoop(bench.id, body);
      open = false;
      onstarted(loop);
    } catch (err) {
      error = err;
    } finally {
      busy = false;
    }
  }
</script>

<Dialog bind:open size="lg" title="New loop" description="Does a model’s own labeling train a better model? Each generation the model labels a pool of images by itself, a student trains on those labels alone, and {bench.name} grades the student, the next generation’s teacher.">
  <form id="{uid}-form" class="form" onsubmit={submit} novalidate>
    <Field
      label="Teacher"
      for="{uid}-teacher"
      hint={teacher && eligible(teacher) ? `\`${teacher.arch}\` at ${num(teacher.imgsz)} px. ${teacherEntry ? 'Its complete entry here is generation 0.' : 'It has no complete entry here yet: one runs first, as generation 0.'}` : 'A model whose classes are this benchmark’s, the same names in the same order: its labels are written by class index.'}
    >
      <Select id="{uid}-teacher" bind:value={teacherId} disabled={!models?.length} options={teacherOptions} />
    </Field>

    <fieldset class="group">
      <legend class="group-label">The pool it labels</legend>
      <Segmented
        label="Where the pool's images come from"
        bind:value={poolKind}
        options={[
          { value: 'dataset', label: 'Dataset split' },
          { value: 'label_job', label: 'Labeling job' },
        ]}
      />
      {#if poolKind === 'dataset'}
        <div class="row2">
          <Field label="Dataset" for="{uid}-ds">
            <Select id="{uid}-ds" bind:value={datasetId} disabled={!datasets?.length} options={datasetOptions} />
          </Field>
          <Field label="Split">
            <Segmented
              label="Split"
              bind:value={split}
              options={(['train', 'val', 'test'] as const).map((s) => ({ value: s, label: SPLIT_WORD[s], meta: dataset ? num(dataset.splits[s]) : undefined }))}
            />
          </Field>
        </div>
      {:else}
        <Field label="Labeling job" for="{uid}-job">
          <Select id="{uid}-job" bind:value={jobId} disabled={!jobs?.length} options={(jobs ?? []).map((j) => ({ value: j.id, label: `${j.name} · ${num(j.counts.images)} images`, short: j.name }))} />
        </Field>
      {/if}
      <p class="note">
        The pool’s own labels are never read: the teacher labels every image by itself. Images that are also in this benchmark are left out{overlaps ? ', and this is the split the benchmark was made from: pick another' : ''}.
      </p>
    </fieldset>

    <Field label="Generations" hint="Each generation is one student: labeled by the one before, trained, benchmarked.">
      <Segmented label="Generations" bind:value={generations} options={Array.from({ length: GENERATIONS_MAX }, (_, i) => ({ value: String(i + 1), label: String(i + 1) }))} />
    </Field>

    <fieldset class="group">
      <legend class="group-label">The students</legend>
      <div class="row2">
        <Field label="Architecture" for="{uid}-arch">
          <Select id="{uid}-arch" bind:value={arch} options={archOptions} />
        </Field>
        <Field label="Starting weights" for="{uid}-w" hint={weights === 'none' ? 'Trained from scratch on the teacher’s labels.' : undefined}>
          <Select id="{uid}-w" bind:value={weights} options={weightOptions} />
        </Field>
        <Field label="Epochs" for="{uid}-ep" error={show(errs.epochs)}>
          <Input id="{uid}-ep" bind:value={epochs} inputmode="numeric" invalid={!!show(errs.epochs)} />
        </Field>
        <Field label="Image size" for="{uid}-sz" error={show(errs.imgsz)} hint={teacher ? `The teacher’s is ${teacher.imgsz} px.` : undefined}>
          <Input id="{uid}-sz" bind:value={imgsz} inputmode="numeric" suffix="px" invalid={!!show(errs.imgsz)} />
        </Field>
      </div>
    </fieldset>

    <div class="conf">
      <Switch label="Keep the teacher’s labels from its own threshold" bind:checked={ownThreshold} hint={ownThreshold ? `From ${thresholdText}.` : 'Every teacher keeps its detections from the confidence below.'} />
      {#if !ownThreshold}
        <Field label="Keep labels from confidence" for="{uid}-conf" error={show(errs.conf)}>
          <div class="narrow"><Input id="{uid}-conf" bind:value={minConf} inputmode="decimal" invalid={!!show(errs.conf)} /></div>
        </Field>
      {/if}
    </div>

    <Field label="Name" for="{uid}-name" error={show(errs.name)} hint="Its datasets and runs are named after it: “<name> gen 1”.">
      <Input id="{uid}-name" bind:value={name} placeholder={teacher ? `${teacher.name} loop` : 'Loop name'} autocomplete="off" invalid={!!show(errs.name)} />
    </Field>

    <p class="note">Its runs queue with the others, its entries with the other model entries. The datasets, runs and models it makes stay when the loop is deleted.</p>
    {#if error}<ErrorState {error} title="Could not start the loop" compact />{/if}
  </form>
  {#snippet footer()}
    <!-- What starting it does and costs, beside the button that does it. -->
    {#if missingWhy}<p class="why" data-gate-why>{missingWhy}</p>{:else}<p class="why s-text num" role="status">{summary}</p>{/if}
    <Button variant="ghost" onclick={() => (open = false)}>Cancel</Button>
    <Gate {reason}>
      {#snippet children(why)}
        <Button variant="primary" type="submit" form="{uid}-form" icon={Repeat} loading={busy} disabled={!!why} aria-describedby={why}>Start loop</Button>
      {/snippet}
    </Gate>
  {/snippet}
</Dialog>

<style>
  .form {
    display: grid;
    gap: var(--space-4);
  }
  .group legend {
    margin-bottom: var(--space-2);
    padding: 0;
  }
  .group {
    display: grid;
    gap: var(--space-3);
    min-width: 0;
    margin: 0;
    padding: 0;
    border: 0;
  }
  .group :global(.seg) {
    justify-self: start;
  }
  .row2 {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: var(--space-3) var(--space-4);
    align-items: start;
  }
  .narrow {
    max-width: 160px;
  }
  .note {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
  }
  .conf {
    display: grid;
    gap: var(--space-3);
  }
  .s-text {
    color: var(--text-primary);
  }
  .why {
    margin-right: auto;
    align-self: center;
    font-size: var(--type-sm-size);
    color: var(--text-secondary);
  }
  @media (max-width: 767px) {
    .row2 {
      grid-template-columns: minmax(0, 1fr);
    }
    .why {
      flex-basis: 100%;
    }
  }
</style>
