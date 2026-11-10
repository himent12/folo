<script lang="ts">
  /**
   * "New loop" (`POST /api/bench/benchmarks/{id}/loops`, docs/BENCH.md, Self-improvement loops): does a model's own
   * labeling train a better model? The form: the first teacher (a model whose classes are the benchmark's, same names in
   * the same order; the others listed, off, with the reason), the pool it labels (a dataset split, `train` by default, or
   * a labeling job's images; its own labels are never read and images also in the benchmark are left out), how many
   * generations, how the students train (architecture, the teacher's by default; epochs; image size; starting
   * weights), and the confidence from which the teacher's labels are kept (by default the threshold its benchmark entry
   * found). When the teacher has a finished entry, what its labels would be is read off that entry's self-labeling
   * curve beside the confidence: its threshold, the share of objects it labels there and its verdict, and about how many
   * pool images would get a box; at the confidence a person sets, the precision that costs, live. A teacher whose labels
   * would be too few to learn from (not ready, or under a tenth of the objects) is said so plainly before the loop
   * starts, with the lower confidence as the way out, and Start reads "Start anyway" (docs/BENCH.md: a teacher that
   * keeps boxes on fewer than 10 pool images ends the loop at labeling). Before starting, what will happen and what it
   * costs, in a sentence. Opened from the Loops tab and from an entry's "start a loop" (`teacher` preset).
   */
  import Info from '@lucide/svelte/icons/info';
  import Repeat from '@lucide/svelte/icons/repeat';
  import TriangleAlert from '@lucide/svelte/icons/triangle-alert';
  import { tick, untrack } from 'svelte';
  import { ApiError } from '$lib/api/client';
  import { hub } from '$lib/api/index.svelte';
  import type { Arch, Bench, BenchEntry, BenchLoop, BenchSelfLabeling, Dataset, LabelJob, Model, Split, StartBenchLoop } from '$lib/api/types';
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
  import { forecastText, labelForecast, labelsText, loopErrorField, loopPlanText, lowerConfidence, teacherEligibility } from './words';

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

  // What a person chose; null follows the default, derived from the rest (the teacher, the benchmark's source), so
  // nothing chases anything: reopening the dialog clears the choices and every default is there again at once.
  let teacherChoice = $state<string | null>(null);
  let datasetChoice = $state<string | null>(null);
  let jobChoice = $state<string | null>(null);
  let archChoice = $state<string | null>(null);
  let sizeChoice = $state<string | null>(null);
  let weightsChoice = $state('none');
  let poolKind = $state<'dataset' | 'label_job'>('dataset');
  let split = $state<Split>('train');
  let generations = $state('1');
  let epochs = $state('30');
  let ownThreshold = $state(true);
  let minConf = $state('');
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
  /** The teacher: the one picked, else the preset (an entry's "start a loop"), else the first eligible model. */
  const teacherId = $derived(teacherChoice ?? preset ?? ranked.find(eligible)?.id ?? '');
  const teacher = $derived(models?.find((m) => m.id === teacherId));

  // Every opening starts fresh: no choice made, every field at its default.
  $effect(() => {
    if (!open) return;
    untrack(() => {
      touched = false;
      busy = false;
      error = undefined;
      teacherChoice = datasetChoice = jobChoice = archChoice = sizeChoice = null;
      weightsChoice = 'none';
      generations = '1';
      epochs = '30';
      ownThreshold = true;
      minConf = '';
      name = '';
      poolKind = 'dataset';
      split = 'train';
    });
  });
  /** The students follow the teacher (architecture, image size) until a person picks theirs. */
  const arch = $derived(archChoice ?? teacher?.arch ?? '');
  const imgsz = $derived(sizeChoice ?? (teacher ? String(teacher.imgsz) : ''));
  const sameClasses = (classes: readonly string[]) => classes.length === bench.classes.length && classes.every((c, i) => c.toLowerCase() === bench.classes[i]!.toLowerCase());
  const datasetOptions = $derived(
    [...(datasets ?? [])]
      .sort((a, b) => Number(sameClasses(b.classes)) - Number(sameClasses(a.classes)) || a.name.localeCompare(b.name))
      .map((d) => ({ value: d.id, label: d.name, group: sameClasses(d.classes) ? 'The benchmark’s classes' : 'Other datasets (their labels are not read anyway)' })),
  );
  /** The pool's dataset: the one picked, else the benchmark's own source dataset, else the first. */
  const datasetId = $derived(
    datasetChoice ?? (datasetOptions.find((o) => bench.source.kind === 'dataset' && o.value === bench.source.dataset_id) ?? datasetOptions[0])?.value ?? '',
  );
  const jobId = $derived(jobChoice ?? jobs?.[0]?.id ?? '');
  const dataset = $derived(datasets?.find((d) => d.id === datasetId));
  const job = $derived(jobs?.find((j) => j.id === jobId));
  const poolImages = $derived(poolKind === 'dataset' ? (dataset ? dataset.splits[split] : null) : job ? job.counts.images : null);
  /** The benchmark's own source split is in it: the pool loses those images. */
  const overlaps = $derived(
    poolKind === 'dataset' ? bench.source.kind === 'dataset' && bench.source.dataset_id === datasetId && (bench.source.split ?? 'val') === split : bench.source.kind === 'label_job' && bench.source.job_id === jobId,
  );

  // The teacher's own complete entry on this benchmark (reused by the loop) and the threshold it found.
  const ofTeacher = (e: BenchEntry) => e.contestant.kind === 'model' && e.contestant.model_id === teacherId && e.status === 'done' && e.fingerprint === bench.fingerprint;
  const teacherEntry = $derived(entries.find((e) => ofTeacher(e) && !e.score?.partial));
  /** What the forecast reads: the complete entry, else a finished one on part of the benchmark (an estimate). */
  const readEntry = $derived(teacherEntry ?? entries.find(ofTeacher));
  let found = $state.raw<{ entry: string; images: number; self: BenchSelfLabeling; bestConf: number } | null>(null);
  $effect(() => {
    const e = readEntry;
    found = null;
    if (!e) return;
    let gone = false;
    hub.api.benchResult(bench.id, e.id).then(
      (r) => !gone && (found = { entry: e.id, images: r.entry.images, self: r.self_labeling, bestConf: r.accuracy.best_conf }),
      () => {},
    );
    return () => (gone = true);
  });
  /**
   * Its complete entry found no threshold: a loop never guesses a confidence, so one is required here (the switch is
   * off and cannot be turned on; Start waits for a confidence).
   */
  const needConf = $derived(!!teacherEntry && found?.entry === teacherEntry.id && found.self.threshold === null);
  /** Whether the teacher's own threshold is used. */
  const useOwn = $derived(ownThreshold && !needConf);
  const thresholdText = $derived(
    !teacherEntry
      ? 'From the threshold its entry finds (that entry runs first). If it finds none, the loop stops after that entry unless you set a confidence.'
      : !found || found.entry !== teacherEntry.id
        ? 'From its entry’s threshold (reading its entry…).'
        : found.self.threshold !== null
          ? `From ${conf(found.self.threshold)}, where its entry keeps ${labelsText(found.self.labels)}, 90% right.`
          : 'Off: its entry found no threshold, and a loop never guesses one.',
  );
  /** The confidence a person set, once it is a valid one; null while the teacher's own threshold is used. */
  const chosen = $derived(!useOwn && minConf.trim() !== '' && Number(minConf) >= 0 && Number(minConf) <= 1 ? Number(minConf) : null);
  /** What its labels would be (./words labelForecast): read off the entry's curve, live as the confidence changes. */
  const forecast = $derived(found && (useOwn || chosen !== null || needConf) ? labelForecast(found.self, found.images, poolImages, chosen) : null);
  const forecastWords = $derived(forecast ? forecastText(forecast, poolImages) : null);
  /** "Keep labels from a lower confidence": the switch off, a lower confidence proposed, its field focused. */
  async function lower() {
    if (!found) return;
    const from = forecast?.threshold ?? null;
    minConf = (from !== null ? lowerConfidence(from, found.bestConf) : Math.max(0.05, Math.round(found.bestConf * 100) / 100)).toFixed(2);
    ownThreshold = false;
    await tick();
    document.getElementById(`${uid}-conf`)?.focus();
  }

  const bigger = $derived(teacher ? ARCHS[ARCHS.indexOf(teacher.arch) + 1] : undefined);
  const archOptions = $derived(
    ARCHS.map((a) => ({ value: a, label: a === teacher?.arch ? `${a} · the teacher’s size` : a === bigger ? `${a} · the next bigger model` : a })),
  );
  const weightOptions = $derived([
    { value: 'none', label: FROM_SCRATCH },
    ...(models ?? []).filter((m) => m.arch === arch && sameClasses(m.classes)).map((m) => ({ value: m.id, label: m.id === teacherId ? `${m.name} · the teacher’s weights` : m.name })),
  ]);
  /** The starting weights picked, while they still fit the architecture; else from scratch. */
  const weights = $derived(weightOptions.some((o) => o.value === weightsChoice) ? weightsChoice : 'none');

  // ---------- validation ----------
  const whole = (t: string, lo: number, hi: number) => {
    const n = Number(t);
    return t.trim() !== '' && Number.isInteger(n) && n >= lo && n <= hi;
  };
  const errs = $derived({
    epochs: whole(epochs, 1, 1000) ? null : 'A whole number from 1 to 1,000.',
    imgsz: whole(imgsz, 32, 4096) && Number(imgsz) % 32 === 0 ? null : 'A multiple of 32 from 32 to 4,096.',
    conf: useOwn
      ? null
      : minConf.trim() === ''
        ? needConf
          ? 'Its entry found no threshold: choose the confidence its labels are kept from.'
          : 'Choose a confidence from 0 to 1.'
        : Number.isFinite(Number(minConf)) && Number(minConf) >= 0 && Number(minConf) <= 1
          ? null
          : 'From 0 to 1.',
    name: name.trim().length > 100 ? 'At most 100 characters.' : null,
  });
  const show = (e: string | null) => (touched ? e : null);
  const missingWhy = $derived(
    (!models
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
                  : null) ?? (needConf && chosen === null ? 'Choose a confidence: its entry found no threshold' : null),
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
    serverErr = null;
    if (Object.values(errs).some((x) => x) || reason || busy || !teacher) return;
    const body: StartBenchLoop = {
      teacher: teacher.id,
      pool: poolKind === 'dataset' ? { kind: 'dataset', dataset_id: datasetId, split } : { kind: 'label_job', job_id: jobId },
      generations: Number(generations),
      ...(useOwn ? {} : { min_confidence: Number(minConf) }),
      student: { arch: arch as Arch, epochs: Number(epochs), imgsz: Number(imgsz), ...(weights !== 'none' ? { weights } : {}) },
      ...(name.trim() ? { name: name.trim() } : {}),
    };
    busy = true;
    try {
      const loop = await hub.api.startBenchLoop(bench.id, body);
      open = false;
      onstarted(loop);
    } catch (err) {
      // At the field it belongs to when the server's error says which (the students' settings, checked as a run's),
      // else as the form's error.
      const f = err instanceof ApiError ? loopErrorField(err, weights !== 'none') : null;
      if (f) serverErr = { field: f, message: (err as ApiError).message };
      else error = err;
    } finally {
      busy = false;
    }
  }
  /** A server error shown at its field, until that field changes. */
  let serverErr = $state<{ field: 'epochs' | 'imgsz' | 'weights' | 'arch'; message: string } | null>(null);
  const fieldErr = (f: 'epochs' | 'imgsz' | 'weights' | 'arch') => (serverErr?.field === f ? serverErr.message : null);
  $effect(() => {
    void [epochs, imgsz, weights, arch];
    untrack(() => (serverErr = null));
  });
</script>

<Dialog bind:open size="lg" title="New loop" description="Does a model’s own labeling train a better model? Each generation the model labels a pool of images by itself, a student trains on those labels alone, and {bench.name} grades the student, the next generation’s teacher.">
  <form id="{uid}-form" class="form" onsubmit={submit} novalidate>
    <Field
      label="Teacher"
      for="{uid}-teacher"
      hint={teacher && eligible(teacher) ? `\`${teacher.arch}\` at ${num(teacher.imgsz)} px. ${teacherEntry ? 'Its complete entry here is generation 0.' : 'It has no complete entry here yet: one runs first, as generation 0.'}` : 'A model whose classes are this benchmark’s, the same names in the same order: its labels are written by class index.'}
    >
      <Select id="{uid}-teacher" bind:value={() => teacherId, (v) => (teacherChoice = v)} disabled={!models?.length} options={teacherOptions} />
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
            <Select id="{uid}-ds" bind:value={() => datasetId, (v) => (datasetChoice = v)} disabled={!datasets?.length} options={datasetOptions} />
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
          <Select id="{uid}-job" bind:value={() => jobId, (v) => (jobChoice = v)} disabled={!jobs?.length} options={(jobs ?? []).map((j) => ({ value: j.id, label: `${j.name} · ${num(j.counts.images)} images`, short: j.name }))} />
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
        <Field label="Architecture" for="{uid}-arch" error={fieldErr('arch')}>
          <Select id="{uid}-arch" bind:value={() => arch, (v) => (archChoice = v)} options={archOptions} />
        </Field>
        <Field label="Starting weights" for="{uid}-w" error={fieldErr('weights')} hint={weights === 'none' ? 'Trained from scratch on the teacher’s labels.' : undefined}>
          <Select id="{uid}-w" bind:value={() => weights, (v) => (weightsChoice = v)} options={weightOptions} />
        </Field>
        <Field label="Epochs" for="{uid}-ep" error={fieldErr('epochs') ?? show(errs.epochs)}>
          <Input id="{uid}-ep" bind:value={epochs} inputmode="numeric" invalid={!!show(errs.epochs)} />
        </Field>
        <Field label="Image size" for="{uid}-sz" error={fieldErr('imgsz') ?? show(errs.imgsz)} hint={teacher ? `The teacher’s is ${teacher.imgsz} px.` : undefined}>
          <Input id="{uid}-sz" bind:value={() => imgsz, (v) => (sizeChoice = v)} inputmode="numeric" suffix="px" invalid={!!show(errs.imgsz)} />
        </Field>
      </div>
    </fieldset>

    <div class="conf">
      <Switch
        label="Keep the teacher’s labels from its own threshold"
        bind:checked={() => useOwn, (v) => (ownThreshold = v)}
        disabled={needConf}
        hint={useOwn || needConf ? thresholdText : 'Every teacher keeps its detections from the confidence below.'}
      />
      {#if !useOwn}
        <Field label="Keep labels from confidence" for="{uid}-conf" error={show(errs.conf)} hint={needConf ? 'Required for this teacher.' : undefined}>
          <div class="narrow"><Input id="{uid}-conf" bind:value={minConf} inputmode="decimal" placeholder="e.g. 0.30" invalid={!!show(errs.conf)} /></div>
        </Field>
      {/if}
      {#if forecast && forecastWords}
        <!-- What the first teacher's labels would be, from its entry's curve: said plainly when they are too few. -->
        <div class="fc" class:warn={forecast.starved} role={forecast.starved ? 'alert' : 'status'} data-forecast={forecast.from === 'none' ? 'none' : forecast.starved ? (forecast.stops ? 'stops' : 'starved') : 'ok'}>
          <span class="fc-icon" aria-hidden="true">{#if forecast.starved}<TriangleAlert size={16} strokeWidth={1.75} />{:else}<Info size={16} strokeWidth={1.75} />{/if}</span>
          <div class="fc-body">
            {#if forecastWords.title}<p class="fc-title">{forecastWords.title}</p>{/if}
            <p class="fc-text">{forecastWords.text}{#if forecastWords.remedy}{' '}{forecastWords.remedy}{/if}{#if !teacherEntry}{' '}Read from its entry on part of the benchmark; the loop runs a complete one first.{/if}</p>
            {#if forecast.starved && useOwn}<div class="fc-act"><Button size="sm" onclick={lower}>Keep labels from a lower confidence</Button></div>{/if}
          </div>
        </div>
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
    {#if missingWhy}<p class="why" data-gate-why>{missingWhy}</p>{:else if forecast?.starved}<p class="why warn-text num" role="status"><TriangleAlert size={14} strokeWidth={1.75} aria-hidden="true" />{forecast.threshold === null ? 'Without a confidence the loop would stop before labeling' : forecast.stops ? 'Its labels would be too few: the loop would stop at labeling' : 'Its labels would be too few to train a student on'}</p>{:else}<p class="why s-text num" role="status">{summary}</p>{/if}
    <Button variant="ghost" onclick={() => (open = false)}>Cancel</Button>
    <Gate {reason}>
      {#snippet children(why)}
        <Button variant="primary" type="submit" form="{uid}-form" icon={Repeat} loading={busy} disabled={!!why} aria-describedby={why}>{forecast?.starved && forecast.threshold !== null ? 'Start anyway' : 'Start loop'}</Button>
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
  .why.warn-text {
    display: inline-flex;
    align-items: center;
    gap: var(--space-2);
    color: var(--text-primary);
  }
  .why.warn-text :global(svg) {
    flex-shrink: 0;
    color: var(--warning-solid);
  }
  /* The forecast: a quiet line of facts; too few labels, the panel surface with a 2 px warning edge (the callout's). */
  .fc {
    display: grid;
    grid-template-columns: 16px minmax(0, 1fr);
    gap: var(--space-2);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
  }
  .fc-icon {
    display: grid;
    padding-top: 1px;
    color: var(--text-tertiary);
  }
  .fc.warn {
    padding: var(--space-3) var(--space-3) var(--space-3) calc(var(--space-3) - 2px);
    border: 1px solid var(--line-panel);
    border-left: 2px solid var(--warning-solid);
    border-radius: var(--radius-md);
    background: var(--bg-panel);
  }
  .fc.warn .fc-icon {
    color: var(--warning-solid);
  }
  .fc-body {
    display: grid;
    gap: var(--space-1);
    min-width: 0;
  }
  .fc-title {
    font-size: var(--type-md-size);
    line-height: var(--type-md-line);
    font-weight: var(--weight-medium);
    color: var(--text-primary);
  }
  .fc-act {
    margin-top: var(--space-2);
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
