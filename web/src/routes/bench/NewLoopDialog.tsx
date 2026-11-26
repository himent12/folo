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
import Info from '$icons/info';
import Repeat from '$icons/repeat';
import TriangleAlert from '$icons/triangle-alert';
import { bindable, computed, createUniqueId, css, effect, onCleanup, signal, tick, untrack } from 'fw';
import { ApiError } from '$lib/api/client';
import { hub } from '$lib/api/index';
import type { Arch, Bench, BenchEntry, BenchLoop, BenchSelfLabeling, Dataset, LabelJob, Model, Split, StartBenchLoop } from '$lib/api/types';
import { GENERATIONS_MAX } from '$lib/bench/rules';
import { conf, count, num } from '$lib/format';
import { FROM_SCRATCH } from '$lib/lineage';
import Button from '$lib/ui/Button';
import Dialog from '$lib/ui/Dialog';
import ErrorState from '$lib/ui/ErrorState';
import Field from '$lib/ui/Field.tsx';
import Gate from '$lib/ui/Gate';
import Input from '$lib/ui/Input';
import Segmented from '$lib/ui/Segmented';
import Select from '$lib/ui/Select';
import Switch from '$lib/ui/Switch';
import { bestF1Conf, forecastText, labelForecast, labelsText, loopErrorField, loopPlanText, lowerConfidence, teacherEligibility } from './words';

export interface NewLoopDialogProps {
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
}

const ARCHS: Arch[] = ['folo-n', 'folo-s', 'folo-m', 'folo-l', 'folo-x'];
const SPLIT_WORD: Record<Split, string> = { train: 'Train', val: 'Validation', test: 'Test' };
const POOL_OPTIONS: { value: 'dataset' | 'label_job'; label: string }[] = [
  { value: 'dataset', label: 'Dataset split' },
  { value: 'label_job', label: 'Labeling job' },
];
const GENERATION_OPTIONS = Array.from({ length: GENERATIONS_MAX }, (_, i) => ({ value: String(i + 1), label: String(i + 1) }));
type ServerField = 'epochs' | 'imgsz' | 'weights' | 'arch';

export default function NewLoopDialog(props: NewLoopDialogProps) {
  const open = bindable(props, 'open', false);
  const uid = createUniqueId();

  // What a person chose; null follows the default, derived from the rest (the teacher, the benchmark's source), so
  // nothing chases anything: reopening the dialog clears the choices and every default is there again at once.
  const teacherChoice = signal<string | null>(null);
  const datasetChoice = signal<string | null>(null);
  const jobChoice = signal<string | null>(null);
  const archChoice = signal<string | null>(null);
  const sizeChoice = signal<string | null>(null);
  const weightsChoice = signal('none');
  const poolKind = signal<'dataset' | 'label_job'>('dataset');
  const split = signal<Split>('train');
  const generations = signal('1');
  const epochs = signal('30');
  const ownThreshold = signal(true);
  const minConf = signal('');
  /** "Keep labels from a lower confidence" found no confidence to propose: the teacher's entry has no best F1. */
  const unproposed = signal(false);
  const name = signal('');
  const touched = signal(false);
  const busy = signal(false);
  const error = signal<unknown>(undefined);

  const eligible = (m: Model) => teacherEligibility(m, props.bench).ok;
  const ranked = computed(() => [...(props.models ?? [])].sort((a, b) => Number(eligible(b)) - Number(eligible(a)) || a.name.localeCompare(b.name)));
  const teacherOptions = computed(() =>
    ranked.get().map((m) => {
      const e = teacherEligibility(m, props.bench);
      return { value: m.id, label: e.ok ? m.name : `${m.name} · ${e.why}`, short: m.name, group: e.ok ? 'The benchmark’s classes, in its order' : 'Not eligible', disabled: !e.ok };
    }),
  );
  /** The teacher: the one picked, else the preset (an entry's "start a loop"), else the first eligible model. */
  const teacherId = computed(() => teacherChoice.get() ?? props.teacher ?? ranked.get().find(eligible)?.id ?? '');
  const teacher = computed(() => props.models?.find((m) => m.id === teacherId.get()));

  let alive = true;
  let submitSeq = 0;
  onCleanup(() => {
    alive = false;
    submitSeq++;
  });

  // Every opening starts fresh: no choice made, every field at its default.
  effect(() => {
    if (!open.get()) return;
    untrack(() => {
      submitSeq++;
      touched.set(false);
      busy.set(false);
      error.set(undefined);
      teacherChoice.set(null);
      datasetChoice.set(null);
      jobChoice.set(null);
      archChoice.set(null);
      sizeChoice.set(null);
      weightsChoice.set('none');
      generations.set('1');
      epochs.set('30');
      ownThreshold.set(true);
      minConf.set('');
      unproposed.set(false);
      name.set('');
      poolKind.set('dataset');
      split.set('train');
    });
  });
  /** The students follow the teacher (architecture, image size) until a person picks theirs. */
  const arch = computed(() => archChoice.get() ?? teacher.get()?.arch ?? '');
  const imgsz = computed(() => {
    const t = teacher.get();
    return sizeChoice.get() ?? (t ? String(t.imgsz) : '');
  });
  const sameClasses = (classes: readonly string[]) => {
    const bc = props.bench.classes;
    return classes.length === bc.length && classes.every((c, i) => c.toLowerCase() === bc[i]!.toLowerCase());
  };
  const datasetOptions = computed(() =>
    [...(props.datasets ?? [])]
      .sort((a, b) => Number(sameClasses(b.classes)) - Number(sameClasses(a.classes)) || a.name.localeCompare(b.name))
      .map((d) => ({ value: d.id, label: d.name, group: sameClasses(d.classes) ? 'The benchmark’s classes' : 'Other datasets (their labels are not read anyway)' })),
  );
  /** The pool's dataset: the one picked, else the benchmark's own source dataset, else the first. */
  const datasetId = computed(() => {
    const src = props.bench.source;
    const opts = datasetOptions.get();
    return datasetChoice.get() ?? (opts.find((o) => src.kind === 'dataset' && o.value === src.dataset_id) ?? opts[0])?.value ?? '';
  });
  const jobId = computed(() => jobChoice.get() ?? props.jobs?.[0]?.id ?? '');
  const dataset = computed(() => props.datasets?.find((d) => d.id === datasetId.get()));
  const job = computed(() => props.jobs?.find((j) => j.id === jobId.get()));
  const poolImages = computed(() => {
    if (poolKind.get() === 'dataset') {
      const d = dataset.get();
      return d ? d.splits[split.get()] : null;
    }
    const j = job.get();
    return j ? j.counts.images : null;
  });
  /** The benchmark's own source split is in it: the pool loses those images. */
  const overlaps = computed(() => {
    const src = props.bench.source;
    return poolKind.get() === 'dataset'
      ? src.kind === 'dataset' && src.dataset_id === datasetId.get() && (src.split ?? 'val') === split.get()
      : src.kind === 'label_job' && src.job_id === jobId.get();
  });

  // The teacher's own complete entry on this benchmark (reused by the loop) and the threshold it found.
  const ofTeacher = (e: BenchEntry) =>
    e.contestant.kind === 'model' && e.contestant.model_id === teacherId.get() && e.status === 'done' && e.fingerprint === props.bench.fingerprint;
  const teacherEntry = computed(() => props.entries.find((e) => ofTeacher(e) && !e.score?.partial));
  /** What the forecast reads: the complete entry, else a finished one on part of the benchmark (an estimate). */
  const readEntry = computed(() => teacherEntry.get() ?? props.entries.find(ofTeacher));
  const found = signal<{ entry: string; images: number; self: BenchSelfLabeling; bestConf: number | null } | null>(null);
  const entryReadFailed = signal(false);
  // By id: the page hands down a new benchmark and new entries on every `bench` event and every load of the lists, and
  // a finished entry's result never changes.
  const benchId = computed(() => props.bench.id);
  const readId = computed(() => readEntry.get()?.id ?? null);
  effect(() => {
    const id = readId.get();
    const isOpen = open.get();
    found.set(null);
    entryReadFailed.set(false);
    if (!isOpen || !id) return;
    let gone = false;
    hub.api.benchResult(benchId.get(), id).then(
      (r) => !gone && found.set({ entry: id, images: r.entry.images, self: r.self_labeling, bestConf: bestF1Conf(r.accuracy) }),
      () => !gone && entryReadFailed.set(true),
    );
    return () => (gone = true);
  });
  /**
   * Its complete entry found no threshold: a loop never guesses a confidence, so one is required here (the switch is
   * off and cannot be turned on; Start waits for a confidence).
   */
  const needConf = computed(() => {
    const te = teacherEntry.get();
    const f = found.get();
    return !!te && f?.entry === te.id && f.self.threshold === null;
  });
  /** While the teacher's complete entry result is still being read and the user relies on its threshold, hold Start until we know whether it found one. */
  const readingThreshold = computed(() => {
    const te = teacherEntry.get();
    const f = found.get();
    return open.get() && !!te && ownThreshold.get() && f?.entry !== te.id && !entryReadFailed.get();
  });
  /** Whether the teacher's own threshold is used. */
  const useOwn = computed(() => ownThreshold.get() && !needConf.get());
  const thresholdText = computed(() => {
    const te = teacherEntry.get();
    const f = found.get();
    return !te
      ? 'From the threshold its entry finds (that entry runs first). If it finds none, the loop stops after that entry unless you set a confidence.'
      : !f || f.entry !== te.id
        ? 'From its entry’s threshold (reading its entry…).'
        : f.self.threshold !== null
          ? `From ${conf(f.self.threshold)}, where its entry keeps ${labelsText(f.self.labels)}, 90% right.`
          : 'Off: its entry found no threshold, and a loop never guesses one.';
  });
  /** The confidence a person set, once it is a valid one; null while the teacher's own threshold is used. */
  const chosen = computed(() => {
    const m = minConf.get();
    return !useOwn.get() && m.trim() !== '' && Number(m) >= 0 && Number(m) <= 1 ? Number(m) : null;
  });
  /** What its labels would be (./words labelForecast): read off the entry's curve, live as the confidence changes. */
  const forecast = computed(() => {
    const f = found.get();
    return f && (useOwn.get() || chosen.get() !== null || needConf.get()) ? labelForecast(f.self, f.images, poolImages.get(), chosen.get()) : null;
  });
  const forecastWords = computed(() => {
    const fc = forecast.get();
    return fc ? forecastText(fc, poolImages.get()) : null;
  });
  /** "Keep labels from a lower confidence": the switch off, a lower confidence proposed, its field focused. */
  async function lower() {
    const f = found.peek();
    if (!f) return;
    const from = forecast.peek()?.threshold ?? null;
    // Without a best-F1 confidence (F1 0 at every one) nothing is measured to propose from: the person sets one.
    unproposed.set(f.bestConf === null);
    if (f.bestConf !== null) minConf.set((from !== null ? lowerConfidence(from, f.bestConf) : Math.max(0.05, Math.round(f.bestConf * 100) / 100)).toFixed(2));
    ownThreshold.set(false);
    await tick();
    if (!alive || !open.peek()) return;
    document.getElementById(`${uid}-conf`)?.focus();
  }

  const bigger = computed(() => {
    const t = teacher.get();
    return t ? ARCHS[ARCHS.indexOf(t.arch) + 1] : undefined;
  });
  const archOptions = computed(() => {
    const ta = teacher.get()?.arch;
    const big = bigger.get();
    return ARCHS.map((a) => ({ value: a, label: a === ta ? `${a} · the teacher’s size` : a === big ? `${a} · the next bigger model` : a }));
  });
  const weightOptions = computed(() => {
    const a = arch.get();
    const tid = teacherId.get();
    return [
      { value: 'none', label: FROM_SCRATCH },
      ...(props.models ?? []).filter((m) => m.arch === a && sameClasses(m.classes)).map((m) => ({ value: m.id, label: m.id === tid ? `${m.name} · the teacher’s weights` : m.name })),
    ];
  });
  /** The starting weights picked, while they still fit the architecture; else from scratch. */
  const weights = computed(() => {
    const w = weightsChoice.get();
    return weightOptions.get().some((o) => o.value === w) ? w : 'none';
  });

  // ---------- validation ----------
  const whole = (t: string, lo: number, hi: number) => {
    const n = Number(t);
    return t.trim() !== '' && Number.isInteger(n) && n >= lo && n <= hi;
  };
  const errs = computed(() => {
    const ep = epochs.get();
    const sz = imgsz.get();
    const m = minConf.get();
    return {
      epochs: whole(ep, 1, 1000) ? null : 'A whole number from 1 to 1,000.',
      imgsz: whole(sz, 32, 4096) && Number(sz) % 32 === 0 ? null : 'A multiple of 32 from 32 to 4,096.',
      conf: useOwn.get()
        ? null
        : m.trim() === ''
          ? needConf.get()
            ? 'Its entry found no threshold: choose the confidence its labels are kept from.'
            : 'Choose a confidence from 0 to 1.'
          : Number.isFinite(Number(m)) && Number(m) >= 0 && Number(m) <= 1
            ? null
            : 'From 0 to 1.',
      name: name.get().trim().length > 100 ? 'At most 100 characters.' : null,
    };
  });
  const show = (e: string | null) => (touched.get() ? e : null);
  /** A number field says a value out of range as soon as it is typed, not on submit (an empty one is said then). */
  const showNow = (text: string, e: string | null) => (touched.get() || text.trim() ? e : null);
  const liveErr = computed(
    () => showNow(epochs.get(), errs.get().epochs) ?? showNow(imgsz.get(), errs.get().imgsz) ?? showNow(minConf.get(), errs.get().conf),
  );
  const missingWhy = computed(() => {
    const models = props.models;
    const t = teacher.get();
    const j = job.get();
    const jobs = props.jobs;
    return (
      (!models
        ? 'Loading models'
        : !models.some(eligible)
          ? `No model of this server has ${props.bench.name}’s classes in its order`
          : !t
            ? 'Pick a teacher'
            : !eligible(t)
              ? 'That model cannot teach on this benchmark'
              : poolKind.get() === 'dataset'
                ? !dataset.get()
                  ? 'Pick a dataset'
                  : !poolImages.get()
                    ? `Its ${SPLIT_WORD[split.get()].toLowerCase()} split has no images`
                    : null
                : !j
                  ? jobs && !jobs.length
                    ? 'No labeling job on this server'
                    : 'Pick a labeling job'
                  : !j.counts.images
                    ? 'That job has no images'
                    : null) ??
      (readingThreshold.get()
        ? 'Reading the teacher’s entry…'
        : needConf.get() && chosen.get() === null
          ? 'Choose a confidence: its entry found no threshold'
          : null)
    );
  });
  const shownErr = computed(() => (touched.get() ? Object.values(errs.get()).find((e) => e) : (liveErr.get() ?? undefined)));
  const reason = computed(() => missingWhy.get() ?? (shownErr.get() ? 'Fix the fields first' : null));
  const summary = computed(() => {
    const ep = epochs.get();
    return loopPlanText({
      generations: Number(generations.get()),
      epochs: whole(ep, 1, 100000) ? Number(ep) : 30,
      poolImages: poolImages.get() ?? null,
      teacherEntry: !!teacherEntry.get(),
    });
  });

  /** A server error shown at its field, until that field changes. */
  const serverErr = signal<{ field: ServerField; message: string } | null>(null);
  const fieldErr = (f: ServerField) => {
    const s = serverErr.get();
    return s?.field === f ? s.message : null;
  };

  async function submit(e: SubmitEvent) {
    e.preventDefault();
    touched.set(true);
    error.set(undefined);
    serverErr.set(null);
    const t = teacher.peek();
    if (Object.values(errs.peek()).some((x) => x) || reason.peek() || busy.peek() || !t) return;
    const useOwnNow = useOwn.peek();
    const w = weights.peek();
    const n = name.peek().trim();
    const body: StartBenchLoop = {
      teacher: t.id,
      pool: poolKind.peek() === 'dataset' ? { kind: 'dataset', dataset_id: datasetId.peek(), split: split.peek() } : { kind: 'label_job', job_id: jobId.peek() },
      generations: Number(generations.peek()),
      ...(useOwnNow ? {} : { min_confidence: Number(minConf.peek()) }),
      student: { arch: arch.peek() as Arch, epochs: Number(epochs.peek()), imgsz: Number(imgsz.peek()), ...(w !== 'none' ? { weights: w } : {}) },
      ...(n ? { name: n } : {}),
    };
    const seq = ++submitSeq;
    const epoch = hub.epoch;
    busy.set(true);
    try {
      const loop = await hub.api.startBenchLoop(props.bench.id, body);
      if (hub.epoch !== epoch) return;
      if (alive && seq === submitSeq) open.set(false);
      if (alive) props.onstarted(loop);
    } catch (err) {
      if (!alive || seq !== submitSeq || hub.epoch !== epoch) return;
      // At the field it belongs to when the server's error says which (the students' settings, checked as a run's),
      // else as the form's error.
      const f = err instanceof ApiError ? loopErrorField(err, weights.peek() !== 'none') : null;
      if (f) serverErr.set({ field: f, message: (err as ApiError).message });
      else error.set(err);
    } finally {
      if (alive && seq === submitSeq) busy.set(false);
    }
  }
  effect(() => {
    void [epochs.get(), imgsz.get(), weights.get(), arch.get()];
    untrack(() => serverErr.set(null));
  });

  // Svelte's function bindings: the shown value is the derived default until a person picks.
  const teacherBind = { get: () => teacherId.get(), set: (v: string) => teacherChoice.set(v) };
  const datasetBind = { get: () => datasetId.get(), set: (v: string) => datasetChoice.set(v) };
  const jobBind = { get: () => jobId.get(), set: (v: string) => jobChoice.set(v) };
  const archBind = { get: () => arch.get(), set: (v: string) => archChoice.set(v) };
  const weightsBind = { get: () => weights.get(), set: (v: string) => weightsChoice.set(v) };
  const sizeBind = { get: () => imgsz.get(), set: (v: string) => sizeChoice.set(v) };
  const ownBind = { get: () => useOwn.get(), set: (v: boolean) => ownThreshold.set(v) };

  const splitOptions = computed(() => {
    const d = dataset.get();
    return (['train', 'val', 'test'] as const).map((s) => ({ value: s, label: SPLIT_WORD[s], meta: d ? num(d.splits[s]) : undefined }));
  });
  const jobOptions = computed(() => (props.jobs ?? []).map((j) => ({ value: j.id, label: `${j.name} · ${count(j.counts.images, 'image')}`, short: j.name })));
  const teacherHint = computed(() => {
    const t = teacher.get();
    return t && eligible(t)
      ? `\`${t.arch}\` at ${num(t.imgsz)} px. ${teacherEntry.get() ? 'Its complete entry here is generation 0.' : 'It has no complete entry here yet: one runs first, as generation 0.'}`
      : 'A model whose classes are this benchmark’s, the same names in the same order: its labels are written by class index.';
  });
  const forecastState = computed(() => {
    const fc = forecast.get()!;
    return fc.from === 'none' ? 'none' : fc.starved ? (fc.stops ? 'stops' : 'starved') : 'ok';
  });

  return (
    <Dialog
      bind:open={open}
      onbeforeclose={() => !busy.peek()}
      size="lg"
      title="New loop"
      description={`Does a model’s own labeling train a better model? Each generation the model labels a pool of images by itself, a student trains on those labels alone, and ${props.bench.name} grades the student, the next generation’s teacher.`}
      footer={() => (
        <>
          {/* What starting it does and costs, beside the button that does it. */}
          {missingWhy.get() ? (
            <p class="why" data-gate-why>
              {missingWhy.get()}
            </p>
          ) : forecast.get()?.starved ? (
            <p class="why warn-text num" role="status">
              <TriangleAlert size={14} strokeWidth={1.75} aria-hidden="true" />
              {forecast.get()!.threshold === null
                ? 'Without a confidence the loop would stop before labeling'
                : forecast.get()!.stops
                  ? 'Its labels would be too few: the loop would stop at labeling'
                  : 'Its labels would be too few to train a student on'}
            </p>
          ) : (
            <p class="why s-text num" role="status">
              {summary.get()}
            </p>
          )}
          <Button variant="ghost" disabled={busy.get()} onclick={() => !busy.peek() && open.set(false)}>
            Cancel
          </Button>
          <Gate reason={reason.get()}>
            {(why) => (
              <Button variant="primary" type="submit" form={`${uid}-form`} icon={Repeat} loading={busy.get()} disabled={!!why()} aria-describedby={why()}>
                {forecast.get()?.starved && forecast.get()!.threshold !== null ? 'Start anyway' : 'Start loop'}
              </Button>
            )}
          </Gate>
        </>
      )}
    >
      <form id={`${uid}-form`} class="form" onsubmit={submit} novalidate>
        <Field label="Teacher" for={`${uid}-teacher`} hint={teacherHint.get()}>
          <Select id={`${uid}-teacher`} bind:value={teacherBind} disabled={!props.models?.length} options={teacherOptions.get()} />
        </Field>

        <fieldset class="group">
          <legend class="group-label">The pool it labels</legend>
          <Segmented label="Where the pool's images come from" bind:value={poolKind} options={POOL_OPTIONS} />
          {poolKind.get() === 'dataset' ? (
            <div class="row2">
              <Field label="Dataset" for={`${uid}-ds`}>
                <Select id={`${uid}-ds`} bind:value={datasetBind} disabled={!props.datasets?.length} options={datasetOptions.get()} />
              </Field>
              <Field label="Split">
                <Segmented label="Split" bind:value={split} options={splitOptions.get()} />
              </Field>
            </div>
          ) : (
            <Field label="Labeling job" for={`${uid}-job`}>
              <Select id={`${uid}-job`} bind:value={jobBind} disabled={!props.jobs?.length} options={jobOptions.get()} />
            </Field>
          )}
          <p class="note">
            The pool’s own labels are never read: the teacher labels every image by itself. Images that are also in this benchmark are left out
            {overlaps.get() ? ', and this is the split the benchmark was made from: pick another' : ''}.
          </p>
        </fieldset>

        <Field label="Generations" hint="Each generation is one student: labeled by the one before, trained, benchmarked.">
          <Segmented label="Generations" bind:value={generations} options={GENERATION_OPTIONS} />
        </Field>

        <fieldset class="group">
          <legend class="group-label">The students</legend>
          <div class="row2">
            <Field label="Architecture" for={`${uid}-arch`} error={fieldErr('arch')}>
              <Select id={`${uid}-arch`} bind:value={archBind} options={archOptions.get()} />
            </Field>
            <Field
              label="Starting weights"
              for={`${uid}-w`}
              error={fieldErr('weights')}
              hint={weights.get() === 'none' ? 'Trained from scratch on the teacher’s labels.' : undefined}
            >
              <Select id={`${uid}-w`} bind:value={weightsBind} options={weightOptions.get()} />
            </Field>
            <Field label="Epochs" for={`${uid}-ep`} error={fieldErr('epochs') ?? showNow(epochs.get(), errs.get().epochs)}>
              <Input id={`${uid}-ep`} bind:value={epochs} inputmode="numeric" invalid={!!showNow(epochs.get(), errs.get().epochs)} />
            </Field>
            <Field
              label="Image size"
              for={`${uid}-sz`}
              error={fieldErr('imgsz') ?? showNow(imgsz.get(), errs.get().imgsz)}
              hint={teacher.get() ? `The teacher’s is ${teacher.get()!.imgsz} px.` : undefined}
            >
              <Input id={`${uid}-sz`} bind:value={sizeBind} inputmode="numeric" suffix="px" invalid={!!showNow(imgsz.get(), errs.get().imgsz)} />
            </Field>
          </div>
        </fieldset>

        <div class="conf">
          <Switch
            label="Keep the teacher’s labels from its own threshold"
            bind:checked={ownBind}
            disabled={needConf.get()}
            hint={useOwn.get() || needConf.get() ? thresholdText.get() : 'Every teacher keeps its detections from the confidence below.'}
          />
          {!useOwn.get() ? (
            <Field
              label="Keep labels from confidence"
              for={`${uid}-conf`}
              error={showNow(minConf.get(), errs.get().conf)}
              hint={
                unproposed.get() && found.get()?.bestConf === null
                  ? 'This teacher has no measured best confidence: set one yourself.'
                  : needConf.get()
                    ? 'Required for this teacher.'
                    : undefined
              }
            >
              <div class="narrow">
                <Input id={`${uid}-conf`} bind:value={minConf} inputmode="decimal" placeholder="e.g. 0.30" invalid={!!showNow(minConf.get(), errs.get().conf)} />
              </div>
            </Field>
          ) : null}
          {forecast.get() && forecastWords.get() ? (
            // What the first teacher's labels would be, from its entry's curve: said plainly when they are too few.
            <div
              class="fc"
              class:warn={forecast.get()!.starved}
              role={forecast.get()!.starved ? 'alert' : 'status'}
              data-forecast={forecastState.get()}
            >
              <span class="fc-icon" aria-hidden="true">
                {forecast.get()!.starved ? <TriangleAlert size={16} strokeWidth={1.75} /> : <Info size={16} strokeWidth={1.75} />}
              </span>
              <div class="fc-body">
                {forecastWords.get()!.title ? <p class="fc-title">{forecastWords.get()!.title}</p> : null}
                <p class="fc-text">
                  {forecastWords.get()!.text}
                  {forecastWords.get()!.remedy ? (
                    <>
                      {' '}
                      {forecastWords.get()!.remedy}
                    </>
                  ) : null}
                  {!teacherEntry.get() ? <>{' '}Read from its entry on part of the benchmark; the loop runs a complete one first.</> : null}
                </p>
                {forecast.get()!.starved && useOwn.get() ? (
                  <div class="fc-act">
                    <Button size="sm" onclick={lower}>
                      Keep labels from a lower confidence
                    </Button>
                  </div>
                ) : null}
              </div>
            </div>
          ) : null}
        </div>

        <Field label="Name" for={`${uid}-name`} error={show(errs.get().name)} hint="Its datasets and runs are named after it: “<name> gen 1”.">
          <Input
            id={`${uid}-name`}
            bind:value={name}
            placeholder={teacher.get() ? `${teacher.get()!.name} loop` : 'Loop name'}
            autocomplete="off"
            invalid={!!show(errs.get().name)}
          />
        </Field>

        <p class="note">
          Its runs queue with the others, its entries with the other model entries. The datasets, runs and models it makes stay when the loop is deleted.
        </p>
        {error.get() ? <ErrorState error={error.get()} title="Could not start the loop" compact /> : null}
      </form>
    </Dialog>
  );
}

css`
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
`;
