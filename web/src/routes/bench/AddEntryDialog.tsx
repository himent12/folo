/**
 * "Add entry": one contestant's run of the benchmark (docs/BENCH.md, Who can run), three kinds.
 * - A model of this server: its classes against the benchmark's (matched by name; it scores 0 on the ones it lacks),
 *   how many images, and whether to run the plan's conditions.
 * - A coding agent, picked as agent labeling picks it (../label/AgentFields), with honest defaults (48 images,
 *   unchanged only) and what they mean: the score is partial (agents give no confidences; without the conditions
 *   there is no robustness), and each image is a request.
 * - Any other model: its predictions as a JSON file, checked here with the server's rules ($lib/bench/external)
 *   before anything is sent, whether it gives confidences, and how to make the file (the example script for anything
 *   but a small benchmark; the API's requests for this benchmark, copyable). The file is read and parsed once, after
 *   "Reading the file…" has painted (a 200 MB file takes seconds), and its records are held once. A file whose
 *   request would be large (over PART_BYTES of JSON; the server takes 64 MiB a request) goes as an open entry
 *   filled in parts, a pass at a time, with the records sent of all of them as a bar, then finished; any failure once
 *   the entry exists stops it (which discards it) and shows the server's message, and closing the dialog mid-send
 *   asks first, then stops it too. A small file is one request, scored in it.
 * On success the dialog closes and the page shows the new entry.
 */
import FileJson from '$icons/file-json';
import Plus from '$icons/plus';
import { bindable, computed, createUniqueId, css, effect, For, signal, tick, untrack } from 'fw';
import { hub } from '$lib/api/index';
import type { Bench, BenchCatalog, BenchEntry, BenchPrediction, LabelAgent, Model, StartBenchEntry } from '$lib/api/types';
import { checkPredictions, estimateBytes, ExternalSendError, PART_BYTES, parsePredictions, partsOf, sendExternal, type PredictionSummary } from '$lib/bench/external';
import { planPasses } from '$lib/bench/plan';
import { AGENT_IMAGES, IMAGES_PAGE_MAX, isEntryActive, BENCH_NAME_MAX } from '$lib/bench/rules';
import { classColor } from '$lib/design/tokens';
import { bytes, num } from '$lib/format';
import { href } from '$lib/router';
import Button from '$lib/ui/Button';
import CommandLine from '$lib/ui/CommandLine';
import Dialog from '$lib/ui/Dialog';
import ErrorState from '$lib/ui/ErrorState';
import Field from '$lib/ui/Field.tsx';
import Gate from '$lib/ui/Gate';
import Input from '$lib/ui/Input';
import Progress from '$lib/ui/Progress';
import Segmented from '$lib/ui/Segmented';
import Select from '$lib/ui/Select';
import Spinner from '$lib/ui/Spinner';
import Switch from '$lib/ui/Switch';
import AgentFields from '../label/AgentFields';
import type { AgentChoice } from '../label/agentChoice';
import { modelLabel } from '../label/words';
import { imagesText, noModelHelp } from './words';

export type EntryKind = 'model' | 'agent' | 'external';

export interface AddEntryDialogProps {
  open?: boolean;
  kind?: EntryKind;
  bench: Bench;
  catalog: BenchCatalog | undefined;
  /** The leaderboard, to say what a model entry waits for. */
  entries: readonly BenchEntry[];
  models: readonly Model[] | undefined;
  modelsError: unknown;
  agents: readonly LabelAgent[] | undefined;
  agentsError: unknown;
  retryAgents: () => void;
  choice: AgentChoice;
  /** The new entry, as the server answered (the page shows it at once). */
  onstarted: (entry: BenchEntry) => void;
}

type Check = { ok: true; predictions: BenchPrediction[]; summary: PredictionSummary } | { ok: false; error: string };

const PER_CALL: Record<string, number> = { claude: 4, codex: 8 };

export default function AddEntryDialog(props: AddEntryDialogProps) {
  const open = bindable(props, 'open', false);
  const kind = bindable(props, 'kind', 'model' as EntryKind);

  const uid = createUniqueId();
  const busy = signal(false);
  const error = signal<unknown>(undefined);
  const touched = signal(false);

  // ---------- a model ----------
  const modelId = signal('');
  const modelImages = signal('');
  const modelConditions = signal(true);
  // ---------- an agent ----------
  const agentImages = signal(String(AGENT_IMAGES));
  const agentConditions = signal(false);
  // ---------- external ----------
  const extName = signal('');
  const scored = signal(true);
  const file = signal<File | null>(null);
  const reading = signal(false);
  const check = signal<Check | null>(null);
  let fileInput: HTMLInputElement | undefined;
  /** The benchmark's image ids, for the predictions' check (loaded once the external kind is shown). */
  const imageIds = signal<string[] | null>(null);
  /** A large file on its way in parts: the records sent of all of them; `confirm` while closing asks first. */
  const sending = signal<{ sent: number; total: number } | null>(null);
  /** After a failed send: what it left on the server, when its clean-up failed too (null: nothing is left). */
  const left = signal<string | null>(null);
  const confirmStop = signal(false);
  let abort: AbortController | null = null;

  // Every opening starts fresh on the kind it was opened with; the first model that shares classes is picked.
  effect(() => {
    if (!open.get()) return;
    untrack(() => {
      error.set(undefined);
      touched.set(false);
      busy.set(false);
      modelImages.set('');
      modelConditions.set(true);
      agentImages.set(String(Math.min(AGENT_IMAGES, props.bench.images)));
      agentConditions.set(false);
      extName.set('');
      scored.set(true);
      file.set(null);
      check.set(null);
      modelId.set('');
      sending.set(null);
      left.set(null);
      confirmStop.set(false);
    });
  });

  const lower = computed(() => new Set(props.bench.classes.map((c) => c.toLowerCase())));
  const shared = (m: Model) => m.classes.filter((c) => lower.get().has(c.toLowerCase())).length;
  // Models with the most classes in common first; those with none listed, off, at the end.
  const ranked = computed(() => [...(props.models ?? [])].sort((a, b) => shared(b) - shared(a) || a.name.localeCompare(b.name)));
  effect(() => {
    if (!open.get() || modelId.get() || !ranked.get().length) return;
    const first = ranked.get().find((m) => shared(m) > 0);
    if (first) untrack(() => modelId.set(first.id));
  });
  const model = computed(() => props.models?.find((m) => m.id === modelId.get()));
  const knows = computed(() => {
    const m = model.get();
    return m ? new Set(m.classes.map((c) => c.toLowerCase())) : new Set<string>();
  });
  const missing = computed(() => (model.get() ? props.bench.classes.filter((c) => !knows.get().has(c.toLowerCase())) : []));
  const modelOptions = computed(() =>
    ranked.get().map((m) => {
      const n = shared(m);
      const all = props.bench.classes.length;
      return {
        value: m.id,
        label: `${m.name} · ${n === all ? 'every class' : `${n} of ${all} classes`}`,
        short: m.name,
        group: n === all ? 'Every class of the benchmark' : n > 0 ? 'Some of its classes' : 'None of its classes',
        disabled: n === 0,
      };
    }),
  );
  /** Models are there, but none knows a class of the benchmark: the picker has nothing to offer, the dialog says why. */
  const noModel = computed(() => (props.models?.length && !props.models.some((m) => shared(m) > 0) ? noModelHelp(props.bench) : null));
  /** A model entry waits for the model entry that runs (they take the GPU one at a time). */
  const ahead = computed(() => props.entries.find((e) => e.contestant.kind === 'model' && isEntryActive(e)));
  const passes = computed(() => planPasses(props.bench.plan));

  // ---------- the agent ----------
  const agent = computed(() => props.choice.agent);
  const agentN = computed(() => Number(agentImages.get()));
  const agentCalls = computed(() => {
    const a = agent.get();
    const n = agentN.get();
    return a && Number.isInteger(n) && n > 0 ? Math.ceil((n * (1 + (agentConditions.get() ? passes.get() : 0))) / (PER_CALL[a.id] ?? 4)) : 0;
  });

  // ---------- external: the file, read and checked as it is picked ----------
  effect(() => {
    if (!open.get() || kind.get() !== 'external' || imageIds.get()) return;
    const id = props.bench.id;
    untrack(async () => {
      const ids: string[] = [];
      try {
        for (let off = 0; off < props.bench.images; off += IMAGES_PAGE_MAX) {
          const p = await hub.api.benchImages(id, { offset: off, limit: IMAGES_PAGE_MAX });
          ids.push(...p.images.map((i) => i.id));
          if (!p.images.length) break;
        }
        imageIds.set(ids);
      } catch {
        // The server checks them anyway; the rest of the file is still checked here.
      }
    });
  });
  async function pick(e: Event & { currentTarget: HTMLInputElement }) {
    const input = e.currentTarget;
    const f = input.files?.[0];
    input.value = '';
    if (!f) return;
    file.set(f);
    reading.set(true);
    check.set(null);
    try {
      // "Reading the file…" paints before the parse holds the page (seconds for a 200 MB file); the text is let go
      // as soon as it is parsed, and the records are checked where they are, never copied.
      await tick();
      await new Promise((r) => requestAnimationFrame(() => setTimeout(r, 0)));
      const parsed = parsePredictions(await f.text());
      if ('error' in parsed) check.set({ ok: false, error: parsed.error });
      else {
        const catalog = props.catalog;
        const c = checkPredictions(parsed.records, {
          imageIds: imageIds.peek(),
          classes: props.bench.classes.length,
          plan: props.bench.plan,
          catalog: catalog?.conditions.map((x) => x.id) ?? props.bench.plan.conditions.map((x) => x.condition),
          names: Object.fromEntries((catalog?.conditions ?? []).map((x) => [x.id, x.name])),
        });
        check.set(c);
        if (c.ok) scored.set(c.summary.scores);
        if (c.ok && !extName.peek().trim()) extName.set(f.name.replace(/\.json$/i, '').replace(/[-_ ]?predictions?$/i, '') || 'external');
      }
    } catch {
      check.set({ ok: false, error: 'The file could not be read.' });
    } finally {
      reading.set(false);
    }
  }
  const condName = (id: string) => props.catalog?.conditions.find((c) => c.id === id)?.name ?? id;
  /** How a checked file goes: in one request, or (over PART_BYTES of JSON) as an open entry in this many parts. */
  const parts = computed(() => {
    const c = check.get();
    return c?.ok && estimateBytes(c.predictions) > PART_BYTES ? partsOf(c.predictions, props.bench.plan).length : 0;
  });
  const api = computed(() => `${hub.serverUrl.replace(/\/$/, '')}/api/bench/benchmarks/${props.bench.id}`);

  // ---------- validation, the request ----------
  const imagesError = (text: string, empty: boolean) => {
    if (!text.trim()) return empty ? null : 'Required.';
    const n = Number(text);
    return Number.isInteger(n) && n >= 1 && n <= props.bench.images ? null : `A whole number from 1 to ${num(props.bench.images)}.`;
  };
  const errs = computed(() => {
    const k = kind.get();
    const name = extName.get().trim();
    return {
      images: k === 'model' ? imagesError(modelImages.get(), true) : k === 'agent' ? imagesError(agentImages.get(), false) : null,
      name: k === 'external' ? (!name ? 'An external entry needs a name.' : name.length > BENCH_NAME_MAX ? `At most ${BENCH_NAME_MAX} characters.` : null) : null,
    };
  });
  const show = (e: string | null) => (touched.get() ? e : null);
  /** The Images field says a value out of range as soon as it is typed, not on submit (an empty one is said then). */
  const imagesShown = computed(() => {
    const text = kind.get() === 'model' ? modelImages.get() : agentImages.get();
    return touched.get() || text.trim() ? errs.get().images : null;
  });
  const missingWhy = computed(() => {
    const k = kind.get();
    if (k === 'model') {
      const models = props.models;
      const m = model.get();
      return props.modelsError && !models
        ? 'Could not load the models'
        : !models
          ? 'Loading models'
          : !models.length
            ? 'Train or import a model first'
            : noModel.get()
              ? 'No model knows these classes'
              : !m
              ? 'Pick a model'
              : missing.get().length === props.bench.classes.length
                ? `None of ${m.name}'s classes is a class of this benchmark`
                : null;
    }
    if (k === 'agent') return !agent.get() ? (props.agents ? 'Install Claude Code or Codex on the server’s machine' : 'Checking which agents are installed') : null;
    const c = check.get();
    return reading.get()
      ? 'Reading the file'
      : !file.get()
        ? 'Choose a predictions file'
        : c && !c.ok
          ? 'Fix the predictions file first'
          : !extName.get().trim()
            ? 'Enter a name'
            : null;
  });
  const shownErr = computed(() => (touched.get() ? Object.values(errs.get()).find((e) => e) : (imagesShown.get() ?? undefined)));
  const reason = computed(() => missingWhy.get() ?? (shownErr.get() ? 'Fix the fields first' : null));

  /** Sends the file: one request, or an open entry in parts with its bar; nothing is left behind if it fails. */
  async function sendFile() {
    const c = check.peek();
    if (!c?.ok) return;
    abort = new AbortController();
    left.set(null);
    sending.set({ sent: 0, total: c.predictions.length });
    const benchId = props.bench.id;
    try {
      const entry = await sendExternal(
        {
          start: (b) => hub.api.startBenchEntry(benchId, b),
          send: (id, ps) => hub.api.sendBenchPredictions(benchId, id, ps),
          finish: (id) => hub.api.finishBenchEntry(benchId, id),
          get: (id) => hub.api.benchEntry(benchId, id),
          stop: (id) => hub.api.stopBenchEntry(benchId, id),
          remove: (id) => hub.api.deleteBenchEntry(benchId, id),
        },
        { name: extName.peek().trim(), scored: scored.peek(), predictions: c.predictions, plan: props.bench.plan },
        (p) => sending.set({ sent: p.sent, total: p.total }),
        abort.signal,
      );
      open.set(false);
      props.onstarted(entry);
    } catch (err) {
      // What the server said, and what it left: nothing (stopped and deleted), or exactly what to clean up and where.
      left.set(err instanceof ExternalSendError ? err.left : null);
      // Stopped by the person with nothing left: the dialog closes; anything else is said in it.
      if (abort?.signal.aborted && !left.peek()) open.set(false);
      else error.set(err instanceof ExternalSendError ? err.cause : err);
    } finally {
      sending.set(null);
      confirmStop.set(false);
      abort = null;
    }
  }
  /** Closing while a large file is being sent asks first (it would discard the entry). */
  const beforeClose = () => {
    if (!sending.peek()) return true;
    confirmStop.set(true);
    return false;
  };
  const stopSending = () => abort?.abort(new DOMException('Stopped', 'AbortError'));

  function body(): StartBenchEntry | null {
    const k = kind.peek();
    if (k === 'model') {
      const m = model.peek();
      if (!m) return null;
      const images = modelImages.peek();
      return { contestant: { kind: 'model', model_id: m.id }, ...(images.trim() ? { max_images: Number(images) } : {}), conditions: modelConditions.peek() };
    }
    if (k === 'agent') {
      const a = props.choice.request();
      if (!a) return null;
      return { contestant: { kind: 'agent', agent: a.agent, model: a.model, effort: a.effort }, max_images: Number(agentImages.peek()), conditions: agentConditions.peek() };
    }
    // An external file goes by sendFile.
    return null;
  }

  async function submit(e: SubmitEvent) {
    e.preventDefault();
    touched.set(true);
    error.set(undefined);
    if (Object.values(errs.peek()).some((x) => x) || reason.peek() || busy.peek()) return;
    if (kind.peek() === 'external') {
      busy.set(true);
      try {
        await sendFile();
      } finally {
        busy.set(false);
      }
      return;
    }
    const b = body();
    if (!b) return;
    busy.set(true);
    try {
      const entry = await hub.api.startBenchEntry(props.bench.id, b);
      open.set(false);
      props.onstarted(entry);
    } catch (err) {
      error.set(err);
    } finally {
      busy.set(false);
    }
  }

  const kindOptions = [
    { value: 'model' as const, label: 'Model' },
    { value: 'agent' as const, label: 'Agent' },
    { value: 'external' as const, label: 'External' },
  ];

  return (
    <Dialog
      bind:open={open}
      onbeforeclose={beforeClose}
      title="Add entry"
      description={`Run a contestant on ${props.bench.name}: a model of this server, a coding agent looking at the images, or the predictions of any other model.`}
      footer={() => (
        <>
          {confirmStop.get() ? (
            /* Closing while a large file is on its way: the entry would be discarded, so it asks first. */
            <>
              <p class="why" role="alert">
                Stop sending? The entry is discarded.
              </p>
              <Button variant="ghost" onclick={() => confirmStop.set(false)}>
                Keep sending
              </Button>
              <Button variant="destructive" onclick={stopSending}>
                Stop and discard
              </Button>
            </>
          ) : (
            <>
              {missingWhy.get() ? (
                <p class="why" data-gate-why>
                  {missingWhy.get()}
                </p>
              ) : null}
              <Button variant="ghost" onclick={() => (beforeClose() ? open.set(false) : undefined)}>
                Cancel
              </Button>
              <Gate reason={reason.get()}>
                {(why) => (
                  <Button variant="primary" type="submit" form={`${uid}-form`} icon={Plus} loading={busy.get()} disabled={!!why()} aria-describedby={why()}>
                    {kind.get() === 'external' ? 'Score and add' : 'Add entry'}
                  </Button>
                )}
              </Gate>
            </>
          )}
        </>
      )}
    >
      <form id={`${uid}-form`} class="form" onsubmit={submit} novalidate>
        <Segmented label="Contestant" bind:value={kind} options={kindOptions} />

        {kind.get() === 'model' ? (
          <>
            <p class="note lead-note">FOLO runs a model of this server on the images, unchanged and under the plan's conditions, and reports its speed.</p>
            <Field
              label="Model"
              for={`${uid}-model`}
              hint={model.get() ? `\`${model.get()!.arch}\` at ${num(model.get()!.imgsz)} px, matched to the benchmark's classes by name.` : undefined}
            >
              <Select id={`${uid}-model`} bind:value={modelId} disabled={!props.models?.length || !!noModel.get()} options={modelOptions.get()} />
            </Field>
            {noModel.get() ? (
              <div class="no-model" role="status">
                <p class="nm-title">{noModel.get()!.title}</p>
                <p class="note">{noModel.get()!.text}</p>
                <div class="nm-actions">
                  <Button size="sm" href={href(noModel.get()!.action.path)}>
                    {noModel.get()!.action.label}
                  </Button>
                  <Button size="sm" variant="ghost" onclick={() => kind.set('agent')}>
                    Use an agent
                  </Button>
                  <Button size="sm" variant="ghost" onclick={() => kind.set('external')}>
                    Send predictions
                  </Button>
                </div>
              </div>
            ) : null}
            {props.modelsError && !props.models ? <ErrorState error={props.modelsError} title="Could not load the models" compact /> : null}
            {model.get() ? (
              <div class="classes">
                <p class="cl-head">
                  {!missing.get().length
                    ? 'It knows every class of the benchmark.'
                    : missing.get().length === props.bench.classes.length
                      ? "It knows none of the benchmark's classes: the server refuses it."
                      : `It knows ${props.bench.classes.length - missing.get().length} of the ${props.bench.classes.length} classes; on the ${missing.get().length} it lacks it scores 0.`}
                </p>
                <ul class="chips" aria-label="The benchmark's classes">
                  <For each={props.bench.classes} key={(c) => c}>
                    {(c, i) => {
                      const has = () => knows.get().has(c().toLowerCase());
                      return (
                        <li class="chip" class:miss={!has()}>
                          {has() ? <span class="sw" style:background={classColor(i())}></span> : null}
                          <span>{c()}</span>
                          <span class="sr-only">{has() ? ', known' : ', missing: scores 0'}</span>
                        </li>
                      );
                    }}
                  </For>
                </ul>
              </div>
            ) : null}
            <Field
              label="Images"
              for={`${uid}-mimages`}
              error={imagesShown.get()}
              hint="Empty runs all of them. Fewer runs the first of the benchmark's seeded order, a fair sample, but the entry is partial."
            >
              <div class="narrow">
                <Input
                  id={`${uid}-mimages`}
                  bind:value={modelImages}
                  inputmode="numeric"
                  placeholder={`All ${num(props.bench.images)}`}
                  invalid={!!imagesShown.get()}
                />
              </div>
            </Field>
            {props.bench.plan.conditions.length ? (
              <Switch
                label="Run the plan's conditions"
                bind:checked={modelConditions}
                hint={
                  modelConditions.get()
                    ? `${passes.get()} more passes over each image: ${props.bench.plan.conditions.length} conditions at their severities.`
                    : 'Off: the unchanged images only. There is no robustness axis, and the entry is partial.'
                }
              />
            ) : null}
            {ahead.get() ? <p class="note">Model entries run one at a time: it waits for {ahead.get()!.name}.</p> : null}
          </>
        ) : kind.get() === 'agent' ? (
          <>
            <p class="note lead-note">
              The agent looks at each image and draws boxes, as agent labeling asks. Its score is partial: agents give no confidences, so calibration is not measured{agentConditions.get() ? '' : ', and on the unchanged images alone neither is robustness'}.
            </p>
            <div class="agent">
              <AgentFields
                agents={props.agents}
                agentsError={props.agentsError}
                retryAgents={props.retryAgents}
                choice={props.choice}
                effortHint={(a) => `More effort means more reasoning tokens per image and a slower entry. The default, ${a.default_effort}, is the best value for ${a.name}.`}
              />
            </div>
            <Field
              label="Images"
              for={`${uid}-aimages`}
              error={imagesShown.get()}
              hint={
                agent.get() && agentCalls.get()
                  ? `The first of the benchmark's seeded order, a fair sample: about ${num(agentCalls.get())} calls to ${agent.get()!.name} (${modelLabel(props.choice.model || null, agent.get())}).`
                  : "The first of the benchmark's seeded order, a fair sample."
              }
            >
              <div class="narrow">
                <Input id={`${uid}-aimages`} bind:value={agentImages} inputmode="numeric" invalid={!!imagesShown.get()} />
              </div>
            </Field>
            {props.bench.plan.conditions.length ? (
              <Switch
                label="Also run the plan's conditions"
                bind:checked={agentConditions}
                hint={
                  agentConditions.get()
                    ? `Each image again under ${passes.get()} conditions and severities: ${1 + passes.get()} times the calls.`
                    : 'Off by default: every condition is another call per image.'
                }
              />
            ) : null}
          </>
        ) : (
          <>
            <p class="note lead-note">Any model that runs elsewhere: fetch the images, run it, and send its boxes as a file. It is scored once all of it has arrived.</p>
            <div class="file">
              <input ref={(e) => (fileInput = e)} type="file" accept="application/json,.json" hidden onchange={pick} />
              <div class="f-row">
                <Button icon={FileJson} onclick={() => fileInput?.click()}>
                  {file.get() ? 'Choose another file' : 'Choose predictions file'}
                </Button>
                {file.get() ? (
                  <span class="f-name">
                    <span class="truncate" title={file.get()!.name}>
                      {file.get()!.name}
                    </span>
                    <span class="faint num">{bytes(file.get()!.size)}</span>
                  </span>
                ) : null}
              </div>
              {reading.get() ? (
                <p class="f-state busy" role="status">
                  <Spinner size={12} /> Reading the file{file.get() && file.get()!.size > 8 * 2 ** 20 ? ` (${bytes(file.get()!.size)})` : ''}…
                </p>
              ) : sending.get() ? (
                /* The records sent of all of them, part by part: a real bar, not a spinner. */
                <div class="f-send" role="status">
                  <Progress value={sending.get()!.total ? sending.get()!.sent / sending.get()!.total : 0} size="sm" tone="accent" label="Predictions sent" />
                  <p class="f-state num">
                    Sending predictions · {num(sending.get()!.sent)} of {num(sending.get()!.total)} records
                  </p>
                </div>
              ) : check.get()?.ok ? (
                <p class="f-state ok num" role="status">
                  {okText()}
                  {okSummary().missing.length && okSummary().conditions.length ? <>. Left out: {okSummary().missing.map(condName).join(', ')}; the entry is partial.</> : null}
                  {parts.get() ? <>{` Sent in ${num(parts.get())} parts, a pass at a time.`}</> : null}
                </p>
              ) : check.get() ? (
                <p class="f-state bad" role="alert">
                  {(check.get() as { error: string }).error}
                </p>
              ) : (
                <p class="f-state">
                  A JSON file of <span class="mono">{'{ "predictions": [...] }'}</span> or a bare array: one record per image and condition, boxes in pixels. Checked here before it is sent.
                </p>
              )}
            </div>
            <Field label="Name on the leaderboard" for={`${uid}-name`} error={show(errs.get().name)}>
              <Input id={`${uid}-name`} bind:value={extName} placeholder="e.g. yolo26n" autocomplete="off" invalid={!!show(errs.get().name)} />
            </Field>
            <Switch
              label="The model gives confidences"
              bind:checked={scored}
              hint={scored.get() ? 'Every box’s score is read; boxes without one count as 1.' : 'Every score is taken as 1: calibration is not measured, and self-labeling has one operating point.'}
            />
            <details class="how">
              <summary>
                <span class="chev" aria-hidden="true"></span>How to make the file
              </summary>
              <div class="how-body">
                <p>
                  For anything but a small benchmark, use the example script <span class="mono">examples/bench_external.py</span> from the FOLO repository: put your model behind its one Python function, and it fetches every image under every condition, runs the model and sends the predictions in parts.
                </p>
                <CommandLine command={`python examples/bench_external.py --server ${hub.serverUrl.replace(/\/$/, '')} --bench ${props.bench.id} --name my-detector`} label="Run the example script" />
                <p>By hand: list the images and fetch each one, unchanged and under the plan's conditions; run your model on them and save its boxes, in pixels, to choose above. In a shell:</p>
                <CommandLine command={`api=${api.get()}`} label="This benchmark's address" />
                <CommandLine command={`curl -s "$api/images?limit=1000"`} label="List the images" />
                <CommandLine command={`curl -s "$api/images/1?condition=haze&severity=3" -o haze3.jpg`} label="Fetch an image under a condition" />
                <p>
                  A small entry goes in one request (up to 64 MiB), its name and predictions in <span class="mono">entry.json</span>; a larger one is opened with <span class="mono">"open": true</span>, sent in parts to its <span class="mono">predictions</span>, and finished. This dialog does that for you.
                </p>
                <CommandLine command={`curl -s "$api/entries" -H 'Content-Type: application/json' -d @entry.json`} label="Send a small entry" />
              </div>
            </details>
          </>
        )}
        {props.bench.entries === 0 && props.bench.plan.conditions.length ? (
          <p class="note first">The first entry fixes the plan ({props.bench.plan.conditions.length} conditions) so that every entry is graded on the same test.</p>
        ) : null}
        {error.get() ? (
          <ErrorState
            error={error.get()}
            title={kind.get() === 'external' && parts.get() ? (left.get() ? 'Could not add the entry' : 'Could not add the entry: nothing was kept') : 'Could not add the entry'}
            compact
          />
        ) : null}
        {left.get() ? (
          <p class="note warn-left" role="alert">
            {left.get()!.charAt(0).toUpperCase() + left.get()!.slice(1)}
          </p>
        ) : null}
      </form>
    </Dialog>
  );

  /** The checked file's summary (only rendered while the check is ok; empty otherwise, never read then). */
  function okSummary(): PredictionSummary {
    const c = check.get();
    return c?.ok ? c.summary : { images: 0, conditions: [], missing: [], boxes: 0, scores: false };
  }
  function okText() {
    const s = okSummary();
    return `${imagesText(s.images)} answered · ${s.conditions.length ? `${s.conditions.length} of ${props.bench.plan.conditions.length} conditions` : 'unchanged only'} · ${num(s.boxes)} boxes${s.scores ? ' with confidences' : ', no confidences'}`;
  }
}

css`
  .form {
    display: grid;
    gap: var(--space-4);
  }
  .form :global(.seg) {
    justify-self: start;
  }
  .agent {
    display: grid;
    gap: var(--space-4);
  }
  .narrow {
    max-width: 200px;
  }
  .classes,
  .no-model {
    display: grid;
    gap: var(--space-2);
  }
  .nm-title {
    font-weight: var(--weight-medium);
  }
  .nm-actions {
    display: flex;
    flex-wrap: wrap;
    gap: var(--space-2);
  }
  .cl-head {
    color: var(--text-secondary);
  }
  .chips {
    display: flex;
    flex-wrap: wrap;
    gap: var(--space-1);
  }
  .chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    height: 24px;
    padding: 0 var(--space-2);
    border-radius: var(--radius-sm);
    background: var(--bg-active);
    font-size: var(--type-sm-size);
  }
  /* A class the model lacks: an outline, its name struck through, never a colour alone. */
  .chip.miss {
    background: none;
    border: 1px dashed var(--line-strong);
    color: var(--text-tertiary);
    text-decoration: line-through;
  }
  .sw {
    width: 8px;
    height: 8px;
    border-radius: 2px;
  }
  .note,
  .f-state {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
  }
  .file {
    display: grid;
    gap: var(--space-2);
  }
  .f-row {
    display: flex;
    align-items: center;
    gap: var(--space-3);
    min-width: 0;
  }
  .f-name {
    display: flex;
    align-items: baseline;
    gap: var(--space-2);
    min-width: 0;
  }
  .warn-left {
    color: var(--warning-text);
  }
  .f-send {
    display: grid;
    gap: var(--space-2);
  }
  .f-state.busy {
    display: flex;
    align-items: center;
    gap: var(--space-2);
  }
  .f-state.ok {
    color: var(--text-primary);
  }
  .f-state.bad {
    color: var(--danger-text);
  }
  /* A disclosure in the one style (Models' engine benchmarks): a chevron turned while open, no marker. */
  .how summary {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    width: fit-content;
    min-height: var(--control-md);
    cursor: pointer;
    font-weight: var(--weight-medium);
    list-style: none;
  }
  .how summary::-webkit-details-marker {
    display: none;
  }
  .chev {
    flex-shrink: 0;
    width: 6px;
    height: 6px;
    margin: 0 var(--space-1) 0 2px;
    border-right: 1.5px solid var(--text-tertiary);
    border-bottom: 1.5px solid var(--text-tertiary);
    transform: rotate(-45deg);
    transition: transform var(--dur-fast) var(--ease-standard);
  }
  .how[open] .chev {
    transform: rotate(45deg);
  }
  .how-body {
    display: grid;
    gap: var(--space-2);
    margin-top: var(--space-2);
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
