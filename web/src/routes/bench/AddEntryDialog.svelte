<script lang="ts" module>
  export type EntryKind = 'model' | 'agent' | 'external';
</script>

<script lang="ts">
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
  import FileJson from '@lucide/svelte/icons/file-json';
  import Plus from '@lucide/svelte/icons/plus';
  import { tick, untrack } from 'svelte';
  import { hub } from '$lib/api/index.svelte';
  import type { Bench, BenchCatalog, BenchEntry, BenchPrediction, LabelAgent, Model, StartBenchEntry } from '$lib/api/types';
  import { checkPredictions, estimateBytes, ExternalSendError, PART_BYTES, parsePredictions, partsOf, sendExternal, type PredictionSummary } from '$lib/bench/external';
  import { planPasses } from '$lib/bench/plan';
  import { AGENT_IMAGES, IMAGES_PAGE_MAX, isEntryActive, BENCH_NAME_MAX } from '$lib/bench/rules';
  import { classColor } from '$lib/design/tokens';
  import { bytes, num } from '$lib/format';
  import Button from '$lib/ui/Button.svelte';
  import CommandLine from '$lib/ui/CommandLine.svelte';
  import Dialog from '$lib/ui/Dialog.svelte';
  import ErrorState from '$lib/ui/ErrorState.svelte';
  import Field from '$lib/ui/Field.svelte';
  import Gate from '$lib/ui/Gate.svelte';
  import Input from '$lib/ui/Input.svelte';
  import Progress from '$lib/ui/Progress.svelte';
  import Segmented from '$lib/ui/Segmented.svelte';
  import Select from '$lib/ui/Select.svelte';
  import Spinner from '$lib/ui/Spinner.svelte';
  import Switch from '$lib/ui/Switch.svelte';
  import AgentFields from '../label/AgentFields.svelte';
  import type { AgentChoice } from '../label/agentChoice.svelte';
  import { modelLabel } from '../label/words';
  import { imagesText } from './words';


  let {
    open = $bindable(false),
    kind = $bindable('model'),
    bench,
    catalog,
    entries,
    models,
    modelsError,
    agents,
    agentsError,
    retryAgents,
    choice,
    onstarted,
  }: {
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
  } = $props();

  const uid = $props.id();
  let busy = $state(false);
  let error = $state.raw<unknown>(undefined);
  let touched = $state(false);

  // ---------- a model ----------
  let modelId = $state('');
  let modelImages = $state('');
  let modelConditions = $state(true);
  // ---------- an agent ----------
  let agentImages = $state(String(AGENT_IMAGES));
  let agentConditions = $state(false);
  // ---------- external ----------
  let extName = $state('');
  let scored = $state(true);
  let file = $state.raw<File | null>(null);
  let reading = $state(false);
  let check = $state.raw<{ ok: true; predictions: BenchPrediction[]; summary: PredictionSummary } | { ok: false; error: string } | null>(null);
  let fileInput = $state<HTMLInputElement>();
  /** The benchmark's image ids, for the predictions' check (loaded once the external kind is shown). */
  let imageIds = $state.raw<string[] | null>(null);
  /** A large file on its way in parts: the records sent of all of them; `confirm` while closing asks first. */
  let sending = $state<{ sent: number; total: number } | null>(null);
  /** After a failed send: what it left on the server, when its clean-up failed too (null: nothing is left). */
  let left = $state<string | null>(null);
  let confirmStop = $state(false);
  let abort: AbortController | null = null;

  // Every opening starts fresh on the kind it was opened with; the first model that shares classes is picked.
  $effect(() => {
    if (!open) return;
    untrack(() => {
      error = undefined;
      touched = false;
      busy = false;
      modelImages = '';
      modelConditions = true;
      agentImages = String(Math.min(AGENT_IMAGES, bench.images));
      agentConditions = false;
      extName = '';
      scored = true;
      file = null;
      check = null;
      modelId = '';
      sending = null;
      left = null;
      confirmStop = false;
    });
  });

  const lower = $derived(new Set(bench.classes.map((c) => c.toLowerCase())));
  const shared = (m: Model) => m.classes.filter((c) => lower.has(c.toLowerCase())).length;
  // Models with the most classes in common first; those with none listed, off, at the end.
  const ranked = $derived(
    [...(models ?? [])].sort((a, b) => shared(b) - shared(a) || a.name.localeCompare(b.name)),
  );
  $effect(() => {
    if (!open || modelId || !ranked.length) return;
    const first = ranked.find((m) => shared(m) > 0);
    if (first) untrack(() => (modelId = first.id));
  });
  const model = $derived(models?.find((m) => m.id === modelId));
  const knows = $derived(model ? new Set(model.classes.map((c) => c.toLowerCase())) : new Set<string>());
  const missing = $derived(model ? bench.classes.filter((c) => !knows.has(c.toLowerCase())) : []);
  const modelOptions = $derived(
    ranked.map((m) => {
      const n = shared(m);
      return {
        value: m.id,
        label: `${m.name} · ${n === bench.classes.length ? 'every class' : `${n} of ${bench.classes.length} classes`}`,
        short: m.name,
        group: n === bench.classes.length ? 'Every class of the benchmark' : n > 0 ? 'Some of its classes' : 'None of its classes',
        disabled: n === 0,
      };
    }),
  );
  /** A model entry waits for the model entry that runs (they take the GPU one at a time). */
  const ahead = $derived(entries.find((e) => e.contestant.kind === 'model' && isEntryActive(e)));
  const passes = $derived(planPasses(bench.plan));

  // ---------- the agent ----------
  const agent = $derived(choice.agent);
  const agentN = $derived(Number(agentImages));
  const PER_CALL: Record<string, number> = { claude: 4, codex: 8 };
  const agentCalls = $derived(agent && Number.isInteger(agentN) && agentN > 0 ? Math.ceil((agentN * (1 + (agentConditions ? passes : 0))) / (PER_CALL[agent.id] ?? 4)) : 0);

  // ---------- external: the file, read and checked as it is picked ----------
  $effect(() => {
    if (!open || kind !== 'external' || imageIds) return;
    const id = bench.id;
    untrack(async () => {
      const ids: string[] = [];
      try {
        for (let off = 0; off < bench.images; off += IMAGES_PAGE_MAX) {
          const p = await hub.api.benchImages(id, { offset: off, limit: IMAGES_PAGE_MAX });
          ids.push(...p.images.map((i) => i.id));
          if (!p.images.length) break;
        }
        imageIds = ids;
      } catch {
        // The server checks them anyway; the rest of the file is still checked here.
      }
    });
  });
  async function pick(e: Event) {
    const input = e.currentTarget as HTMLInputElement;
    const f = input.files?.[0];
    input.value = '';
    if (!f) return;
    file = f;
    reading = true;
    check = null;
    try {
      // "Reading the file…" paints before the parse holds the page (seconds for a 200 MB file); the text is let go
      // as soon as it is parsed, and the records are checked where they are, never copied.
      await tick();
      await new Promise((r) => requestAnimationFrame(() => setTimeout(r, 0)));
      const parsed = parsePredictions(await f.text());
      if ('error' in parsed) check = { ok: false, error: parsed.error };
      else {
        const c = checkPredictions(parsed.records, {
          imageIds,
          classes: bench.classes.length,
          plan: bench.plan,
          catalog: catalog?.conditions.map((x) => x.id) ?? bench.plan.conditions.map((x) => x.condition),
          names: Object.fromEntries((catalog?.conditions ?? []).map((x) => [x.id, x.name])),
        });
        check = c;
        if (c.ok) scored = c.summary.scores;
        if (c.ok && !extName.trim()) extName = f.name.replace(/\.json$/i, '').replace(/[-_ ]?predictions?$/i, '') || 'external';
      }
    } catch {
      check = { ok: false, error: 'The file could not be read.' };
    } finally {
      reading = false;
    }
  }
  const condName = (id: string) => catalog?.conditions.find((c) => c.id === id)?.name ?? id;
  /** How a checked file goes: in one request, or (over PART_BYTES of JSON) as an open entry in this many parts. */
  const parts = $derived(check?.ok && estimateBytes(check.predictions) > PART_BYTES ? partsOf(check.predictions, bench.plan).length : 0);
  const api = $derived(`${hub.serverUrl.replace(/\/$/, '')}/api/bench/benchmarks/${bench.id}`);

  // ---------- validation, the request ----------
  const imagesError = (text: string, empty: boolean) => {
    if (!text.trim()) return empty ? null : 'Required.';
    const n = Number(text);
    return Number.isInteger(n) && n >= 1 && n <= bench.images ? null : `A whole number from 1 to ${num(bench.images)}.`;
  };
  const errs = $derived({
    images: kind === 'model' ? imagesError(modelImages, true) : kind === 'agent' ? imagesError(agentImages, false) : null,
    name: kind === 'external' ? (!extName.trim() ? 'An external entry needs a name.' : extName.trim().length > BENCH_NAME_MAX ? `At most ${BENCH_NAME_MAX} characters.` : null) : null,
  });
  const show = (e: string | null) => (touched ? e : null);
  const missingWhy = $derived(
    kind === 'model'
      ? modelsError && !models
        ? 'Could not load the models'
        : !models
          ? 'Loading models'
          : !models.length
            ? 'Train or import a model first'
            : !model
              ? 'Pick a model'
              : missing.length === bench.classes.length
                ? `None of ${model.name}'s classes is a class of this benchmark`
                : null
      : kind === 'agent'
        ? !agent
          ? agents
            ? 'Install Claude Code or Codex on the server’s machine'
            : 'Checking which agents are installed'
          : null
        : reading
          ? 'Reading the file'
          : !file
            ? 'Choose a predictions file'
            : check && !check.ok
              ? 'Fix the predictions file first'
              : !extName.trim()
                ? 'Enter a name'
                : null,
  );
  const shownErr = $derived(touched ? Object.values(errs).find((e) => e) : undefined);
  const reason = $derived(missingWhy ?? (shownErr ? 'Fix the fields first' : null));

  /** Sends the file: one request, or an open entry in parts with its bar; nothing is left behind if it fails. */
  async function sendFile() {
    if (!check?.ok) return;
    abort = new AbortController();
    left = null;
    sending = { sent: 0, total: check.predictions.length };
    try {
      const entry = await sendExternal(
        {
          start: (b) => hub.api.startBenchEntry(bench.id, b),
          send: (id, ps) => hub.api.sendBenchPredictions(bench.id, id, ps),
          finish: (id) => hub.api.finishBenchEntry(bench.id, id),
          stop: (id) => hub.api.stopBenchEntry(bench.id, id),
          remove: (id) => hub.api.deleteBenchEntry(bench.id, id),
        },
        { name: extName.trim(), scored, predictions: check.predictions, plan: bench.plan },
        (p) => (sending = { sent: p.sent, total: p.total }),
        abort.signal,
      );
      open = false;
      onstarted(entry);
    } catch (err) {
      // What the server said, and what it left: nothing (stopped and deleted), or exactly what to clean up and where.
      left = err instanceof ExternalSendError ? err.left : null;
      // Stopped by the person with nothing left: the dialog closes; anything else is said in it.
      if (abort?.signal.aborted && !left) open = false;
      else error = err instanceof ExternalSendError ? err.cause : err;
    } finally {
      sending = null;
      confirmStop = false;
      abort = null;
    }
  }
  /** Closing while a large file is being sent asks first (it would discard the entry). */
  const beforeClose = () => {
    if (!sending) return true;
    confirmStop = true;
    return false;
  };
  const stopSending = () => abort?.abort(new DOMException('Stopped', 'AbortError'));

  function body(): StartBenchEntry | null {
    if (kind === 'model') {
      if (!model) return null;
      return { contestant: { kind: 'model', model_id: model.id }, ...(modelImages.trim() ? { max_images: Number(modelImages) } : {}), conditions: modelConditions };
    }
    if (kind === 'agent') {
      const a = choice.request();
      if (!a) return null;
      return { contestant: { kind: 'agent', agent: a.agent, model: a.model, effort: a.effort }, max_images: Number(agentImages), conditions: agentConditions };
    }
    // An external file goes by sendFile.
    return null;
  }

  async function submit(e: SubmitEvent) {
    e.preventDefault();
    touched = true;
    error = undefined;
    if (Object.values(errs).some((x) => x) || reason || busy) return;
    if (kind === 'external') {
      busy = true;
      try {
        await sendFile();
      } finally {
        busy = false;
      }
      return;
    }
    const b = body();
    if (!b) return;
    busy = true;
    try {
      const entry = await hub.api.startBenchEntry(bench.id, b);
      open = false;
      onstarted(entry);
    } catch (err) {
      error = err;
    } finally {
      busy = false;
    }
  }
</script>

<Dialog bind:open onbeforeclose={beforeClose} title="Add entry" description="Run a contestant on {bench.name}: a model of this server, a coding agent looking at the images, or the predictions of any other model.">
  <form id="{uid}-form" class="form" onsubmit={submit} novalidate>
    <Segmented
      label="Contestant"
      bind:value={kind}
      options={[
        { value: 'model', label: 'Model' },
        { value: 'agent', label: 'Agent' },
        { value: 'external', label: 'External' },
      ]}
    />

    {#if kind === 'model'}
      <p class="note lead-note">FOLO runs a model of this server on the images, unchanged and under the plan's conditions, and reports its speed.</p>
      <Field label="Model" for="{uid}-model" hint={model ? `\`${model.arch}\` at ${num(model.imgsz)} px, matched to the benchmark's classes by name.` : undefined}>
        <Select id="{uid}-model" bind:value={modelId} disabled={!models?.length} options={modelOptions} />
      </Field>
      {#if modelsError && !models}<ErrorState error={modelsError} title="Could not load the models" compact />{/if}
      {#if model}
        <div class="classes">
          <p class="cl-head">
            {#if !missing.length}It knows every class of the benchmark.{:else if missing.length === bench.classes.length}It knows none of the benchmark's classes: the server refuses it.{:else}It knows {bench.classes.length - missing.length} of the {bench.classes.length} classes; on the {missing.length} it lacks it scores 0.{/if}
          </p>
          <ul class="chips" aria-label="The benchmark's classes">
            {#each bench.classes as c, i (c)}
              {@const has = knows.has(c.toLowerCase())}
              <li class="chip" class:miss={!has}>
                {#if has}<span class="sw" style:background={classColor(i)}></span>{/if}<span>{c}</span><span class="sr-only">{has ? ', known' : ', missing: scores 0'}</span>
              </li>
            {/each}
          </ul>
        </div>
      {/if}
      <Field label="Images" for="{uid}-mimages" error={show(errs.images)} hint="Empty runs all of them. Fewer runs the first of the benchmark's seeded order, a fair sample, but the entry is partial.">
        <div class="narrow"><Input id="{uid}-mimages" bind:value={modelImages} inputmode="numeric" placeholder="All {num(bench.images)}" invalid={!!show(errs.images)} /></div>
      </Field>
      {#if bench.plan.conditions.length}
        <Switch
          label="Run the plan's conditions"
          bind:checked={modelConditions}
          hint={modelConditions ? `${passes} more passes over each image: ${bench.plan.conditions.length} conditions at their severities.` : 'Off: the unchanged images only. There is no robustness axis, and the entry is partial.'}
        />
      {/if}
      {#if ahead}<p class="note">Model entries run one at a time: it waits for {ahead.name}.</p>{/if}
    {:else if kind === 'agent'}
      <p class="note lead-note">
        The agent looks at each image and draws boxes, as agent labeling asks. Its score is partial: agents give no confidences, so calibration is not measured{agentConditions ? '' : ', and on the unchanged images alone neither is robustness'}.
      </p>
      <div class="agent">
        <AgentFields {agents} {agentsError} {retryAgents} {choice} effortHint={(a) => `More effort means more reasoning tokens per image and a slower entry. The default, ${a.default_effort}, is the best value for ${a.name}.`} />
      </div>
      <Field
        label="Images"
        for="{uid}-aimages"
        error={show(errs.images)}
        hint={agent && agentCalls ? `The first of the benchmark's seeded order, a fair sample: about ${num(agentCalls)} calls to ${agent.name} (${modelLabel(choice.model || null, agent)}).` : "The first of the benchmark's seeded order, a fair sample."}
      >
        <div class="narrow"><Input id="{uid}-aimages" bind:value={agentImages} inputmode="numeric" invalid={!!show(errs.images)} /></div>
      </Field>
      {#if bench.plan.conditions.length}
        <Switch
          label="Also run the plan's conditions"
          bind:checked={agentConditions}
          hint={agentConditions ? `Each image again under ${passes} conditions and severities: ${1 + passes} times the calls.` : 'Off by default: every condition is another call per image.'}
        />
      {/if}
    {:else}
      <p class="note lead-note">Any model that runs elsewhere: fetch the images, run it, and send its boxes as a file. It is scored once all of it has arrived.</p>
      <div class="file">
        <input bind:this={fileInput} type="file" accept="application/json,.json" hidden onchange={pick} />
        <div class="f-row">
          <Button icon={FileJson} onclick={() => fileInput?.click()}>{file ? 'Choose another file' : 'Choose predictions file'}</Button>
          {#if file}<span class="f-name"><span class="truncate" title={file.name}>{file.name}</span><span class="faint num">{bytes(file.size)}</span></span>{/if}
        </div>
        {#if reading}
          <p class="f-state busy" role="status"><Spinner size={12} /> Reading the file{file && file.size > 8 * 2 ** 20 ? ` (${bytes(file.size)})` : ''}…</p>
        {:else if sending}
          <!-- The records sent of all of them, part by part: a real bar, not a spinner. -->
          <div class="f-send" role="status">
            <Progress value={sending.total ? sending.sent / sending.total : 0} size="sm" tone="accent" label="Predictions sent" />
            <p class="f-state num">Sending predictions · {num(sending.sent)} of {num(sending.total)} records</p>
          </div>
        {:else if check?.ok}
          <p class="f-state ok num" role="status">
            {imagesText(check.summary.images)} answered · {check.summary.conditions.length ? `${check.summary.conditions.length} of ${bench.plan.conditions.length} conditions` : 'unchanged only'} · {num(check.summary.boxes)} boxes{check.summary.scores ? ' with confidences' : ', no confidences'}{#if check.summary.missing.length && check.summary.conditions.length}. Left out: {check.summary.missing.map(condName).join(', ')}; the entry is partial.{/if}{#if parts}{' '}Sent in {num(parts)} parts, a pass at a time.{/if}
          </p>
        {:else if check}
          <p class="f-state bad" role="alert">{check.error}</p>
        {:else}
          <p class="f-state">A JSON file of <span class="mono">{'{ "predictions": [...] }'}</span> or a bare array: one record per image and condition, boxes in pixels. Checked here before it is sent.</p>
        {/if}
      </div>
      <Field label="Name on the leaderboard" for="{uid}-name" error={show(errs.name)}>
        <Input id="{uid}-name" bind:value={extName} placeholder="e.g. yolo26n" autocomplete="off" invalid={!!show(errs.name)} />
      </Field>
      <Switch label="The model gives confidences" bind:checked={scored} hint={scored ? 'Every box’s score is read; boxes without one count as 1.' : 'Every score is taken as 1: calibration is not measured, and self-labeling has one operating point.'} />
      <details class="how">
        <summary><span class="chev" aria-hidden="true"></span>How to make the file</summary>
        <div class="how-body">
          <p>For anything but a small benchmark, use the example script <span class="mono">examples/bench_external.py</span> from the FOLO repository: put your model behind its one Python function, and it fetches every image under every condition, runs the model and sends the predictions in parts.</p>
          <CommandLine command={`python examples/bench_external.py --server ${hub.serverUrl.replace(/\/$/, '')} --bench ${bench.id} --name my-detector`} label="Run the example script" />
          <p>By hand: list the images and fetch each one, unchanged and under the plan's conditions; run your model on them and save its boxes, in pixels, to choose above. In a shell:</p>
          <CommandLine command={`api=${api}`} label="This benchmark's address" />
          <CommandLine command={`curl -s "$api/images?limit=1000"`} label="List the images" />
          <CommandLine command={`curl -s "$api/images/1?condition=haze&severity=3" -o haze3.jpg`} label="Fetch an image under a condition" />
          <p>A small entry goes in one request (up to 64 MiB), its name and predictions in <span class="mono">entry.json</span>; a larger one is opened with <span class="mono">"open": true</span>, sent in parts to its <span class="mono">predictions</span>, and finished. This dialog does that for you.</p>
          <CommandLine command={`curl -s "$api/entries" -H 'Content-Type: application/json' -d @entry.json`} label="Send a small entry" />
        </div>
      </details>
    {/if}
    {#if bench.entries === 0 && bench.plan.conditions.length}<p class="note first">The first entry fixes the plan ({bench.plan.conditions.length} conditions) so that every entry is graded on the same test.</p>{/if}
    {#if error}<ErrorState {error} title={kind === 'external' && parts ? (left ? 'Could not add the entry' : 'Could not add the entry: nothing was kept') : 'Could not add the entry'} compact />{/if}
    {#if left}<p class="note warn-left" role="alert">{left.charAt(0).toUpperCase() + left.slice(1)}</p>{/if}
  </form>
  {#snippet footer()}
    {#if confirmStop}
      <!-- Closing while a large file is on its way: the entry would be discarded, so it asks first. -->
      <p class="why" role="alert">Stop sending? The entry is discarded.</p>
      <Button variant="ghost" onclick={() => (confirmStop = false)}>Keep sending</Button>
      <Button variant="destructive" onclick={stopSending}>Stop and discard</Button>
    {:else}
      {#if missingWhy}<p class="why" data-gate-why>{missingWhy}</p>{/if}
      <Button variant="ghost" onclick={() => (beforeClose() ? (open = false) : undefined)}>Cancel</Button>
      <Gate {reason}>
        {#snippet children(why)}
          <Button variant="primary" type="submit" form="{uid}-form" icon={Plus} loading={busy} disabled={!!why} aria-describedby={why}>{kind === 'external' ? 'Score and add' : 'Add entry'}</Button>
        {/snippet}
      </Gate>
    {/if}
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
  .agent {
    display: grid;
    gap: var(--space-4);
  }
  .narrow {
    max-width: 200px;
  }
  .classes {
    display: grid;
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
</style>
