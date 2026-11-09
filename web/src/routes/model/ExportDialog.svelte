<script lang="ts">
  /**
   * Export a model to a deployable file: pick a format and run it on the server. The f16 file can then be downloaded
   * to this computer (GET /api/models/{id}/weights?format=f16), under the name the server gave it; ONNX has no
   * download endpoint, so its path on the server is shown, and the dialog says so. Servers that list
   * `export_onnx_switches` in /api/system's `features` get folo export's three ONNX switches as checkboxes.
   */
  import Check from '@lucide/svelte/icons/check';
  import Copy from '@lucide/svelte/icons/copy';
  import Download from '@lucide/svelte/icons/download';
  import { HttpApi } from '$lib/api/client';
  import { hub } from '$lib/api/index.svelte';
  import type { ExportFormat, ExportResult, Model, OnnxSwitches } from '$lib/api/types';
  import { bytes } from '$lib/format';
  import { prefs } from '$lib/prefs.svelte';
  import Button from '$lib/ui/Button.svelte';
  import Dialog from '$lib/ui/Dialog.svelte';
  import ErrorState from '$lib/ui/ErrorState.svelte';
  import { toast } from '$lib/ui/toast.svelte';

  let { model, open = $bindable(false) }: { model: Model; open?: boolean } = $props();

  const formats: { value: ExportFormat; label: string; ext: string; desc: string; bytesPerParam: number }[] = [
    {
      value: 'folo-f16',
      label: 'FOLO f16',
      ext: '.folo',
      desc: 'Native weights in half precision. Loads instantly in `folo predict` and `folo serve`.',
      bytesPerParam: 2.03,
    },
    {
      value: 'onnx',
      label: 'ONNX',
      ext: '.onnx',
      desc: 'Open standard graph for ONNX Runtime, TensorRT or OpenVINO. Full precision (float32) unless you pick half precision.',
      bytesPerParam: 4.1,
    },
  ];

  /** folo export's ONNX switches (docs/CLI.md, ONNX export); off is its default. */
  const switchOptions: { key: keyof OnnxSwitches; label: string; desc: string }[] = [
    { key: 'decode', label: 'Decode boxes in the graph', desc: 'Outputs boxes and class scores instead of the raw head. NMS still runs outside.' },
    { key: 'dynamic', label: 'Dynamic input size', desc: 'Any batch, and any height and width in multiples of 32.' },
    { key: 'half', label: 'Half precision (f16)', desc: 'f16 weights, input and outputs: about half the file size.' },
  ];

  let format = $state<ExportFormat>('folo-f16');
  let switches = $state<Required<OnnxSwitches>>({ decode: false, dynamic: false, half: false });
  /** Whether the server takes the switches (older ones would ignore them): asked once the dialog opens. */
  let switchesOk = $state(false);
  $effect(() => {
    if (!open) return;
    let live = true;
    hub.api.system().then(
      (s) => {
        if (live) switchesOk = !!s.features?.includes('export_onnx_switches');
      },
      () => {},
    );
    return () => {
      live = false;
    };
  });
  /** The estimate follows the switches: an f16 ONNX file is about half the size. */
  const perParam = (f: (typeof formats)[number]) => (f.value === 'onnx' && switchesOk && switches.half ? 2.05 : f.bytesPerParam);
  /** The app's one size format (`bytes`: one decimal, "≈ 18.3 MiB", "≈ 37.0 MiB"). */
  const estimate = (b: number) => bytes(b, prefs.units);
  /** The format of the finished export (the radio may not be shown any more). */
  let done = $state<ExportFormat | null>(null);
  let phase = $state<'idle' | 'running' | 'done' | 'error'>('idle');
  let result = $state<ExportResult | null>(null);
  const announcement = $derived(
    phase === 'running'
      ? `Exporting ${model.name}…`
      : phase === 'done' && result
        ? `Export finished: ${result.path.split('/').pop() ?? result.path}, ${bytes(result.bytes, prefs.units)}.`
        : '',
  );
  /** The file's name as the server wrote it, which is also the name it downloads under. */
  const fileName = $derived(result ? (result.path.split('/').pop() ?? result.path) : '');
  let error = $state<unknown>(null);
  let copied = $state(false);
  let run = 0;

  async function start() {
    const id = ++run;
    phase = 'running';
    error = null;
    try {
      const r = await hub.api.exportModel(model.id, format, format === 'onnx' && switchesOk ? { ...switches } : undefined);
      if (id !== run) return;
      result = r;
      done = format;
      phase = 'done';
      // The open dialog already says "Export finished" with the path: a toast would say it twice at once.
      if (!open) toast.success(`Exported ${model.name}`, `${r.path} · ${bytes(r.bytes, prefs.units)}`);
    } catch (e) {
      if (id !== run) return;
      error = e;
      phase = 'error';
    }
  }

  async function copy() {
    if (!result) return;
    try {
      await navigator.clipboard.writeText(result.path);
      copied = true;
      setTimeout(() => (copied = false), 1600);
    } catch {
      toast.info('Copy failed', 'Your browser blocked clipboard access; select the path and copy it by hand.');
    }
  }

  /** Saves the f16 file through the browser, under the server's file name. The demo's models have no file to save. */
  function download() {
    const api = hub.api;
    if (!(api instanceof HttpApi)) {
      toast.info('Nothing to download in the demo', 'Demo models are not real networks, so they have no weights file.');
      return;
    }
    const a = document.createElement('a');
    a.href = api.weightsUrl(model.id);
    a.download = fileName;
    a.rel = 'noopener';
    a.click();
  }

  function reset() {
    run++;
    done = null;
    switches = { decode: false, dynamic: false, half: false };
    phase = 'idle';
    result = null;
    error = null;
    copied = false;
  }
</script>

<Dialog bind:open title="Export model" description="Write {model.name} to a file you can deploy." onclose={reset}>
  <!-- One live region that stays in the dialog, so the start and the end of an export are both announced (the toast is
       held back while the dialog is open). Failures announce themselves through ErrorState's alert. -->
  <p class="sr-only" role="status">{announcement}</p>
  {#if phase === 'done' && result}
    <div class="done">
      <div class="done-head">
        <span class="ok" aria-hidden="true"><Check size={14} strokeWidth={2.25} /></span>
        <span>Export finished</span>
        <span class="size num">{bytes(result.bytes, prefs.units)}</span>
      </div>
      <div class="where">
        <span class="where-label">Saved on the server at</span>
        <div class="path">
          <code class="mono">{result.path}</code>
          <Button size="sm" variant="ghost" icon={copied ? Check : Copy} aria-label="Copy path" onclick={copy} />
        </div>
      </div>
      {#if done === 'onnx'}
        <p class="note">
          Studio cannot download ONNX files yet: the file stays on the machine running <span class="mono">folo serve</span>. Copy it from there, or
          run <span class="mono">folo export --format onnx</span> where you need it.
        </p>
      {/if}
    </div>
  {:else}
    <fieldset disabled={phase === 'running'}>
      <legend class="overline">Format</legend>
      {#each formats as f (f.value)}
        <label class="opt" class:on={format === f.value}>
          <input type="radio" name="export-format" value={f.value} bind:group={format} />
          <span class="radio" aria-hidden="true"></span>
          <span class="opt-text">
            <span class="opt-top">
              <span class="opt-name">{f.label}</span>
              <span class="ext mono">{f.ext}</span>
              <span class="est num">≈ {estimate(model.params * perParam(f))}</span>
            </span>
            <!-- Commands in `backticks` are code: mono. -->
            <span class="opt-desc">{#each f.desc.split('`') as part, i (i)}{#if i % 2}<span class="mono">{part}</span>{:else}{part}{/if}{/each}</span>
          </span>
        </label>
        {#if f.value === 'onnx' && format === 'onnx' && switchesOk}
          <!-- folo export's switches, under the format they belong to. -->
          <div class="switches" role="group" aria-label="ONNX options">
            {#each switchOptions as o (o.key)}
              <label class="sw-opt">
                <input type="checkbox" bind:checked={switches[o.key]} />
                <span class="box" aria-hidden="true"><Check size={12} strokeWidth={2.5} /></span>
                <span class="opt-text">
                  <span class="sw-name">{o.label}</span>
                  <span class="opt-desc">{o.desc}</span>
                </span>
              </label>
            {/each}
          </div>
        {/if}
      {/each}
    </fieldset>
    {#if phase === 'running'}
      <div class="progress">
        <div class="bar" aria-hidden="true"><span></span></div>
        <span class="faint">Exporting {model.name}…</span>
      </div>
    {:else if phase === 'error'}
      <div class="err"><ErrorState {error} title="Export failed" compact /></div>
    {/if}
  {/if}

  {#snippet footer()}
    {#if phase === 'done'}
      <!-- The f16 file's next step is Download, the one primary; Done only closes. ONNX has nothing to download. -->
      <Button variant="ghost" onclick={reset}>Export another</Button>
      {#if done === 'folo-f16'}
        <Button variant="ghost" onclick={() => (open = false)}>Done</Button>
        <Button variant="primary" icon={Download} onclick={download}>Download</Button>
      {:else}
        <Button variant="primary" onclick={() => (open = false)}>Done</Button>
      {/if}
    {:else}
      <Button variant="ghost" onclick={() => (open = false)}>Cancel</Button>
      <Button variant="primary" icon={Download} loading={phase === 'running'} onclick={start}>
        {phase === 'error' ? 'Retry' : 'Export'}
      </Button>
    {/if}
  {/snippet}
</Dialog>

<style>
  fieldset {
    border: 0;
    padding: 0;
    margin: 0;
    display: grid;
    gap: var(--space-2);
    min-width: 0;
  }
  legend {
    padding: 0;
    margin-bottom: var(--space-2);
  }
  .opt {
    position: relative;
    display: flex;
    align-items: flex-start;
    gap: var(--space-3);
    padding: var(--space-3);
    border-radius: var(--radius-md);
    border: 1px solid var(--line-default);
    background: var(--bg-panel);
    cursor: pointer;
    transition:
      border-color var(--dur-fast) var(--ease-standard),
      background-color var(--dur-fast) var(--ease-standard);
  }
  .opt:hover {
    border-color: var(--line-strong);
  }
  /* Selected is neutral (lime is for the primary action, focus and live state): a raised card with a 1.5 px edge
     (the inset ring adds the half pixel without moving the content). */
  .opt.on {
    border-color: var(--line-strong);
    box-shadow: inset 0 0 0 0.5px var(--line-strong);
    background: var(--bg-raised);
  }
  fieldset:disabled .opt {
    cursor: default;
    opacity: 0.6;
  }
  input {
    position: absolute;
    opacity: 0;
    width: 1px;
    height: 1px;
  }
  .radio {
    flex-shrink: 0;
    width: 14px;
    height: 14px;
    margin-top: 4px;
    border-radius: 50%;
    /* The empty ring marks the option: line.control (>= 3:1 on every surface). */
    border: 1px solid var(--line-control);
    background: var(--bg-inset);
    transition: border-color var(--dur-fast) var(--ease-standard);
  }
  .on .radio {
    border: 4px solid var(--text-primary);
    background: var(--bg-panel);
  }
  input:focus-visible + .radio {
    outline: var(--size-focus-ring) solid var(--focus);
    outline-offset: 2px;
  }
  .opt-text {
    flex: 1;
    min-width: 0;
    display: grid;
    gap: 2px;
  }
  .opt-top {
    display: flex;
    align-items: baseline;
    gap: var(--space-2);
  }
  .opt-name {
    font-weight: var(--weight-medium);
  }
  .ext {
    color: var(--text-tertiary);
    font-size: var(--type-sm-size);
  }
  .est {
    margin-left: auto;
    font-size: var(--type-sm-size);
    color: var(--text-secondary);
  }
  .opt-desc {
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
  }
  .progress {
    display: grid;
    gap: var(--space-2);
    margin-top: var(--space-4);
    font-size: var(--type-sm-size);
  }
  .bar {
    position: relative;
    height: 4px;
    border-radius: var(--radius-full);
    background: var(--bg-active);
    overflow: hidden;
  }
  .bar span {
    position: absolute;
    inset: 0 auto 0 0;
    width: 36%;
    border-radius: inherit;
    background: var(--accent-line);
    animation: slide 1.1s var(--ease-standard) infinite;
  }
  @keyframes slide {
    from {
      transform: translateX(-100%);
    }
    to {
      transform: translateX(280%);
    }
  }
  @media (prefers-reduced-motion: reduce) {
    .bar span {
      width: 100%;
      opacity: 0.5;
    }
  }
  .err {
    margin-top: var(--space-4);
  }
  .done {
    display: grid;
    gap: var(--space-3);
  }
  .done-head {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    font-weight: var(--weight-medium);
  }
  .ok {
    display: grid;
    place-items: center;
    width: 20px;
    height: 20px;
    border-radius: 50%;
    background: var(--success-muted);
    color: var(--success-text);
  }
  .size {
    margin-left: auto;
    color: var(--text-secondary);
    font-weight: var(--weight-regular);
  }
  .path {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    padding: 4px 4px 4px var(--space-3);
    border-radius: var(--radius-md);
    background: var(--bg-inset);
    border: 1px solid var(--line-subtle);
  }
  .path code {
    flex: 1;
    min-width: 0;
    overflow-wrap: anywhere;
    font-size: var(--type-sm-size);
    color: var(--text-primary);
  }
  .note {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-tertiary);
  }
  .note .mono {
    color: var(--text-secondary);
  }
  /* The ONNX switches hang under their format, indented to its text (12 px padding + 14 px radio + 12 px gap). */
  .switches {
    display: grid;
    padding: 0 var(--space-3) var(--space-1) calc(var(--space-6) + 14px);
  }
  .sw-opt {
    position: relative;
    display: flex;
    align-items: flex-start;
    gap: var(--space-3);
    padding: var(--space-1) 0;
    cursor: pointer;
  }
  .box {
    flex-shrink: 0;
    display: grid;
    place-items: center;
    width: 14px;
    height: 14px;
    margin-top: 3px; /* grid-ok: centres the 14 px box on the 20 px line */
    border-radius: var(--radius-xs);
    /* The empty box marks the option: line.control (>= 3:1 on every surface). */
    border: 1px solid var(--line-control);
    background: var(--bg-inset);
    color: transparent;
    transition:
      background-color var(--dur-fast) var(--ease-standard),
      border-color var(--dur-fast) var(--ease-standard);
  }
  .sw-opt:hover .box {
    border-color: var(--line-control-hover);
  }
  input:checked + .box {
    border-color: var(--text-primary);
    background: var(--text-primary);
    color: var(--bg-panel);
  }
  input:focus-visible + .box {
    outline: var(--size-focus-ring) solid var(--focus);
    outline-offset: 2px;
  }
  @media (pointer: coarse) {
    .sw-opt {
      min-height: 44px;
      align-items: center;
    }
    .box {
      margin-top: 0;
    }
  }
  .where {
    display: grid;
    gap: var(--space-1);
  }
  .where-label {
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
  }
</style>
