<script lang="ts">
  /** Export a model to a deployable file: pick a format, run it on the server, then copy the resulting path. */
  import Check from '@lucide/svelte/icons/check';
  import Copy from '@lucide/svelte/icons/copy';
  import Download from '@lucide/svelte/icons/download';
  import { hub } from '$lib/api/index.svelte';
  import type { ExportFormat, ExportResult, Model } from '$lib/api/types';
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
      ext: '.f16.folo',
      desc: 'Native weights in half precision. Loads instantly in folo predict and folo serve.',
      bytesPerParam: 2.03,
    },
    {
      value: 'onnx',
      label: 'ONNX',
      ext: '.onnx',
      desc: 'Open standard, float32 graph for ONNX Runtime, TensorRT or OpenVINO.',
      bytesPerParam: 4.1,
    },
  ];

  let format = $state<ExportFormat>('folo-f16');
  let phase = $state<'idle' | 'running' | 'done' | 'error'>('idle');
  let result = $state<ExportResult | null>(null);
  let error = $state<unknown>(null);
  let copied = $state(false);
  let run = 0;

  async function start() {
    const id = ++run;
    phase = 'running';
    error = null;
    try {
      const r = await hub.api.exportModel(model.id, format);
      if (id !== run) return;
      result = r;
      phase = 'done';
      toast.success(`Exported ${model.name}`, `${r.path} · ${bytes(r.bytes, prefs.units)}`);
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

  function reset() {
    run++;
    phase = 'idle';
    result = null;
    error = null;
    copied = false;
  }
</script>

<Dialog bind:open title="Export model" description="Write {model.name} to a file you can deploy." onclose={reset}>
  {#if phase === 'done' && result}
    <div class="done">
      <div class="done-head">
        <span class="ok" aria-hidden="true"><Check size={14} strokeWidth={2.25} /></span>
        <span>Export finished</span>
        <span class="size num">{bytes(result.bytes, prefs.units)}</span>
      </div>
      <div class="path">
        <code class="mono">{result.path}</code>
        <Button size="sm" variant="ghost" icon={copied ? Check : Copy} aria-label="Copy path" onclick={copy} />
      </div>
      <p class="note">The file is on the machine running folo serve.</p>
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
              <span class="est num">≈ {bytes(model.params * f.bytesPerParam, prefs.units)}</span>
            </span>
            <span class="opt-desc">{f.desc}</span>
          </span>
        </label>
      {/each}
    </fieldset>
    {#if phase === 'running'}
      <div class="progress" role="status">
        <div class="bar" aria-hidden="true"><span></span></div>
        <span class="faint">Exporting {model.name}…</span>
      </div>
    {:else if phase === 'error'}
      <div class="err"><ErrorState {error} title="Export failed" compact /></div>
    {/if}
  {/if}

  {#snippet footer()}
    {#if phase === 'done'}
      <Button variant="ghost" onclick={reset}>Export another</Button>
      <Button variant="primary" onclick={() => (open = false)}>Done</Button>
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
    font-size: var(--type-xs-size);
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
    color: var(--text-tertiary);
  }
</style>
