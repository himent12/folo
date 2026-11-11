/**
 * Export a model to a deployable file: pick a format and run it on the server. The f16 file can then be downloaded
 * to this computer (GET /api/models/{id}/weights?format=f16), under the name the server gave it. Servers that list
 * `export_onnx_switches` in /api/system's `features` get folo export's three ONNX switches as checkboxes. Servers
 * that list `weights_onnx` serve the ONNX graph itself (GET /api/models/{id}/weights?format=onnx, the switches as
 * query parameters): ONNX then downloads straight to this computer, the button naming the file
 * ("Download fasteners-n-best.onnx"). On older servers ONNX is written on the server and its path shown.
 *
 * The demo (the in-browser mock) creates no file and says so before and after: "Demo: no file is created" under the
 * formats, then a neutral preview with no success check, no saved-file claim and no Download, whose one action is
 * "Connect to a server" (Settings › Connection). FOLO and ONNX alike.
 */
import Check from '$icons/check';
import Plug from '$icons/plug';
import Copy from '$icons/copy';
import Download from '$icons/download';
import Info from '$icons/info';
import { bindable, computed, css, effect, For, signal } from 'fw';
import { HttpApi, onnxFileName } from '$lib/api/client';
import { hub } from '$lib/api/index';
import type { ExportFormat, ExportResult, Model, OnnxSwitches } from '$lib/api/types';
import { bytes } from '$lib/format';
import { prefs } from '$lib/prefs';
import { href } from '$lib/router';
import Button from '$lib/ui/Button';
import Dialog from '$lib/ui/Dialog';
import ErrorState from '$lib/ui/ErrorState';
import { DialogStatus } from '$lib/ui/feedback';
import { toast } from '$lib/ui/toast';

export interface ExportDialogProps {
  model: Model;
  open?: boolean;
}

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

const OFF: Required<OnnxSwitches> = { decode: false, dynamic: false, half: false };

export default function ExportDialog(props: ExportDialogProps) {
  const open = bindable(props, 'open', false);
  const model = () => props.model;

  /** The demo's models are not real networks: an export there is a preview that creates no file. */
  const demo = computed(() => hub.api.mock);
  const format = signal<ExportFormat>('folo-f16');
  const switches = signal<Required<OnnxSwitches>>(OFF);
  /** Whether the server takes the switches (older ones would ignore them): asked once the dialog opens. */
  const switchesOk = signal(false);
  /** Whether the server serves the ONNX graph for download (`weights_onnx`), so ONNX skips the server-side file. */
  const onnxDirect = signal(false);
  effect(() => {
    if (!open.get()) return;
    let live = true;
    hub.api.system().then(
      (s) => {
        if (!live) return;
        switchesOk.set(!!s.features?.includes('export_onnx_switches'));
        onnxDirect.set(!!s.features?.includes('weights_onnx'));
      },
      () => {},
    );
    return () => {
      live = false;
    };
  });
  /** The estimate follows the switches: an f16 ONNX file is about half the size. */
  const perParam = (f: (typeof formats)[number]) => (f.value === 'onnx' && switchesOk.get() && switches.get().half ? 2.05 : f.bytesPerParam);
  /** The app's one size format (`bytes`: one decimal, "≈ 18.3 MiB", "≈ 37.0 MiB"). */
  const estimate = (b: number) => bytes(b, prefs.units);
  /** The format of the finished export (the radio may not be shown any more); `onnx-file` a direct ONNX download. */
  const done = signal<ExportFormat | 'onnx-file' | null>(null);
  /** The ONNX switches go with the request when the server takes them (`weights_onnx` takes them as query parameters). */
  const onnxSwitches = computed(() => switchesOk.get() || onnxDirect.get());
  /** Direct ONNX download: the file it will be, by the server's naming rule, and the file once it arrived. */
  const onnxName = computed(() => onnxFileName(model().name, onnxSwitches.get() && switches.get().half, onnxSwitches.get() && switches.get().decode));
  const direct = computed(() => format.get() === 'onnx' && onnxDirect.get() && !demo.get());
  /** The format as listed, for the demo preview's size. */
  const picked = computed(() => formats.find((f) => f.value === format.get())!);
  const file = signal<File | null>(null);
  const phase = signal<'idle' | 'running' | 'done' | 'error'>('idle');
  const result = signal<ExportResult | null>(null);
  const announcement = computed(() => {
    const ph = phase.get();
    const f = file.get();
    const r = result.get();
    return ph === 'running'
      ? direct.get()
        ? `Preparing ${onnxName.get()}…`
        : `Exporting ${model().name}…`
      : ph === 'done' && demo.get()
        ? 'Demo preview: no file was created.'
        : ph === 'done' && f
          ? `Downloaded ${f.name}, ${bytes(f.size, prefs.units)}.`
          : ph === 'done' && r
            ? `Export finished: ${r.path.split('/').pop() ?? r.path}, ${bytes(r.bytes, prefs.units)}.`
            : '';
  });
  /** The file's name as the server wrote it, which is also the name it downloads under. */
  const fileName = computed(() => {
    const r = result.get();
    return r ? (r.path.split('/').pop() ?? r.path) : '';
  });
  /** The file a server would create, for the demo's preview (the server's naming rule for ONNX). */
  const previewName = computed(() => (format.get() === 'onnx' ? onnxName.get() : fileName.get()));
  const error = signal<unknown>(null);
  const copied = signal(false);
  /** Results inside the open dialog (a blocked copy, the demo's missing file) show in its status region, not a toast. */
  const status = new DialogStatus();
  let run = 0;

  /**
   * ONNX on a server with `weights_onnx`: asks for the graph (built on the first request, about as long as an export)
   * and saves it through the browser under the server's name. The demo's models have no graph, so nothing is saved.
   */
  async function downloadOnnx() {
    const id = ++run;
    phase.set('running');
    error.set(null);
    try {
      const api = hub.api;
      const f = await api.onnxFile(model().id, onnxSwitches.peek() ? { ...switches.peek() } : {});
      if (id !== run) return;
      file.set(f);
      done.set('onnx-file');
      phase.set('done');
      if (api.mock) return;
      const url = URL.createObjectURL(f);
      const a = document.createElement('a');
      a.href = url;
      a.download = f.name;
      a.rel = 'noopener';
      a.click();
      setTimeout(() => URL.revokeObjectURL(url), 60_000);
      if (!open.peek()) toast.success(`Downloaded ${f.name}`, bytes(f.size, prefs.units));
    } catch (e) {
      if (id !== run) return;
      error.set(e);
      phase.set('error');
    }
  }

  async function start() {
    if (direct.peek()) return downloadOnnx();
    const id = ++run;
    phase.set('running');
    error.set(null);
    try {
      const fmt = format.peek();
      const r = await hub.api.exportModel(model().id, fmt, fmt === 'onnx' && onnxSwitches.peek() ? { ...switches.peek() } : undefined);
      if (id !== run) return;
      result.set(r);
      done.set(fmt);
      phase.set('done');
      // The open dialog already says "Export finished" with the path: a toast would say it twice at once.
      if (!open.peek()) {
        if (demo.peek()) toast.info('Demo preview', `Exporting ${model().name} created no file.`);
        else toast.success(`Exported ${model().name}`, `${r.path} · ${bytes(r.bytes, prefs.units)}`);
      }
    } catch (e) {
      if (id !== run) return;
      error.set(e);
      phase.set('error');
    }
  }

  async function copy() {
    const r = result.peek();
    if (!r) return;
    try {
      await navigator.clipboard.writeText(r.path);
      copied.set(true);
      setTimeout(() => copied.set(false), 1600);
    } catch {
      status.error(new Error('Your browser blocked clipboard access; select the path and copy it by hand.'), 'Could not copy the path');
    }
  }

  /** Saves the f16 file through the browser, under the server's file name. The demo's models have no file to save. */
  function download() {
    const api = hub.api;
    if (!(api instanceof HttpApi)) {
      status.info('Nothing to download in the demo', 'Demo models are not real networks, so they have no weights file.');
      return;
    }
    const a = document.createElement('a');
    a.href = api.weightsUrl(model().id);
    a.download = fileName.peek();
    a.rel = 'noopener';
    a.click();
  }

  function reset() {
    run++;
    status.clear();
    done.set(null);
    switches.set(OFF);
    phase.set('idle');
    result.set(null);
    file.set(null);
    error.set(null);
    copied.set(false);
  }

  /** `bind:checked` of one switch (the switches are one immutable object). */
  const switchOf = (key: keyof OnnxSwitches) => ({
    get: () => switches.get()[key],
    set: (v: boolean) => switches.set({ ...switches.peek(), [key]: v }),
  });
  const close = () => open.set(false);

  const footer = () => (
    <>
      {phase.get() === 'done' ? (
        // One shape for every result: "Export another" a quiet link at the footer's start (a way back, not an action on
        // the result), Done the secondary, and the result's next step the one primary at the end: Connect to a server in
        // the demo, Download for the f16 file; ONNX has nothing to download, so Done is the primary.
        <>
          <button type="button" class="another" onclick={reset}>
            Export another
          </button>
          {demo.get() ? (
            <>
              <Button onclick={close}>Done</Button>
              <Button variant="primary" icon={Plug} href={href('/settings')} onclick={close}>
                Connect to a server
              </Button>
            </>
          ) : done.get() === 'folo-f16' ? (
            <>
              <Button onclick={close}>Done</Button>
              <Button variant="primary" icon={Download} onclick={download}>
                Download
              </Button>
            </>
          ) : (
            <Button variant="primary" onclick={close}>
              Done
            </Button>
          )}
        </>
      ) : (
        <>
          <Button variant="ghost" onclick={close}>
            Cancel
          </Button>
          <Button variant="primary" icon={Download} loading={phase.get() === 'running'} onclick={start}>
            {phase.get() === 'error' ? (
              'Retry'
            ) : direct.get() ? (
              <>
                Download <span class="mono file">{onnxName.get()}</span>
              </>
            ) : (
              'Export'
            )}
          </Button>
        </>
      )}
    </>
  );

  return (
    <Dialog bind:open={open} title="Export model" description={`Write ${model().name} to a file you can deploy.`} onclose={reset} status={status} footer={footer}>
      {/* One live region that stays in the dialog, so the start and the end of an export are both announced (the toast is
          held back while the dialog is open). Failures announce themselves through ErrorState's alert. */}
      <p class="sr-only" role="status">
        {announcement.get()}
      </p>
      {phase.get() === 'done' && demo.get() ? (
        // The demo created nothing: a neutral mark (not the success check), what a server would write, no path and no
        // Download. The way to a real file is the footer's Connect to a server.
        <div class="done">
          <div class="done-head">
            <span class="ok quiet" aria-hidden="true">
              <Info size={14} />
            </span>
            <span>Demo preview: no file was created</span>
          </div>
          <p class="note">
            On a server, this writes <span class="mono">{previewName.get()}</span>, about {estimate(model().params * perParam(picked.get()))}. The demo's
            models are not real networks, so there is no file to save or download.
          </p>
        </div>
      ) : phase.get() === 'done' && file.get() ? (
        // The ONNX graph, downloaded straight to this computer (servers with weights_onnx).
        <div class="done">
          <div class="done-head">
            <span class="ok" aria-hidden="true">
              <Check size={14} strokeWidth={2.25} />
            </span>
            <span>Downloaded</span>
            <span class="size num">{bytes(file.get()!.size, prefs.units)}</span>
          </div>
          <div class="path">
            <code class="mono">{file.get()!.name}</code>
          </div>
          <p class="note">Your browser saved it to its downloads folder. Nothing was written on the server.</p>
        </div>
      ) : phase.get() === 'done' && result.get() ? (
        <div class="done">
          <div class="done-head">
            <span class="ok" aria-hidden="true">
              <Check size={14} strokeWidth={2.25} />
            </span>
            <span>Export finished</span>
            <span class="size num">{bytes(result.get()!.bytes, prefs.units)}</span>
          </div>
          <div class="where">
            <span class="where-label">Saved on the server at</span>
            <div class="path">
              <code class="mono">{result.get()!.path}</code>
              <Button size="sm" variant="ghost" icon={copied.get() ? Check : Copy} aria-label="Copy path" onclick={copy} />
            </div>
          </div>
          {done.get() === 'onnx' ? (
            <p class="note">
              Studio cannot download ONNX files yet: the file stays on the machine running <span class="mono">folo serve</span>. Copy it from there, or
              run <span class="mono">folo export --format onnx</span> where you need it.
            </p>
          ) : null}
        </div>
      ) : (
        <>
          <fieldset disabled={phase.get() === 'running'}>
            <legend class="overline">Format</legend>
            <For each={formats} key={(f) => f.value}>
              {(f) => (
                <>
                  <label class="opt" class:on={format.get() === f().value}>
                    <input type="radio" name="export-format" value={f().value} bind:group={format} />
                    <span class="radio" aria-hidden="true"></span>
                    <span class="opt-text">
                      <span class="opt-top">
                        <span class="opt-name">{f().label}</span>
                        <span class="ext mono">{f().ext}</span>
                        <span class="est num">≈ {estimate(model().params * perParam(f()))}</span>
                      </span>
                      {/* Commands in `backticks` are code: mono. */}
                      <span class="opt-desc">
                        <For each={f().desc.split('`')} key={(_, i) => i}>
                          {(part, i) => <>{i() % 2 ? <span class="mono">{part()}</span> : part()}</>}
                        </For>
                      </span>
                    </span>
                  </label>
                  {f().value === 'onnx' && onnxSwitches.get() ? (
                    // folo export's switches, under the format they belong to: always there, off until ONNX is
                    // picked, so choosing it does not grow the centred dialog under the pointer.
                    <div class="switches" class:off={format.get() !== 'onnx'} role="group" aria-label="ONNX options">
                      <For each={switchOptions} key={(o) => o.key}>
                        {(o) => (
                          <label class="sw-opt">
                            <input type="checkbox" bind:checked={switchOf(o().key)} disabled={format.get() !== 'onnx'} />
                            <span class="box" aria-hidden="true">
                              <Check size={12} strokeWidth={2.5} />
                            </span>
                            <span class="opt-text">
                              <span class="sw-name">{o().label}</span>
                              <span class="opt-desc">{o().desc}</span>
                            </span>
                          </label>
                        )}
                      </For>
                    </div>
                  ) : null}
                </>
              )}
            </For>
          </fieldset>
          {demo.get() && phase.get() !== 'running' ? (
            // Said before the export, not only after it.
            <p class="demo-note">
              <Info size={14} aria-hidden="true" />
              <span>Demo: no file is created. Connect to a server to export real weights.</span>
            </p>
          ) : null}
          {phase.get() === 'running' ? (
            <div class="progress">
              <div class="bar" aria-hidden="true">
                <span></span>
              </div>
              {/* The server builds the graph on its first request (about as long as an export), then sends it. */}
              <span class="faint">
                {direct.get() ? (
                  <>
                    Preparing <span class="mono">{onnxName.get()}</span>…
                  </>
                ) : (
                  <>Exporting {model().name}…</>
                )}
              </span>
            </div>
          ) : phase.get() === 'error' ? (
            <div class="err">
              <ErrorState error={error.get()} title="Export failed" compact />
            </div>
          ) : null}
        </>
      )}
    </Dialog>
  );
}

css`
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
    /* Loading motion is bounded (DESIGN_DIRECTION §2.9): the bar exists only while the export runs (phase 'running'),
       and even then it sweeps a minute at most (55 × 1.1 s, longer than an export takes) before it holds still. */
    animation: slide 1.1s var(--ease-standard) 55;
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
  .ok.quiet {
    background: var(--bg-active);
    color: var(--text-secondary);
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
    max-inline-size: var(--measure);
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
  .off .sw-opt {
    cursor: default;
  }
  .off .sw-name,
  .off .opt-desc {
    color: var(--text-disabled);
  }
  .off .sw-opt:hover .box {
    border-color: var(--line-control);
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
  /* The demo's word before an export: a neutral note with its mark, under the formats. */
  .demo-note {
    display: flex;
    max-inline-size: calc(var(--measure) + 22px);
    align-items: flex-start;
    gap: var(--space-2);
    margin-top: var(--space-4);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
  }
  .demo-note > :global(svg) {
    flex-shrink: 0;
    margin-top: 3px; /* grid-ok: the 14 px icon centred on the 20 px first line */
    color: var(--text-tertiary);
  }
  .where {
    display: grid;
    gap: var(--space-1);
  }
  /* The file the button downloads, by name: mono (a file name), in the button's own colour. */
  .file {
    font-size: var(--type-sm-size);
  }
  .where-label {
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
  }
  /* "Export another": a link at the footer's start, the buttons kept at its end. */
  .another {
    position: relative;
    margin-right: auto;
    align-self: center;
    color: var(--text-primary);
    font-size: var(--type-base-size);
    line-height: var(--type-base-line);
    text-decoration: underline;
    text-decoration-color: var(--line-strong);
    text-underline-offset: 3px;
  }
  .another:hover {
    text-decoration-color: currentColor;
  }
  /* A 44 px target around the link's text line (DESIGN_DIRECTION 9). */
  .another::after {
    content: '';
    position: absolute;
    inset: -12px -8px;
  }
  /* Phones: the buttons share the row's width; the link takes a line of its own above them. */
  @media (max-width: 767px) {
    .another {
      flex: 0 0 100%;
      margin: 0 0 var(--space-1);
      text-align: center;
    }
  }
`;
