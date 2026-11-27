/**
 * Export a model to a deployable file. FOLO f16 downloads straight to this computer (GET
 * /api/models/{id}/weights?format=f16), the button naming the file ("Download fasteners-n-best-f16.folo"); nothing is
 * written on the server but its cache. Servers that list `export_onnx_switches` in /api/system's `features` get folo
 * export's three ONNX switches as checkboxes. Servers that list `weights_onnx` serve the ONNX graph the same way
 * (GET /api/models/{id}/weights?format=onnx, the switches as query parameters). On older servers ONNX is written on
 * the server (POST /api/models/{id}/export) and its path shown.
 *
 * Closing the dialog while a file is on its way does not drop it: it is still saved, and a toast says so (or why not).
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
import { bindable, computed, css, effect, For, onCleanup, signal, tick } from 'fw';
import { f16FileName, onnxFileName } from '$lib/api/client';
import { hub } from '$lib/api/index';
import type { ExportFormat, ExportResult, Model, OnnxSwitches } from '$lib/api/types';
import { bytes } from '$lib/format';
import { prefs } from '$lib/prefs';
import { href } from '$lib/router';
import Button from '$lib/ui/Button';
import Dialog from '$lib/ui/Dialog';
import ErrorState from '$lib/ui/ErrorState';
import Segmented from '$lib/ui/Segmented';
import { DialogStatus } from '$lib/ui/feedback';
import { toast } from '$lib/ui/toast';
import ExportCliPanel from './ExportCliPanel';
import { ONNX_AXES_OPTIONS, onnxSwitchesForRequest, type OnnxDynamicAxes } from './modelCli';

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
const switchOptions: { key: 'decode' | 'dynamic' | 'half'; label: string; desc: string }[] = [
  { key: 'decode', label: 'Decode boxes in the graph', desc: 'Outputs boxes and class scores instead of the raw head. NMS still runs outside.' },
  { key: 'dynamic', label: 'Dynamic input size', desc: 'Any batch, and any height and width in multiples of 32.' },
  { key: 'half', label: 'Half precision (f16)', desc: 'f16 weights, input and outputs: about half the file size.' },
];

const AXES_SEG = ONNX_AXES_OPTIONS.map((o) => ({ value: o.value, label: o.label }));
const OFF = { decode: false, dynamic: false, half: false };

export default function ExportDialog(props: ExportDialogProps) {
  const open = bindable(props, 'open', false);
  const model = () => props.model;

  /** The demo's models are not real networks: an export there is a preview that creates no file. */
  const demo = computed(() => hub.api.mock);
  const format = signal<ExportFormat>('folo-f16');
  const switches = signal<{ decode: boolean; dynamic: boolean; half: boolean }>(OFF);
  /** Independent ONNX dynamic axes (`static`, `batch`, `hw`, `full`), synced with the `dynamic` switch. */
  const axes = signal<OnnxDynamicAxes>('static');
  /** Whether the server takes the switches (older ones would ignore them): asked once the dialog opens. */
  const switchesOk = signal(false);
  /** Whether the server supports independent `dynamic_batch` / `dynamic_hw` axes on `POST /api/models/{id}/export`. */
  const dynamicAxesOk = signal(false);
  /** Whether the server serves the ONNX graph for download (`weights_onnx`), so ONNX skips the server-side file. */
  const onnxDirect = signal(false);
  /** Capability discovery state for `/api/system` on the current open/server epoch. */
  const capsState = signal<'idle' | 'loading' | 'ready' | 'error'>('idle');
  const capsError = signal<unknown>(null);
  const capsAttempt = signal(0);
  effect(() => {
    void hub.epoch;
    void capsAttempt.get();
    if (!open.get()) {
      capsState.set('idle');
      capsError.set(null);
      return;
    }
    const ctrl = new AbortController();
    capsState.set('loading');
    capsError.set(null);
    hub.api.system(ctrl.signal).then(
      (s) => {
        if (ctrl.signal.aborted) return;
        switchesOk.set(!!s.features?.includes('export_onnx_switches'));
        dynamicAxesOk.set(!!s.features?.includes('export_dynamic_axes'));
        onnxDirect.set(!!s.features?.includes('weights_onnx'));
        capsState.set('ready');
      },
      (e) => {
        if (ctrl.signal.aborted) return;
        switchesOk.set(false);
        dynamicAxesOk.set(false);
        onnxDirect.set(false);
        capsError.set(e);
        capsState.set('error');
      },
    );
    return () => {
      ctrl.abort();
    };
  });
  /** While ONNX is chosen on a live server and `/api/system` has not answered yet, hold export so we do not fall back to writing on the server. */
  const checkingOnnx = computed(() => !demo.get() && format.get() === 'onnx' && capsState.get() === 'loading');
  /** Capability discovery failed while ONNX is chosen: keep ONNX blocked and offer Retry instead of writing on the server. */
  const onnxCapsFailed = computed(() => !demo.get() && format.get() === 'onnx' && capsState.get() === 'error');
  function retryCaps() {
    capsAttempt.set(capsAttempt.peek() + 1);
    void tick().then(() => {
      if (destroyed || !open.peek()) return;
      if (!document.activeElement || document.activeElement === document.body) {
        statusEl?.closest('dialog')?.querySelector<HTMLElement>('input[name="export-format"]:checked')?.focus({ preventScroll: true });
      }
    });
  }
  /** The estimate follows the switches: an f16 ONNX file is about half the size. */
  const perParam = (f: (typeof formats)[number]) => (f.value === 'onnx' && onnxSwitches.get() && switches.get().half ? 2.05 : f.bytesPerParam);
  /** The app's one size format (`bytes`: one decimal, "≈ 18.3 MiB", "≈ 37.0 MiB"). */
  const estimate = (b: number) => bytes(b, prefs.units);
  /** The format of the finished export (the radio may not be shown any more); `file` a direct download. */
  const done = signal<ExportFormat | 'file' | null>(null);
  /** The ONNX switches go with the request when the server takes them (`weights_onnx` takes them as query parameters). */
  const onnxSwitches = computed(() => switchesOk.get() || onnxDirect.get());
  /** Direct download: the file it will be, by the server's naming rule, and the file once it arrived. */
  const onnxName = computed(() => onnxFileName(model().name, onnxSwitches.get() && switches.get().half, onnxSwitches.get() && switches.get().decode));
  /** Split dynamic axes (`batch` or `hw`) are served via `POST /api/models/{id}/export` (not `/weights?format=onnx`). */
  const splitAxesActive = computed(() => dynamicAxesOk.get() && (axes.get() === 'batch' || axes.get() === 'hw'));
  const direct = computed(() => !demo.get() && (format.get() === 'folo-f16' || (onnxDirect.get() && !splitAxesActive.get())));
  const directName = computed(() => (format.get() === 'onnx' ? onnxName.get() : f16FileName(model().name)));
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
        ? `Preparing ${directName.get()}…`
        : `Exporting ${model().name}…`
      : ph === 'done' && demo.get()
        ? 'Demo preview: no file was created.'
        : ph === 'done' && f
          ? `Downloaded ${f.name}, ${bytes(f.size, prefs.units)}.`
          : ph === 'done' && r
            ? `Export finished: ${r.path.split('/').pop() ?? r.path}, ${bytes(r.bytes, prefs.units)}.`
            : '';
  });
  /** The file a server would create, for the demo's preview (the name its export path has). */
  const previewName = computed(() => {
    const r = result.get();
    return format.get() === 'onnx' ? onnxName.get() : r ? (r.path.split('/').pop() ?? r.path) : '';
  });
  const error = signal<unknown>(null);
  const copied = signal(false);
  /** Results inside the open dialog (a blocked copy, the demo's missing file) show in its status region, not a toast. */
  const status = new DialogStatus();
  let uiRun = 0;
  let destroyed = false;
  let copiedTimer: ReturnType<typeof setTimeout> | undefined;
  let statusEl: HTMLElement | undefined;
  let lastModelId = model().id;
  let lastEpoch = hub.epoch;
  effect(() => {
    const ep = hub.epoch;
    if (ep !== lastEpoch) {
      lastEpoch = ep;
      switchesOk.set(false);
      dynamicAxesOk.set(false);
      onnxDirect.set(false);
      reset();
    }
  });
  effect(() => {
    const mid = model().id;
    if (mid !== lastModelId) {
      lastModelId = mid;
      reset();
    }
  });
  // Disabling the Export button while running or swapping the footer to the result drops focus to <body>: keep focus
  // on the dialog's primary next action when the result or error lands while the dialog is open.
  effect(() => {
    const ph = phase.get();
    if (!open.get() || (ph !== 'done' && ph !== 'error')) return;
    void tick().then(() => {
      if (destroyed || !open.peek() || phase.peek() !== ph) return;
      const dlg = statusEl?.closest('dialog');
      if (!dlg) return;
      const active = document.activeElement;
      if (active && active !== document.body && active !== dlg && dlg.contains(active)) return;
      const next = dlg.querySelector<HTMLElement>('footer .primary') ?? dlg.querySelector<HTMLElement>('footer button:not(:disabled), footer a[href]');
      (next ?? dlg).focus({ preventScroll: true });
    });
  });
  onCleanup(() => {
    destroyed = true;
    clearTimeout(copiedTimer);
    copiedTimer = undefined;
  });

  /** An export that ended after the dialog closed (or moved to another model): its toast is out, and if it still owns the dialog UI, reset it. */
  function endedClosed(modelName: string, ownsUi: boolean, e?: unknown) {
    // page-level: only called once the dialog is closed, moved to another model, or unmounted.
    if (e !== undefined) toast.error(e, `Could not export ${modelName}`);
    if (ownsUi) reset();
  }

  /**
   * FOLO f16, and ONNX on a server with `weights_onnx`: asks for the file (an ONNX graph is built on the first
   * request, about as long as an export) and saves it through the browser under the server's name.
   */
  async function downloadFile() {
    if (phase.peek() === 'running') return;
    const id = ++uiRun;
    const m = model();
    const epoch = hub.epoch;
    phase.set('running');
    error.set(null);
    try {
      const api = hub.api;
      const f = await (format.peek() === 'onnx' ? api.onnxFile(m.id, onnxSwitches.peek() ? { ...switches.peek() } : {}) : api.f16File(m.id));
      if (hub.epoch !== epoch) return;
      const ownsUi = !destroyed && id === uiRun && model().id === m.id;
      if (ownsUi) {
        file.set(f);
        done.set('file');
        phase.set('done');
      }
      if (api.mock) return;
      const url = URL.createObjectURL(f);
      const a = document.createElement('a');
      a.href = url;
      a.download = f.name;
      a.rel = 'noopener';
      a.click();
      setTimeout(() => URL.revokeObjectURL(url), 60_000);
      if (!ownsUi || !open.peek()) {
        toast.success(`Downloaded ${f.name}`, bytes(f.size, prefs.units));
        endedClosed(m.name, ownsUi);
      }
    } catch (e) {
      if (hub.epoch !== epoch) return;
      const ownsUi = !destroyed && id === uiRun && model().id === m.id;
      if (!ownsUi || !open.peek()) return endedClosed(m.name, ownsUi, e);
      error.set(e);
      phase.set('error');
    }
  }

  async function start() {
    if (phase.peek() === 'running' || checkingOnnx.peek() || onnxCapsFailed.peek()) return;
    if (direct.peek()) return downloadFile();
    const id = ++uiRun;
    const m = model();
    const epoch = hub.epoch;
    const isDemo = demo.peek();
    phase.set('running');
    error.set(null);
    try {
      const fmt = format.peek();
      const sw = switches.peek();
      const reqSwitches: OnnxSwitches | undefined =
        fmt === 'onnx' && onnxSwitches.peek()
          ? onnxSwitchesForRequest({
              decode: sw.decode,
              half: sw.half,
              axes: axes.peek(),
              supportsSplitAxes: dynamicAxesOk.peek(),
            })
          : undefined;
      const r = await hub.api.exportModel(m.id, fmt, reqSwitches);
      if (hub.epoch !== epoch) return;
      const ownsUi = !destroyed && id === uiRun && model().id === m.id;
      if (ownsUi) {
        result.set(r);
        done.set(fmt);
        phase.set('done');
      }
      // The open dialog already says "Export finished" with the path: a toast would say it twice at once.
      if (!ownsUi || !open.peek()) {
        if (isDemo) toast.info('Demo preview', `Exporting ${m.name} created no file.`);
        else toast.success(`Exported ${m.name}`, `${r.path} · ${bytes(r.bytes, prefs.units)}`);
        endedClosed(m.name, ownsUi);
      }
    } catch (e) {
      if (hub.epoch !== epoch) return;
      const ownsUi = !destroyed && id === uiRun && model().id === m.id;
      if (!ownsUi || !open.peek()) return endedClosed(m.name, ownsUi, e);
      error.set(e);
      phase.set('error');
    }
  }

  async function copy() {
    const r = result.peek();
    if (!r) return;
    try {
      await navigator.clipboard.writeText(r.path);
      if (destroyed) return;
      copied.set(true);
      clearTimeout(copiedTimer);
      copiedTimer = setTimeout(() => {
        copiedTimer = undefined;
        if (!destroyed) copied.set(false);
      }, 1600);
    } catch {
      if (!destroyed) status.error(new Error('Your browser blocked clipboard access; select the path and copy it by hand.'), 'Could not copy the path');
    }
  }

  function reset() {
    uiRun++;
    clearTimeout(copiedTimer);
    copiedTimer = undefined;
    status.clear();
    done.set(null);
    switches.set(OFF);
    axes.set('static');
    phase.set('idle');
    result.set(null);
    file.set(null);
    error.set(null);
    copied.set(false);
  }

  function setAxes(next: OnnxDynamicAxes) {
    axes.set(next);
    switches.set({ ...switches.peek(), dynamic: next !== 'static' });
  }

  /** `bind:checked` of one switch (the switches are one immutable object). */
  const switchOf = (key: 'decode' | 'dynamic' | 'half') => ({
    get: () => switches.get()[key],
    set: (v: boolean) => {
      if (key === 'dynamic') {
        axes.set(v ? 'full' : 'static');
      }
      switches.set({ ...switches.peek(), [key]: v });
    },
  });
  const axesBind = {
    get: () => axes.get(),
    set: (v: OnnxDynamicAxes) => setAxes(v),
  };
  const close = () => open.set(false);
  // A file on its way is not dropped by closing: it goes on, and a toast says how it ended.
  const onclose = () => {
    if (phase.peek() !== 'running') reset();
  };

  function exportAnother() {
    reset();
    void tick().then(() => {
      if (destroyed || !open.peek() || phase.peek() !== 'idle') return;
      const dlg = statusEl?.closest('dialog');
      const next = dlg?.querySelector<HTMLElement>('input[name="export-format"]:checked') ?? dlg?.querySelector<HTMLElement>('input[name="export-format"]');
      next?.focus({ preventScroll: true });
    });
  }

  const footer = () => (
    <>
      {phase.get() === 'done' ? (
        // One shape for every result: "Export another" a quiet link at the footer's start (a way back, not an action on
        // the result), and the result's next step the one primary at the end: Connect to a server in the demo (Done the
        // secondary), else Done (the file is saved already, or stays on the server).
        <>
          <button type="button" class="another" onclick={exportAnother}>
            Export another
          </button>
          {demo.get() ? (
            <>
              <Button onclick={close}>Done</Button>
              <Button variant="primary" icon={Plug} href={href('/settings')} onclick={close}>
                Connect to a server
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
          <Button variant="primary" icon={Download} loading={phase.get() === 'running' || checkingOnnx.get()} disabled={onnxCapsFailed.get()} onclick={start}>
            {checkingOnnx.get() ? (
              'Checking server…'
            ) : phase.get() === 'error' ? (
              'Retry'
            ) : direct.get() ? (
              <>
                Download <span class="mono file">{directName.get()}</span>
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
    <Dialog bind:open={open} title="Export model" description={`Write ${model().name} to a file you can deploy.`} onclose={onclose} status={status} footer={footer}>
      {/* One live region that stays in the dialog, so the start and the end of an export are both announced (the toast is
          held back while the dialog is open). Failures announce themselves through ErrorState's alert. */}
      <p class="sr-only" role="status" ref={(e) => (statusEl = e)}>
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
        // The file, downloaded straight to this computer.
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
            onnxDirect.get() ? (
              <p class="note">
                Split dynamic-axis ONNX exports (<span class="mono">--dynamic-batch</span> / <span class="mono">--dynamic-hw</span>) stay on the machine
                running <span class="mono">folo serve</span>. Pick Static or Full dynamic for direct browser download.
              </p>
            ) : (
              <p class="note">
                Studio cannot download ONNX files yet: the file stays on the machine running <span class="mono">folo serve</span>. Copy it from there, or
                run <span class="mono">folo export --format onnx</span> where you need it.
              </p>
            )
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
                              <span class="opt-desc">
                                {o().key === 'dynamic' && dynamicAxesOk.get() && axes.get() === 'batch'
                                  ? 'Symbolic batch only, fixed height and width.'
                                  : o().key === 'dynamic' && dynamicAxesOk.get() && axes.get() === 'hw'
                                    ? 'Fixed batch 1, symbolic height and width in multiples of 32.'
                                    : o().desc}
                              </span>
                            </span>
                          </label>
                        )}
                      </For>
                      {dynamicAxesOk.get() && format.get() === 'onnx' ? (
                        <div class="axes-row">
                          <span class="axes-label">Dynamic axes</span>
                          <Segmented
                            label="Dynamic ONNX axes"
                            size="sm"
                            disabled={phase.get() === 'running'}
                            bind:value={axesBind}
                            options={AXES_SEG}
                          />
                          {splitAxesActive.get() && onnxDirect.get() && !demo.get() ? (
                            <span class="opt-desc">Exports to the server exports folder (direct download supports Static or Full).</span>
                          ) : null}
                        </div>
                      ) : null}
                    </div>
                  ) : null}
                </>
              )}
            </For>
          </fieldset>
          {phase.get() !== 'running' ? (
            <ExportCliPanel
              model={model()}
              format={format.get()}
              decode={switches.get().decode}
              half={switches.get().half}
              axes={axes.get()}
              showAxesPicker={!dynamicAxesOk.get()}
            />
          ) : null}
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
                    Preparing <span class="mono">{directName.get()}</span>…
                  </>
                ) : (
                  <>Exporting {model().name}…</>
                )}
              </span>
            </div>
          ) : onnxCapsFailed.get() ? (
            <div class="err">
              <ErrorState error={capsError.get()} title="Could not check ONNX support" retry={retryCaps} compact />
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
  .axes-row {
    display: grid;
    gap: var(--space-1);
    margin-top: var(--space-2);
    padding-top: var(--space-2);
    border-top: 1px solid var(--line-subtle);
  }
  .axes-label {
    font-size: var(--type-sm-size);
    color: var(--text-secondary);
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
