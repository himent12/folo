/**
 * A run's outcome under the header, every state in one pattern (./Callout): icon, title, cause, one action, and a
 * disclosure for the rest. Out of GPU memory says so plainly, with the smaller setting the memory forecast says fits
 * as the action, named with its forecast ("Retry with batch 8 · ~13.6 GiB", lib/oom `fitFixes`), and the other fixes
 * and the driver's report behind "More fixes". A queued run leads with what it waits for and when it
 * should start; on servers with queue control, with its place ("2nd in queue") and the way to the front. The trainer's warning that a run is too short to
 * learn from (./warnings) is one line, "Too short to learn: 125 optimizer steps", with the engine's sentences in plain
 * words behind "Why?"; never a toast. Once the run has an outcome, that line joins the outcome's callout as its second
 * line ("Completed in 30s · best …", then "Too short to learn: …"), so one run never stacks two banners.
 */
import CircleCheck from '$icons/circle-check';
import Clock from '$icons/clock';
import Play from '$icons/play';
import RotateCcw from '$icons/rotate-ccw';
import ScanSearch from '$icons/scan-search';
import Square from '$icons/square';
import TriangleAlert from '$icons/triangle-alert';
import { computed, css, effect, For, signal } from 'fw';
import { hub } from '$lib/api/index';
import type { Run, SystemInfo } from '$lib/api/types';
import { bytes, duration, metric } from '$lib/format';
import { fitFixes, isOom, oomCalibration, oomDetail, oomFixes, retryPath, vramForecaster, type OomFix } from '$lib/oom';
import { prefs } from '$lib/prefs';
import { href } from '$lib/router';
import Button from '$lib/ui/Button';
import Ident from '$lib/ui/Ident';
import type { IconType } from '$lib/ui/types';
import Callout from './Callout';
import { plain, scheduleTitle, shortCause, suggestedEpochs } from './warnings';

export interface OutcomeProps {
  run: Run;
  elapsed: number | null;
  resuming?: boolean;
  onresume: () => void;
  /** A queued run: what it waits for and when it should start. */
  waiting?: () => JSX.Element;
  /** A queued run's place, "2nd in queue", where the server reports it; a cloud run's phase. */
  waitingTitle?: string;
  /** The waiting state's mark: a clock in the queue, a cloud for a cloud run's hand-over. */
  waitingIcon?: IconType;
  /** A queued run's action: Start next. */
  waitingActions?: () => JSX.Element;
  /** The trainer's schedule warnings, as it wrote them (./warnings). */
  warnings?: string[];
  /**
   * `/api/system`: the GPU's memory and whether the server estimates runs, to forecast the fixes. Undefined while it
   * loads; null when it could not be read (the fixes are then offered unforecast).
   */
  system?: SystemInfo | null;
}

const NONE: string[] = [];

export default function Outcome(props: OutcomeProps) {
  const warnings = () => props.warnings ?? NONE;
  // Out of GPU memory: the same config fails again, so the ways out are smaller settings (a new run), most likely first.
  const oom = computed(() => props.run.status === 'failed' && isOom(props.run.error));
  // Each fix is forecast as New run will forecast it (lib/oom `fitFixes`): the batch is the largest that fits the GPU
  // with 10% to spare, and the fixes that fit lead. Null while the forecast loads.
  const fixes = signal<OomFix[] | null>(null);
  const gpuBytes = computed(() => {
    const d = props.run.config.device;
    const g = props.system?.gpus.find((x) => (d.startsWith('gpu:') ? `gpu:${x.ordinal}` === d : true));
    return g && g.memory_bytes > 0 ? g.memory_bytes : null;
  });
  effect(() => {
    const system = props.system;
    if (!oom.get() || !system) {
      fixes.set(!oom.get() ? [] : system === null ? oomFixes(props.run.config) : null);
      return;
    }
    const ctrl = new AbortController();
    const c = props.run.config;
    const gb = gpuBytes.get();
    fitFixes(c, gb, vramForecaster(hub.api, system, ctrl.signal), oomCalibration(c, gb)).then(
      (f) => !ctrl.signal.aborted && fixes.set(f),
      () => {},
    );
    return () => ctrl.abort();
  });
  /** A fix with what it costs: "Retry with batch 8 · ~13.6 GiB". */
  const fixLabel = (f: OomFix) => (f.vram !== undefined ? `${f.label} · ~${bytes(f.vram, prefs.units)}` : f.label);
  const driver = computed(() => (oom.get() ? oomDetail(props.run.error) : null));
  /** The epochs that clear the warning (⌈200 / steps per epoch⌉): its action opens New run with them filled in. */
  const suggested = computed(() => suggestedEpochs(warnings()));
  /**
   * The trainer's schedule warning as a callout line, "Too short to learn: 125 optimizer steps": the second line of the
   * run's outcome callout once there is one, else a callout of its own.
   */
  const short = computed(() => {
    const w = warnings();
    return w.length ? { icon: TriangleAlert, tone: 'warning' as const, title: scheduleTitle(w), msg: shortCause(w) } : undefined;
  });
  /** The other fixes behind the disclosure: those that may fit. */
  const others = computed(() => fixes.get()?.slice(1).filter((f) => f.fits !== false) ?? []);

  const scheduleFix = () => (
    <>
      {suggested.get() && suggested.get()! > props.run.config.epochs ? (
        <Button size="sm" icon={RotateCcw} href={href(`${retryPath(props.run.id)}&epochs=${suggested.get()}`)}>
          Train {suggested.get()} epochs
        </Button>
      ) : (
        <Button size="sm" icon={RotateCcw} href={href(retryPath(props.run.id))}>
          New run like this
        </Button>
      )}
    </>
  );
  const why = () => (
    <For each={warnings()} key={(w) => w}>
      {(w) => <p class="why">{plain(w())}</p>}
    </For>
  );

  const status = () => props.run.status;

  return (
    <>
      {status() === 'failed' && oom.get() ? (
        // Out of memory blocks any run: its fix stays the action; a schedule warning joins as the second line, its fix
        // and reasons behind More fixes.
        <Callout
          tone="danger"
          icon={TriangleAlert}
          title="Ran out of GPU memory"
          role="alert"
          stack
          moreLabel="More fixes"
          also={short.get()}
          actions={() => (
            <>
              {fixes.get() === null ? (
                // The forecast is on its way: the action keeps its place and says it is choosing.
                <Button size="sm" variant="primary" icon={RotateCcw} loading>
                  Retry with smaller settings
                </Button>
              ) : fixes.get()![0] ? (
                <Button size="sm" variant="primary" icon={RotateCcw} href={href(retryPath(props.run.id, fixes.get()![0]))}>
                  {fixLabel(fixes.get()![0]!)}
                </Button>
              ) : (
                <Button size="sm" icon={RotateCcw} href={href(retryPath(props.run.id))}>
                  New run like this
                </Button>
              )}
            </>
          )}
          more={() => (
            <>
              {others.get().length > 0 || short.get() ? (
                <div class="alts cta-row">
                  <For each={others.get()} key={(f) => f.kind}>
                    {(f) => (
                      <Button size="sm" href={href(retryPath(props.run.id, f()))}>
                        {fixLabel(f())}
                      </Button>
                    )}
                  </For>
                  {short.get() ? scheduleFix() : null}
                </div>
              ) : null}
              {driver.get() ? (
                <>
                  <p class="d-head">Driver report</p>
                  <code class="mono">{driver.get()}</code>
                </>
              ) : null}
              {short.get() ? why() : null}
            </>
          )}
        >
          Batch {props.run.config.batch} at {props.run.config.imgsz} px with <Ident value={props.run.config.arch} /> did not fit, at epoch {props.run.epoch}.
        </Callout>
      ) : status() === 'failed' ? (
        <Callout
          tone="danger"
          icon={TriangleAlert}
          title={`Failed at epoch ${props.run.epoch}`}
          role="alert"
          moreLabel={short.get() ? 'Why?' : 'More fixes'}
          also={short.get()}
          actions={() => (
            <>
              {short.get() ? scheduleFix() : null}
              {/* Anything but out of memory can continue from the last checkpoint. */}
              <Button size="sm" icon={Play} loading={props.resuming ?? false} onclick={props.onresume}>
                Resume
              </Button>
            </>
          )}
          more={() => (
            <>
              {short.get() ? (
                why()
              ) : (
                <div class="alts cta-row">
                  <Button size="sm" icon={RotateCcw} href={href(retryPath(props.run.id))}>
                    New run like this
                  </Button>
                </div>
              )}
            </>
          )}
        >
          {props.run.error}
        </Callout>
      ) : status() === 'completed' && short.get() ? (
        // A run that completed but was too short to learn: the outcome and the warning in one callout, the fix its action
        // (trying a model that learned nothing in the Playground is not the next step).
        <Callout
          tone="success"
          icon={CircleCheck}
          title={`Completed in ${duration(props.elapsed)}`}
          moreLabel="Why?"
          also={short.get()}
          facts={() => (
            <span>
              best mAP50-95 <span class="num">{metric(props.run.best?.map50_95)}</span> at epoch {props.run.best?.epoch ?? 0}
            </span>
          )}
          actions={scheduleFix}
          more={why}
        />
      ) : status() === 'completed' ? (
        <Callout
          tone="success"
          icon={CircleCheck}
          title={`Completed in ${duration(props.elapsed)}`}
          brief
          facts={() => (
            <>
              <span>
                best mAP50-95 <span class="num">{metric(props.run.best?.map50_95)}</span> at epoch {props.run.best?.epoch ?? 0}
              </span>
              <span>saved as a model</span>
            </>
          )}
          actions={() => (
            <>
              {props.run.model_id ? (
                <Button size="sm" icon={ScanSearch} href={href(`/playground?model=${props.run.model_id}`)}>
                  Try in Playground
                </Button>
              ) : null}
            </>
          )}
        />
      ) : status() === 'stopped' ? (
        <Callout
          tone="neutral"
          icon={Square}
          title={`Stopped at epoch ${props.run.epoch}`}
          moreLabel="Why?"
          also={short.get()}
          more={short.get() ? why : undefined}
          actions={() => (
            <>
              {short.get() ? scheduleFix() : null}
              <Button size="sm" icon={Play} loading={props.resuming ?? false} onclick={props.onresume}>
                Resume
              </Button>
            </>
          )}
        >
          The last checkpoint was kept{props.run.model_id ? ' and saved as a model' : ''}.
        </Callout>
      ) : status() === 'queued' && props.waiting ? (
        // The dependency and the start estimate lead; the frames below stay compact until there is data.
        <Callout tone="neutral" icon={props.waitingIcon ?? Clock} title={props.waitingTitle ?? 'Waiting to start'} actions={props.waitingActions}>
          {props.waiting}
        </Callout>
      ) : short.get() ? (
        // No outcome yet (a live run): the warning alone, one line: the cause in numbers and the fix; the engine's
        // sentences, in plain words, behind "Why?".
        <Callout
          tone="warning"
          icon={TriangleAlert}
          title={short.get()!.title}
          role="note"
          label="Schedule warning"
          moreLabel="Why?"
          runIn
          more={why}
          actions={scheduleFix}
        >
          {short.get()!.msg}
        </Callout>
      ) : null}
    </>
  );
}

css`
  .alts {
    display: flex;
    flex-wrap: wrap;
    gap: var(--space-2);
    padding-top: var(--space-1);
  }
  .d-head {
    margin-top: var(--space-2);
    color: var(--text-tertiary);
  }
  code {
    display: block;
    overflow-wrap: anywhere;
    color: var(--text-secondary);
  }
  /* The engine's sentences at a readable measure. */
  .why {
    max-inline-size: var(--measure);
  }
  .why + .why {
    margin-top: var(--space-2);
  }
  @media (max-width: 767px) {
    /* Phones: the other fixes as full-width secondary buttons, one per row (each names its setting and its forecast,
       "Retry at 640 px, batch 16 · ~12.1 GiB", too long to share a row), in the one phone button size (.cta-row) of the
       fix above them. */
    .alts {
      flex-direction: column;
      align-items: stretch;
    }
    .alts > :global(.btn.btn) {
      min-width: 0;
    }
  }
`;
