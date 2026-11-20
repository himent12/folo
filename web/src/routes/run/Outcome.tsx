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
import Box from '$icons/box';
import CircleCheck from '$icons/circle-check';
import Clock from '$icons/clock';
import Database from '$icons/database';
import Play from '$icons/play';
import RotateCcw from '$icons/rotate-ccw';
import ScanSearch from '$icons/scan-search';
import CircleStop from '$icons/circle-stop';
import TriangleAlert from '$icons/triangle-alert';
import { computed, css, effect, For, signal } from 'fw';
import { hub } from '$lib/api/index';
import type { Run, SystemInfo } from '$lib/api/types';
import { stoppedEarly } from '$lib/api/progress';
import { bytes, duration, metric } from '$lib/format';
import { fitFixes, isOom, oomCalibration, oomDetail, oomFixes, retryPath, vramForecaster, type OomFix } from '$lib/oom';
import { prefs } from '$lib/prefs';
import { href } from '$lib/router';
import Button from '$lib/ui/Button';
import Ident from '$lib/ui/Ident';
import type { IconType } from '$lib/ui/types';
import Callout from './Callout';
import { failedEpoch, failedTitle, failureSummary } from './failure';
import { LEARNED_MAP, plain, scheduleTitle, shortCause, suggestedEpochs } from './warnings';

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
  /** The run's dataset's name, for a failure that names it. */
  datasetName?: string;
  /**
   * The run's dataset is known to be removed from FOLO (../overview/newDatasets `datasetPresence`): Check dataset is then
   * not offered (its page is gone).
   */
  datasetRemoved?: boolean;
  /**
   * `/api/system`: the GPU's memory and whether the server estimates runs, to forecast the fixes. Undefined while it
   * loads; null when it could not be read (the fixes are then offered unforecast).
   */
  system?: SystemInfo | null;
}

const NONE: string[] = [];

/**
 * A run has a checkpoint to continue from: a stopped run with weights has a model, a failed one a best epoch (an
 * epoch finished, so its checkpoint was written).
 */
export const canContinue = (run: Pick<Run, 'best' | 'model_id'>) => !!run.model_id || !!run.best;

/** `POST /api/runs` takes `resume_run_id` (docs/API.md, `features`): the mock server lists it not, and answers 501. */
export const RESUME_FEATURE = 'run_resume';
/**
 * Whether Resume (or Start again) is offered: not on a server that said it cannot resume (`folo serve --mock`; issue
 * #382: the palette's Resume answered 501 there). Unknown (an older server, the in-browser demo) is offered, as before.
 */
export const canResume = () => !hub.lacks(RESUME_FEATURE);

/**
 * A stopped run's title: a stop the server's shutdown or restart made is said as such, never as the user's own stop. A
 * cloud run's is its worker's (a remote server or an instance restarted), not this server's, which ran on.
 */
function stoppedTitle(run: Run) {
  const who = run.cloud ? 'the worker' : 'the server';
  if (run.error_code === 'interrupted') return run.started ? `Interrupted: ${who} stopped during training` : `Interrupted: ${who} stopped before it started`;
  return canContinue(run) ? `Stopped at epoch ${run.epoch}` : run.started ? 'Stopped during its first epoch' : 'Stopped before it started';
}

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
  /** A clearly longer run that clears the warning (./warnings suggestedEpochs): its action opens New run with it. */
  const suggested = computed(() => suggestedEpochs(warnings(), props.run.config.epochs));
  /** It scored well despite the warning: the warning advises more epochs instead of saying it could not learn. */
  const learned = computed(() => (props.run.best?.map50_95 ?? 0) >= LEARNED_MAP);
  /**
   * The trainer's schedule warning as a callout line, "Too short to learn: 125 optimizer steps": the second line of the
   * run's outcome callout once there is one, else a callout of its own.
   */
  const short = computed(() => {
    const w = warnings();
    return w.length ? { icon: TriangleAlert, tone: 'warning' as const, title: scheduleTitle(w, learned.get()), msg: shortCause(w) } : undefined;
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
      {(w) => <p class="why">{plain(w(), learned.get())}</p>}
    </For>
  );

  const status = () => props.run.status;
  /** "best mAP50-95 0.604 at epoch 150"; a model that detected nothing says so instead of "best 0.000 at epoch 1". */
  const bestFact = () => (
    <span>
      {props.run.best && props.run.best.map50_95 > 0 ? (
        <>
          best mAP50-95 <span class="num">{metric(props.run.best.map50_95)}</span> at epoch {props.run.best.epoch}
        </>
      ) : (
        'no detections'
      )}
    </span>
  );
  /**
   * Why a completed run ended before its last planned epoch: the trainer's early stop, in the words of what it saw ("no
   * better score after epoch 156"), so "Epoch 256 of 2000" never reads as a crash or lost epochs.
   */
  const earlyFact = () =>
    stoppedEarly(props.run) ? (
      <span>
        stopped early at epoch {props.run.epoch} of {props.run.epochs}
        {props.run.best ? `: no better score after epoch ${props.run.best.epoch}` : ''}
      </span>
    ) : null;
  /** A failure's cause in plain words (./failure), by its error code; null: the server's message is the cause. */
  const summary = computed(() => (props.run.status === 'failed' ? failureSummary(props.run.error_code, props.datasetName ?? 'this run', failedEpoch(props.run) !== null) : null));
  /**
   * The hand-off of a run with a model, beside its result whatever else the callout says (a too-short warning keeps
   * it): Open model, and Try in Playground unless the model scored exactly 0 (nothing to see there).
   */
  const handOff = () => (
    <>
      {props.run.model_id ? (
        <>
          <Button size="sm" icon={Box} href={href(`/models/${props.run.model_id}`)}>
            Open model
          </Button>
          {props.run.best?.map50_95 !== 0 ? (
            <Button size="sm" icon={ScanSearch} href={href(`/playground?model=${props.run.model_id}`)}>
              Try in Playground
            </Button>
          ) : null}
        </>
      ) : null}
    </>
  );
  /** Resume continues from the last finished epoch's checkpoint; with none (no epoch finished) it starts the run over. */
  const resumeAction = () =>
    canResume() ? (
      <Button size="sm" icon={canContinue(props.run) ? Play : RotateCcw} loading={props.resuming ?? false} onclick={props.onresume}>
        {canContinue(props.run) ? 'Resume' : 'Start again'}
      </Button>
    ) : null;

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
          Batch {props.run.config.batch} at {props.run.config.imgsz} px with <Ident value={props.run.config.arch} /> did not fit{failedEpoch(props.run) === null ? ' before its first step' : `, at epoch ${failedEpoch(props.run)}`}.
        </Callout>
      ) : status() === 'failed' ? (
        // What went wrong in plain words by its error code, the server's own message under Details. A failure is not
        // about the run's length: a schedule warning stays a line of its own, its fix behind Details with its reasons.
        <Callout
          tone="danger"
          icon={TriangleAlert}
          title={failedTitle(props.run)}
          role="alert"
          moreLabel="Details"
          also={short.get()}
          actions={() => (
            <>
              {props.run.error_code === 'data_error' && !props.datasetRemoved ? (
                <Button size="sm" icon={Database} href={href(`/datasets/${props.run.config.dataset_id}`)}>
                  Check dataset
                </Button>
              ) : null}
              {/* Anything but out of memory can continue from the last checkpoint. */}
              {resumeAction()}
            </>
          )}
          more={() => (
            // The server's own message in a labelled box of its own, then the way on; the schedule warning, which is not
            // why it failed, is a section of its own under its name, never beside the message.
            <>
              {summary.get() && props.run.error ? (
                <>
                  <p class="d-head first">Error message</p>
                  <code class="mono raw">{props.run.error}</code>
                </>
              ) : null}
              <div class="alts cta-row">
                <Button size="sm" icon={RotateCcw} href={href(retryPath(props.run.id))}>
                  New run like this
                </Button>
              </div>
              {short.get() ? (
                <>
                  <p class="d-head">{short.get()!.title.replace(/:$/, '')}</p>
                  {why()}
                  <div class="alts cta-row">{scheduleFix()}</div>
                </>
              ) : null}
            </>
          )}
        >
          {summary.get() ?? props.run.error}
        </Callout>
      ) : status() === 'completed' && short.get() ? (
        // A run that completed but was too short to learn: the outcome and the warning in one callout, the fix its
        // action (the model's hand-off is the header's Open model and Try in Playground).
        <Callout
          tone="success"
          icon={CircleCheck}
          title={`Completed in ${duration(props.elapsed)}`}
          moreLabel="Why?"
          also={short.get()}
          facts={() => (
            <>
              {bestFact()}
              {earlyFact()}
            </>
          )}
          actions={scheduleFix}
          more={why}
        />
        // A plain completion has no callout (design/UNIFIED.md §4 Run): its duration is the Duration tile, its best the
        // mAP tile's caption, an early stop the Epoch tile's, and its model the header's Open model.
      ) : status() === 'stopped' ? (
        // A run stopped before its first epoch ended has no checkpoint: it says so, and starts again rather than resumes.
        <Callout
          tone="neutral"
          icon={CircleStop}
          title={stoppedTitle(props.run)}
          moreLabel="Why?"
          also={short.get()}
          more={short.get() ? why : undefined}
          actions={() => (
            <>
              {short.get() ? scheduleFix() : null}
              {resumeAction()}
              {handOff()}
            </>
          )}
        >
          {canContinue(props.run)
            ? `The last checkpoint was kept${props.run.model_id ? ' and saved as a model' : ''}.`
            : props.run.started
              ? 'No epoch had finished, so nothing was saved.'
              : 'It never trained, so nothing was saved.'}
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
  .d-head.first {
    margin-top: 0;
  }
  /* A failure's raw message: a box of its own, apart from the advice. */
  .raw {
    margin-top: var(--space-1);
    padding: var(--space-2) var(--space-3);
    background: var(--bg-inset);
    border: 1px solid var(--line-panel);
    border-radius: var(--radius-sm);
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
