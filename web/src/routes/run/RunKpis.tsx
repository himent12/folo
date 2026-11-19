/**
 * The run's KPI strip (design/UNIFIED.md §4 Run, row 1): one panel cut into five tiles by hairlines. Epoch with its
 * progress rail (the last epochs without mosaic hatched at its end), mAP50-95 (the change since the last epoch, the
 * best), Precision / Recall at the epoch's best-F1 confidence, Throughput with the trend of its last steps, and the
 * time left (Duration once the run has ended). A tile is not a button: what the tabs once switched between is in the
 * four charts under it, their legends and their headers.
 */
import { computed, css, createUniqueId } from 'fw';
import { ETA_PENDING, progressText } from '$lib/api/progress';
import Sparkline from '$lib/charts/Sparkline';
import { ago, conf, duration, eta, metric, num } from '$lib/format';
import Stat from '$lib/ui/Stat';
import { tooltip } from '$lib/ui/tooltip';
import { failedEpoch } from './failure';
import { scoreDelta } from './readings';
import type { RunModel } from './runModel';

export interface RunKpisProps {
  model: RunModel;
  /** The run's history is on its way: the tiles' values hold their place. */
  loading: boolean;
}

export default function RunKpis(props: RunKpisProps) {
  const m = props.model;
  const run = () => m.run();
  const status = () => run().status;
  const uid = createUniqueId();
  const closeId = `${uid}-close`;
  const completed = () => status() === 'completed';
  const running = () => status() === 'running' || status() === 'stopping';
  /** Nothing to read yet (a queued run, or the history still loading). */
  const waiting = () => status() === 'queued' && run().step === 0;

  const epochCap = computed(() => {
    const r = run();
    if (r.status === 'queued') return 'Not started';
    // Running with no step yet: the trainer is built and its first batches are on the way.
    if (running() && r.step === 0) return 'Starting the first epoch…';
    // The trainer's early stop, in the words of what it saw, so "Epoch 256 of 2000" never reads as lost epochs.
    if (completed() && r.epoch < r.epochs) return r.best ? `stopped early: no better after ep ${r.best.epoch}` : 'stopped early';
    return m.mosaicOff.get() ? `${progressText(r)} · ${m.closeText.get().cap}` : progressText(r);
  });
  const nothing = computed(() => run().best?.map50_95 === 0);
  // Nothing is final, or last, while the epochs the run validated are on their way, and "final" only once the server's
  // ended snapshot confirms the last epoch held is the run's last validated one (runModel `pending`, `final`).
  const mapLabel = computed(() =>
    running() ? 'mAP50-95 · latest' : waiting() || m.pending.get() ? 'mAP50-95' : completed() ? (m.final.get() ? 'mAP50-95 · final' : 'mAP50-95') : 'mAP50-95 · last',
  );
  const mapCap = computed(() => {
    const b = run().best;
    const le = m.lastEpoch.get();
    const prev = m.prevEpoch.get();
    if (nothing.get()) return m.noneYet.get();
    if (running() && le && prev) return `${scoreDelta(le.val.map50_95, prev.val.map50_95)} vs last${b ? ` · best ${metric(b.map50_95)}` : ''}`;
    return b ? `best ${metric(b.map50_95)} · ep ${b.epoch}` : m.firstEpoch.get();
  });
  const prCap = computed(() => {
    const le = m.lastEpoch.get();
    const pc = m.prConf.get();
    if (le && le.val.precision === 0 && le.val.recall === 0) return m.noneYet.get();
    if (!le) return m.firstEpoch.get();
    return pc != null ? `conf ${conf(pc)} · ep ${le.epoch}` : `best-F1 conf · ep ${le.epoch}`;
  });
  const ipsCap = computed(() => {
    const t = m.ips.get();
    // The average over the sparkline's window (the last 120 steps, as Desktop's tile), said as Desktop says it.
    return t.avg !== null ? `avg ${num(t.avg, 0)}` : waiting() ? 'Not started' : 'after the first step';
  });
  const ended = () => !m.live.get();
  /** The per-epoch tiles: loading too while an ended run's validated epochs are on their way (runModel `pending`). */
  const epochLoading = () => props.loading || m.pending.get();
  const timeValue = computed(() => {
    const r = run();
    if (ended()) return duration(m.elapsed.get());
    return running() && m.etaReady.get() && r.eta_s !== null ? eta(r.eta_s) : '—';
  });
  const timeCap = computed(() => {
    const r = run();
    if (ended()) return r.finished ? `ended ${ago(r.finished)}` : 'ended';
    if (r.status === 'queued') return 'Not started';
    if (!m.etaReady.get()) return ETA_PENDING;
    return m.elapsed.get() !== null ? `${duration(m.elapsed.get())} elapsed` : '';
  });
  const shownEpoch = () => (status() === 'failed' ? (failedEpoch(run()) ?? 0) : run().epoch);

  return (
    <section class="kpis" aria-label="Run summary">
      <div class="tile">
        <Stat label="Epoch" value={num(shownEpoch())} unit={`/ ${run().epochs}`} sub={epochCap.get()} />
        {/* "38 / 100": the epoch in progress, its rail lime only while the run trains. */}
        <div class="bar">
          <div
            class="track"
            role="progressbar"
            aria-label="Training progress"
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={completed() ? 100 : Math.floor(m.progress.get() * 100 + 1e-9)}
            aria-describedby={m.mosaicOff.get() ? closeId : undefined}
          >
            <div class="fill" class:live={running()} style:transform={`scaleX(${completed() ? 1 : m.progress.get()})`}></div>
            {/* The hatched end, the last epochs without mosaic; over the filled part it is cut out of the fill. */}
            {m.mosaicOff.get() ? (
              <div class="close" class:over={completed() || m.progress.get() >= m.closeFrom.get()} style:left={`${m.closeFrom.get() * 100}%`} use:tooltip={m.closeText.get().full}></div>
            ) : null}
          </div>
          {m.mosaicOff.get() ? (
            <span class="sr-only" id={closeId}>
              {m.closeText.get().full}
            </span>
          ) : null}
        </div>
      </div>
      <div class="tile">
        <Stat label={mapLabel.get()} value={metric(m.lastEpoch.get()?.val.map50_95)} sub={mapCap.get()} loading={epochLoading()} />
      </div>
      <div class="tile">
        <Stat
          label="Precision / Recall"
          value={m.lastEpoch.get() ? `${metric(m.lastEpoch.get()!.val.precision)} / ${metric(m.lastEpoch.get()!.val.recall)}` : '—'}
          sub={prCap.get()}
          loading={epochLoading()}
        />
      </div>
      <div class="tile ips">
        <Stat
          label={ended() && !waiting() ? 'Final throughput' : 'Throughput'}
          value={num(m.lastStep.get()?.imgs_per_s, 0)}
          unit={m.lastStep.get() ? 'img/s' : undefined}
          sub={ipsCap.get()}
          loading={props.loading}
        />
        {/* A glance, not a chart: the trend of the last steps, neutral, the end lime while the run trains. */}
        {m.ips.get().trend.length > 2 ? (
          <span class="spark" aria-hidden="true">
            <Sparkline values={m.ips.get().trend} width={96} height={24} color="var(--text-tertiary)" endColor={running() ? 'var(--accent-line)' : undefined} fill={false} />
          </span>
        ) : null}
      </div>
      <div class="tile">
        <Stat label={ended() ? 'Duration' : 'Time left'} value={timeValue.get()} sub={timeCap.get()} />
      </div>
    </section>
  );
}

css`
  /* One panel cut into tiles: 1 px gaps over a hairline-coloured background draw the dividers in any column count. */
  .kpis {
    display: grid;
    grid-template-columns: repeat(5, minmax(0, 1fr));
    gap: 1px;
    background: var(--line-subtle);
    background-clip: padding-box;
    border: 1px solid var(--line-panel);
    border-radius: var(--radius-lg);
    overflow: hidden;
    min-width: 0;
  }
  .tile {
    position: relative;
    display: grid;
    align-content: start;
    gap: var(--space-2);
    padding: var(--space-3) var(--space-4);
    background: var(--bg-panel);
    min-width: 0;
  }
  /* A caption is a fact: where a tile is too narrow for it (the Epoch's "37% · Mosaic off after epoch 90" on a phone)
     it wraps, never ends in an ellipsis. */
  .tile :global(.sub) {
    white-space: normal;
  }
  /* The throughput's trend beside its value, right-aligned in the tile. The value and its unit never give way: where
     the tile is short of room the trend narrows to its latest steps (clipped on the left, its end kept). */
  .tile.ips {
    grid-template-columns: max-content minmax(0, 96px);
    justify-content: space-between;
    column-gap: var(--space-3);
  }
  .spark {
    align-self: center;
    display: flex;
    justify-content: flex-end;
    min-width: 0;
    overflow: hidden;
  }
  .spark :global(svg) {
    flex-shrink: 0;
  }
  .bar {
    position: relative;
    min-width: 0;
  }
  .track {
    position: relative;
    height: 4px;
    border-radius: 2px;
    background: var(--bg-active);
    overflow: hidden;
  }
  .fill {
    position: absolute;
    inset: 0;
    transform-origin: left;
    background: var(--text-tertiary);
    border-radius: 2px;
    transition: transform var(--dur-slow) var(--ease-standard);
  }
  /* Lime is now: the rail of a run that trains. */
  .fill.live {
    background: var(--accent-line);
  }
  .close {
    position: absolute;
    top: 0;
    bottom: 0;
    right: 0;
    background: repeating-linear-gradient(135deg, transparent 0 3px, var(--line-strong) 3px 4px);
    border-left: 1px solid var(--line-strong);
  }
  .close.over {
    background: repeating-linear-gradient(135deg, transparent 0 3px, var(--bg-panel) 3px 4.5px);
    border-left-color: var(--bg-panel);
  }
  /* Tablets: three tiles over two, Throughput (whose trend fills it) two columns wide, so its divider lines up with
     the first row's second. */
  @media (max-width: 1023px) {
    .kpis {
      grid-template-columns: repeat(6, minmax(0, 1fr));
    }
    .tile {
      grid-column: span 2;
    }
    .tile.ips {
      grid-column: span 4;
    }
  }
  /* Phones: 2 × 2, then the time left across both columns. */
  @media (max-width: 767px) {
    .kpis {
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
    .tile,
    .tile.ips {
      grid-column: auto;
      padding: var(--space-3);
    }
    .tile:last-child {
      grid-column: 1 / -1;
    }
    .spark {
      display: none;
    }
  }
`;
