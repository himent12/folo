/**
 * How fast the engine is, against the incumbent of the same size on the same machine: every row of
 * docs/BENCHMARKS.md's summary where both have a number, including the ones FOLO trails. Values in ink; the lead is a
 * word ("1.66× faster", "17% less memory"), never a colour, and a row where FOLO is behind says so plainly. Only rows
 * measured under the same conditions state a winner: a row whose engines ran at different precision or did different
 * work keeps both measurements, and its conclusion is "—" beside the specific difference ("f32 vs TensorFloat-32"),
 * with one legend over the table saying what "—" means ("No winner: conditions differ"), never the same words in every
 * row. Units sit in the row's header, never in the cells. The table's default (both in f32) is said once, over the
 * columns; a row keeps a short qualifier only where it differs from that default, and the method and
 * caveats sit in a "How this was measured" fold.
 * Phones are a compact three-column table: one header row (folo-n, YOLO26n, the lead), then each row's metric over
 * its three values on the header's edges, and a no-winner row's own condition on a line under them. Every value
 * names its metric and column through `headers`, and explicit cell roles keep the table's semantics through the grid
 * layout.
 * Every value in the table is docs/BENCHMARKS.md's (the reference); where this server has measured folo-n itself
 * (`table`, GET /api/system/throughput), those figures follow the table under their own heading, never mixed into
 * it: YOLO26n was not run here, so they compare with nothing.
 */
import Check from '$icons/check';
import ChevronDown from '$icons/chevron-down';
import Copy from '$icons/copy';
import Download from '$icons/download';
import { computed, css, For, onCleanup, signal, tick } from 'fw';
import { advantage, BENCH_SOURCE, comparable, unlike, unlikeCondition, versusYolo26n, type Versus as VersusRow } from '$lib/benchmarks';
import type { ThroughputTable } from '$lib/api/types';
import { deviceName, num } from '$lib/format';
import { exportTableCsv } from '$lib/tableExport';
import { latencyMs, latencyRange, measuredLatency, measuredTraining } from '$lib/throughput';
import Button from '$lib/ui/Button';
import Panel from '$lib/ui/Panel';
import { toast } from '$lib/ui/toast';
import { latencyMetricLabel, versusCsv, versusMarkdown } from './versusExport';

const fmt = (v: VersusRow, x: number) => num(x, v.unit === 'ms' ? (x < 10 ? 2 : 1) : 0);
/** One format per kind: speed as a ratio ("1.66× faster", "1.09× slower"), memory as a share ("17% less memory"). */
function lead(v: VersusRow): string {
  const a = advantage(v);
  const ahead = a >= 1;
  if (v.unit === 'MiB') return `${Math.round(Math.abs(1 - v.folo / v.yolo) * 100)}% ${ahead ? 'less' : 'more'} memory`;
  return ahead ? `${a.toFixed(2)}× faster` : `${(1 / a).toFixed(2)}× slower`;
}
const caveats = versusYolo26n.filter((v) => v.caveat);
// What keeps a row from being like for like: one list, lib/benchmarks.ts `unlike` (Models' summary counts it too).
const mixed = versusYolo26n.filter((v) => !comparable(v)).length;

export interface VersusProps {
  table?: ThroughputTable | null;
}

export default function Versus(props: VersusProps) {
  const table = () => props.table ?? null;
  /** folo-n at 640 px as this server measured it: end to end (any precision; the line says which) and training. */
  const hereLat = computed(() => measuredLatency(table(), 'folo-n'));
  const hereTrain = computed(() => measuredTraining(table(), 'folo-n'));
  /** Those figures as rows of the table: the value in the folo-n column, its spread and count on the qualifier line (the
   *  precision only where it is not the table's f32). */
  const hereRows = computed(() => {
    const lat = hereLat.get();
    const train = hereTrain.get();
    return [
      ...(lat
        ? [
            {
              id: 'lat',
              metric: latencyMetricLabel(lat.device),
              value: latencyMs(lat.latency_ms),
              note: `${latencyRange(lat.latency_ms_low, lat.latency_ms_high)} · ${num(lat.samples)} predictions${lat.precision === 'f32' ? '' : ` · ${lat.precision}`}`,
            },
          ]
        : []),
      ...(train
        ? [
            {
              id: 'train',
              metric: `Training, batch ${train.batch}`,
              value: `${num(train.imgs_per_s)} img/s`,
              note: `${train.mosaic ? `${num(train.mosaic.low)}–${num(train.mosaic.high)} img/s · ` : ''}${num(train.samples)} samples · whole trainer`,
            },
          ]
        : []),
    ];
  });

  let how: HTMLDetailsElement | undefined;
  let alive = true;
  let copiedTimer: ReturnType<typeof setTimeout> | undefined;
  const copiedMd = signal(false);
  onCleanup(() => {
    alive = false;
    clearTimeout(copiedTimer);
    copiedTimer = undefined;
  });

  /** "See how this was measured": opens the fold and takes the reader there. */
  async function showHow() {
    if (!how) return;
    how.open = true;
    await tick();
    const s = how.querySelector('summary');
    s?.focus({ preventScroll: true });
    s?.scrollIntoView({ block: 'nearest', behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth' });
  }

  async function copyMarkdown() {
    try {
      await navigator.clipboard.writeText(versusMarkdown(table()));
      if (!alive) return;
      copiedMd.set(true);
      clearTimeout(copiedTimer);
      copiedTimer = setTimeout(() => {
        copiedTimer = undefined;
        if (alive) copiedMd.set(false);
      }, 1600);
      toast.success('Markdown copied', 'Engine speed comparison table copied to clipboard.', undefined, { key: 'versus-export' });
    } catch {
      if (alive) toast.error(new Error('Your browser blocked clipboard access.'), 'Could not copy Markdown', { key: 'versus-export' });
    }
  }

  function downloadCsv() {
    const csv = versusCsv(table());
    if (exportTableCsv(csv.filenameBase, csv.headers, csv.rows, 'versus-export')) {
      toast.success('CSV downloaded', `${csv.rows.length} benchmark rows saved.`, undefined, { key: 'versus-export' });
    }
  }

  return (
    <Panel
      title="Engine speed"
      pad="none"
      id="engine-speed"
      actions={
        <>
          <Button size="sm" variant="ghost" icon={copiedMd.get() ? Check : Copy} onclick={() => void copyMarkdown()}>
            {copiedMd.get() ? 'Copied' : 'Copy Markdown'}
          </Button>
          <Button size="sm" variant="ghost" icon={Download} onclick={downloadCsv}>
            Download CSV
          </Button>
        </>
      }
    >
      <p class="lede">
        folo-n vs YOLO26n on the same RTX 4080 SUPER, 640 px
        {hereLat.get() || hereTrain.get() ? <>: the reference figures, with this server's own folo-n figures last</> : null}. Both in f32 unless a row
        says otherwise.
      </p>{' '}
      {/* What "—" means, said once over every row that has it: those rows then carry only their own condition. */}
      {mixed ? (
        <p class="legend">
          <span class="dc" aria-hidden="true">
            —
          </span>
          &nbsp;No winner: conditions differ
          <span class="lg-n">
            {' · '}
            {mixed} of {versusYolo26n.length} rows
          </span>
          {' · '}
          <button type="button" class="see" onclick={showHow}>
            How this was measured
          </button>
        </p>
      ) : null}{' '}
      <table class="table vs">
        <thead>
          <tr>
            <th scope="col" role="columnheader" id="vs-h-m">
              Measured
            </th>
            <th scope="col" role="columnheader" class="r" id="vs-h-folo">
              folo-n
            </th>
            <th scope="col" role="columnheader" class="r" id="vs-h-yolo">
              YOLO26n
            </th>
            <th scope="col" role="columnheader" class="r lead-h" id="vs-h-lead">
              folo-n vs YOLO26n
            </th>
          </tr>
        </thead>
        <tbody>
          <For each={versusYolo26n} key={(v) => v.metric}>
            {(v, i) => {
              const ahead = () => advantage(v()) >= 1;
              const diff = () => unlike[v().metric];
              return (
                <tr class:behind={!diff() && !ahead()} class:unlike={!!diff()}>
                  <th scope="row" role="rowheader" id={`vs-r-${i()}`} headers="vs-h-m">
                    <span class="m">
                      {v().metric}
                      <span class="unit">, {v().unit}</span>
                    </span>{' '}
                    {v().note ? <span class="note">{v().note}</span> : null}
                  </th>
                  <td role="cell" class="r num v" headers={`vs-r-${i()} vs-h-folo`}>
                    {fmt(v(), v().folo)}
                  </td>
                  <td role="cell" class="r num v yolo" headers={`vs-r-${i()} vs-h-yolo`}>
                    {fmt(v(), v().yolo)}
                  </td>
                  <td role="cell" class="r lead" headers={`vs-r-${i()} vs-h-lead`}>
                    {diff() ? (
                      <>
                        <span class="dc" aria-label="No winner: different conditions">
                          —
                        </span>
                        <span class="why">{diff()!.differs}</span>
                        <span class="cond" data-fact>
                          <span aria-hidden="true">—&nbsp;</span>
                          {unlikeCondition(diff()!)}
                        </span>
                      </>
                    ) : (
                      lead(v())
                    )}
                  </td>
                </tr>
              );
            }}
          </For>
        </tbody>
        {table() && hereRows.get().length ? (
          // This server's own folo-n figures under the same columns, each with its spread and count. YOLO26n was not run
          // here, so they state no comparison: the last column says where they come from.
          <tbody class="here">
            <tr class="here-h">
              <th scope="colgroup" colspan={4} role="rowheader" id="vs-here">
                Measured on this server <span class="here-dev">{deviceName(table()!.device_name)}</span>
              </th>
            </tr>
            <For each={hereRows.get()} key={(h) => h.id}>
              {(h) => (
                <tr>
                  <th scope="row" role="rowheader" id={`vs-here-${h().id}`} headers="vs-h-m vs-here">
                    <span class="m">{h().metric}</span>{' '}
                    <span class="note num">{h().note}</span>
                  </th>
                  <td role="cell" class="r num v" headers={`vs-here-${h().id} vs-h-folo`}>
                    {h().value}
                  </td>
                  <td role="cell" class="r yolo nr" headers={`vs-here-${h().id} vs-h-yolo`}>
                    not run here
                  </td>
                  <td role="cell" class="r lead here-src" headers={`vs-here-${h().id} vs-h-lead`}>
                    Measured here
                  </td>
                </tr>
              )}
            </For>
          </tbody>
        ) : null}
      </table>{' '}
      {/* How the rows were made and what qualifies them, one step away. */}
      <details class="how" ref={(e) => (how = e)}>
        <summary>
          <ChevronDown size={14} aria-hidden="true" />
          How this was measured
        </summary>{' '}
        <div class="how-body">
          {/* The setup as facts, one per line, then the caveats as paragraphs at the measure: in a full-width panel they
              take two columns where they fit (each at least 36ch), so the text fills the frame instead of one narrow
              column at its left edge. */}
          <dl class="facts">
            <div>
              <dt>Hardware</dt>
              <dd>RTX 4080 SUPER; an i7-14700KF for the CPU rows</dd>
            </div>{' '}
            <div>
              <dt>Input</dt>
              <dd class="num">640 × 640 px</dd>
            </div>{' '}
            <div>
              <dt>Precision</dt>
              <dd>FOLO f32 throughout; YOLO26n f32 unless a row names TensorFloat-32 or fp16, its fastest</dd>
            </div>{' '}
            <div>
              <dt>End to end</dt>
              <dd>Decoded image in, boxes out, NMS included</dd>
            </div>{' '}
            <div>
              <dt>CPU threads</dt>
              <dd>Each engine at its best thread count</dd>
            </div>{' '}
            <div>
              <dt>Weights</dt>
              <dd>folo-n untrained; YOLO26n COCO-pretrained</dd>
            </div>{' '}
            <div>
              <dt>Load</dt>
              <dd>FOLO's CPU forward ran under load from other jobs</dd>
            </div>{' '}
            <div>
              <dt>Reading</dt>
              <dd>Times and memory: lower is better. Images per second: higher is better.</dd>
            </div>
          </dl>{' '}
          <div class="caveats">
            <p>
              <span class="cv-m">Untrained weights:</span> almost no FOLO candidates pass the confidence threshold, so its
              end-to-end times include less NMS work than a trained model's (about 0.1–0.3 ms per image).
            </p>{' '}
            <p>
              <span class="cv-m">Rows marked —:</span> the engines ran at different numeric precision (f32, TensorFloat-32,
              f16), did different work, or ran under different load. Those rows keep both measurements and name the
              difference, but no winner: the ratio would mix the engine with its conditions.
            </p>{' '}
            <For each={caveats} key={(v) => v.metric}>
              {(v) => (
                <p>
                  <span class="cv-m">{v().metric}:</span> {v().caveat}
                </p>
              )}
            </For>
          </div>{' '}
          <p class="src">Source: {BENCH_SOURCE.split(' · ')[0]}.</p>
        </div>
      </details>
    </Panel>
  );
}

css`
  /* Rows of one or two lines (metric, and a qualifier where the row differs): 8 px above and below, values on the
     metric's line. The row header looks like
     the table system's first cell (13 / 500 primary), not a column header. */
  .vs tbody :is(td, th) {
    vertical-align: top;
    padding-block: var(--space-2);
  }
  .vs tbody th {
    height: auto;
    text-align: left;
    font-size: var(--type-base-size);
    line-height: var(--type-base-line);
    color: var(--text-primary);
    white-space: normal;
    border-bottom: 1px solid var(--line-subtle);
  }
  .vs tbody tr:last-child th {
    border-bottom: 0;
  }
  .m {
    display: block;
    font-weight: var(--weight-medium);
  }
  .lede {
    padding: 0 var(--space-4) var(--space-3);
    color: var(--text-secondary);
  }
  /* The legend for "—", over the rows it explains, one step quieter than the lede. */
  .legend {
    padding: 0 var(--space-4) var(--space-3);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-tertiary);
  }
  .unit {
    font-weight: var(--weight-regular);
    color: var(--text-tertiary);
  }
  .see {
    color: var(--text-secondary);
    text-decoration: underline;
    text-decoration-color: var(--line-strong);
    text-underline-offset: 3px;
  }
  .see:hover {
    color: var(--text-primary);
    text-decoration-color: currentColor;
  }
  @media (pointer: coarse) {
    .see {
      display: inline-block;
      padding-block: var(--space-3);
      margin-block: calc(var(--space-3) * -1);
    }
  }
  /* One 12 px line per row (qualifiers are ≤ 60 characters, tested), in secondary: it holds figures people read
     ("YOLO26n with AMP: 252", "5.50–6.40 ms · 214 predictions"), not a label to skim (DESIGN_DIRECTION §2.3). */
  .note {
    display: block;
    font-weight: var(--weight-regular);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .vs tbody th {
    max-width: 0;
    width: 50%;
  }
  .r {
    text-align: right;
  }
  .v {
    white-space: nowrap;
    font-variant-numeric: tabular-nums;
  }
  .yolo {
    color: var(--text-secondary);
  }
  .lead {
    white-space: nowrap;
    font-weight: var(--weight-medium);
  }
  .behind .lead {
    font-weight: var(--weight-regular);
    color: var(--text-secondary);
  }
  /* No winner: one line like every other cell of the column, "—" in secondary ink and the specific difference beside
     it at 12 px tertiary; what "—" means is the one footnote under the table. */
  .unlike .lead {
    font-weight: var(--weight-regular);
  }
  .dc {
    color: var(--text-secondary);
  }
  .why {
    margin-left: var(--space-2);
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
  }
  /* The phone form of a no-winner row's conclusion (a line of its own there). */
  .cond {
    display: none;
  }
  /* The server's own rows: a group heading over them, the values in the folo-n column. */
  .vs tbody.here tr.here-h th {
    max-width: none;
    padding-top: var(--space-3);
    padding-bottom: var(--space-1);
    border-top: 1px solid var(--line-subtle);
    border-bottom: 0;
  }
  .vs tbody.here tr:last-child th {
    border-bottom: 0;
  }
  .here-dev {
    margin-left: var(--space-2);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    font-weight: var(--weight-regular);
    color: var(--text-tertiary);
  }
  .nr {
    color: var(--text-tertiary);
    white-space: nowrap;
  }
  .lead.here-src {
    font-weight: var(--weight-regular);
    color: var(--text-secondary);
  }
  .how {
    border-top: 1px solid var(--line-subtle);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
  }
  .how summary {
    display: flex;
    align-items: center;
    gap: var(--space-1);
    min-height: 40px;
    padding: 0 var(--space-4);
    list-style: none;
    cursor: pointer;
    color: var(--text-secondary);
    border-radius: 0 0 calc(var(--radius-lg) - 1px) calc(var(--radius-lg) - 1px);
  }
  .how summary::-webkit-details-marker {
    display: none;
  }
  .how summary:hover {
    color: var(--text-primary);
    background: var(--bg-hover);
  }
  .how summary:focus-visible {
    outline-offset: -2px;
  }
  .how summary :global(svg) {
    color: var(--text-tertiary);
    transform: rotate(-90deg);
    transition: transform var(--dur-fast) var(--ease-standard);
  }
  .how[open] summary :global(svg) {
    transform: none;
  }
  /* The fold's body spans the panel: a facts grid (term column, value), then the caveats in columns of at least 36ch,
     each paragraph at the measure, so a 1,200 px panel holds two readable columns rather than one 40ch strip. */
  .how-body {
    display: grid;
    gap: var(--space-4);
    padding: var(--space-1) var(--space-4) var(--space-4) calc(var(--space-4) + 18px);
    color: var(--text-secondary);
  }
  .facts {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(36ch, 1fr));
    gap: var(--space-1) var(--space-6);
  }
  .facts > div {
    display: grid;
    grid-template-columns: 80px minmax(0, 1fr);
    gap: var(--space-3);
    max-inline-size: var(--measure);
  }
  .facts dt {
    color: var(--text-tertiary);
  }
  .facts dd {
    color: var(--text-primary);
  }
  .caveats {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(36ch, 1fr));
    gap: var(--space-3) var(--space-6);
    align-items: start;
  }
  .caveats p {
    max-inline-size: var(--measure);
  }
  .src {
    color: var(--text-tertiary);
  }
  .cv-m {
    color: var(--text-primary);
  }
  @media (pointer: coarse) {
    .how summary {
      min-height: 44px;
    }
  }
  /* Phones: a compact three-column table. One header row names the columns; each row is its metric on a line of its
     own, then its three values on the header's edges. */
  @media (max-width: 767px) {
    .vs thead tr,
    .vs tbody tr {
      display: grid;
      /* One column template for every row, so folo-n, YOLO26n and the lead sit on the same three edges all the way
         down (a content-sized lead column would move them row by row); the lead's column is the widest. */
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr) minmax(0, 1.5fr);
      gap: var(--space-1) var(--space-3);
      padding: var(--space-3) var(--space-4);
      border-bottom: 1px solid var(--line-subtle);
    }
    .vs tbody tr:last-child {
      border-bottom: 0;
    }
    .vs thead tr {
      padding: 0 var(--space-4);
      border-bottom: 1px solid var(--line-subtle);
    }
    .vs thead th {
      height: 32px;
      padding: 0;
      border: 0;
      text-align: left;
    }
    /* The row headers' column has no cell of its own on phones: its name is read, not seen. */
    .vs thead th:first-child {
      position: absolute;
      width: 1px;
      height: 1px;
      overflow: hidden;
      clip-path: inset(50%);
      white-space: nowrap;
    }
    .vs tbody tr {
      padding-block: var(--space-2);
    }
    /* Rows grow with their lines (the table's fixed row height would let a note run into the divider). */
    .vs tbody :is(td, th) {
      height: auto;
      padding: 0;
      border: 0;
    }
    .vs tbody th {
      grid-column: 1 / -1;
      max-width: none;
      width: auto;
    }
    .vs .r {
      text-align: left;
    }
    .vs .lead {
      white-space: normal;
    }
    /* No winner: the conditions are a full-width line under the two values ("Different precision: f32 vs
       TensorFloat-32, no winner"), never a three-line "—" in the ratio column, which keeps real ratios only. */
    .vs tbody tr.unlike .lead {
      grid-column: 1 / -1;
    }
    .unlike .lead :is(.dc, .why) {
      display: none;
    }
    .cond {
      display: block;
      font-size: var(--type-sm-size);
      line-height: var(--type-sm-line);
      color: var(--text-secondary);
    }
    /* The legend's count is the least useful part on a phone: "— No winner: conditions differ · How this was measured". */
    .lg-n {
      display: none;
    }
  }
`;
