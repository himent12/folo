<script lang="ts">
  /**
   * What the model confuses and misses: the confusion matrix beside its two ranked lists (each line shows its cell)
   * and how to read it; on phones the lists lead and the matrix is one step away.
   */
  import ArrowLeft from '@lucide/svelte/icons/arrow-left';
  import ChevronDown from '@lucide/svelte/icons/chevron-down';
  import { tick } from 'svelte';
  import { MediaQuery } from 'svelte/reactivity';
  import { reduceConfusion, type Evaluation } from '$lib/api/evaluation';
  import type { Resource } from '$lib/api/resource.svelte';
  import type { Model, ModelEval } from '$lib/api/types';
  import ConfusionMatrix from '$lib/charts/ConfusionMatrix.svelte';
  import { sharePct } from '$lib/charts/confusionInk';
  import { classInk } from '$lib/design/classInk.svelte';
  import { matrixCounted, matrixVsPr, type EvalSource } from '$lib/evaluated';
  import { atBestF1, num } from '$lib/format';
  import Button from '$lib/ui/Button.svelte';
  import Panel from '$lib/ui/Panel.svelte';
  import Skeleton from '$lib/ui/Skeleton.svelte';
  import { cmIndex, rankConfusions, type ConfusionEntry } from './confusion';
  import type { ModelSection } from './sections';

  let {
    m,
    ev,
    source,
    phone,
    section = $bindable(),
    evaluation,
  }: {
    m: Model;
    ev: Evaluation | null;
    source: EvalSource;
    /** Phones: the lists lead, the matrix one step away. */
    phone: MediaQuery;
    /** The page's phone section: a ranked line brings Confusion up first. */
    section: ModelSection;
    evaluation: Resource<ModelEval>;
  } = $props();

  // The matrix's own thresholds and what it counted, never precision and recall's operating point.
  const matrixLine = $derived(ev ? matrixCounted(source, ev.images, ev.matrixAt) : '');
  const matrixHow = $derived(ev ? matrixVsPr(ev.matrixAt, atBestF1(ev.bestF1Conf)) : '');
  // The matrix's thresholds in its header, as the curves' IoU is in theirs: the KPIs above are at another operating
  // point (conf at best F1, IoU 0.50), so a panel whose figures depend on a threshold names it where it is titled.
  const matrixAtMeta = $derived(ev?.matrixAt ? `conf ${ev.matrixAt.conf.toFixed(2)} · IoU ${ev.matrixAt.iou.toFixed(2)}` : null);

  // Large class sets (COCO) get a readable matrix of the most frequent classes.
  const CM_MAX = 12;
  const cm = $derived.by(() => {
    if (!m || !ev) return null;
    const n = m.classes.length;
    if (n <= CM_MAX) return { classes: m.classes, ids: undefined, matrix: ev.confusion, partial: false };
    const keep = ev.perClass
      .map((c) => c)
      .sort((a, b) => b.instances - a.instances)
      .slice(0, CM_MAX)
      .map((c) => c.class)
      .sort((a, b) => a - b);
    return {
      classes: [...keep.map((i) => m.classes[i] ?? `class ${i}`), 'Other'],
      ids: [...keep, -1],
      matrix: reduceConfusion(ev.confusion, keep),
      partial: true,
    };
  });

  /**
   * Missed objects: the share of real instances whose only match was background (the matrix's last column), overall
   * and for the class that misses most (rows with very few instances skipped, as above).
   */
  const missed = $derived.by(() => {
    if (!ev) return null;
    const conf = ev.confusion;
    const k = conf.length - 1;
    let total = 0;
    let miss = 0;
    let worst: { r: number; v: number } | null = null;
    for (let r = 0; r < k; r++) {
      const row = conf[r] ?? [];
      const sum = row.reduce((a, b) => a + b, 0);
      const m = row[k] ?? 0;
      total += sum;
      miss += m;
      if (sum >= 20 && m > 0 && (!worst || m / sum > worst.v)) worst = { r, v: m / sum };
    }
    return total ? { v: miss / total, worst } : null;
  });

  // The ranked lists point into the matrix: each line is a button that reveals its cell (on a phone the matrix scrolls
  // sideways, so the background column starts out of view).
  let matrixView = $state<ConfusionMatrix>();
  // What goes wrong most, ranked: beside the matrix in a rail on wider screens; on phones the Confusion section opens on
  // these lists and the full matrix is one step further.
  let cmExplore = $state(false);
  // Two short groups, so misses (usually the larger shares) never crowd out the confusions between classes.
  const RANK_MAX = 5;
  const ranked = $derived.by(() => {
    const all = ev ? rankConfusions(ev.confusion, Infinity) : [];
    return [
      { kind: 'confused', title: 'Most confused', rows: all.filter((e) => e.kind === 'confused').slice(0, RANK_MAX) },
      { kind: 'missed', title: 'Most missed', rows: all.filter((e) => e.kind === 'missed').slice(0, RANK_MAX) },
    ].filter((g) => g.rows.length);
  });
  /** Any confusion or miss at all, counting the classes too small to rank (fewer than 20 objects). */
  const anyErrors = $derived(!!ev && rankConfusions(ev.confusion, 1, 1).length > 0);
  /** A ranked line's cell in the shown matrix (null when its class is folded into Other). */
  function cellOf(e: ConfusionEntry): { r: number; c: number } | null {
    if (!cm) return null;
    const r = cmIndex(e.r, cm.ids);
    const c = e.kind === 'missed' ? cm.classes.length : cmIndex(e.c, cm.ids);
    return r === null || c === null ? null : { r, c };
  }

  /** Points the matrix at a cell, first bringing the Confusion section and its matrix up on a phone. */
  async function reveal(at: { r: number; c: number } | null) {
    if (phone.current && (section !== 'confusion' || !cmExplore)) {
      section = 'confusion';
      cmExplore = true;
      await tick();
    }
    // The pressed line is gone with the list: focus lands on the cell, or (a class folded into Other) on the way back.
    if (at) matrixView?.reveal(at.r, at.c);
    else focusBack();
  }
  /** Phones: the whole matrix, focus on its way back to the list (the pressed button is gone with the list). */
  async function openMatrix() {
    cmExplore = true;
    await tick();
    focusBack();
  }
  const focusBack = () => document.querySelector<HTMLElement>('#cm .to-list')?.focus({ preventScroll: true });
  /** Back from the matrix to the ranked list, keeping the reader's place at the section's top. */
  async function closeMatrix() {
    cmExplore = false;
    await tick();
    document.querySelector<HTMLElement>('#cm .explore')?.focus({ preventScroll: true });
  }

</script>

<!-- Phones, exploring the matrix: the way back is in the panel's heading row, so the only control row above the
     cells is the matrix's own (overview and pages). -->
{#snippet cmBack()}<button type="button" class="to-list" onclick={closeMatrix}><ArrowLeft size={14} aria-hidden="true" />Back to lists</button>{/snippet}
<Panel
  title={phone.current && !cmExplore ? 'Confusions and misses' : 'Confusion matrix'}
  meta={phone.current && !cmExplore
    ? 'largest share of a class first'
    : [
        cm?.partial ? (phone.current ? `top ${CM_MAX} + Other` : `${CM_MAX} most common classes + Other + background`) : phone.current ? null : 'share of each true class',
        matrixAtMeta,
      ]
        .filter(Boolean)
        .join(' · ') || undefined}
  actions={phone.current && cmExplore && cm ? cmBack : undefined}
  id="cm"
>
  {#if cm && phone.current && !cmExplore}
    <!-- Phones: what goes wrong most, as a list (share of the true class and the counts behind it); each line
         opens the matrix at its cell, and the whole matrix is one explicit step away. -->
    {#each ranked as g (g.kind)}
      <h3 class="rank-h" id="cm-rank-{g.kind}">{g.title}</h3>
      <ol class="rank" aria-labelledby="cm-rank-{g.kind}">
        {#each g.rows as e (`${e.r}-${e.c}`)}
          <li>
            <button type="button" class="rk" onclick={() => reveal(cellOf(e))}>
              <span class="rk-what">
                <span class="sw" style:background={classInk(e.r)} aria-hidden="true"></span>
                {#if e.kind === 'missed'}<b>{m.classes[e.r]}</b> missed{:else}<b>{m.classes[e.r]}</b> → <b>{m.classes[e.c]}</b>{/if}
              </span>
              <span class="rk-pct num">{sharePct(e.v)}%</span>
              <span class="rk-sub num">
                {num(e.count)} of {num(e.total)}
                {e.kind === 'missed' ? 'objects found no box' : `objects predicted as ${m.classes[e.c]}`}
              </span>
            </button>
          </li>
        {/each}
      </ol>
    {:else}
      <p class="none">
        {anyErrors
          ? 'No class has the 20 objects it takes to rank its confusions here; the matrix shows every cell.'
          : 'No class is confused with another or missed.'}
      </p>
    {/each}
    <div class="rank-foot">
      <p>{#if missed}<span class="num">{sharePct(missed.v) || '0'}%</span> of all objects missed overall.{' '}{/if}{matrixLine}</p>
      <Button class="explore" onclick={openMatrix}>Explore the matrix</Button>
    </div>
  {:else if cm}
    <!-- One layout whatever the class count (DESIGN_DIRECTION, Model detail): from the laptop tier the matrix on
         the left 8 columns (cells up to 48 px) and a sticky rail on the right 4, what goes wrong most as two
         ranked lists stacked (each line shows its cell), then how to read the matrix; under 5 classes, 6 + 6.
         Tablets: the matrix, the two lists side by side under it. Phones lead with the lists; the matrix is a
         step away. -->
    <div class="cm-layout" class:few={cm.classes.length < 5}>
      <div class="cm-main"><ConfusionMatrix bind:this={matrixView} classes={cm.classes} classIds={cm.ids} matrix={cm.matrix} label="Confusion matrix of {m.name}" minCell={32} maxCell={48} scale /></div>
      <div class="cm-rail">
        <div class="rail-in">
          {#if !phone.current}
            {#each ranked as g (g.kind)}
              <section class="rl" aria-labelledby="cm-rl-{g.kind}">
                <h3 class="rl-h" id="cm-rl-{g.kind}">{g.title}</h3>
                <ol>
                  {#each g.rows as e (`${e.r}-${e.c}`)}
                    {@const at = cellOf(e)}
                    <li>
                      <button
                        type="button"
                        class="rl-row"
                        disabled={!at}
                        title={at ? 'Show in the matrix' : undefined}
                        aria-label="{e.kind === 'missed' ? `${m.classes[e.r]} missed` : `${m.classes[e.r]} taken for ${m.classes[e.c]}`}: {num(e.count)} of {num(e.total)}, {sharePct(e.v)}%{at ? '. Show in the matrix' : ''}"
                        onclick={() => reveal(at)}
                      >
                        <span class="rl-what">
                          <span class="sw" style:background={classInk(e.r)} aria-hidden="true"></span>
                          {#if e.kind === 'missed'}<b class="truncate">{m.classes[e.r]}</b>{:else}<span class="truncate"><b>{m.classes[e.r]}</b> → <b>{m.classes[e.c]}</b></span>{/if}
                        </span>
                        <span class="rl-n num">{num(e.count)} of {num(e.total)}</span>
                        <span class="rl-pct num">{sharePct(e.v)}%</span>
                      </button>
                    </li>
                  {/each}
                </ol>
              </section>
            {/each}
          {/if}
          <div class="cm-notes">
            {#if missed && !phone.current}<p><span class="num strong">{sharePct(missed.v) || '0'}%</span> of all objects missed overall.</p>{/if}
            <p>{matrixLine} Each row is one true class, split by what the model predicted for its objects. The diagonal
              is right; the background column is objects it missed.</p>
            <details class="how">
              <summary><ChevronDown size={14} aria-hidden="true" />How the shares are computed</summary>
              <div class="how-body">
                <p>{matrixHow}</p>
                <p>Each row sums to 100% of that class's ground-truth boxes, in whole percentages ("&lt;1"
                  under half a percent). A cell too small for its number shows a dot instead; hover, tap or focus any cell
                  for its count. An empty cell had no objects at all.</p>
                <p>The background row splits the false detections (boxes on no object) by the class they were given.</p>
                {#if cm.partial}
                  <p>The {CM_MAX} classes with the most ground-truth boxes are shown by name; the other {m.classes.length - CM_MAX} are folded into Other.</p>
                {/if}
              </div>
            </details>
          </div>
        </div>
      </div>
    </div>
  {:else if evaluation.error}
    <p class="none">No evaluation available.</p>
  {:else}
    <Skeleton h="300px" />
  {/if}
</Panel>

<style>
  /* Phones' confusion list: two-line 56 px rows (the pair and its share, then the counts behind it). */
  .rank-h {
    display: flex;
    align-items: center;
    min-height: 32px;
    margin: 0 calc(var(--space-4) * -1);
    padding: 0 var(--space-4);
    border-top: 1px solid var(--line-subtle);
    background: var(--bg-raised);
    font-size: var(--type-sm-size);
    font-weight: var(--weight-medium);
    color: var(--text-secondary);
  }
  .rank {
    margin: 0 calc(var(--space-4) * -1);
    border-top: 1px solid var(--line-subtle);
  }
  .rank li + li {
    border-top: 1px solid var(--line-subtle);
  }
  .rk {
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto;
    align-items: baseline;
    gap: 2px var(--space-3);
    width: 100%;
    min-height: 56px;
    padding: var(--space-2) var(--space-4);
    text-align: left;
  }
  .rk:hover {
    background: var(--bg-hover);
  }
  .rk:focus-visible {
    outline-offset: -2px;
  }
  .rk-what {
    display: inline-flex;
    align-items: center;
    gap: var(--space-2);
    min-width: 0;
    color: var(--text-secondary);
  }
  .rk-what b {
    color: var(--text-primary);
    font-weight: var(--weight-medium);
  }
  .rk-pct {
    font-weight: var(--weight-medium);
  }
  .rk-sub {
    grid-column: 1 / -1;
    padding-left: calc(8px + var(--space-2));
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
  }
  .rank-foot {
    display: grid;
    justify-items: start;
    gap: var(--space-3);
    padding-top: var(--space-3);
    border-top: 1px solid var(--line-subtle);
    margin: 0 calc(var(--space-4) * -1);
    padding-inline: var(--space-4);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
  }
  .to-list {
    display: inline-flex;
    align-items: center;
    gap: var(--space-1);
    min-height: 44px;
    /* A 44 px target in the 40 px heading row: it overhangs the row instead of deepening it. */
    margin: -6px calc(var(--space-1) * -1) -6px 0;
    padding: 0 var(--space-1);
    font-size: var(--type-sm-size);
    font-weight: var(--weight-medium);
    color: var(--text-secondary);
  }
  .to-list:hover {
    color: var(--text-primary);
  }
  /* Under the laptop tier the matrix takes the width its cells want (up to 48 px each) and the rail wraps under it at
     full width, its two lists side by side where 560 px allows. */
  .cm-layout {
    display: flex;
    flex-wrap: wrap;
    gap: var(--space-4) var(--space-8);
    align-items: flex-start;
  }
  .cm-main {
    flex: 0 1 auto;
    min-width: 0;
    max-width: 100%;
  }
  .cm-rail {
    flex: 1 1 280px;

    position: sticky;
    top: calc(var(--size-topbar) + var(--space-4));
    min-width: 0;
    container: cm-rail / inline-size;
  }
  .rail-in {
    display: grid;
    gap: var(--space-5);
  }
  /* Room for two lists (a wide rail beside a small matrix, or the rail under the matrix): side by side, the notes
     under both. */
  @container cm-rail (min-width: 560px) {
    .rail-in {
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: var(--space-4) var(--space-8);
    }
    .rail-in .cm-notes {
      grid-column: 1 / -1;
    }
  }
  /* From the laptop tier, one layout for 3 classes or 80 (round 26: a 3-class matrix beside lists spread as three
     columns over 1160 px, an 8-class one beside them stacked): the matrix on 8 of the page grid's 12 columns, the rail
     on 4, its lists stacked and the notes under them; under 5 classes the small matrix and the rail share 6 + 6. The
     rail sticks under the top bar while a tall matrix scrolls past. */
  @media (min-width: 1024px) {
    .cm-layout {
      display: grid;
      grid-template-columns: repeat(12, minmax(0, 1fr));
      gap: var(--space-4);
    }
    .cm-main {
      grid-column: span 8;
    }
    .cm-rail {
      grid-column: span 4;
    }
    .cm-layout.few .cm-main,
    .cm-layout.few .cm-rail {
      grid-column: span 6;
    }
    .cm-layout .rail-in {
      grid-template-columns: minmax(0, 1fr);
      gap: var(--space-5);
    }
    .cm-layout .rail-in .cm-notes {
      grid-column: auto;
    }
  }
  /* A ranked list: 32 px rows (the pair, its counts, its share); each line shows its cell in the matrix. */
  .rl-h {
    display: flex;
    align-items: center;
    height: 28px;
    font-size: var(--type-sm-size);
    font-weight: var(--weight-medium);
    color: var(--text-secondary);
    border-bottom: 1px solid var(--line-subtle);
  }
  .rl li + li {
    border-top: 1px solid var(--line-subtle);
  }
  .rl-row {
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto 40px;
    align-items: center;
    gap: var(--space-3);
    width: calc(100% + var(--space-4));
    height: 32px;
    margin: 0 calc(var(--space-2) * -1);
    padding: 0 var(--space-2);
    border-radius: var(--radius-sm);
    font-size: var(--type-sm-size);
    text-align: left;
    color: var(--text-secondary);
  }
  .rl-row:hover:not(:disabled) {
    background: var(--bg-hover);
    color: var(--text-primary);
  }
  .rl-row:focus-visible {
    outline-offset: -2px;
  }
  .rl-row:disabled {
    cursor: default;
  }
  .rl-what {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    min-width: 0;
  }
  .rl-what b {
    color: var(--text-primary);
    font-weight: var(--weight-medium);
  }
  .rl-n {
    color: var(--text-tertiary);
  }
  .rl-pct {
    text-align: right;
    color: var(--text-primary);
    font-weight: var(--weight-medium);
  }
  @media (pointer: coarse) {
    .rl-row {
      height: 44px;
    }
  }
  .cm-notes {
    display: grid;
    gap: var(--space-2);
    max-inline-size: var(--measure);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
  }
  .cm-notes .strong {
    color: var(--text-primary);
    font-weight: var(--weight-medium);
  }
  /* Secondary explanation: how the figures are made, folded until asked for. */
  .how summary {
    display: inline-flex;
    align-items: center;
    gap: var(--space-1);
    min-height: 24px;
    list-style: none;
    cursor: pointer;
    color: var(--text-tertiary);
    border-radius: var(--radius-sm);
  }
  .how summary::-webkit-details-marker {
    display: none;
  }
  .how summary:hover {
    color: var(--text-primary);
  }
  .how summary :global(svg) {
    transform: rotate(-90deg);
    transition: transform var(--dur-fast) var(--ease-standard);
  }
  .how[open] summary :global(svg) {
    transform: none;
  }
  .how-body {
    display: grid;
    gap: var(--space-2);
    padding-top: var(--space-1);
    color: var(--text-tertiary);
  }
  @media (pointer: coarse) {
    .how summary {
      min-height: 44px;
    }
  }
  .none {
    color: var(--text-tertiary);
  }
  .sw {
    width: 8px;
    height: 8px;
    border-radius: 2px;
    flex-shrink: 0;
  }
</style>
