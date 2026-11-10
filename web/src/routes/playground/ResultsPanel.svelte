<script lang="ts">
  /** The detections kept by the thresholds, as a table (each row a way to inspect its box) or the prediction as JSON. */
  import type { Prediction } from '$lib/api/types';
  import { classColor } from '$lib/design/tokens';
  import Panel from '$lib/ui/Panel.svelte';
  import Tabs from '$lib/ui/Tabs.svelte';
  import JsonView from './JsonView.svelte';
  import { FETCH, px, score } from './predict';
  import Swatch from './Swatch.svelte';

  type Det = Prediction['boxes'][number];

  let {
    result,
    tab = $bindable(),
    rows,
    predicting,
    notice,
    json,
    pred,
    conf = $bindable(),
    hidden,
    tbody = $bindable(),
    tabRow,
    activeIdx,
    selectedIdx,
    hover = $bindable(),
    picked = $bindable(),
    onrowkey,
  }: {
    result: { boxes: Det[]; ms: number } | null;
    tab: 'detections' | 'json';
    rows: { b: Det; i: number }[];
    predicting: boolean;
    notice: 'file' | 'error' | null;
    json: unknown;
    pred: Prediction | null;
    conf: number;
    hidden: Set<number>;
    tbody: HTMLTableSectionElement | undefined;
    tabRow: number | null;
    activeIdx: number | null;
    selectedIdx: number | null;
    hover: Det | null;
    picked: Det | null;
    onrowkey: (e: KeyboardEvent, k: number) => void;
  } = $props();
</script>

<div class="results">
  <Panel pad="none">
    <div class="tabs-row">
      <Tabs
        label="Result views"
        idPrefix="pg"
        bind:value={tab}
        tabs={[
          { value: 'detections', label: 'Detections', meta: result ? String(rows.length) : undefined },
          { value: 'json', label: 'JSON' },
        ]}
      />
    </div>
    <div id="pg-panel" role="tabpanel" aria-labelledby="pg-{tab}">
      {#if !result}
        <p class="placeholder">
          {predicting ? 'Running inference…' : notice === 'error' ? 'No detections: the last prediction failed.' : 'Detections appear here once you run the model on an image.'}
        </p>
      {:else if tab === 'json'}
        <div class="json-wrap"><JsonView value={json} label="Prediction as JSON" /></div>
      {:else if !pred?.boxes.length}
        <!-- Candidates are fetched down to the lowest threshold (FETCH): none means no threshold would show more. -->
        <p class="placeholder">The model found nothing in this image, even at conf ≥ {FETCH.conf}.</p>
      {:else if rows.length === 0}
        <p class="placeholder">
          No detections at conf ≥ {conf.toFixed(2)}{hidden.size ? ' in the visible classes' : ''}.
          {#if conf > 0.05}<button type="button" class="inline" onclick={() => (conf = Math.max(0.01, +(conf / 2).toFixed(2)))}>Lower the threshold</button>{/if}
        </p>
      {:else}
        <div class="scroll">
          <table class="table dense">
            <thead>
              <tr>
                <th scope="col" class="r idx">#</th>
                <th scope="col">Class</th>
                <th scope="col">Score</th>
                <th scope="col" class="r hide-sm coord" title="Box left edge, in image pixels">x1</th>
                <th scope="col" class="r hide-sm coord" title="Box top edge, in image pixels">y1</th>
                <th scope="col" class="r hide-sm coord" title="Box right edge, in image pixels">x2</th>
                <th scope="col" class="r hide-sm coord" title="Box bottom edge, in image pixels">y2</th>
                <th scope="col" class="r">Size (px)</th>
              </tr>
            </thead>
            <tbody bind:this={tbody}>
              {#each rows as { b, i }, k (i)}
                <tr
                  data-det={i}
                  tabindex={i === tabRow ? 0 : -1}
                  class:hot={activeIdx === i}
                  class:on={selectedIdx === i}
                  aria-selected={selectedIdx === i ? 'true' : undefined}
                  aria-label="{b.name}, score {score(b.score)}, {px(b.x2 - b.x1)} by {px(b.y2 - b.y1)} pixels"
                  onpointerenter={(e) => {
                    // A tap is not a hover: on touch the selection alone marks the row.
                    if (e.pointerType !== 'touch') hover = b;
                  }}
                  onpointerleave={() => (hover = null)}
                  onfocus={() => (picked = b)}
                  onclick={() => (picked = b)}
                  onkeydown={(e) => onrowkey(e, k)}
                >
                  <td class="r idx faint">{i + 1}</td>
                  <td class="lead">
                    <span class="cls">
                      <Swatch color={classColor(b.class)} />
                      <span class="truncate">{b.name}</span>
                    </span>
                  </td>
                  <td>
                    <span class="score">
                      <span class="s-track" aria-hidden="true"><span style:transform="scaleX({b.score})"></span></span>
                      <span class="num">{score(b.score)}</span>
                    </span>
                  </td>
                  <td class="r muted hide-sm coord">{px(b.x1)}</td>
                  <td class="r muted hide-sm coord">{px(b.y1)}</td>
                  <td class="r muted hide-sm coord">{px(b.x2)}</td>
                  <td class="r muted hide-sm coord">{px(b.y2)}</td>
                  <td class="r muted">{px(b.x2 - b.x1)}×{px(b.y2 - b.y1)}</td>
                </tr>
              {/each}
            </tbody>
          </table>
        </div>
      {/if}
    </div>
  </Panel>
</div>

<style>
  .tabs-row {
    padding: 0 var(--space-4);
  }
  /* A state in the frame (DESIGN_DIRECTION 8): one line with the panel's 16 px padding, on its left edge, never a
     centred block held open. */
  .placeholder {
    padding: var(--space-4);
    color: var(--text-tertiary);
  }

  /* An inline action is a link: text.primary, underlined on hover, never lime (DESIGN_DIRECTION §2.6). */
  .inline {
    color: var(--text-primary);
    font-weight: var(--weight-medium);
  }
  .inline:hover {
    text-decoration: underline;
    text-underline-offset: 3px;
  }
  .json-wrap {
    padding: var(--space-3);
  }
  .scroll {
    overflow: auto;
    max-height: 420px;
  }
  .scroll thead th {
    position: sticky;
    top: 0;
    background: var(--bg-panel);
    z-index: 1;
  }
  .idx {
    width: 40px;
  }
  /* x1 y1 x2 y2 read as one group: tighter gaps between the four coordinate columns. */
  .table .coord + .coord {
    padding-left: var(--space-2);
  }
  .table .coord {
    font-variant-numeric: tabular-nums;
  }
  tbody tr {
    transition: background-color var(--dur-fast) var(--ease-standard);
  }
  tbody tr {
    cursor: pointer;
  }
  /* Hover (the pointer, or the box under it on the image) is a light step, only where a pointer can hover, and never
     on the selected row, which keeps the selection's fill (base.css .on): one highlight per meaning. */
  @media (hover: hover) {
    tbody tr:hover:not(.on),
    tbody tr.hot:not(.on) {
      background: var(--bg-hover);
    }
  }
  .cls {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    max-width: 180px;
  }
  .score {
    display: inline-flex;
    align-items: center;
    gap: var(--space-2);
  }
  .s-track {
    position: relative;
    width: 72px;
    height: 4px;
    border-radius: var(--radius-full);
    background: var(--bg-active);
    overflow: hidden;
  }
  .s-track span {
    position: absolute;
    inset: 0;
    transform-origin: left;
    border-radius: inherit;
  }
  /* The table's one inline bar is neutral ink (the hue is on the swatch); the hovered or selected row's in primary. */
  .s-track span {
    background: var(--text-tertiary);
    transition: background-color var(--dur-fast) var(--ease-standard);
  }
  tbody tr:hover .s-track span,
  tbody tr.hot .s-track span,
  tbody tr.on .s-track span {
    background: var(--text-primary);
  }
  @media (pointer: coarse) {
    .table.dense td {
      height: 44px;
    }
  }
  @media (max-width: 1023px) {
    .results {
      order: 3;
    }
  }
</style>
