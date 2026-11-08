<script lang="ts">
  /** Per-class validation metrics of the latest epoch, with the change since the previous epoch. */
  import type { Epoch } from '$lib/api/types';
  import { classInk } from '$lib/design/classInk.svelte';
  import Panel from '$lib/ui/Panel.svelte';
  import Skeleton from '$lib/ui/Skeleton.svelte';

  let {
    classes,
    epochs,
    version,
    loading,
    compact = false,
  }: { classes: string[]; epochs: Epoch[]; version: number; loading: boolean; /** Rail width: class, mAP50-95, change, bar. */ compact?: boolean } =
    $props();

  const rows = $derived.by(() => {
    void version;
    const last = epochs[epochs.length - 1];
    const prev = epochs[epochs.length - 2];
    if (!last) return [];
    return last.val.per_class
      .map((c) => ({
        ...c,
        name: classes[c.class] ?? `class ${c.class}`,
        delta: prev ? c.map50_95 - (prev.val.per_class.find((p) => p.class === c.class)?.map50_95 ?? c.map50_95) : 0,
      }))
      .sort((a, b) => b.map50_95 - a.map50_95);
  });
  const epochNo = $derived.by(() => {
    void version;
    return epochs.length ? epochs[epochs.length - 1]!.epoch : null;
  });
</script>

<Panel title="Per class" meta={epochNo ? `epoch ${epochNo}` : undefined} pad="none" id="per-class">
  {#if loading}
    <div class="sk">{#each [0, 1, 2, 3] as i (i)}<Skeleton h="18px" />{/each}</div>
  {:else if rows.length === 0}
    <p class="none">Per-class metrics appear after the first validation pass.</p>
  {:else}
    <div class="scroll">
      <table class="table dense">
        <thead>
          <tr>
            <th>Class</th>
            <th class="r" title="mAP50-95 of the latest epoch; the small number is the change since the previous epoch">mAP50-95 <span class="delta-h">Δ prev</span></th>
            <th class="bar-col"><span class="sr-only">mAP50-95 bar</span></th>
            {#if !compact}
              <th class="r">mAP50</th>
              <th class="r">P</th>
              <th class="r">R</th>
            {/if}
          </tr>
        </thead>
        <tbody>
          {#each rows as c (c.class)}
            <tr>
              <td><span class="cls"><span class="sw" style:background={classInk(c.class)}></span>{c.name}</span></td>
              <td class="r num">
                {c.map50_95.toFixed(3)}
                {#if Math.abs(c.delta) >= 0.0005}
                  <span class="delta" class:up={c.delta > 0} class:down={c.delta < 0} aria-label="{c.delta > 0 ? 'up' : 'down'} {Math.abs(c.delta).toFixed(3)} since the previous epoch"
                    >{c.delta > 0 ? '▲' : '▼'}&#8201;{Math.abs(c.delta).toFixed(3)}</span
                  >
                {:else}<span class="delta">·</span>{/if}
              </td>
              <td class="bar-col"><span class="bar"><span style:width="{c.map50_95 * 100}%" style:background={classInk(c.class)}></span></span></td>
              {#if !compact}
                <td class="r num">{c.map50.toFixed(3)}</td>
                <td class="r num faint">{c.precision.toFixed(2)}</td>
                <td class="r num faint">{c.recall.toFixed(2)}</td>
              {/if}
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
  {/if}
</Panel>

<style>
  .scroll {
    overflow-x: auto;
  }
  .sk {
    display: grid;
    gap: var(--space-3);
    padding: var(--space-4);
  }
  .none {
    padding: var(--space-6) var(--space-4);
    color: var(--text-tertiary);
    text-align: center;
  }
  .cls {
    display: inline-flex;
    align-items: center;
    gap: var(--space-2);
  }
  .sw {
    width: 8px;
    height: 8px;
    border-radius: 2px;
    box-shadow: 0 0 0 1px var(--class-halo);
  }
  .delta {
    display: inline-block;
    width: 52px;
    text-align: left;
    padding-left: 6px;
    font-size: var(--type-sm-size);
    color: var(--text-disabled);
  }
  /* Direction by mark (▲ / ▼) in ink, not by status hue: a metric moving is not a status. */
  .delta.up {
    color: var(--text-secondary);
  }
  .delta.down {
    color: var(--text-tertiary);
  }
  .bar-col {
    width: 30%;
    /* Low enough that the table fits the rail at 960 px (about 278 px wide). */
    min-width: 48px;
  }
  .bar {
    display: block;
    height: 4px;
    border-radius: 2px;
    background: var(--bg-active);
    overflow: hidden;
  }
  .delta-h {
    font-weight: var(--weight-regular);
    color: var(--text-tertiary);
  }
  .bar span {
    display: block;
    height: 100%;
    border-radius: 2px;
    box-shadow: inset 0 0 0 1px var(--chart-swatch-edge);
    transition: width var(--dur-chart) var(--ease-standard);
  }
</style>
