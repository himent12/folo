<script lang="ts">
  /**
   * A benchmark's conditions (its plan): the catalog's conditions by group, each on or off, at which severities and how
   * much it counts, with its reason when the plan has one; who designed the plan and why; and a preview of any
   * condition on the benchmark's own images. The plan can change only while the benchmark has no entries: then it is
   * edited here (saved with Save plan; the page holds the edits and asks before they are lost) or designed by an agent;
   * after, it reads as a fixed test, with the one-line reason why.
   */
  import { untrack } from 'svelte';
  import Eye from '@lucide/svelte/icons/eye';
  import Lock from '@lucide/svelte/icons/lock';
  import Sparkles from '@lucide/svelte/icons/sparkles';
  import type { Bench, BenchCatalog, BenchCondition, BenchPlan } from '$lib/api/types';
  import { planCondition, planError, planPasses, setWeight, toggleCondition, toggleSeverity, weightError } from '$lib/bench/plan';
  import { GROUP_NAME, GROUPS, SEVERITIES } from '$lib/bench/rules';
  import { AGENT_NAME } from '$lib/label';
  import Button from '$lib/ui/Button.svelte';
  import Gate from '$lib/ui/Gate.svelte';
  import Panel from '$lib/ui/Panel.svelte';
  import Switch from '$lib/ui/Switch.svelte';
  import { tooltip } from '$lib/ui/tooltip';
  import Callout from '../run/Callout.svelte';
  import Preview from './Preview.svelte';
  import { FROZEN } from './words';

  let {
    bench,
    editable,
    catalog,
    draft = $bindable(),
    dirty,
    saving,
    onsave,
    ondiscard,
    ondesign,
    designWhy,
  }: {
    bench: Bench;
    /** The plan can still change: no entry finished or is under way ($lib/bench/rules planEditable). */
    editable: boolean;
    catalog: BenchCatalog;
    /** The plan as edited (the page holds it, saved or not). */
    draft: BenchPlan;
    dirty: boolean;
    saving: boolean;
    onsave: () => void;
    ondiscard: () => void;
    /** Opens "Design with an agent". */
    ondesign: () => void;
    /** Why designing cannot start now, or null. */
    designWhy: string | null;
  } = $props();

  /** The plan shown: the edits while it can change, else the benchmark's own. */
  const plan = $derived(editable ? draft : bench.plan);
  /** The condition the preview shows: the plan's first at the start, then the one last looked at. */
  const first = () => bench.plan.conditions[0]?.condition ?? catalog.conditions[0]?.id ?? 'haze';
  let preview = $state(first());
  let severity = $state(3);
  const error = $derived(editable ? planError(draft, catalog.conditions) : null);
  /** Weights as typed, per condition, so a half-typed number is not thrown away. */
  let typed = $state<Record<string, string>>({});
  $effect(() => {
    void bench.id;
    typed = {};
  });
  // A draft changed from outside the field (Discard, a condition off and on, a save) drops what was typed for a weight
  // it no longer has: the field shows the plan's.
  $effect(() => {
    const d = draft;
    untrack(() => {
      const kept = Object.entries(typed).filter(([id, t]) => Number(t) === planCondition(d, id)?.weight);
      if (kept.length !== Object.keys(typed).length) typed = Object.fromEntries(kept);
    });
  });

  const groups = $derived(GROUPS.map((g) => ({ g, conditions: catalog.conditions.filter((c) => c.group === g) })).filter((x) => x.conditions.length));
  const designer = $derived(plan.designed_by ? `${AGENT_NAME[plan.designed_by.agent]}${plan.designed_by.model ? ` (${plan.designed_by.model[0]!.toUpperCase()}${plan.designed_by.model.slice(1)})` : ''}` : null);
  const isDefault = $derived(!plan.designed_by && plan.conditions.length === catalog.default_plan.conditions.length && plan.conditions.every((c) => c.weight === 1 && c.severities.join() === '1,3,5'));

  function setTyped(id: string, v: string) {
    typed = { ...typed, [id]: v };
    const w = Number(v);
    if (v.trim() && Number.isFinite(w)) draft = setWeight(draft, id, w);
  }
  function show(id: string) {
    preview = id;
    const c = planCondition(plan, id);
    if (c && !c.severities.includes(severity)) severity = c.severities.at(-1) ?? 3;
  }
</script>

{#snippet row(c: BenchCondition)}
  {@const pc = planCondition(plan, c.id)}
  {@const on = !!pc}
  {@const wErr = pc && editable ? weightError(pc.weight) : null}
  <li class="cond" class:off={!on} class:shown={preview === c.id}>
    <div class="c-main">
      {#if editable}
        <Switch label={c.name} hint={c.description} checked={on} onchange={(v) => (draft = toggleCondition(draft, c.id, v, catalog, bench.plan))} />
      {:else}
        <span class="c-name">{c.name}</span>
        <span class="c-desc">{c.description}</span>
      {/if}
    </div>
    {#if pc}
      <div class="c-sev" role="group" aria-label="Severities of {c.name}">
        {#each SEVERITIES as s (s)}
          {@const has = pc.severities.includes(s)}
          {#if editable}
            <button
              type="button"
              class="sev"
              class:has
              aria-pressed={has}
              aria-label="Severity {s}"
              disabled={has && pc.severities.length === 1}
              use:tooltip={has && pc.severities.length === 1 ? 'A condition runs at one severity at least: turn it off instead' : null}
              onclick={() => (draft = toggleSeverity(draft, c.id, s))}>{s}</button
            >
          {:else}
            <span class="sev static" class:has aria-label={has ? `Severity ${s}: runs` : `Severity ${s}: not run`}>{s}</span>
          {/if}
        {/each}
      </div>
      <div class="c-weight">
        {#if editable}
          <label class="w">
            <span class="w-x" aria-hidden="true">×</span>
            <input
              type="number"
              min="0.5"
              max="10"
              step="0.5"
              inputmode="decimal"
              aria-label="Weight of {c.name}"
              aria-invalid={wErr ? 'true' : undefined}
              value={typed[c.id] ?? String(pc.weight)}
              oninput={(e) => setTyped(c.id, (e.currentTarget as HTMLInputElement).value)}
            />
          </label>
        {:else}
          <span class="w-static num" aria-label="Weight {pc.weight}">× {pc.weight}</span>
        {/if}
      </div>
    {:else}
      <div class="c-not">Not in the plan</div>
    {/if}
    <div class="c-eye">
      <Button variant="ghost" size="sm" icon={Eye} aria-label="Preview {c.name}" aria-pressed={preview === c.id} onclick={() => show(c.id)} />
    </div>
    {#if pc?.why}<span class="c-why"><Sparkles size={12} strokeWidth={1.75} aria-hidden="true" /><span>{pc.why}</span></span>{/if}
    {#if wErr}<p class="c-err" role="alert">{c.name}: {wErr}.</p>{/if}
  </li>
{/snippet}

<div class="tab">
  {#if !editable}
    <div class="frozen">
      <Callout tone="neutral" icon={Lock} title={FROZEN} role="note">To change it, delete those entries (failed and stopped ones do not count) or make a new benchmark with the plan you want.</Callout>
    </div>
  {/if}
  <div class="grid">
    <div class="plan">
      <Panel title="Plan" meta="{plan.conditions.length} of {catalog.conditions.length} conditions · {planPasses(plan)} severities" pad="none">
        {#snippet actions()}
          {#if editable}
            {#if dirty}
              <span class="unsaved" data-gate-why>Unsaved changes</span>
              <Button variant="ghost" size="sm" onclick={ondiscard}>Discard</Button>
              <Gate reason={error ? 'Fix the plan first' : null}>
                {#snippet children(why)}
                  <Button variant="primary" size="sm" loading={saving} disabled={!!why} aria-describedby={why} onclick={onsave}>Save plan</Button>
                {/snippet}
              </Gate>
            {:else}
              <Gate reason={designWhy}>
                {#snippet children(why)}
                  <Button size="sm" icon={Sparkles} disabled={!!why} aria-describedby={why} onclick={ondesign}>Design with an agent</Button>
                {/snippet}
              </Gate>
            {/if}
          {/if}
        {/snippet}
        <div class="about">
          {#if designer}
            <p class="by"><Sparkles size={14} strokeWidth={1.75} aria-hidden="true" /><span>Designed by {designer}{dirty ? ' · not saved yet' : ''}</span></p>
            {#if plan.rationale}<p class="rationale">{plan.rationale}</p>{/if}
          {:else if isDefault}
            <p class="rationale">The default plan: every condition at severities 1, 3 and 5, each counting the same. Weight what your camera meets, or let an agent design it from the description.</p>
          {:else if !plan.conditions.length}
            <p class="rationale">No conditions: entries run the unchanged images only, and the score has no robustness axis.</p>
          {:else}
            <p class="rationale">Written by hand.</p>
          {/if}
          {#if error}<p class="plan-err" role="alert">{error}</p>{/if}
        </div>
        {#each groups as { g, conditions } (g)}
          <section class="group" aria-labelledby="grp-{g}">
            <h3 class="group-label" id="grp-{g}">{GROUP_NAME[g]}</h3>
            <ul>{#each conditions as c (c.id)}{@render row(c)}{/each}</ul>
          </section>
        {/each}
      </Panel>
    </div>
    <aside class="rail" aria-label="Preview">
      <Preview {bench} {catalog} bind:condition={preview} bind:severity />
    </aside>
  </div>
</div>

<style>
  .tab {
    display: grid;
    gap: var(--space-4);
  }
  /* The plan on 8 columns, the preview a sticky rail on 4 (DESIGN_DIRECTION §2.2). */
  .grid {
    display: grid;
    grid-template-columns: minmax(0, 2fr) minmax(0, 1fr);
    gap: var(--space-4);
    align-items: start;
  }
  .rail {
    position: sticky;
    top: calc(var(--size-topbar) + var(--space-4));
    min-width: 0;
  }
  .about {
    display: grid;
    gap: var(--space-2);
    padding: 0 var(--space-4) var(--space-4);
    border-bottom: 1px solid var(--line-subtle);
  }
  .by {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    font-weight: var(--weight-medium);
  }
  .by :global(svg) {
    color: var(--text-tertiary);
  }
  .rationale {
    color: var(--text-secondary);
    max-inline-size: none;
  }
  .plan-err,
  .c-err {
    font-size: var(--type-sm-size);
    color: var(--danger-text);
  }
  .unsaved {
    font-size: var(--type-sm-size);
    color: var(--text-secondary);
    margin-right: var(--space-1);
  }
  .group + .group {
    border-top: 1px solid var(--line-subtle);
  }
  .group .group-label {
    padding: var(--space-3) var(--space-4) var(--space-1);
  }
  /* A row: the condition (its switch at the end of the text while the plan can change), its severities, its weight,
     the preview button. */
  .cond {
    display: grid;
    grid-template-columns: minmax(0, 1fr) 136px 72px auto;
    align-items: center;
    column-gap: var(--space-4);
    padding: var(--space-3) var(--space-3) var(--space-3) var(--space-4);
  }
  .cond.shown {
    background: var(--bg-active);
    box-shadow: var(--select-bar);
  }
  .c-main {
    display: grid;
    gap: 2px;
    min-width: 0;
  }
  .c-main :global(.switch .label) {
    font-weight: var(--weight-medium);
  }
  .cond.off .c-main :global(.switch .label) {
    color: var(--text-secondary);
  }
  .c-main :global(.switch .hint) {
    max-inline-size: none;
  }
  .c-name {
    font-weight: var(--weight-medium);
  }
  .cond.off .c-name {
    color: var(--text-secondary);
  }
  .c-desc,
  .c-why {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-tertiary);
  }
  /* The agent's reason is a line of its own under the condition, so the switch, the severities and the weight share
     the row's first line. */
  .c-why {
    grid-column: 1;
    display: grid;
    grid-template-columns: 12px minmax(0, 1fr);
    gap: 6px;
    margin-top: var(--space-2);
    color: var(--text-secondary);
  }
  .c-why :global(svg) {
    margin-top: 2px;
    color: var(--text-tertiary);
  }
  .c-sev {
    display: flex;
    gap: var(--space-1);
  }
  .c-not {
    grid-column: span 2;
    justify-self: start;
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
  }
  .sev {
    position: relative;
    display: grid;
    place-items: center;
    width: 24px;
    height: 24px;
    border-radius: var(--radius-sm);
    border: 1px solid var(--line-control);
    background: none;
    color: var(--text-tertiary);
    font-size: var(--type-sm-size);
    font-variant-numeric: tabular-nums;
  }
  /* On: an ink edge on the active fill, the number in ink at 500 (a selection is ink, never lime); off, a faint
     outline and a tertiary number, so the five read as one set with the run ones standing out. */
  .sev.has {
    border-color: var(--text-primary);
    background: var(--bg-active);
    color: var(--text-primary);
    font-weight: var(--weight-medium);
  }
  @media (hover: hover) {
    button.sev:hover:not(:disabled):not(.has) {
      border-color: var(--line-control-hover);
      color: var(--text-primary);
    }
  }
  button.sev:disabled {
    cursor: not-allowed;
  }
  /* Touch: 40 px squares, each a 44 px target (the gap between them is its margin). */
  @media (pointer: coarse) {
    .sev {
      width: 40px;
      height: 40px;
    }
    button.sev::after {
      content: '';
      position: absolute;
      inset: -2px;
    }
  }
  /* Read-only: a run severity is a filled mark, one that does not run its bare number, faint. */
  .sev.static {
    border-color: transparent;
    color: var(--text-tertiary);
  }
  .sev.static.has {
    border-color: transparent;
    background: var(--bg-active);
    color: var(--text-primary);
    font-weight: var(--weight-medium);
  }
  .c-weight {
    justify-self: end;
  }
  .w {
    display: flex;
    align-items: center;
    gap: 2px;
    height: var(--control-md);
    padding: 0 var(--space-2);
    border-radius: var(--radius-md);
    border: 1px solid var(--line-control);
    background: var(--bg-panel);
  }
  .w:has(input:focus-visible) {
    outline: var(--size-focus-ring) solid var(--focus);
    outline-offset: var(--size-focus-offset);
  }
  .w:has(input[aria-invalid='true']) {
    border-color: var(--danger-solid);
  }
  .w-x {
    color: var(--text-tertiary);
  }
  .w input {
    width: 40px;
    border: 0;
    background: none;
    color: var(--text-primary);
    font: inherit;
    font-variant-numeric: tabular-nums;
    text-align: right;
    outline: none;
  }
  @media (pointer: coarse) {
    .w input {
      font-size: var(--type-lg-size);
    }
  }
  .w-static {
    color: var(--text-secondary);
  }
  .c-err {
    grid-column: 1 / -1;
    margin-top: var(--space-1);
  }
  @media (max-width: 1023px) {
    .grid {
      grid-template-columns: minmax(0, 1fr);
    }
    .rail {
      position: static;
      order: -1;
    }
    /* Twelve rows under a preview: denser, each condition's sentence on demand (its eye shows it in the preview, where
       the condition is seen), the row being previewed keeping it. */
    .cond {
      padding-top: var(--space-2);
      padding-bottom: var(--space-2);
    }
    .cond:not(.shown) .c-main :global(.switch .hint),
    .cond:not(.shown) .c-desc {
      display: none;
    }
  }
  /* Phones: the condition (and its switch) over its severities and weight, the preview button beside it. */
  @media (max-width: 767px) {
    .cond {
      grid-template-columns: minmax(0, 1fr) auto auto;
      row-gap: var(--space-2);
      padding-right: var(--space-2);
    }
    .c-main {
      grid-row: 1;
      grid-column: 1 / span 2;
    }
    .c-eye {
      grid-row: 1;
      grid-column: 3;
      align-self: start;
    }
    .c-sev {
      grid-row: 2;
      grid-column: 1;
    }
    .c-weight {
      grid-row: 2;
      grid-column: 2;
    }
    .c-not {
      grid-row: 2;
      grid-column: 1 / span 2;
      justify-self: start;
    }
    .c-why {
      grid-row: 3;
      grid-column: 1 / -1;
      margin-top: 0;
    }
  }
</style>
