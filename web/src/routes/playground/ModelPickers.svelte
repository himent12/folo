<script lang="ts">
  /**
   * The model to run (its score and what it is on under the select) and where it runs: on the title row from 1024 px
   * up, or (`toolbar`) a toolbar row of their own under the title below that.
   */
  import ArrowUpRight from '@lucide/svelte/icons/arrow-up-right';
  import type { Model } from '$lib/api/types';
  import type { Backend } from '$lib/inference/browser.svelte';
  import type { ModelChoice } from '$lib/scoreSummary';
  import { href } from '$lib/router.svelte';
  import Button from '$lib/ui/Button.svelte';
  import Segmented from '$lib/ui/Segmented.svelte';
  import Select from '$lib/ui/Select.svelte';

  let {
    toolbar = false,
    ranked,
    modelId,
    choices,
    chooseModel,
    model,
    pickedMeta,
    runOn = $bindable(),
    webgpu,
    browserOff,
    runOnNote,
  }: {
    toolbar?: boolean;
    ranked: Model[];
    modelId: string | null;
    choices: ModelChoice<Model>[];
    chooseModel: (id: string) => void;
    model: Model | null;
    pickedMeta: string | null;
    runOn: 'server' | Backend;
    webgpu: boolean;
    browserOff: string | null;
    runOnNote: string | null;
  } = $props();
</script>

<!-- The model and where it runs: on the title row from 1024 px up, a toolbar row under the title below that. -->
{#snippet pickers()}
  {#if ranked.length}
    <!-- Closed, the select shows the model's name alone; its score and everything the score is on wrap in the meta line
         under it (never cut at the select's edge), which is also its description. The open list keeps the groups and
         each option's context. -->
    <div class="model-pick">
      <span class="pick-label">Model</span>
      <Select
        label="Model"
        value={modelId ?? ''}
        options={choices.map((c) => ({ value: c.model.id, label: c.label, group: c.group, short: c.short }))}
        onchange={chooseModel}
        describedby={pickedMeta ? 'model-meta' : undefined}
      />
      {#if model}<Button icon={ArrowUpRight} href={href(`/models/${model.id}`)} aria-label="Open {model.name}" />{/if}
      {#if pickedMeta}<p class="pick-meta num" id="model-meta">{pickedMeta}</p>{/if}
    </div>
    <div class="runon" role="group" aria-label="Run on" aria-describedby={runOnNote ? 'runon-note' : undefined}>
      <span class="pick-label" aria-hidden="true">Run on</span>
      <!-- A choice that is off says why in its own tip, and the image panel's status line says it once ("No WebGPU in
           this browser"), so the toolbar stays one row at every tier: never a note wrapping under the group (round
           24's tablet). The group's description is the same words, for assistive technology. -->
      <span class="runon-seg">
        <Segmented
          label="Run the model on"
          size="sm"
          bind:value={runOn}
          options={[
            { value: 'server', label: 'Server' },
            { value: 'webgpu', label: 'WebGPU', disabled: !webgpu || !!browserOff, tip: browserOff ?? 'No WebGPU in this browser' },
            { value: 'cpu', label: 'CPU', disabled: !!browserOff, tip: browserOff ?? undefined },
          ]}
        />
        {#if runOnNote}<span class="sr-only" id="runon-note">{runOnNote}</span>{/if}
      </span>
    </div>
  {/if}
{/snippet}

{#if toolbar}<div class="pg-tools">{@render pickers()}</div>{:else}{@render pickers()}{/if}

<style>
  /* Label, select and open button on one row; the score's meta line under the select, wrapping, never cut. */
  .model-pick {
    display: grid;
    grid-template-columns: auto minmax(0, 1fr) auto;
    align-items: center;
    gap: var(--space-1) var(--space-2);
    width: 400px;
  }
  .pick-meta {
    grid-column: 2 / -1;
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-tertiary);
    text-wrap: pretty;
  }
  /* Below 1024 px: one full-width toolbar row under the title; the model select takes what Run on leaves (a choice
     that is off says why in its tip and on the image panel's status line, never in a note that wraps the row). */
  .pg-tools {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: var(--space-2) var(--space-6);
  }
  .pg-tools .model-pick {
    flex: 1 1 320px;
    width: auto;
    min-width: 0;
  }
  .pg-tools .runon {
    flex-wrap: wrap;
    row-gap: var(--space-1);
  }
  .pg-tools .runon-seg {
    display: contents;
  }
  .model-pick :global(.select) {
    min-width: 0;
  }
  .runon {
    display: flex;
    align-items: center;
    gap: var(--space-2);
  }
  /* On the title row, Run on sits on the model select's row, not centred on the select and its meta line together. */
  .model-pick + .runon {
    align-self: flex-start;
    min-height: var(--control-md);
  }
  .pick-label {
    color: var(--text-tertiary);
    font-size: var(--type-sm-size);
  }
  .runon-seg {
    display: flex;
  }
  @media (max-width: 1023px) {
    /* The toolbar row takes the title row's place for the controls; no extra gap under the title. */
    .pg-tools {
      margin-top: calc(-1 * var(--space-1));
    }
  }
  @media (max-width: 767px) {
    /* Phones: two rows, the model, then Run on with its control on the right; the note a row of its own under them. */
    .pg-tools {
      flex-direction: column;
      align-items: stretch;
      gap: var(--space-2);
      margin-top: 0;
    }
    .pg-tools .model-pick {
      flex: none;
    }
    .pg-tools .runon {
      justify-content: space-between;
    }
  }
</style>
