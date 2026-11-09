<script lang="ts">
  /**
   * A run's config. With `resolved` (docs/API.md, `Run.resolved`) the settings it trains with are the values shown,
   * the request a note where it differs ("AdamW (auto)", "0.001429 (requested 0.01)"), plus the batch it accumulates,
   * the LR decay, the warmup and the optimizer's momentum and weight decay; without it, the request as asked for.
   * Close mosaic is the trainer's capped count ("last 7 epochs (requested 10)"). Copy JSON copies the request: the
   * TrainConfig `POST /api/runs` and MCP `train` take, which resolves the same way again. With `cli` (servers that list
   * `run_cli`), the same run from a terminal on the server's machine closes the panel: "In PowerShell:" and the
   * command on one copyable line (ui/CommandLine).
   */
  import Copy from '@lucide/svelte/icons/copy';
  import type { ResolvedTrain, TrainConfig } from '$lib/api/types';
  import { FROM_SCRATCH, type Origin } from '$lib/lineage';
  import Button from '$lib/ui/Button.svelte';
  import CommandLine from '$lib/ui/CommandLine.svelte';
  import Lineage from '$lib/ui/Lineage.svelte';
  import Note from '$lib/ui/Note.svelte';
  import Panel from '$lib/ui/Panel.svelte';
  import { toast } from '$lib/ui/toast.svelte';
  import { batchSetting, closeMosaicSetting, headSetting, lrSetting, optimizerSetting, precisionSetting, resolvedRows, effectiveBatchSetting, type Setting } from './resolved';

  let {
    config,
    resolved = null,
    datasetName,
    compact = false,
    essential = false,
    weights,
    cli = null,
  }: {
    config: TrainConfig;
    /** What the trainer resolved `config` to; null until it is built, missing from older servers. */
    resolved?: ResolvedTrain | null;
    datasetName?: string;
    /** One column for a rail, augmentation folded away. */
    compact?: boolean;
    /** Phones, a queued run: the six settings that say what will train, the rest of training and augmentation folded. */
    essential?: boolean;
    /** Where the run started, read once ($lib/lineage): the checkpoint by name, its id in the tooltip, as the header says it. */
    weights?: Origin | null;
    /** The run's `folo train` command, quoted for PowerShell (`GET /api/runs/{id}/cli`); null where the server has none. */
    cli?: string | null;
  } = $props();

  const a = $derived(config.augment);
  // Identifiers (model names, weight paths, devices) are mono; numbers, prose ("From scratch") and named optimizers
  // ("AdamW", a name) are sans with tabular numerals (design/README.md, Type).
  const id = (value: string): Setting => ({ value, ident: true });
  const plain = (value: string): Setting => ({ value });
  // Three groups: what trains on what (the essentials), how the optimizer runs, and the rest of the setup. The rail
  // lists them in that order as one Training list; the two-column layout puts the optimizer over Augmentation so the
  // columns end together; a phone shows the essentials and folds the others.
  // The effective batch sits under Batch ("64 (4 × 16)"); a phone keeps its six essentials and folds it
  // with the other settings.
  const effective = $derived<[string, Setting][]>(resolved ? [['Effective batch', effectiveBatchSetting(resolved)]] : []);
  const core = $derived<[string, Setting][]>([
    ['Dataset', plain(datasetName ?? config.dataset_id)],
    ['Model', id(config.arch)],
    ['Epochs', plain(String(config.epochs))],
    ['Batch', batchSetting(config, resolved)],
    ...(essential ? [] : effective),
    ['Image size', plain(`${config.imgsz} px`)],
    ['Device', config.device === 'auto' ? plain('auto') : id(config.device)],
  ]);
  // Under its own group heading ("Optimizer settings") the optimizer's row is "Type", never "Optimizer" twice; in one
  // list (the rail, a phone's fold) it keeps its name.
  const grouped = $derived(!essential && !compact);
  const optim = $derived<[string, Setting][]>([
    [grouped ? 'Type' : 'Optimizer', optimizerSetting(config, resolved)],
    ['Initial LR', lrSetting(config, resolved)],
    ...(resolved ? resolvedRows(resolved) : []),
  ]);
  const setup = $derived<[string, Setting][]>([
    ['Precision', precisionSetting(config)],
    ['Head', headSetting(config)],
    ['Weights', config.weights ? id(config.weights) : plain(FROM_SCRATCH)],
    ['Seed', plain(String(config.seed))],
  ]);
  const aug = $derived<[string, Setting][]>([
    ['Mosaic', plain(String(a.mosaic))],
    ['Mixup', plain(String(a.mixup))],
    ['HSV h · s · v', plain(a.hsv.join(' · '))],
    ['Flip LR', plain(String(a.fliplr))],
    ['Scale', plain(`±${a.scale}`)],
    ['Translate', plain(`±${a.translate}`)],
    ['Close mosaic', closeMosaicSetting({ config, resolved })],
  ]);

  /** A value too long to sit beside its label in a narrow column ("Auto (AdamW for short runs, else SGD)"): in a
   *  column under 360 px it takes the full width under the label instead of wrapping to a right-aligned second line. */
  const isLong = (v: Setting) => v.value.length + (v.note ? v.note.length + 1 : 0) > 32;

  async function copy() {
    try {
      await navigator.clipboard.writeText(JSON.stringify(config, null, 2));
      toast.success('Config copied', 'The request as POST /api/runs and MCP train take it.');
    } catch (e) {
      toast.error(e, 'Could not copy');
    }
  }
</script>

{#snippet list(heading: string | null, items: [string, Setting][])}
  <dl>
    <!-- A heading only beside another group's; alone in the rail it would name the panel twice. -->
    {#if heading}<p class="group group-label">{heading}</p>{/if}
    {#each items as [k, v] (k)}{@const long = isLong(v)}<dt class:long>{k}{#if v.about}<span class="sub">{v.about}</span>{/if}</dt>{#if k === 'Weights' && weights}<dd>{#if weights.kind === 'scratch'}{FROM_SCRATCH}{:else}<Lineage origin={weights} />{/if}</dd>{:else}<dd class="num" class:long><span class:mono={v.ident}>{v.value}</span>{#if v.note}<Note text={v.note} />{/if}</dd>{/if}{/each}
  </dl>
{/snippet}
{#snippet fold(title: string, items: [string, Setting][])}
  <details>
    <summary>{title}</summary>
    {@render list(null, items)}
  </details>
{/snippet}

<Panel title="Config" id="config">
  {#snippet actions()}<Button size="sm" variant="ghost" icon={Copy} onclick={copy}>Copy JSON</Button>{/snippet}
  {#if essential}
    {@render list(null, core)}
    {@render fold('More training settings', [...effective, ...optim, ...setup])}
    {@render fold('Augmentation', aug)}
  {:else if compact}
    {@render list(null, [...core, ...optim, ...setup])}
    {@render fold('Augmentation', aug)}
  {:else}
    <!-- Two columns that end together: the run's setup on the left, the optimizer and augmentation on the right. -->
    <div class="cols">
      {@render list('Training', [...core, ...setup])}
      <div class="col">
        {@render list('Optimizer settings', optim)}
        {@render list('Augmentation', aug)}
      </div>
    </div>
  {/if}
  {#if cli}
    <div class="cli">
      <p class="shell">In PowerShell:</p>
      <CommandLine command={cli} label="PowerShell command" copied="The run's folo train command, for the server's machine." />
    </div>
  {/if}
</Panel>

<style>
  /* The run from a terminal: under the settings, across the panel's width, the command one line that never wraps. */
  .cli {
    display: grid;
    gap: var(--space-1);
    margin-top: var(--space-4);
    min-width: 0;
  }
  /* As ui/Remedy labels its shell. */
  .shell {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-tertiary);
  }
  .cols {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: var(--space-6);
  }
  /* 16 px between the right column's groups: with one heading more than the left, the columns end within a few px. */
  .col {
    display: grid;
    gap: var(--space-4);
    align-content: start;
  }
  /* A container: whether a long value fits beside its label depends on the column's width (half a panel, a rail, a
     phone), not the viewport's (DESIGN_DIRECTION §18). */
  dl {
    display: grid;
    grid-template-columns: auto 1fr;
    gap: 0;
    align-content: start;
    container-type: inline-size;
  }
  /* Type and ink: the shared group label (base.css .group-label). */
  .group {
    grid-column: 1 / -1;
    margin-bottom: var(--space-1);
  }
  dt,
  dd {
    padding: 4px 0;
    border-bottom: 1px solid var(--line-subtle);
  }
  details {
    margin-top: var(--space-3);
  }
  details + details {
    margin-top: var(--space-1);
  }
  summary {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    height: 28px;
    font-size: var(--type-sm-size);
    color: var(--text-secondary);
    cursor: pointer;
    list-style: none;
  }
  @media (pointer: coarse) {
    summary {
      height: 44px;
    }
  }
  summary::-webkit-details-marker {
    display: none;
  }
  summary::before {
    content: '';
    width: 6px;
    height: 6px;
    border-right: 1.5px solid currentColor;
    border-bottom: 1.5px solid currentColor;
    transform: rotate(-45deg);
    transition: transform var(--dur-fast) var(--ease-standard);
  }
  details[open] summary::before {
    transform: rotate(45deg);
  }
  summary:hover {
    color: var(--text-primary);
  }
  dt {
    color: var(--text-tertiary);
  }
  /* What a short value means ("Auto": AdamW for short runs, else SGD), under its label at 12 px: a value cell holds one
     short value and an explanation goes under the label, so every value stays on its row's right edge. */
  dt .sub {
    display: block;
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
  }
  dd {
    padding-left: var(--space-4);
    text-align: right;
    overflow-wrap: anywhere;
  }
  /* A narrow column: a long value goes under its label at the column's full width, left-aligned, one row's rule under
     both, never a value wrapping to a ragged right-aligned second line (round 26's "else SGD"). */
  @container (max-width: 359px) {
    dt.long {
      grid-column: 1 / -1;
      padding-bottom: 0;
      border-bottom: 0;
    }
    dd.long {
      grid-column: 1 / -1;
      padding-top: 2px;
      padding-left: 0;
      text-align: left;
    }
  }
  @media (max-width: 767px) {
    .cols {
      grid-template-columns: 1fr;
    }
  }
</style>
