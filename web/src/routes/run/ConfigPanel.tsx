/**
 * A run's config. With `resolved` (docs/API.md, `Run.resolved`) the settings it trains with are the values shown,
 * the request a note where it differs ("AdamW (auto)", "0.001429 (requested 0.01)"), plus the batch it accumulates,
 * the LR decay, the warmup and the optimizer's momentum and weight decay; without it, the request as asked for.
 * Close mosaic is the trainer's capped count ("last 7 epochs (requested 10)"). Copy JSON copies the request: the
 * TrainConfig `POST /api/runs` and MCP `train` take, which resolves the same way again. With `cli` (servers that list
 * `run_cli`), the same run from a terminal on the server's machine closes the panel: "In PowerShell:" and the
 * command on one copyable line (ui/CommandLine).
 */
import Copy from '$icons/copy';
import { computed, css, For, MediaQuery } from 'fw';
import { media } from '$lib/design/tokens';
import type { ResolvedTrain, TrainConfig } from '$lib/api/types';
import { FROM_SCRATCH, type Origin } from '$lib/lineage';
import Button from '$lib/ui/Button';
import CommandLine from '$lib/ui/CommandLine';
import Lineage from '$lib/ui/Lineage';
import Note from '$lib/ui/Note';
import Ident from '$lib/ui/Ident';
import Panel from '$lib/ui/Panel';
import { toast } from '$lib/ui/toast';
import { batchSetting, closeMosaicSetting, headSetting, lrSetting, optimizerSetting, precisionSetting, resolvedRows, effectiveBatchSetting, type Setting } from './resolved';

export interface ConfigPanelProps {
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
  /** The run's server id (`run_7d1e`): in the header, never in the page's (design/UNIFIED.md §2, Detail header). */
  runId?: string;
}

type Row = [string, Setting];

// Identifiers (model names, weight paths, devices) are mono; numbers, prose ("From scratch") and named optimizers
// ("AdamW", a name) are sans with tabular numerals (design/README.md, Type).
const id = (value: string): Setting => ({ value, ident: true });
const plain = (value: string): Setting => ({ value });

/** A value too long to sit beside its label in a narrow column ("Auto (MuSGD for every run)"): in a
 *  column under 360 px it takes the full width under the label instead of wrapping to a right-aligned second line. */
const isLong = (v: Setting) => v.value.length + (v.note ? v.note.length + 1 : 0) > 32;

export default function ConfigPanel(props: ConfigPanelProps) {
  const resolved = () => props.resolved ?? null;
  const compact = () => props.compact ?? false;
  const essential = () => props.essential ?? false;
  const a = () => props.config.augment;
  // Three groups: what trains on what (the essentials), how the optimizer runs, and the rest of the setup. The rail
  // lists them in that order as one Training list; the two-column layout puts the optimizer over Augmentation so the
  // columns end together; a phone shows the essentials and folds the others.
  // The effective batch sits under Batch ("64 (4 × 16)"); a phone keeps its six essentials and folds it
  // with the other settings.
  const effective = computed<Row[]>(() => {
    const r = resolved();
    return r ? [['Effective batch', effectiveBatchSetting(r)]] : [];
  });
  const core = computed<Row[]>(() => {
    const config = props.config;
    return [
      ['Dataset', plain(props.datasetName ?? config.dataset_id)],
      ['Model', id(config.arch)],
      ['Epochs', plain(String(config.epochs))],
      ['Batch', batchSetting(config, resolved())],
      ...(essential() ? [] : effective.get()),
      ['Image size', plain(`${config.imgsz} px`)],
      ['Device', config.device === 'auto' ? plain('auto') : id(config.device)],
    ];
  });
  // Under its own group heading ("Optimizer settings") the optimizer's row is "Type", never "Optimizer" twice; in one
  // list (the rail, a phone's fold) it keeps its name.
  const grouped = () => !essential() && !compact();
  const optim = computed<Row[]>(() => {
    const r = resolved();
    return [
      [grouped() ? 'Type' : 'Optimizer', optimizerSetting(props.config, r)],
      ['Initial LR', lrSetting(props.config, r)],
      ...(r ? resolvedRows(r) : []),
    ];
  });
  const setup = computed<Row[]>(() => {
    const config = props.config;
    return [
      ['Precision', precisionSetting(config)],
      ['Head', headSetting(config)],
      ['Weights', config.weights ? id(config.weights) : plain(FROM_SCRATCH)],
      ['Seed', plain(String(config.seed))],
    ];
  });
  const aug = computed<Row[]>(() => {
    const au = a();
    return [
      ['Mosaic', plain(String(au.mosaic))],
      ['Mixup', plain(String(au.mixup))],
      ['HSV h · s · v', plain(au.hsv.join(' · '))],
      ['Flip LR', plain(String(au.fliplr))],
      ['Scale', plain(`±${au.scale}`)],
      ['Translate', plain(`±${au.translate}`)],
      ['Close mosaic', closeMosaicSetting({ config: props.config, resolved: resolved() })],
    ];
  });

  async function copy() {
    try {
      await navigator.clipboard.writeText(JSON.stringify(props.config, null, 2));
      toast.success('Config copied', 'The request as POST /api/runs and MCP train take it.');
    } catch (e) {
      toast.error(e, 'Could not copy');
    }
  }

  async function copyCli() {
    try {
      await navigator.clipboard.writeText(props.cli ?? '');
      toast.success('Command copied', "The run's folo train command, for the server's machine.");
    } catch (e) {
      toast.error(e, 'Could not copy');
    }
  }

  const list = (heading: string | null, items: () => Row[]) => (
    <dl>
      {/* A heading only beside another group's; alone in the rail it would name the panel twice. */}
      {heading ? <p class="group group-label">{heading}</p> : null}
      <For each={items()} key={(r) => r[0]}>
        {(row) => {
          const k = () => row()[0];
          const v = () => row()[1];
          const long = () => isLong(v());
          return (
            <>
              <dt class:long={long()}>
                {k()}
                {v().about ? <span class="sub">{v().about}</span> : null}
              </dt>
              {k() === 'Weights' && props.weights ? (
                <dd>{props.weights.kind === 'scratch' ? FROM_SCRATCH : <Lineage origin={props.weights} />}</dd>
              ) : (
                <dd class="num" class:long={long()}>
                  <span class:mono={v().ident ?? false}>{v().value}</span>
                  {v().note ? <Note text={v().note!} /> : null}
                </dd>
              )}
            </>
          );
        }}
      </For>
    </dl>
  );
  const fold = (title: string, items: () => Row[]) => (
    <details>
      <summary>{title}</summary>
      {list(null, items)}
    </details>
  );

  // The 1/3 rail from 1024 to 1279 px, and phones: too narrow for the id and two labelled buttons beside the title.
  const laptop = new MediaQuery(media.laptopUp, false);
  const belowDesktop = new MediaQuery(media.upToDesktop, false);
  const phone = new MediaQuery(media.phone, false);
  const tight = () => (laptop.current && belowDesktop.current) || phone.current;

  return (
    <Panel
      title="Config"
      id="config"
      actions={() => (
        // The run's server id, then the ways to take its settings elsewhere: its command (where the server gives one)
        // and the request as JSON.
        <>
          {props.runId ? <Ident value={props.runId} class="run-id" /> : null}
          {/* Where the column is narrow (the laptop rail, phones) the two are icons, named and titled. */}
          {props.cli ? (
            tight() ? (
              <Button size="sm" variant="ghost" icon={Copy} aria-label="Copy command" title="Copy command" onclick={copyCli} />
            ) : (
              <Button size="sm" variant="ghost" icon={Copy} onclick={copyCli}>
                Copy command
              </Button>
            )
          ) : null}
          {tight() ? (
            <Button size="sm" variant="ghost" icon={Copy} aria-label="Copy JSON" title="Copy JSON" onclick={copy} />
          ) : (
            <Button size="sm" variant="ghost" icon={Copy} onclick={copy}>
              Copy JSON
            </Button>
          )}
        </>
      )}
    >
      {essential() ? (
        <>
          {list(null, () => core.get())}
          {fold('More training settings', () => [...effective.get(), ...optim.get(), ...setup.get()])}
          {fold('Augmentation', () => aug.get())}
        </>
      ) : compact() ? (
        <>
          {list(null, () => [...core.get(), ...optim.get(), ...setup.get()])}
          {fold('Augmentation', () => aug.get())}
        </>
      ) : (
        // Two columns that end together: the run's setup on the left, the optimizer and augmentation on the right.
        <div class="cols">
          {list('Training', () => [...core.get(), ...setup.get()])}
          <div class="col">
            {list('Optimizer settings', () => optim.get())}
            {list('Augmentation', () => aug.get())}
          </div>
        </div>
      )}
      {props.cli ? (
        <div class="cli">
          <p class="shell">In PowerShell:</p>
          <CommandLine command={props.cli} label="PowerShell command" copied="The run's folo train command, for the server's machine." />
        </div>
      ) : null}
    </Panel>
  );
}

css`
  /* The server id in the header: quiet, a 12 px identifier. */
  :global(.ident.run-id) {
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
  }
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
  /* What a short value means ("Auto": MuSGD for every run), under its label at 12 px: a value cell holds one
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
`;
