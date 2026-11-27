/**
 * Model A and Model B selectors for Playground Compare A/B mode, with optional per-model backend switches, a Swap A/B
 * action, and the Side-by-side / Split view mode control. Works in both the 320 px desktop rail (`rail`) and the
 * full-width toolbar below 1024 px (`toolbar`).
 */
import ArrowLeftRight from '$icons/arrow-left-right';
import ArrowUpRight from '$icons/arrow-up-right';
import { bindable, css, type Signal } from 'fw';
import type { Model } from '$lib/api/types';
import { href } from '$lib/router';
import type { ModelChoice } from '$lib/scoreSummary';
import Button from '$lib/ui/Button';
import Segmented from '$lib/ui/Segmented';
import Select from '$lib/ui/Select';
import { tooltip } from '$lib/ui/tooltip';
import type { CompareViewMode } from './compare';
import type { RunOn } from './predict';

export interface ComparePickersProps {
  toolbar?: boolean;
  rail?: boolean;
  ranked: Model[];
  choices: ModelChoice<Model>[];
  modelAId: string | null;
  modelBId: string | null;
  modelA: Model | null;
  modelB: Model | null;
  metaA: string | null;
  metaB: string | null;
  chooseModelA: (id: string) => void;
  chooseModelB: (id: string) => void;
  swapModels: () => void;
  runOnA?: RunOn;
  runOnB?: RunOn;
  compareView?: CompareViewMode;
  webgpu: boolean;
  browserOff: string | null;
}

const VIEW_MODES: { value: CompareViewMode; label: string }[] = [
  { value: 'side-by-side', label: 'Side by side' },
  { value: 'split', label: 'Split' },
];

export default function ComparePickers(props: ComparePickersProps) {
  const runOnA = bindable(props, 'runOnA', 'server');
  const runOnB = bindable(props, 'runOnB', 'server');
  const compareView = bindable(props, 'compareView', 'side-by-side');

  const options = () => props.choices.map((c) => ({ value: c.model.id, label: c.label, group: c.group, short: c.short }));
  const backendOptions = () => [
    { value: 'server' as const, label: 'Server' },
    { value: 'webgpu' as const, label: 'WebGPU', disabled: !props.webgpu, tip: 'No WebGPU in this browser' },
    { value: 'cpu' as const, label: 'CPU' },
  ];

  const sideBlock = (
    badge: 'A' | 'B',
    label: string,
    id: string | null,
    model: Model | null,
    meta: string | null,
    choose: (v: string) => void,
    runOn: Signal<RunOn>,
  ) => (
    <div class="ab-side">
      <div class="ab-head">
        <span class="ab-tag" class:b={badge === 'B'}>
          {badge}
        </span>
        <span class="ab-label">{label}</span>
        {badge === 'A' ? (
          <span class="ab-swap" use:tooltip={'Swap Model A and Model B'}>
            <Button
              size="sm"
              variant="ghost"
              icon={ArrowLeftRight}
              aria-label="Swap Model A and Model B"
              disabled={props.modelAId === props.modelBId && runOnA.get() === runOnB.get()}
              onclick={props.swapModels}
            >
              Swap
            </Button>
          </span>
        ) : null}
      </div>
      <div class="ab-row">
        <Select
          label={label}
          value={id ?? ''}
          options={options()}
          onchange={choose}
          describedby={model ? `compare-meta-${badge}` : undefined}
        />
        {model ? <Button icon={ArrowUpRight} href={href(`/models/${model.id}`)} aria-label={`Open ${model.name}`} /> : null}
      </div>
      {model ? (
        <p class="pick-meta num" id={`compare-meta-${badge}`}>
          {meta ?? 'Not evaluated'}
        </p>
      ) : null}
      {props.browserOff ? null : (
        <div class="ab-runon" role="group" aria-label={`${label} run on`}>
          <Segmented label={`Run ${label} on`} size="sm" bind:value={runOn} options={backendOptions()} />
        </div>
      )}
    </div>
  );

  return (
    <div class={props.rail ? 'ab-rail' : 'ab-tools'}>
      <div class="ab-sides">
        {sideBlock('A', 'Model A', props.modelAId, props.modelA, props.metaA, props.chooseModelA, runOnA)}
        {sideBlock('B', 'Model B', props.modelBId, props.modelB, props.metaB, props.chooseModelB, runOnB)}
      </div>
      <div class="ab-view" role="group" aria-label="Compare layout">
        <span class="ab-label">Layout</span>
        <Segmented label="Compare layout" size="sm" bind:value={compareView} options={VIEW_MODES} />
      </div>
    </div>
  );
}

css`
  .ab-rail {
    display: grid;
    gap: var(--space-4);
  }
  .ab-sides {
    display: grid;
    gap: var(--space-3);
  }
  .ab-side {
    display: grid;
    gap: var(--space-1);
    min-width: 0;
  }
  .ab-head {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    min-height: 24px;
  }
  .ab-tag {
    display: inline-grid;
    place-items: center;
    width: 20px;
    height: 20px;
    border-radius: var(--radius-xs);
    background: var(--accent-muted);
    color: var(--accent-text);
    font-size: var(--type-sm-size);
    font-weight: var(--weight-semibold);
    flex-shrink: 0;
  }
  .ab-tag.b {
    background: var(--info-muted);
    color: var(--info-text);
  }
  .ab-label {
    color: var(--text-secondary);
    font-size: var(--type-sm-size);
    font-weight: var(--weight-medium);
  }
  .ab-swap {
    margin-left: auto;
    display: inline-flex;
  }
  .ab-row {
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto;
    align-items: center;
    gap: var(--space-2);
  }
  .ab-row :global(.select) {
    min-width: 0;
  }
  .pick-meta {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-tertiary);
    text-wrap: pretty;
  }
  .ab-runon {
    display: flex;
    margin-top: 2px;
  }
  .ab-view {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-2);
    padding-top: var(--space-3);
    border-top: 1px solid var(--line-subtle);
  }
  .ab-tools {
    display: grid;
    gap: var(--space-3);
    padding: var(--space-3) var(--space-4);
    border: 1px solid var(--line-panel);
    border-radius: var(--radius-lg);
    background: var(--bg-panel);
  }
  .ab-tools .ab-sides {
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: var(--space-4);
  }
  @media (max-width: 767px) {
    .ab-tools {
      padding: var(--space-3);
    }
    .ab-tools .ab-sides {
      grid-template-columns: minmax(0, 1fr);
      gap: var(--space-3);
    }
  }
`;
