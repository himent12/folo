/**
 * Model and backend picker switcher for Playground: renders `ComparePickers` in A/B Compare mode
 * and `ModelPickers` in single-model mode, either inside the desktop rail or the narrow toolbar row.
 */
import type { Signal } from 'fw';
import type { Model } from '$lib/api/types';
import type { ModelChoice } from '$lib/scoreSummary';
import ComparePickers from './ComparePickers';
import ModelPickers from './ModelPickers';
import type { CompareViewMode } from './compare';
import type { RunOn } from './predict';

export interface PlaygroundPickersProps {
  compare: boolean;
  inRail: boolean;
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
  runOnA: Signal<RunOn>;
  runOnB: Signal<RunOn>;
  compareView: Signal<CompareViewMode>;
  webgpu: boolean;
  browserOff: string | null;
  runOnNote: string | null;
}

export default function PlaygroundPickers(props: PlaygroundPickersProps) {
  return (
    <>
      {props.compare ? (
        <ComparePickers
          rail={props.inRail}
          toolbar={!props.inRail}
          ranked={props.ranked}
          choices={props.choices}
          modelAId={props.modelAId}
          modelBId={props.modelBId}
          modelA={props.modelA}
          modelB={props.modelB}
          metaA={props.metaA}
          metaB={props.metaB}
          chooseModelA={props.chooseModelA}
          chooseModelB={props.chooseModelB}
          swapModels={props.swapModels}
          bind:runOnA={props.runOnA}
          bind:runOnB={props.runOnB}
          bind:compareView={props.compareView}
          webgpu={props.webgpu}
          browserOff={props.browserOff}
        />
      ) : (
        <ModelPickers
          rail={props.inRail}
          toolbar={!props.inRail}
          ranked={props.ranked}
          modelId={props.modelAId}
          choices={props.choices}
          chooseModel={props.chooseModelA}
          model={props.modelA}
          pickedMeta={props.metaA}
          bind:runOn={props.runOnA}
          webgpu={props.webgpu}
          browserOff={props.browserOff}
          runOnNote={props.runOnNote}
        />
      )}
    </>
  );
}
