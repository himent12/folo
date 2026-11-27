/**
 * Playground timing section body: renders either the single-model `TimingBar` (with desktop browser re-filter time)
 * or stacked Model A + Model B `TimingBar` cards in Compare A/B mode.
 */
import { css } from 'fw';
import type { Model, Prediction } from '$lib/api/types';
import { detectorInfoBadge, inferenceMetaLabel, type DetectorInfo } from '$lib/inference/browser';
import Skeleton from '$lib/ui/Skeleton';
import TimingBar from './TimingBar';
import { deviceLabel } from './viewState';

export interface PlaygroundTimingProps {
  compare: boolean;
  narrow: boolean;
  modelA: Model | null;
  modelB: Model | null;
  predA: Prediction | null;
  predB: Prediction | null;
  predictingA: boolean;
  predictingB: boolean;
  resultMsA: number | null;
  adapter?: string | null;
  /** Loaded browser detector metadata (`folo-wasm` `detector.info()`), surfaced for browser predictions. */
  engineInfo?: DetectorInfo | null;
}

function browserBadgeFor(pred: Prediction | null, info: DetectorInfo | null | undefined, model: Model | null): string | null {
  if (!pred?.backend) return null;
  if (info && info.backend === pred.backend && (!info.modelId || info.modelId === pred.model_id)) return detectorInfoBadge(info);
  if (model) {
    const parts: string[] = [model.arch];
    if (model.arch.startsWith('folo2-')) parts.push('FOLO v2');
    if (model.head) parts.push(model.head);
    return parts.join(' · ');
  }
  return null;
}

export default function PlaygroundTiming(props: PlaygroundTimingProps) {
  return (
    <>
      {props.compare ? (
        <div class="ab-timing">
          <div class="ab-t-block">
            <p class="ab-t-head">A · {props.modelA?.name ?? 'Model A'}</p>
            {props.predA ? (
              <TimingBar
                timing={props.predA.timing}
                device={deviceLabel(props.predA, props.adapter)}
                cold={props.predA.cold}
                engineBadge={browserBadgeFor(props.predA, props.engineInfo, props.modelA)}
                mode={inferenceMetaLabel(props.predA.inference)}
              />
            ) : props.predictingA ? (
              <div class="t-sk">
                <Skeleton h="24px" w="50%" />
                <Skeleton h="8px" />
              </div>
            ) : (
              <p class="hint">Waiting for Model A.</p>
            )}
          </div>
          <div class="ab-t-block">
            <p class="ab-t-head">B · {props.modelB?.name ?? 'Model B'}</p>
            {props.predB ? (
              <TimingBar
                timing={props.predB.timing}
                device={deviceLabel(props.predB, props.adapter)}
                cold={props.predB.cold}
                engineBadge={browserBadgeFor(props.predB, props.engineInfo, props.modelB)}
                mode={inferenceMetaLabel(props.predB.inference)}
              />
            ) : props.predictingB ? (
              <div class="t-sk">
                <Skeleton h="24px" w="50%" />
                <Skeleton h="8px" />
              </div>
            ) : (
              <p class="hint">Waiting for Model B.</p>
            )}
          </div>
        </div>
      ) : props.predA ? (
        <>
          <TimingBar
            timing={props.predA.timing}
            device={deviceLabel(props.predA, props.adapter)}
            cold={props.predA.cold}
            engineBadge={browserBadgeFor(props.predA, props.engineInfo, props.modelA)}
            mode={inferenceMetaLabel(props.predA.inference)}
          />
          {!props.narrow && props.resultMsA !== null ? (
            <p class="refilter num">Thresholds re-filtered in {props.resultMsA < 0.01 ? '<0.01' : props.resultMsA.toFixed(2)} ms</p>
          ) : null}
        </>
      ) : props.predictingA ? (
        <div class="t-sk">
          <Skeleton h="28px" w="50%" />
          <Skeleton h="8px" />
          <Skeleton h="14px" />
          <Skeleton h="14px" />
        </div>
      ) : (
        <p class="hint">Timing appears after a run.</p>
      )}
    </>
  );
}

css`
  .refilter {
    margin-top: var(--space-3);
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
  }
  .hint {
    color: var(--text-tertiary);
    font-size: var(--type-sm-size);
  }
  .t-sk {
    display: grid;
    gap: var(--space-3);
  }
  .ab-timing {
    display: grid;
    gap: var(--space-4);
  }
  .ab-t-block {
    display: grid;
    gap: var(--space-2);
  }
  .ab-t-block + .ab-t-block {
    padding-top: var(--space-3);
    border-top: 1px solid var(--line-subtle);
  }
  .ab-t-head {
    font-size: var(--type-sm-size);
    font-weight: var(--weight-medium);
    color: var(--text-secondary);
  }
`;
