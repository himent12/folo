/**
 * Playground timing section body: renders either the single-model `TimingBar` (with desktop browser re-filter time)
 * or stacked Model A + Model B `TimingBar` cards in Compare A/B mode.
 */
import { css } from 'fw';
import type { Model, Prediction } from '$lib/api/types';
import { detectorInfoBadge, inferenceMetaLabel, type BrowserEngine, type DetectorInfo } from '$lib/inference/browser';
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
  /** Browser engine instance used to look up per-prediction `DetectorInfo` (preserves per-side metadata in Compare A/B). */
  engine?: BrowserEngine | null;
  /** Fallback loaded browser detector metadata (`folo-wasm` `detector.info()`), used when `engine` is not passed. */
  engineInfo?: DetectorInfo | null;
}

function matchedEngineInfo(
  pred: Prediction | null,
  engine: BrowserEngine | null | undefined,
  fallbackInfo: DetectorInfo | null | undefined,
): DetectorInfo | null {
  if (!pred?.backend) return null;
  const bound = engine?.infoFor(pred);
  if (bound) return bound;
  if (!fallbackInfo) return null;
  return fallbackInfo.backend === pred.backend && (!fallbackInfo.modelId || fallbackInfo.modelId === pred.model_id)
    ? fallbackInfo
    : null;
}

function browserBadgeFor(
  pred: Prediction | null,
  info: DetectorInfo | null,
  model: Model | null,
): string | null {
  if (!pred?.backend) return null;
  if (info) return detectorInfoBadge(info);
  if (model) {
    const parts: string[] = [model.arch];
    if (model.arch.startsWith('folo2-')) parts.push('FOLO v2');
    if (model.head) parts.push(model.head);
    return parts.join(' · ');
  }
  return null;
}

export default function PlaygroundTiming(props: PlaygroundTimingProps) {
  const infoA = () => matchedEngineInfo(props.predA, props.engine, props.engineInfo);
  const infoB = () => matchedEngineInfo(props.predB, props.engine, props.engineInfo);
  return (
    <>
      {props.compare ? (
        <div class="ab-timing">
          <div class="ab-t-block">
            <p class="ab-t-head">A · {props.modelA?.name ?? 'Model A'}</p>
            {props.predA ? (
              <TimingBar
                timing={props.predA.timing}
                device={deviceLabel(props.predA, infoA()?.adapter ?? props.adapter)}
                cold={props.predA.cold}
                engineBadge={browserBadgeFor(props.predA, infoA(), props.modelA)}
                precision={infoA()?.precision ?? null}
                precisionFallback={infoA()?.precisionFallback ?? null}
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
                device={deviceLabel(props.predB, infoB()?.adapter ?? props.adapter)}
                cold={props.predB.cold}
                engineBadge={browserBadgeFor(props.predB, infoB(), props.modelB)}
                precision={infoB()?.precision ?? null}
                precisionFallback={infoB()?.precisionFallback ?? null}
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
            device={deviceLabel(props.predA, infoA()?.adapter ?? props.adapter)}
            cold={props.predA.cold}
            engineBadge={browserBadgeFor(props.predA, infoA(), props.modelA)}
            precision={infoA()?.precision ?? null}
            precisionFallback={infoA()?.precisionFallback ?? null}
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
