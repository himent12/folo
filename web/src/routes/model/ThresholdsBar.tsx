/**
 * Operating point and server-calibrated threshold bar inside Model Detail (`#pr`):
 * - Always displays the evaluator's measured global `best_f1_conf` operating point with one-click Playground handoff.
 * - On servers advertising `threshold_calibration`, runs `POST /api/val` on demand to obtain the exact 0.001-grid
 *   per-class `ThresholdCalibration` for `thresholds.json` download (`folo predict --thresholds` / Playground import)
 *   and per-class Playground handoff.
 */
import Download from '$icons/download';
import ScanSearch from '$icons/scan-search';
import SlidersHorizontal from '$icons/sliders-horizontal';
import { css } from 'fw';
import { metric } from '$lib/format';
import Button from '$lib/ui/Button';
import { formatConfThreshold } from '../playground/perClassConf';
import type { ModelOperatingSummary, ServerModelCalibration } from './thresholds';

export interface ThresholdsBarProps {
  /** Measured global operating point from `GET /api/models/{id}/eval`. */
  operatingPoint: ModelOperatingSummary;
  /** Exact server-computed per-class calibration from `POST /api/val`, once calibrated. */
  calibrated: ServerModelCalibration | null;
  /** True when the server advertises `threshold_calibration` and the evaluated dataset + split are known. */
  canCalibrate: boolean;
  /** True while `POST /api/val` calibration is in flight. */
  calibrating?: boolean;
  onCalibrate?: () => void;
  onDownload: () => void;
  onApplyPlayground: () => void;
}

export default function ThresholdsBar(props: ThresholdsBarProps) {
  const cal = () => props.calibrated;
  const op = () => props.operatingPoint;
  const globalConf = () => cal()?.payload.global.best_conf ?? op().globalConf;
  const macroF1 = () => cal()?.payload.macro_f1 ?? op().macroF1;
  const count = () => cal()?.payload.per_class.length ?? op().classCount;

  return (
    <div class="thr-bar" data-model-thresholds={cal() ? 'calibrated' : 'operating-point'}>
      <div class="thr-info">
        <span class="thr-title">{cal() ? 'Calibrated thresholds' : 'Operating point'}</span>
        <span class="thr-meta num">
          {cal() ? 'global conf' : 'best-F1 conf'} {formatConfThreshold(globalConf())} · macro-F1 {metric(macroF1())} ·{' '}
          {count()} {count() === 1 ? 'class' : 'classes'}
        </span>
        <span class="thr-hint">
          {cal() ? (
            'Server-calibrated 0.001-grid per-class F1 thresholds ready for folo predict --thresholds or Playground.'
          ) : props.canCalibrate ? (
            'Calibrate via validation for per-class thresholds.json download and Playground overrides, or apply the global threshold now.'
          ) : (
            <>
              Measured global best-F1 threshold; run <span class="mono">folo val --save-thresholds</span> for per-class thresholds.json.
            </>
          )}
        </span>
      </div>
      <div class="thr-actions">
        {!cal() && props.canCalibrate && props.onCalibrate ? (
          <Button
            size="sm"
            variant="ghost"
            icon={SlidersHorizontal}
            loading={!!props.calibrating}
            data-calibrate-thresholds
            onclick={props.onCalibrate}
          >
            {props.calibrating ? 'Calibrating…' : 'Calibrate per-class'}
          </Button>
        ) : null}
        {cal() || props.canCalibrate ? (
          <Button
            size="sm"
            variant="ghost"
            icon={Download}
            loading={!!props.calibrating}
            data-download-thresholds
            onclick={props.onDownload}
          >
            Download thresholds.json
          </Button>
        ) : null}
        <Button size="sm" icon={ScanSearch} disabled={!!props.calibrating} data-apply-playground onclick={props.onApplyPlayground}>
          Apply in Playground
        </Button>
      </div>
    </div>
  );
}

css`
  .thr-bar {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-3);
    padding: var(--space-3) var(--space-4);
    border-top: 1px solid var(--line-subtle);
    background: var(--bg-panel);
  }
  .thr-info {
    display: flex;
    flex-wrap: wrap;
    align-items: baseline;
    gap: var(--space-1) var(--space-3);
    min-width: 0;
  }
  .thr-title {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    font-weight: var(--weight-medium);
    color: var(--text-primary);
  }
  .thr-meta {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
  }
  .thr-hint {
    flex-basis: 100%;
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-tertiary);
  }
  .thr-hint .mono {
    color: var(--text-secondary);
  }
  .thr-actions {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: var(--space-2);
  }
`;
