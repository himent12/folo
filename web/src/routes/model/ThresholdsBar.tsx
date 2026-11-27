/**
 * Calibrated per-class F1 thresholds strip inside Model Detail (`#pr`): summarizes the global operating point
 * and macro-F1 derived from the loaded evaluation, downloads `thresholds.json` (ready for `folo predict --thresholds`
 * or Playground import), and hands off the model + per-class thresholds straight into Playground.
 */
import Download from '$icons/download';
import ScanSearch from '$icons/scan-search';
import { css } from 'fw';
import { metric } from '$lib/format';
import Button from '$lib/ui/Button';
import { formatConfThreshold } from '../playground/perClassConf';
import type { ComputedModelCalibration } from './thresholds';

export interface ThresholdsBarProps {
  calibration: ComputedModelCalibration;
  onDownload: () => void;
  onApplyPlayground: () => void;
}

export default function ThresholdsBar(props: ThresholdsBarProps) {
  const cal = () => props.calibration;
  const g = () => cal().payload.global;
  const count = () => cal().payload.per_class.length;

  return (
    <div class="thr-bar" data-model-thresholds>
      <div class="thr-info">
        <span class="thr-title">Calibrated thresholds</span>
        <span class="thr-meta num">
          global conf {formatConfThreshold(g().best_conf)} · macro-F1 {metric(cal().payload.macro_f1)} ·{' '}
          {count()} {count() === 1 ? 'class' : 'classes'}
        </span>
        <span class="thr-hint">
          Estimated from PR curves; run <span class="mono">folo val --save-thresholds</span> for exact 0.001-grid sweep.
        </span>
      </div>
      <div class="thr-actions">
        <Button size="sm" variant="ghost" icon={Download} data-download-thresholds onclick={props.onDownload}>
          Download thresholds.json
        </Button>
        <Button size="sm" icon={ScanSearch} data-apply-playground onclick={props.onApplyPlayground}>
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
