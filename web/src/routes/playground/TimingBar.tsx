/**
 * Where the time of one prediction went: preprocess / inference / postprocess as a stacked bar in neutral shades
 * (the phases are not entities, so they get no hue; inference, the part that matters, is the darkest ink). Each
 * phase says what it covers (docs/API.md, docs/WASM.md): inference includes waiting for the device to finish.
 */
import { computed, css, For } from 'fw';
import type { Timing } from '$lib/api/types';
import { ms, num } from '$lib/format';
import type { WasmPrecision } from '$lib/inference/browser';

export default function TimingBar(props: {
  timing: Timing;
  device: string;
  cold?: boolean;
  mode?: string | null;
  engineBadge?: string | null;
  precision?: WasmPrecision | null;
  precisionFallback?: string | null;
}) {
  const parts = computed(() => [
    { key: 'pre', label: 'Preprocess', what: 'decode, resize', v: props.timing.preprocess_ms, color: 'var(--text-tertiary)' },
    { key: 'inf', label: 'Inference', what: 'the model', v: props.timing.inference_ms, color: 'var(--text-primary)' },
    { key: 'post', label: 'Postprocess', what: 'boxes, NMS', v: props.timing.postprocess_ms, color: 'var(--line-strong)' },
  ]);
  const sum = computed(() => parts.get().reduce((a, p) => a + p.v, 0) || 1);
  const total = computed(() => props.timing.total_ms || sum.get());
  const split = computed(() => ms(total.get()).split(' '));

  return (
    <div class="timing">
      <div class="head">
        <span class="total num">
          {split.get()[0]}
          <span class="unit">{split.get()[1]}</span>
        </span>
        {/* A first call's rate is the warm-up's, not the model's: none is given for it. */}
        {props.cold ? null : <span class="rate num">{num(1000 / total.get(), 0)} img/s</span>}
      </div>
      {props.cold ? <p class="cold">Includes loading the model: the next image runs faster.</p> : null}
      <div
        class="bar"
        role="img"
        aria-label={`Preprocess ${ms(props.timing.preprocess_ms)}, inference ${ms(props.timing.inference_ms)}, postprocess ${ms(props.timing.postprocess_ms)}`}
      >
        <For each={parts.get()} key={(p) => p.key}>
          {(p) => <span style:flex-grow={p().v / sum.get()} style:background={p().color}></span>}
        </For>
      </div>
      <ul class="legend">
        <For each={parts.get()} key={(p) => p.key}>
          {(p) => (
            <li>
              <span class="sw" style:background={p().color} aria-hidden="true"></span>
              <span class="name">
                {p().label} <span class="what">{p().what}</span>
              </span>
              <span class="v num">{ms(p().v)}</span>
              <span class="pct num">{Math.round((p().v / sum.get()) * 100)}%</span>
            </li>
          )}
        </For>
      </ul>
      <p class="device">
        <span class="faint">Device</span> <span class="mono">{props.device}</span>
      </p>
      {props.engineBadge ? (
        <p class="device engine" data-browser-engine-info>
          <span class="faint">Engine</span>
          <span class="engine-val">
            {props.precision === 'int8' ? (
              <span class="prec-pill" data-browser-precision="int8">
                INT8
              </span>
            ) : null}
            <span class="mono">{props.engineBadge}</span>
          </span>
        </p>
      ) : null}
      {props.precisionFallback ? (
        <p class="prec-fallback" data-precision-fallback>
          {props.precisionFallback}
        </p>
      ) : null}
      {props.mode ? (
        <p class="device mode">
          <span class="faint">Mode</span> <span class="mono">{props.mode}</span>
        </p>
      ) : null}
    </div>
  );
}

css`
  .timing {
    display: grid;
    gap: var(--space-3);
  }
  .cold {
    margin-top: calc(-1 * var(--space-2));
    color: var(--text-tertiary);
    font-size: var(--type-sm-size);
  }
  .head {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
  }
  .total {
    font-size: var(--type-2xl-size);
    line-height: var(--type-2xl-line);
    letter-spacing: var(--type-2xl-tracking);
    font-weight: var(--weight-semibold);
  }
  .unit {
    margin-left: 4px;
    font-size: var(--type-sm-size);
    font-weight: var(--weight-regular);
    color: var(--text-tertiary);
    letter-spacing: 0;
  }
  .rate {
    font-size: var(--type-sm-size);
    color: var(--text-secondary);
  }
  .bar {
    display: flex;
    gap: 2px;
    height: 8px;
    border-radius: var(--radius-full);
    overflow: hidden;
    background: var(--bg-active);
  }
  .bar span {
    flex-basis: 0;
    min-width: 3px;
  }
  .legend {
    display: grid;
    gap: 2px;
  }
  .legend li {
    display: grid;
    grid-template-columns: auto 1fr auto 36px;
    align-items: center;
    gap: var(--space-2);
    height: 22px;
    font-size: var(--type-sm-size);
  }
  .sw {
    width: 8px;
    height: 8px;
    border-radius: 2px;
  }
  .name {
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    color: var(--text-secondary);
  }
  .what {
    margin-left: var(--space-1);
    color: var(--text-tertiary);
  }
  .v {
    font-size: var(--type-sm-size);
    font-variant-numeric: tabular-nums;
  }
  .pct {
    text-align: right;
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
  }
  .device {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: var(--space-2);
    padding-top: var(--space-2);
    border-top: 1px solid var(--line-subtle);
    font-size: var(--type-sm-size);
  }
  .engine-val {
    display: inline-flex;
    align-items: center;
    gap: var(--space-2);
    min-width: 0;
  }
  .prec-pill {
    padding: 0 var(--space-2);
    border-radius: var(--radius-sm);
    background: var(--accent-muted);
    color: var(--accent-text);
    font-size: var(--type-sm-size);
    font-weight: var(--weight-medium);
    white-space: nowrap;
  }
  .prec-fallback {
    margin-top: calc(-1 * var(--space-2));
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--warning-text);
  }
`;
