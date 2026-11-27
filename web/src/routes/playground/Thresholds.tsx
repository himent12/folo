/**
 * The thresholds' sliders in their three homes: the rail's Thresholds section (from 1024 px; each explanation a
 * tooltip), the Thresholds panel's Overlap and Max detections (below 1024 px; each explanation a hint under it), and
 * the phone sheet's or the tablet popover's body.
 */
import { bindable, css } from 'fw';
import Slider from '$lib/ui/Slider';
import { tooltip } from '$lib/ui/tooltip';
import { DEFAULTS } from './predict';

const NMS_TIP = 'Non-max suppression: a lower-scoring box of the same class is dropped when it overlaps a kept one by more than this.';
export const CONF_HINT = 'Hide boxes scored below this.';
const IOU_HINT = 'Suppress same-class overlaps above this.';
const MAX_HINT = 'Keep at most this many boxes.';
const round = (v: number) => String(Math.round(v));

interface SliderValues {
  conf?: number;
  iou?: number;
  maxDet?: number;
}

/** Overlap and Max detections, with their hints (the Thresholds panel), or inline (a sheet's rows). */
export function SecondarySliders(props: SliderValues & { inline: boolean; o2o?: boolean }) {
  const iou = bindable(props, 'iou', DEFAULTS.iou);
  const maxDet = bindable(props, 'maxDet', DEFAULTS.maxDet);
  return (
    <>
      {props.o2o ? (
        <p class="no-nms" class:inline={props.inline}>
          <span>Overlap (IoU)</span>
          <span class="faint">Not used · end-to-end model</span>
        </p>
      ) : (
        <div title={NMS_TIP}>
          <Slider inline={props.inline} label="Overlap (IoU)" bind:value={iou} min={0.05} max={0.95} step={0.05} hint={IOU_HINT} />
        </div>
      )}
      <Slider inline={props.inline} label="Max detections" bind:value={maxDet} min={1} max={300} step={1} format={round} hint={MAX_HINT} />
    </>
  );
}

/**
 * The rail's thresholds: each explanation a tooltip on its slider. An end-to-end model has no NMS: its Overlap row
 * says so instead of offering a control that changes nothing.
 */
export function RailThresholds(props: SliderValues & { o2o: boolean }) {
  const conf = bindable(props, 'conf', DEFAULTS.conf);
  const iou = bindable(props, 'iou', DEFAULTS.iou);
  const maxDet = bindable(props, 'maxDet', DEFAULTS.maxDet);
  return (
    <div class="r-sliders">
      <span use:tooltip={CONF_HINT}>
        <Slider label="Confidence" bind:value={conf} min={0.01} max={0.95} step={0.01} />
      </span>
      {props.o2o ? (
        <p class="no-nms">
          <span>Overlap (IoU)</span>
          <span class="faint">Not used · end-to-end model</span>
        </p>
      ) : (
        <span use:tooltip={`${IOU_HINT} ${NMS_TIP}`}>
          <Slider label="Overlap (IoU)" bind:value={iou} min={0.05} max={0.95} step={0.05} />
        </span>
      )}
      <span use:tooltip={MAX_HINT}>
        <Slider label="Max detections" bind:value={maxDet} min={1} max={300} step={1} format={round} />
      </span>
    </div>
  );
}

/** The phone sheet's every threshold, or the tablet popover's Max detections; then inference options and what is drawn on the image. */
export function MoreBody(props: SliderValues & { tablet: boolean; o2o?: boolean; toggles: () => JSX.Element; inferenceOptions?: (() => JSX.Element) | null }) {
  const conf = bindable(props, 'conf', DEFAULTS.conf);
  const iou = bindable(props, 'iou', DEFAULTS.iou);
  const maxDet = bindable(props, 'maxDet', DEFAULTS.maxDet);
  return (
    <div class="sheet-body">
      <div class="more-sliders">
        {props.tablet ? (
          <>
            {props.o2o ? (
              <p class="no-nms inline">
                <span>Overlap (IoU)</span>
                <span class="faint">Not used · end-to-end model</span>
              </p>
            ) : null}
            <Slider inline label="Max detections" bind:value={maxDet} min={1} max={300} step={1} format={round} />
          </>
        ) : (
          <>
            <Slider inline label="Confidence" bind:value={conf} min={0.01} max={0.95} step={0.01} />
            <SecondarySliders inline o2o={props.o2o} bind:iou={iou} bind:maxDet={maxDet} />
          </>
        )}
      </div>
      {props.inferenceOptions ? props.inferenceOptions() : null}
      <div class="more-toggles">{props.toggles()}</div>
    </div>
  );
}

css`
  .r-sliders {
    display: grid;
    gap: var(--space-4);
  }
  .r-sliders > span {
    display: block;
  }
  /* Threshold values are numbers, not code: sans with tabular numerals. */
  .r-sliders :global(output.val) {
    font-family: var(--font-sans);
    font-variant-numeric: tabular-nums;
  }
  .no-nms {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: var(--space-2);
    font-size: var(--type-sm-size);
    color: var(--text-secondary);
  }
  .no-nms.inline {
    align-items: center;
    min-height: 44px;
  }
  .sheet-body {
    overflow-y: auto;
    overscroll-behavior: contain;
    padding: 0 var(--space-4) var(--space-3);
  }
  /* One label column and one value column: every track (the pinned Confidence, the sheet's three) starts and ends at
     the same x. */
  .more-sliders :global(.slider.inline) {
    grid-template-columns: 112px minmax(0, 1fr) 40px;
    min-height: 44px;
  }
  .more-toggles {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 2px;
    min-height: 44px;
    margin-left: calc(-1 * var(--space-2));
  }
`;
