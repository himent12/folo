// The mock's vision-benchmark catalog (docs/BENCH.md, Conditions and The score): the twelve conditions, the default
// plan and the five axes, worded as a server's catalog would be. How hard each condition is for a typical detector is
// the mock's own (`HARDNESS`): what its results are generated from.

import type { BenchAxisInfo, BenchCatalog, BenchCondition, BenchConditionGroup } from '../types';
import { AXES, AXIS_NAME, AXIS_WEIGHT, MAX_SEVERITY } from '../../bench/rules';

const c = (id: string, name: string, group: BenchConditionGroup, description: string): BenchCondition => ({ id, name, group, description });

export const CONDITIONS: BenchCondition[] = [
  c('low_light', 'Low light', 'light', 'Darker, with the grain a camera adds when it pushes its gain: dusk, night, a dim room.'),
  c('overexposure', 'Overexposure', 'light', 'Brighter until highlights clip: direct sun, a flash, a window behind the scene.'),
  c('low_contrast', 'Low contrast', 'light', 'Tones pulled towards grey: flat light, a dull overcast day, a cheap sensor.'),
  c('defocus', 'Defocus', 'blur', 'A lens out of focus: the whole image softens evenly.'),
  c('motion_blur', 'Motion blur', 'blur', 'Streaks along the movement: a moving camera, a conveyor, a long exposure.'),
  c('sensor_noise', 'Sensor noise', 'noise', 'Grain in every pixel, as a small sensor gives in poor light.'),
  c('haze', 'Haze', 'medium', 'A veil that washes out detail: fog, smoke, murky water.'),
  c('color_cast', 'Colour cast', 'medium', 'A tint over the whole image: green water, sodium lamps, a wrong white balance.'),
  c('jpeg', 'JPEG artifacts', 'digital', 'Blocks and ringing from heavy compression: a low-bitrate stream, a photo saved many times.'),
  c('low_resolution', 'Low resolution', 'digital', 'Fewer pixels, scaled back up: a cheap or digitally zoomed camera.'),
  c('distance', 'Distance', 'geometry', 'Objects farther away: the scene shrinks towards its centre, and the boxes with it.'),
  c('occlusion', 'Occlusion', 'geometry', 'Parts of the objects hidden behind something: leaves, a hand, another object.'),
];

/** What share of a typical detector's accuracy each condition takes away at severity 5 (the mock's results). */
export const HARDNESS: Record<string, number> = {
  low_light: 0.46,
  overexposure: 0.3,
  low_contrast: 0.34,
  defocus: 0.52,
  motion_blur: 0.5,
  sensor_noise: 0.42,
  haze: 0.5,
  color_cast: 0.22,
  jpeg: 0.26,
  low_resolution: 0.44,
  distance: 0.62,
  occlusion: 0.48,
};

const AXIS_DESCRIPTION: Record<(typeof AXES)[number], string> = {
  accuracy: 'mAP@0.5:0.95 on the unchanged images.',
  robustness: "The weighted mean, over the plan's conditions, of the accuracy kept.",
  self_labeling: 'The share of all objects it labels by itself while at least 9 in 10 of its labels are right.',
  calibration: 'One minus the expected calibration error of its confidences.',
  localization: 'The mean IoU of its correct detections, rescaled from 0.5–1 to 0–1.',
};

export const AXIS_INFO: BenchAxisInfo[] = AXES.map((axis) => ({ axis, name: AXIS_NAME[axis], weight: AXIS_WEIGHT[axis], description: AXIS_DESCRIPTION[axis] }));

/** `GET /api/bench/catalog`. */
export function catalog(): BenchCatalog {
  return {
    conditions: CONDITIONS.map((x) => ({ ...x })),
    severities: MAX_SEVERITY,
    default_plan: { conditions: CONDITIONS.map((x) => ({ condition: x.id, severities: [1, 3, 5], weight: 1, why: null })), rationale: null, designed_by: null },
    axes: AXIS_INFO.map((a) => ({ ...a })),
  };
}

export const conditionName = (id: string) => CONDITIONS.find((x) => x.id === id)?.name ?? id;
