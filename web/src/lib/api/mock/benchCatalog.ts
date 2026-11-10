// The mock's vision-benchmark catalog (docs/BENCH.md, Conditions and The score): the twelve conditions, the default
// plan and the five axes, worded as docs/BENCH.md's table (the server's catalog) words them. How hard each condition is for a typical detector is
// the mock's own (`HARDNESS`): what its results are generated from.

import type { BenchAxisInfo, BenchCatalog, BenchCondition, BenchConditionGroup } from '../types';
import { AXES, AXIS_NAME, AXIS_WEIGHT, MAX_SEVERITY } from '../../bench/rules';

const c = (id: string, name: string, group: BenchConditionGroup, description: string): BenchCondition => ({ id, name, group, description });

export const CONDITIONS: BenchCondition[] = [
  c('low_light', 'Low light', 'light', 'Darkens the scene as a short exposure does and adds the sensor noise that comes with it: dusk, indoors, night.'),
  c('overexposure', 'Overexposure', 'light', 'Brightens the scene as a long exposure does, clipping highlights to white: glare, a bright sky, a flash too close.'),
  c('low_contrast', 'Low contrast', 'light', "Pulls every pixel toward the image's mean grey: flat light, an overcast day, a washed-out screen."),
  c('defocus', 'Defocus', 'blur', 'Blurs the image as a lens out of focus does.'),
  c('motion_blur', 'Motion blur', 'blur', 'Smears the image along one direction, as a moving camera or subject does during the exposure.'),
  c('sensor_noise', 'Sensor noise', 'noise', 'Adds grain to every pixel and channel, as a small or hot sensor does.'),
  c('haze', 'Haze', 'medium', 'Scatters light: blends the scene toward a bright veil and softens detail, as fog, smoke or murky water do.'),
  c('color_cast', 'Colour cast', 'medium', 'Absorbs red and tints the scene blue-green, as water does with depth (or tinted light).'),
  c('jpeg', 'JPEG compression', 'digital', 'Re-encodes the image as a low-quality JPEG, with the blocks and smeared colour of strong compression.'),
  c('low_resolution', 'Low resolution', 'digital', 'Downscales the image and scales it back up, so fine detail is lost: a cheap camera, a digital zoom.'),
  c('distance', 'Distance', 'geometry', 'Shrinks the whole scene onto a canvas of the same size, so every object is smaller, as if seen from farther away; the true boxes move with it.'),
  c('occlusion', 'Occlusion', 'geometry', 'Covers part of every object with a flat patch, as something in front of it would; the true boxes stay.'),
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
