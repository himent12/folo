import type { Arch, Augment, Capabilities, Device, Optimizer, TrainConfig } from '$lib/api/types';
import type { CloudPick } from './cloudPick';

/** The run the form describes, as the request (`POST /api/runs`) the estimate, the CLI command and Start send. */
export function trainConfig(
  name: string,
  datasetId: string,
  arch: Arch,
  epochs: number,
  batchN: number,
  imgszN: number,
  onCloud: boolean,
  device: string,
  optimizer: Optimizer,
  lr0: number,
  weights: string,
  augment: Augment,
  seed: number,
  caps: Capabilities | undefined,
  ampOn: boolean,
  e2e: boolean,
  cloudPick: CloudPick,
): TrainConfig {
  return {
    name: name.trim(),
    dataset_id: datasetId,
    arch,
    epochs,
    batch: batchN,
    imgsz: imgszN,
    // A cloud run's worker picks its own GPU (docs/API.md, Cloud runs).
    device: onCloud ? 'auto' : (device as Device),
    optimizer,
    lr0,
    weights: weights === 'none' ? null : weights,
    augment: { ...augment, hsv: [...augment.hsv] as [number, number, number], close_mosaic: Math.min(augment.close_mosaic, epochs) },
    seed,
    // Only to servers that report capabilities (older ones predate `amp`), or train it in the cloud.
    ...(caps || onCloud ? { amp: ampOn } : {}),
    // Left out for the default (missing means `o2m`, docs/API.md), so older servers see the request they know.
    ...(e2e ? { head: 'o2o' as const } : {}),
    // Only once a connection and its GPU are chosen, and only to servers with the feature.
    ...(onCloud && cloudPick.compute ? { compute: cloudPick.compute } : {}),
  };
}
