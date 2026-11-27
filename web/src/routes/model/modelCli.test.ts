import { describe, expect, it } from 'vitest';
import type { Model } from '$lib/api/types';
import { buildExportCli, buildPredictCli, buildQuantizeCli, onnxSwitchesForRequest } from './modelCli';

const baseModel: Model = {
  id: 'm_fast_n',
  name: 'fasteners-n-best',
  arch: 'folo-n',
  imgsz: 640,
  classes: ['bolt', 'nut'],
  params: 2_624_064,
  flops_g: 6.5,
  bytes: 5_400_000,
  created: '2026-11-01T00:00:00Z',
  run_id: 'r_1',
  metrics: null,
  path: 'D:/folo/runs/fasteners-n/weights/best.folo',
};

describe('modelCli', () => {
  it('builds accurate folo export commands for folo-f16 and all ONNX dynamic axis modes', () => {
    expect(buildExportCli(baseModel, { format: 'folo-f16', decode: true, half: true, axes: 'full' })).toEqual({
      line: 'folo export --model D:/folo/runs/fasteners-n/weights/best.folo --format folo-f16',
      placeholder: null,
    });

    expect(buildExportCli(baseModel, { format: 'onnx', decode: false, half: false, axes: 'static' }).line).toBe(
      'folo export --model D:/folo/runs/fasteners-n/weights/best.folo --format onnx',
    );
    expect(buildExportCli(baseModel, { format: 'onnx', decode: true, half: false, axes: 'batch' }).line).toBe(
      'folo export --model D:/folo/runs/fasteners-n/weights/best.folo --format onnx --decode --dynamic-batch',
    );
    expect(buildExportCli(baseModel, { format: 'onnx', decode: false, half: true, axes: 'hw' }).line).toBe(
      'folo export --model D:/folo/runs/fasteners-n/weights/best.folo --format onnx --dynamic-hw --half',
    );
    expect(buildExportCli(baseModel, { format: 'onnx', decode: true, half: true, axes: 'full' }).line).toBe(
      'folo export --model D:/folo/runs/fasteners-n/weights/best.folo --format onnx --decode --dynamic --half',
    );
  });

  it('uses published weights names without placeholders and quotes spaces safely', () => {
    const published: Model = { ...baseModel, id: 'folo-n-coco-v5', name: 'folo-n-coco-v5', path: undefined, source: 'published' };
    expect(buildExportCli(published, { format: 'onnx', decode: true, half: false, axes: 'dynamic' as never })).toMatchObject({
      placeholder: null,
    });
    expect(buildExportCli(published, { format: 'onnx', decode: true, half: false, axes: 'batch' })).toEqual({
      line: 'folo export --model folo-n-coco-v5 --format onnx --decode --dynamic-batch',
      placeholder: null,
    });

    const spaced: Model = { ...baseModel, path: 'D:/my models/best.folo' };
    expect(buildExportCli(spaced, { format: 'folo-f16', decode: false, half: false, axes: 'static' }).line).toBe(
      "folo export --model 'D:/my models/best.folo' --format folo-f16",
    );

    const noPath: Model = { ...baseModel, path: undefined };
    expect(buildExportCli(noPath, { format: 'folo-f16', decode: false, half: false, axes: 'static' })).toEqual({
      line: 'folo export --model <path to fasteners-n-best.folo> --format folo-f16',
      placeholder: 'fasteners-n-best.folo',
    });
  });

  it('builds real folo predict snippets using verified CLI flags (--tile, --tta, --track)', () => {
    expect(buildPredictCli(baseModel, 'standard').line).toBe(
      'folo predict --model D:/folo/runs/fasteners-n/weights/best.folo <images>',
    );
    expect(buildPredictCli(baseModel, 'sahi').line).toBe(
      'folo predict --model D:/folo/runs/fasteners-n/weights/best.folo <images> --tile 640 --tile-overlap 0.2',
    );
    expect(buildPredictCli(baseModel, 'tta').line).toBe(
      'folo predict --model D:/folo/runs/fasteners-n/weights/best.folo <images> --tta',
    );
    expect(buildPredictCli(baseModel, 'track').line).toBe(
      'folo predict --model D:/folo/runs/fasteners-n/weights/best.folo <frames_dir> --track',
    );
  });

  it('builds real folo quantize snippets matching selective-f32 sensitivity flags', () => {
    expect(buildQuantizeCli(baseModel, 'auto')).toEqual({
      line: 'folo quantize --model D:/folo/runs/fasteners-n/weights/best.folo --data <dataset> --f32-policy auto --max-f32-macs 0.05 --sqnr-threshold 35',
      placeholder: null,
    });
    expect(buildQuantizeCli(baseModel, 'accurate').line).toBe(
      'folo quantize --model D:/folo/runs/fasteners-n/weights/best.folo --data <dataset> --f32-policy auto --max-f32-macs 0.10 --sqnr-threshold 40',
    );
    // Auto-only tuning flags must never be emitted when --f32-policy != auto.
    expect(buildQuantizeCli(baseModel, 'p5-tail').line).toBe(
      'folo quantize --model D:/folo/runs/fasteners-n/weights/best.folo --data <dataset> --f32-policy p5-tail',
    );
    expect(buildQuantizeCli(baseModel, 'none').line).toBe(
      'folo quantize --model D:/folo/runs/fasteners-n/weights/best.folo --data <dataset> --f32-policy none',
    );

    const noPath: Model = { ...baseModel, path: undefined };
    expect(buildQuantizeCli(noPath, 'auto')).toEqual({
      line: 'folo quantize --model <path to fasteners-n-best.folo> --data <dataset> --f32-policy auto --max-f32-macs 0.05 --sqnr-threshold 35',
      placeholder: 'fasteners-n-best.folo',
    });
  });

  it('only emits dynamic_batch / dynamic_hw in API payloads when supportsSplitAxes is true', () => {
    expect(onnxSwitchesForRequest({ decode: true, half: false, axes: 'batch', supportsSplitAxes: true })).toEqual({
      decode: true,
      dynamic: false,
      dynamic_batch: true,
      dynamic_hw: false,
      half: false,
    });
    expect(onnxSwitchesForRequest({ decode: false, half: true, axes: 'hw', supportsSplitAxes: true })).toEqual({
      decode: false,
      dynamic: false,
      dynamic_batch: false,
      dynamic_hw: true,
      half: true,
    });
    expect(onnxSwitchesForRequest({ decode: false, half: false, axes: 'batch', supportsSplitAxes: false })).toEqual({
      decode: false,
      dynamic: false,
      half: false,
    });
    expect(onnxSwitchesForRequest({ decode: false, half: false, axes: 'full', supportsSplitAxes: true })).toEqual({
      decode: false,
      dynamic: true,
      half: false,
    });
  });
});
