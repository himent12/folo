// The mock's scenarios (`?scenario=`, listed in ./index) and what the server reports in them: the capabilities, and
// how long the `stale` server answers.

import type { Capabilities } from '../types';
import { GPU_NAME } from './runs';

export type Scenario =
  | 'normal'
  | 'empty'
  | 'error'
  | 'loading'
  | 'oom'
  | 'gpu-unplugged'
  | 'short-run'
  | 'tensor-cores-off'
  | 'no-tensor-cores'
  | 'finishing'
  // A finished run on fasteners-mini, which has no validation images: it validated on its training split.
  | 'train-split'
  | 'no-throughput'
  | 'amp-refused'
  // The server answers the first loads, then stops (STALE_AFTER): pages keep what they have and say it is stale, and
  // the live run's stream drops too, so its card shows the last known state.
  | 'stale'
  // The same, but only the requests stop answering: the live run's stream keeps streaming, so its card stays live
  // and the page says the lists alone are not refreshing.
  | 'stale-lists'
  // A Studio build without folo-wasm: /api/system does not list `web_wasm`, so Playground's browser choices are off.
  | 'no-wasm'
  // `folo serve --mock`: no agent labeling (the feature is not listed; its endpoints answer 501 not_implemented).
  | 'no-labeling'
  // Both labeling agents installed (Codex is not, elsewhere).
  | 'all-agents'
  // Vision benchmarks: a server without them (501 not_implemented, no `vision_bench`); a design whose answer is no plan.
  | 'no-bench'
  | 'bench-agent-fails'
  // A loop at generation 0: its first teacher is being benchmarked (its entry queued behind the running one).
  | 'bench-loop-gen0'
  // The fresh benchmark with one failed entry: its plan still open (a failed entry graded nothing).
  | 'bench-failed-only'
  // Cloud compute: no connection; no `cloud` feature; a revoked key and an unconfirmed termination; a cloud run
  // training; one uploading its dataset.
  | 'cloud-none'
  | 'cloud-off'
  | 'cloud-trouble'
  | 'cloud-training'
  | 'cloud-upload';

/** The `stale` scenario: how long the server answers before it stops. */
export const STALE_AFTER = 3000;

/** folo_gpu::context::TENSOR_CORES_NOT_REQUESTED. */
const TC_NOT_REQUESTED = 'off (not requested: FOLO_GPU_TENSOR_CORES=1 or --tensor-cores)';

/**
 * `/api/system`'s `capabilities` as folo-server's mock reports them (`MockTensorCores`), worded as the real server's:
 * tensor cores on and `amp` available; `tensor-cores-off`, available but not requested; `no-tensor-cores`, the adapter
 * through DX12, where wgpu has no cooperative matrices.
 */
export function capabilities(scenario: Scenario): Capabilities {
  const status = scenario === 'tensor-cores-off' ? TC_NOT_REQUESTED : scenario === 'no-tensor-cores' ? `off: ${GPU_NAME} (Dx12) has no cooperative matrices in wgpu` : null;
  return {
    device: 'gpu:0',
    shader_f16: true,
    tensor_cores: { available: scenario !== 'no-tensor-cores', enabled: status === null, reason: status },
    amp:
      status === null
        ? { available: true, reason: null }
        : {
            available: false,
            // folo_vision::train::amp_not_supported
            reason:
              `mixed-precision training (amp) needs a GPU with tensor cores (f16 cooperative matrices): gpu:0 has tensor cores ${status}; ` +
              'train without amp (f32), or on such a GPU (`folo train --amp` asks for them; start `folo serve` or `folo mcp` with `--tensor-cores`)',
          },
  };
}
