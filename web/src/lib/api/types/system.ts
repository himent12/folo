// System: health, the machine, its GPUs and what it can do (docs/API.md, System).

// ---------- System ----------

export interface Health {
  status: 'ok';
  version: string;
  /** API generation, `v1`. */
  api: string;
  mock: boolean;
}

export type GpuBackend = 'vulkan' | 'dx12' | 'metal';

export interface Gpu {
  ordinal: number;
  name: string;
  backend: GpuBackend;
  memory_bytes: number;
  /** 0..1, or null when the driver does not report it. */
  util: number | null;
}

export interface SystemInfo {
  cpu: { name: string; threads: number; util: number };
  memory: { used_bytes: number; total_bytes: number };
  gpus: Gpu[];
  /** The real server's data dir (`/` separators); absent in mock mode. */
  data_dir?: string;
  /** Optional API features the server implements, e.g. `export_onnx_switches`; empty or absent from older ones. */
  features?: string[];
  /** What `device: auto` runs can use (f16, tensor cores, amp); absent from older servers. */
  capabilities?: Capabilities;
  /** FOLO's GPU buffer pool, once the server has used the GPU; absent otherwise and from older servers. */
  gpu_pool?: GpuPool;
}

/** GPU memory held by FOLO's buffer pool (released after a run and when inference is idle). */
export interface GpuPool {
  device: string;
  live_bytes: number;
  cached_bytes: number;
  peak_live_bytes: number;
}

/** `/api/system`'s `capabilities` (docs/API.md). A `reason` is null exactly when the feature is usable. */
export interface Capabilities {
  /** What `device: auto` resolves to: `gpu:0` or `cpu`. */
  device: string;
  /** f16 inference runs in half precision (else it falls back to f32). */
  shader_f16: boolean;
  tensor_cores: { available: boolean; enabled: boolean; reason: string | null };
  /** Mixed-precision training (`TrainConfig.amp`). */
  amp: { available: boolean; reason: string | null };
}
