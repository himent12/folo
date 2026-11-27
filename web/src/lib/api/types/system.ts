// System: health, the machine, its GPUs and what it can do (docs/API.md, System).

// ---------- System ----------

export interface Health {
  status: 'ok';
  version: string;
  /** API generation, `v1`. */
  api: string;
  mock: boolean;
  /** The optional features `/api/system` lists, without its hardware probe; absent from older servers. */
  features?: string[];
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

/** CPU SIMD ISA availability and active int8 (`q8`) micro-kernel tier on this host (`/api/system`'s `cpu_simd`). */
export interface CpuSimd {
  /** x86-64 AVX2 instruction set detected. */
  avx2: boolean;
  /** x86-64 AVX-VNNI (256-bit VEX `vpdpbusd`) detected. */
  avx_vnni: boolean;
  /** x86-64 AVX-512 VNNI (`avx512vnni`) detected. */
  avx512_vnni: boolean;
  /** ARM64 NEON instruction set detected. */
  neon: boolean;
  /** True when the active int8 micro-kernel uses hardware VNNI (`vpdpbusd` via AVX-512 VNNI or AVX-VNNI). */
  int8_fast: boolean;
  /** Human-readable active CPU int8 micro-kernel tier (e.g. `"AVX-512 VNNI (512-bit)"`, `"AVX-VNNI"`, `"AVX2"`, `"portable"`). */
  int8_kernel: string;
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
  /** CPU SIMD ISA flags and active int8 micro-kernel tier; absent from older servers. */
  cpu_simd?: CpuSimd;
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
