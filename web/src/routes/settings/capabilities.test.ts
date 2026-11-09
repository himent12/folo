import { describe, expect, it } from 'vitest';
import type { Capabilities, Gpu } from '$lib/api/types';
import { capabilityRows, deviceSummary } from './capabilities';

const gpu: Gpu = { ordinal: 0, name: 'NVIDIA GeForce RTX 4080 SUPER', backend: 'vulkan', memory_bytes: 0, util: null };
const on: Capabilities = {
  device: 'gpu:0',
  shader_f16: true,
  tensor_cores: { available: true, enabled: true, reason: null },
  amp: { available: true, reason: null },
};
const offReason = 'off (not requested: FOLO_GPU_TENSOR_CORES=1 or --tensor-cores)';

describe('capabilityRows', () => {
  it('everything on: three marks, no server reasons', () => {
    const rows = capabilityRows(on);
    expect(rows.map((r) => [r.id, r.state, r.word])).toEqual([
      ['f16', 'on', 'Available'],
      ['tensor_cores', 'on', 'On'],
      ['amp', 'on', 'Available'],
    ]);
    expect(rows.every((r) => r.raw === null)).toBe(true);
  });

  it('tensor cores available but not requested: says how to turn them on, and why amp is out', () => {
    const rows = capabilityRows({
      ...on,
      tensor_cores: { available: true, enabled: false, reason: offReason },
      amp: { available: false, reason: `mixed-precision training (amp) needs ...: gpu:0 has tensor cores ${offReason}` },
    });
    const [, tc, amp] = rows;
    expect(tc).toMatchObject({ state: 'off', word: 'Off', raw: offReason });
    expect(tc!.note).toContain('`folo serve --tensor-cores`');
    expect(amp).toMatchObject({ state: 'none', word: 'Not available' });
    expect(amp!.note).toBe('Needs the tensor cores, which are off on this server.');
    expect(amp!.raw).toContain('mixed-precision');
  });

  it('no cooperative matrices: not available, in plain words; unknown reasons pass through as a sentence', () => {
    const none = capabilityRows({
      ...on,
      tensor_cores: { available: false, enabled: false, reason: 'off: RTX (Dx12) has no cooperative matrices in wgpu' },
      amp: { available: false, reason: 'x' },
    });
    expect(none[1]).toMatchObject({ state: 'none', word: 'Not available' });
    expect(none[1]!.note).toMatch(/no f16 cooperative matrices/);
    expect(none[2]!.note).toBe('Needs a GPU with tensor cores.');
    const odd = capabilityRows({ ...on, tensor_cores: { available: true, enabled: false, reason: 'off: something new' } });
    expect(odd[1]!.note).toBe('Off: something new.');
  });

  it('a CPU server and no f16 shaders', () => {
    const rows = capabilityRows({
      device: 'cpu',
      shader_f16: false,
      tensor_cores: { available: false, enabled: false, reason: 'off: no GPU' },
      amp: { available: false, reason: 'needs a GPU' },
    });
    expect(rows[0]).toMatchObject({ state: 'none', note: 'The CPU computes in f32, so f16 inference runs in f32.' });
    expect(rows[1]!.note).toMatch(/CPU/);
    expect(rows[2]!.note).toMatch(/trains on the CPU/);
  });
});

describe('deviceSummary', () => {
  it('names the adapter auto resolves to, or the CPU', () => {
    expect(deviceSummary(on, [gpu], 'Ryzen')).toBe('NVIDIA GeForce RTX 4080 SUPER · Vulkan');
    expect(deviceSummary({ device: 'cpu' }, [gpu], 'AMD Ryzen 9')).toBe('AMD Ryzen 9');
    expect(deviceSummary({ device: 'gpu:3' }, [gpu], '')).toBe('gpu:3');
  });
});
