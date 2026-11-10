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
    expect(rows.every((r) => r.raw === null && r.fix === null)).toBe(true);
  });

  it('tensor cores available but not requested: one sentence, the command that turns them on, and why amp is out', () => {
    const rows = capabilityRows({
      ...on,
      tensor_cores: { available: true, enabled: false, reason: offReason },
      amp: { available: false, reason: `mixed-precision training (amp) needs ...: gpu:0 has tensor cores ${offReason}` },
    });
    const [, tc, amp] = rows;
    expect(tc).toMatchObject({ state: 'off', word: 'Off', raw: offReason });
    expect(tc!.fix).toEqual({ say: 'Restart the server with them on:', command: 'folo serve --tensor-cores' });
    expect(amp).toMatchObject({ state: 'none', word: 'Not available' });
    // The command shows once, under Tensor cores; Mixed precision points to it.
    expect(amp!.note).toContain('See the command under Tensor cores.');
    expect(amp!.fix).toBeNull();
    expect(amp!.raw).toContain('mixed-precision');
  });

  it('no cooperative matrices: not available, in plain words; Vulkan only where DirectX 12 is the cause', () => {
    const none = capabilityRows({
      ...on,
      tensor_cores: { available: false, enabled: false, reason: 'off: RTX (Dx12) has no cooperative matrices in wgpu' },
      amp: { available: false, reason: 'x' },
    });
    expect(none[1]).toMatchObject({ state: 'none', word: 'Not available' });
    // The command whole, for one unwrapped line; the shell named apart from it.
    expect(none[1]!.fix).toEqual({ say: 'Restart the server on Vulkan to use them.', shell: 'PowerShell', command: "$env:FOLO_GPU_BACKEND='vulkan'; folo serve --tensor-cores" });
    // Mixed precision ends on the way out the row above shows, never on a dead end (DESIGN_DIRECTION §16).
    expect(none[2]!.note).toContain('See the command under Tensor cores.');
    expect(none[2]!.fix).toBeNull();
    // Another adapter without them (Metal, an older GPU): no action docs/GPU.md would back.
    const metal = capabilityRows({ ...on, tensor_cores: { available: false, enabled: false, reason: 'off: no cooperative matrices' } }, [{ ...gpu, backend: 'metal' }]);
    expect(metal[1]!.fix).toBeNull();
    // No command above, so nothing to point to.
    const metalAmp = capabilityRows(
      { ...on, tensor_cores: { available: false, enabled: false, reason: 'off: no cooperative matrices' }, amp: { available: false, reason: 'x' } },
      [{ ...gpu, backend: 'metal' }],
    );
    expect(metalAmp[2]!.note).not.toContain('See the command');
    expect(metalAmp[2]!.fix).toBeNull();
    // f16 shaders missing on DirectX 12 (docs/GPU.md: it lacks shader-f16): Vulkan, by the adapter's backend.
    const dx = capabilityRows({ ...on, shader_f16: false }, [{ ...gpu, backend: 'dx12' }]);
    expect(dx[0]!.fix).toEqual({ say: 'Restart the server on Vulkan to compute in f16.', shell: 'PowerShell', command: "$env:FOLO_GPU_BACKEND='vulkan'; folo serve" });
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
    expect(rows[0]).toMatchObject({ state: 'none', fix: null });
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

  it('after a failed tensor-core self-test, mixed precision points at the diagnosis under Tensor cores, not at --tensor-cores', () => {
    const reason = 'off: the tensor-core self-test failed (results differ from the standard kernels)';
    const rows = capabilityRows({
      device: 'gpu:0',
      shader_f16: true,
      tensor_cores: { available: true, enabled: false, reason },
      amp: { available: false, reason: `mixed-precision training (amp) needs a GPU with tensor cores: gpu:0 has tensor cores ${reason}` },
    });
    expect(rows[1]!.fix).toEqual({ say: 'To see why, run:', command: 'folo doctor --tensor-cores' });
    expect(rows[2]!.fix).toBeNull();
    expect(rows[2]!.note).toContain('self-test failed');
    expect(rows[2]!.note).toContain('See the command under Tensor cores.');
  });
});
