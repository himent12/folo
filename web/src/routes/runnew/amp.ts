// Why mixed precision cannot be switched on for a device (New run's Mixed precision field), and how to enable it.
import type { Capabilities, SystemInfo } from '$lib/api/types';
import { VULKAN } from '../settings/capabilities';

// Each reason is the field's one sentence (`why`), the same before and after a refusal: once the capabilities are read
// again the switch is off, so the refusal is answered and the field says only why and how to turn it back on.
export type AmpBlock = { why: string; shell?: string; command?: string; raw?: string };

/** Why mixed precision cannot be switched on for `device` (RunNew's Mixed precision field); null where it can. */
export function ampBlockOf(caps: Capabilities | undefined, device: string, onCloud: boolean, system: SystemInfo | undefined): AmpBlock | null {
  // A cloud run's mixed precision is the worker's to check when the instance boots, not this machine's.
  if (!caps || onCloud) return null;
  if (device === 'cpu') return { why: 'The CPU has no tensor cores. Pick a GPU to train in mixed precision.' };
  // Auto on a server without a usable GPU is the CPU too: no GPU advice.
  if (device === 'auto' && caps.device === 'cpu') return { why: 'This server has no usable GPU, so runs train on the CPU in f32.' };
  // `capabilities` describe the device `auto` resolves to; another GPU is left to the server.
  if (caps.amp.available || (device !== 'auto' && device !== caps.device)) return null;
  const raw = caps.amp.reason ?? caps.tensor_cores.reason ?? undefined;
  const t = caps.tensor_cores;
  if (t.available && !t.enabled) {
    if (/self-test/i.test(t.reason ?? '')) return { why: "The server's tensor-core self-test failed. Train in f32 on this GPU.", raw };
    return {
      why: 'Tensor cores are off on this server; restart it with them on:',
      command: 'folo serve --tensor-cores',
      raw,
    };
  }
  if (!t.available) {
    // docs/GPU.md: wgpu has cooperative matrices on Vulkan only (NVIDIA tensor cores through SPV_KHR_cooperative_matrix;
    // an RTX 4080 SUPER has them there), none on DX12 or Metal. So only an NVIDIA RTX card seen through DX12 has a way.
    const g = system?.gpus.find((x) => `gpu:${x.ordinal}` === caps.device);
    if (g?.backend === 'dx12' && /nvidia/i.test(g.name) && /\brtx\b/i.test(g.name))
      return {
        why: 'DirectX 12 does not reach this GPU’s tensor cores; restart the server on Vulkan.',
        shell: 'PowerShell',
        command: `${VULKAN} --tensor-cores`,
        raw,
      };
    return { why: 'This GPU has no tensor cores FOLO can use. Train in f32 on this GPU.', raw };
  }
  return { why: 'Not available on this server. Train in f32 on this GPU.', raw };
}
