// Gallery twin of Remedy.svelte.
import Remedy from '$lib/ui/Remedy';

const report = 'adapter: Intel(R) UHD Graphics 630 (Vulkan, integrated)\nshader-f16: not supported by this adapter';

export default function RemedySection() {
  return (
    <div class="g-stack g-medium" data-parity-coarse data-parity-phone>
      <Remedy
        cause="Mixed precision needs `shader-f16`, which this GPU's driver does not offer."
        id="amp-why"
        shell="PowerShell"
        command="$env:WGPU_BACKEND='vulkan'; folo serve"
        report={report}
      />
      <Remedy cause="Set `FOLO_DATA` to a folder on a drive with room." command="export FOLO_DATA=/mnt/data/folo" compact />
      <Remedy cause="The server runs on the CPU only." report={null} />
      <div data-parity-step="open" data-parity-target="summary">
        <Remedy cause="No `vulkan` adapter." shell="bash" command="sudo apt install mesa-vulkan-drivers" report={report} compact />
      </div>
    </div>
  );
}
