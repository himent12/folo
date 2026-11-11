// Gallery twin of DriverReport.svelte.
import DriverReport from '$lib/ui/DriverReport';

const report =
  'adapter: NVIDIA GeForce RTX 4080 SUPER (Vulkan, discrete)\n\nshader-f16: not supported by this adapter (VK_KHR_shader_float16_int8 missing from the device extensions the driver reports)\r\nsubgroups: supported, size 32\n   \ncooperative matrices: not enabled';

export default function DriverReportSection() {
  return (
    <div class="g-stack g-medium" data-parity-coarse data-parity-phone>
      <DriverReport report={report} />
      <div data-parity-step="open" data-parity-target="summary">
        <DriverReport report={report} />
      </div>
      <DriverReport report="one line only" />
    </div>
  );
}
