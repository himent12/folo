// Gallery twin of ErrorState.svelte.
import { ApiError } from '$lib/api/client';
import Button from '$lib/ui/Button';
import ErrorState from '$lib/ui/ErrorState';

const offline = new ApiError(0, 'network', 'Failed to fetch');
const oom = new ApiError(507, 'gpu_out_of_memory', 'the GPU ran out of memory (wgpu backend: Out of Memory)', 'lower `batch` or `imgsz`, or use a smaller `arch` (folo-n, folo-s)');
const device = new ApiError(404, 'device_not_found', 'device gpu:3 is not available', 'this machine has gpu:0, cpu; use device `auto` or `cpu`');
const bare = new ApiError(500, 'internal', 'boom', null);
const plain = new Error('The labels file is not valid JSON at line 3.');
const retry = () => {};

export default function ErrorStateSection() {
  return (
    <div class="g-stack g-medium" data-parity-coarse data-parity-phone>
      <ErrorState error={offline} retry={retry} />
      <ErrorState
        error={oom}
        title="Prediction failed"
        actions={() => (
          <Button size="sm" variant="primary">
            Run in this browser
          </Button>
        )}
      />
      <div data-parity-step="open" data-parity-target="summary">
        <ErrorState error={oom} title="Prediction failed" hint="Run it in this browser instead." retry={retry} />
      </div>
      <ErrorState error={device} title="Could not start" compact />
      <ErrorState error={bare} />
      <ErrorState error={plain} title="Could not read the labels" retry={retry} compact />
      <ErrorState error="a string error" />
    </div>
  );
}
