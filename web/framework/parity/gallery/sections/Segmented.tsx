// Gallery twin of Segmented.svelte.
import Monitor from '$icons/monitor';
import Sun from '$icons/sun';
import Moon from '$icons/moon';
import { signal } from 'fw';
import Segmented from '$lib/ui/Segmented';

export default function SegmentedSection() {
  const runOn = signal<'server' | 'webgpu' | 'cpu'>('server');
  const theme = signal<'system' | 'light' | 'dark'>('dark');
  const range = signal<'10' | 'all'>('all');
  const pick = signal<'a' | 'b' | 'c'>('a');
  const keys = signal<'a' | 'b' | 'c'>('a');
  return (
      <div class="g-col" data-parity-coarse>
        <Segmented
          label="Run the model on"
          bind:value={runOn}
          options={[
            { value: 'server', label: 'Server', meta: '2' },
            { value: 'webgpu', label: 'WebGPU', disabled: true, tip: 'No WebGPU in this browser' },
            { value: 'cpu', label: 'CPU' },
          ]}
        />
        <Segmented
          label="Theme"
          size="sm"
          bind:value={theme}
          options={[
            { value: 'system', label: 'System', icon: Monitor },
            { value: 'light', label: 'Light', icon: Sun },
            { value: 'dark', label: 'Dark', icon: Moon },
          ]}
        />
        <Segmented
          label="Theme icons"
          iconOnly
          bind:value={theme}
          options={[
            { value: 'system', label: 'System', icon: Monitor },
            { value: 'light', label: 'Light', icon: Sun },
            { value: 'dark', label: 'Dark', icon: Moon },
          ]}
        />
        <Segmented
          label="Range"
          disabled
          bind:value={range}
          options={[
            { value: '10', label: 'Last 10' },
            { value: 'all', label: 'All 60' },
          ]}
        />
        <Segmented
          label="Range"
          invalid
          id="range-invalid"
          describedby="range-why"
          bind:value={range}
          options={[
            { value: '10', label: 'Last 10' },
            { value: 'all', label: 'All 60' },
          ]}
        />
        <span data-parity-step="click" data-parity-target="button:last-child">
          <Segmented
            label="Pick"
            bind:value={pick}
            options={[
              { value: 'a', label: 'Alpha' },
              { value: 'b', label: 'Beta' },
              { value: 'c', label: 'Gamma' },
            ]}
          />
        </span>
        <span data-parity-step="keys" data-parity-do="press:ArrowRight" data-parity-target="button[aria-checked='true']">
          <Segmented
            label="Keys"
            bind:value={keys}
            options={[
              { value: 'a', label: 'One' },
              { value: 'b', label: 'Two' },
              { value: 'c', label: 'Three' },
            ]}
          />
        </span>
      </div>
  );
}
