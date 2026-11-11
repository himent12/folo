// Gallery twin of Tabs.svelte.
import { signal } from 'fw';
import Tabs from '$lib/ui/Tabs';

export default function TabsSection() {
  const view = signal<'overview' | 'metrics' | 'config' | 'logs'>('metrics');
  const small = signal<'a' | 'b'>('a');
  return (
    <div class="g-stack g-medium" data-parity-coarse>
      <span data-parity-step="arrow" data-parity-do="press:ArrowRight" data-parity-target="[aria-selected='true']">
        <Tabs
          label="Run views"
          bind:value={view}
          tabs={[
            { value: 'overview', label: 'Overview' },
            { value: 'metrics', label: 'Metrics', meta: '12' },
            { value: 'config', label: 'Config' },
            { value: 'logs', label: 'Logs', meta: '1,204' },
          ]}
        />
      </span>
      <div class="g-row">
        <span data-parity-step="hover" data-parity-do="hover" data-parity-target="[aria-selected='false']">
          <Tabs
            label="Two"
            idPrefix="two"
            bind:value={small}
            tabs={[
              { value: 'a', label: 'Images' },
              { value: 'b', label: 'Labels' },
            ]}
          />
        </span>
      </div>
      <p class="g-note">{`Selected: ${view.get()}`}</p>
    </div>
  );
}
