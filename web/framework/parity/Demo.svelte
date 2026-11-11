<!-- The proof page's Svelte twin: Demo.tsx with the Svelte originals. Keep the two in step. -->
<script lang="ts">
  import Plus from '@lucide/svelte/icons/plus';
  import Copy from '@lucide/svelte/icons/copy';
  import Trash2 from '@lucide/svelte/icons/trash-2';
  import ChevronDown from '@lucide/svelte/icons/chevron-down';
  import Monitor from '@lucide/svelte/icons/monitor';
  import Sun from '@lucide/svelte/icons/sun';
  import Moon from '@lucide/svelte/icons/moon';
  import Badge from '$lib/ui/Badge.svelte';
  import Button from '$lib/ui/Button.svelte';
  import Menu from '$lib/ui/Menu.svelte';
  import Segmented from '$lib/ui/Segmented.svelte';
  import Sparkline from '$lib/charts/Sparkline.svelte';
  import './demo.css';

  let runOn = $state<'server' | 'webgpu' | 'cpu'>('server');
  let theme = $state<'system' | 'light' | 'dark'>('dark');
  let range = $state<'10' | 'all'>('all');
  const loss = [2.4, 2.1, 1.9, 1.7, 1.62, 1.5, 1.41, 1.33, 1.3, 1.24];
  const map = [0.12, 0.2, 0.31, 0.36, 0.4, 0.44, 0.47];
</script>

<main class="demo">
  <section>
    <h2>Button</h2>
    <div class="row">
      <Button variant="primary">Primary</Button>
      <Button>Secondary</Button>
      <Button variant="ghost">Ghost</Button>
      <Button variant="danger" icon={Trash2}>Delete</Button>
      <Button variant="destructive">Destroy</Button>
    </div>
    <div class="row">
      <Button size="sm" icon={Plus}>New run</Button>
      <Button size="md" icon={Copy}>Copy</Button>
      <Button size="lg" iconRight={ChevronDown}>More</Button>
      <Button icon={Plus} aria-label="Add" />
      <Button variant="ghost" size="sm" icon={Copy} aria-label="Copy" />
    </div>
    <div class="row">
      <Button variant="primary" loading>Saving</Button>
      <Button variant="primary" disabled>Disabled</Button>
      <Button disabled>Disabled</Button>
      <Button variant="ghost" disabled>Ghost off</Button>
      <Button href="#/runs">A link</Button>
    </div>
    <div class="row narrow">
      <Button variant="primary" block>Block</Button>
    </div>
  </section>
  <section>
    <h2>Badge</h2>
    <div class="row">
      <Badge>neutral</Badge>
      <Badge tone="accent">accent</Badge>
      <Badge tone="success">success</Badge>
      <Badge tone="warning">warning</Badge>
      <Badge tone="danger">danger</Badge>
      <Badge tone="info">info</Badge>
      <Badge tone="outline">outline</Badge>
      <Badge mono>run_7d1e</Badge>
      <Badge tone="accent" mono>0.535</Badge>
    </div>
  </section>
  <section>
    <h2>Segmented</h2>
    <div class="col">
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
        bind:value={range}
        options={[
          { value: '10', label: 'Last 10' },
          { value: 'all', label: 'All 60' },
        ]}
      />
    </div>
  </section>
  <section>
    <h2>Menu</h2>
    <div class="row end">
      <Menu
        label="More run actions"
        items={[
          { label: 'Delete run', danger: true, onclick: () => {} },
          { label: 'Open model', href: '#/models', icon: Copy },
          { label: 'Duplicate', icon: Plus, onclick: () => {} },
        ]}
      />
      <Menu label="Row actions" variant="ghost" items={[{ label: 'Compare', onclick: () => {} }]} />
    </div>
  </section>
  <section>
    <h2>Sparkline</h2>
    <div class="row">
      <Sparkline values={loss} />
      <Sparkline values={map} color="var(--chart-2)" length={12} />
      <Sparkline values={loss} fill={false} width={120} height={32} />
      <Sparkline values={map} domain={[0, 1]} color="var(--chart-3)" />
      <Sparkline values={[1]} />
    </div>
  </section>
</main>
