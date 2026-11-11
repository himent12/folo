<!-- Gallery twin of Select.tsx. -->
<script lang="ts">
  import Select from '$lib/ui/Select.svelte';

  let device = $state<'auto' | 'cpu' | 'cuda'>('auto');
  let model = $state('s');
  let small = $state('a');
  let grouped = $state('n');
  let off = $state('a');
  let bad = $state('b');
  let changed = $state('');
  const devices = [
    { value: 'auto' as const, label: 'Auto (fastest available)' },
    { value: 'cpu' as const, label: 'CPU', title: 'Slow, works everywhere' },
    { value: 'cuda' as const, label: 'CUDA GPU', disabled: true },
  ];
  const models = [
    { value: 'n', label: 'folo-n · mAP50-95 0.412 on COCO val', short: 'folo-n' },
    { value: 's', label: 'folo-s · mAP50-95 0.535 on COCO val', short: 'folo-s' },
  ];
  const ab = [
    { value: 'a', label: 'Alpha' },
    { value: 'b', label: 'Bravo' },
  ];
  const groups = [
    { value: 'none', label: 'From scratch' },
    { value: 'n', label: 'folo-n', group: 'Pretrained' },
    { value: 's', label: 'folo-s', group: 'Pretrained' },
    { value: 'mine', label: 'my-run best.folo', group: 'Your runs' },
    { value: 'file', label: 'A file…' },
  ];
</script>

<div class="g-stack g-narrow" data-parity-coarse>
  <Select bind:value={device} options={devices} label="Device" onchange={(v) => (changed = v)} />
  <Select bind:value={model} options={models} label="Model" />
  <Select bind:value={small} options={ab} size="sm" label="Small" />
  <Select bind:value={model} options={models} size="sm" label="Small short" />
  <Select bind:value={grouped} options={groups} label="Start from" id="g-sel-grouped" describedby="g-sel-note" />
  <Select bind:value={off} options={ab} disabled label="Disabled" />
  <Select bind:value={model} options={models} disabled label="Disabled short" />
  <Select bind:value={bad} options={ab} invalid label="Invalid" />
  <div data-parity-step="focus" data-parity-do="focus" data-parity-target="select">
    <Select bind:value={small} options={ab} label="Focus" />
  </div>
  <div data-parity-step="key" data-parity-do="press:ArrowDown" data-parity-target="select">
    <Select bind:value={device} options={devices} label="Key" onchange={(v) => (changed = v)} />
  </div>
  <p id="g-sel-note">Chose {changed || 'nothing'}, device {device}.</p>
</div>
