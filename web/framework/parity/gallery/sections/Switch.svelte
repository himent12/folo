<!-- Gallery twin of Switch.tsx. -->
<script lang="ts">
  import Switch from '$lib/ui/Switch.svelte';

  let amp = $state(false);
  let on = $state(true);
  let count = $state(0);
</script>

<div class="g-stack g-medium">
  <div data-parity-step="click" data-parity-target="label">
    <Switch label="Mixed precision" bind:checked={amp} onchange={() => count++} />
  </div>
  <Switch label="Cache images in memory" checked={true} />
  <Switch label="Mosaic augmentation" hint="Four images in one, for the first 90% of the epochs." bind:checked={on} />
  <Switch label="Uses the GPU" hint="No GPU on this server: set `--device cpu`." disabled id="g-sw-off" describedby="g-sw-why" />
  <Switch label="Locked on" checked={true} disabled />
  <Switch
    label="A long label that keeps going so the switch has to sit at the far end of a wrapped line of text"
    hint="And a long hint under it, past the readable measure, that wraps onto more lines than one as hints sometimes do."
  />
  <div data-parity-step="focus" data-parity-do="focus" data-parity-target="input">
    <Switch label="Focused" />
  </div>
  <p id="g-sw-why">Changed {count} times.</p>
</div>
