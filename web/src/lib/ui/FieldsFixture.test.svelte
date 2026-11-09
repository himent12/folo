<script lang="ts">
  /** Every shared control inside a Field, with a hint or an error, for `fields.svelte.test.ts`. */
  import Field from './Field.svelte';
  import Input from './Input.svelte';
  import Segmented from './Segmented.svelte';
  import Select from './Select.svelte';
  import Slider from './Slider.svelte';
  import Switch from './Switch.svelte';

  let { error = null }: { error?: string | null } = $props();
  let seg = $state('a');
  let sel = $state('a');
  let on = $state(false);
  let level = $state(0.5);
  const options = [
    { value: 'a', label: 'A' },
    { value: 'b', label: 'B' },
  ];
</script>

<Field label="Name" for="f-input" hint="Letters and digits." {error}><Input id="f-input" aria-describedby="extra" /></Field>
<Field label="Size" for="f-seg" hint="Per batch." {error}><Segmented id="f-seg" label="Size" bind:value={seg} {options} /></Field>
<Field label="Model" for="f-select" hint="Trained on shapes." {error}><Select id="f-select" bind:value={sel} {options} /></Field>
<Field label="Mode" hint="Without an id of its own." {error}><Segmented label="Mode" bind:value={seg} {options} /></Field>
<Field label="Fast" for="f-switch" hint="Uses tensor cores." {error}><Switch id="f-switch" label="Fast" hint="On the GPU." bind:checked={on} /></Field>
<Field label="Level" hint="How much." {error}><Slider label="Level" bind:value={level} /></Field>
