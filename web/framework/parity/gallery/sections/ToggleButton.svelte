<!-- Gallery twin of ToggleButton.tsx. -->
<script lang="ts">
  import SquareDashed from '@lucide/svelte/icons/square-dashed';
  import Tag from '@lucide/svelte/icons/tag';
  import ToggleButton from '$lib/ui/ToggleButton.svelte';

  let boxes = $state(true);
  let labels = $state(false);
</script>

<div class="g-row" data-parity-coarse>
  <span data-parity-step="click" data-parity-target="button">
    <ToggleButton pressed={boxes} label="Boxes" icon={SquareDashed} onchange={(on) => (boxes = on)} />
  </span>
  <span data-parity-step="click" data-parity-target="button">
    <ToggleButton pressed={labels} label="Labels" icon={Tag} onchange={(on) => (labels = on)} />
  </span>
  <span data-parity-step="hover" data-parity-do="hover" data-parity-target="button">
    <ToggleButton pressed={false} label="Hover" icon={Tag} onchange={() => {}} />
  </span>
  <span data-parity-step="focus" data-parity-do="focus" data-parity-target="button">
    <ToggleButton pressed={true} label="Focus" icon={SquareDashed} onchange={() => {}} />
  </span>
</div>
