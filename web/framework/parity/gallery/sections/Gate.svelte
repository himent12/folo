<!-- Gallery twin of Gate.tsx. -->
<script lang="ts">
  import Plus from '@lucide/svelte/icons/plus';
  import Button from '$lib/ui/Button.svelte';
  import Gate from '$lib/ui/Gate.svelte';

  let reason = $state<string | null>('Server offline');
</script>

<div class="g-stack">
  <div class="g-row">
    <Gate reason="Server offline">
      {#snippet children(why)}<Button variant="primary" icon={Plus} disabled aria-describedby={why}>Add dataset</Button>{/snippet}
    </Gate>
    <Gate reason={null}>
      {#snippet children(why)}<Button aria-describedby={why}>Compare</Button>{/snippet}
    </Gate>
    <Gate reason="Select 2–4 runs to compare">
      {#snippet children(why)}<Button disabled aria-describedby={why}>Compare</Button>{/snippet}
    </Gate>
  </div>
  <div class="g-row">
    <span data-parity-step="tooltip" data-parity-do="hover" data-parity-wait="700" data-parity-target=":document .gate.off"></span>
    <span data-parity-step="toggle" data-parity-target="button">
      <Button onclick={() => (reason = reason ? null : 'Server offline')}>Toggle the reason</Button>
    </span>
    <Gate {reason}>
      {#snippet children(why)}<Button variant="primary" disabled={!!reason} aria-describedby={why}>Start training</Button>{/snippet}
    </Gate>
  </div>
</div>
