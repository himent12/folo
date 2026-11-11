<!-- Gallery twin of Viewfinder.tsx. -->
<script lang="ts">
  import Button from '$lib/ui/Button.svelte';
  import Viewfinder from '$lib/ui/Viewfinder.svelte';
</script>

<div class="g-stack g-medium" data-parity-phone>
  <Viewfinder title="Drop an image to detect objects">
    {#snippet description()}PNG or JPEG, up to 20 MB. Or <b>paste</b> one.{/snippet}
    {#snippet actions()}
      <Button variant="primary">Choose image</Button>
      <Button>Use a sample</Button>
    {/snippet}
  </Viewfinder>
  <Viewfinder dragging title="Drop to add">
    {#snippet description()}Images are added to this job.{/snippet}
  </Viewfinder>
  <div style="--vf-pad: 8px">
    <Viewfinder title="Small room">
      {#snippet description()}A tight frame.{/snippet}
    </Viewfinder>
  </div>
</div>
