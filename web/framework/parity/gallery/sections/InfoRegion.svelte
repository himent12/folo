<!-- Gallery twin of InfoRegion.tsx: a chart's ⓘ in a Panel's header, in place under the heading on phones. -->
<script lang="ts">
  import InfoButton from '$lib/ui/InfoButton.svelte';
  import Panel from '$lib/ui/Panel.svelte';
</script>

<div class="g-stack g-medium" data-parity-phone data-parity-coarse>
  <Panel title="Box size">
    {#snippet actions()}<span data-parity-step="text" data-parity-target="button"
        ><InfoButton chart label="How box size is measured" text="Size = √(box area ÷ image area)." /></span
      >{/snippet}
    <p class="readout">44,433 boxes</p>
  </Panel>
  <Panel title="Loss terms" meta="3">
    {#snippet actions()}<span data-parity-step="list" data-parity-target="button"
        ><InfoButton chart label="What the loss terms mean" how="Each term is averaged over the epoch's batches."
          ><dl><dt>Box</dt><dd>how far predicted boxes are from the true ones</dd></dl></InfoButton
        ></span
      >{/snippet}
    <p class="readout">Box 1.21 · Cls 0.84 · DFL 1.02</p>
  </Panel>
</div>
