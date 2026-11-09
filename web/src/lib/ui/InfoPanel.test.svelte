<script lang="ts">
  // A chart panel with its ⓘ in the header, for InfoButton.svelte.test.ts.
  import InfoButton from './InfoButton.svelte';
  import Panel from './Panel.svelte';
</script>

<Panel title="Box size">
  {#snippet actions()}<InfoButton chart label="How box size is measured" text="Size = √(box area ÷ image area)." />{/snippet}
  <p class="readout">44,433 boxes</p>
</Panel>
