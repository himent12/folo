<!-- Gallery twin of Dialog.tsx (with DialogStatus inside it). -->
<script lang="ts">
  import Button from '$lib/ui/Button.svelte';
  import Dialog from '$lib/ui/Dialog.svelte';
  import { DialogStatus } from '$lib/ui/feedback.svelte';

  let md = $state(false);
  let sm = $state(false);
  let lg = $state(false);
  let st = $state(false);
  const lines = Array.from({ length: 40 }, (_, i) => `Line ${i + 1} of the report: adapter, driver, backend and limits.`);
  const status = new DialogStatus();
  status.success('Path copied', 'D:/folo/runs/r_7d1e/best.folo');
</script>

<div class="g-row" data-parity-phone data-parity-coarse>
  <span data-parity-step="md"><Button onclick={() => (md = true)}>Open md</Button></span>
  <span data-parity-step="sm"><Button onclick={() => (sm = true)}>Open sm</Button></span>
  <span data-parity-step="lg"><Button onclick={() => (lg = true)}>Open lg</Button></span>
  <span data-parity-step="status"><Button onclick={() => (st = true)}>Open status</Button></span>
  <Dialog bind:open={md} title="Rename run" description="A run's name is how it reads in every list.">
    <p>The new name applies at once.</p>
    {#snippet footer()}
      <Button variant="ghost" onclick={() => (md = false)}>Cancel</Button>
      <Button variant="primary">Rename</Button>
    {/snippet}
  </Dialog>
  <Dialog bind:open={sm} title="Delete run?" size="sm" class="extra">
    {#snippet footer()}
      <Button variant="ghost">Cancel</Button>
      <Button variant="destructive">Delete</Button>
    {/snippet}
  </Dialog>
  <Dialog bind:open={lg} title="Driver report" size="lg" description="The server's own words.">
    {#each lines as l (l)}<p>{l}</p>{/each}
  </Dialog>
  <Dialog bind:open={st} title="Export model" {status}>
    <p>Copy the path or download the file.</p>
    {#snippet footer()}
      <Button variant="primary">Download</Button>
    {/snippet}
  </Dialog>
</div>
