<!-- Gallery twin of DialogStatus.tsx. -->
<script lang="ts">
  import DialogStatusRegion from '$lib/ui/DialogStatus.svelte';
  import { DialogStatus } from '$lib/ui/feedback.svelte';

  const ok = new DialogStatus();
  ok.success('Path copied', 'D:/folo/runs/r_7d1e/best.folo');
  const info = new DialogStatus();
  info.info('The demo cannot download files');
  const err = new DialogStatus();
  err.error(new Error('The server refused the stop.'), 'Could not stop the run');
  const none = new DialogStatus();
  const gone = new DialogStatus();
  gone.success('Dismiss me');
</script>

<div class="g-stack g-medium">
  <DialogStatusRegion status={ok} />
  <DialogStatusRegion status={info} class="extra" />
  <DialogStatusRegion status={err} />
  <DialogStatusRegion status={none} />
  <span data-parity-step="dismiss" data-parity-target="button[aria-label='Dismiss']"><DialogStatusRegion status={gone} /></span>
</div>
