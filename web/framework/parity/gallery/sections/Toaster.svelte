<!-- Gallery twin of Toaster.tsx. -->
<script lang="ts">
  import Toaster from '$lib/ui/Toaster.svelte';
  import { toast } from '$lib/ui/toast.svelte';

  toast.show({ tone: 'success', title: 'Run finished', detail: ['Finished training', 'best mAP50-95 0.535 at epoch 60'], action: { label: 'Open run', run: () => {} } });
  toast.show({ tone: 'info', title: 'Copied the command', detail: 'Paste it into PowerShell.', timeout: 0 });
  toast.error(new Error('The server answered 500.'), 'Prediction failed');
  toast.show({ tone: 'success', title: 'Saved', timeout: 0 });
</script>

<div data-parity-coarse data-parity-phone>
  <span data-parity-step="dismiss" data-parity-target=":document .toast:last-child .close"></span>
  <span data-parity-step="hover" data-parity-do="hover" data-parity-target=":document .toast .action"></span>
  <Toaster />
</div>
