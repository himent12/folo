// Gallery twin of Dialog.svelte (with DialogStatus inside it).
import { For, signal } from 'fw';
import Button from '$lib/ui/Button';
import Dialog from '$lib/ui/Dialog';
import { DialogStatus } from '$lib/ui/feedback';

export default function DialogSection() {
  const md = signal(false);
  const sm = signal(false);
  const lg = signal(false);
  const st = signal(false);
  const lines = Array.from({ length: 40 }, (_, i) => `Line ${i + 1} of the report: adapter, driver, backend and limits.`);
  const status = new DialogStatus();
  status.success('Path copied', 'D:/folo/runs/r_7d1e/best.folo');
  return (
    <div class="g-row" data-parity-phone data-parity-coarse>
      <span data-parity-step="md"><Button onclick={() => md.set(true)}>Open md</Button></span>
      <span data-parity-step="sm"><Button onclick={() => sm.set(true)}>Open sm</Button></span>
      <span data-parity-step="lg"><Button onclick={() => lg.set(true)}>Open lg</Button></span>
      <span data-parity-step="status"><Button onclick={() => st.set(true)}>Open status</Button></span>
      <Dialog
        bind:open={md}
        title="Rename run"
        description="A run's name is how it reads in every list."
        footer={() => (
          <>
            <Button variant="ghost" onclick={() => md.set(false)}>Cancel</Button>
            <Button variant="primary">Rename</Button>
          </>
        )}
      >
        <p>The new name applies at once.</p>
      </Dialog>
      <Dialog
        bind:open={sm}
        title="Delete run?"
        size="sm"
        class="extra"
        footer={() => (
          <>
            <Button variant="ghost">Cancel</Button>
            <Button variant="destructive">Delete</Button>
          </>
        )}
      />
      <Dialog bind:open={lg} title="Driver report" size="lg" description="The server's own words.">
        <For each={lines}>{(l) => <p>{l()}</p>}</For>
      </Dialog>
      <Dialog bind:open={st} title="Export model" status={status} footer={() => <Button variant="primary">Download</Button>}>
        <p>Copy the path or download the file.</p>
      </Dialog>
    </div>
  );
}
