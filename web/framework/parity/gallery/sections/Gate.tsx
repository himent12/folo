// Gallery twin of Gate.svelte.
import Plus from '$icons/plus';
import { signal } from 'fw';
import Button from '$lib/ui/Button';
import Gate from '$lib/ui/Gate';

export default function GateSection() {
  const reason = signal<string | null>('Server offline');
  return (
    <div class="g-stack">
      <div class="g-row">
        <Gate reason="Server offline">
          {(why) => (
            <Button variant="primary" icon={Plus} disabled aria-describedby={why()}>
              Add dataset
            </Button>
          )}
        </Gate>
        <Gate reason={null}>{(why) => <Button aria-describedby={why()}>Compare</Button>}</Gate>
        <Gate reason="Select 2–4 runs to compare">
          {(why) => (
            <Button disabled aria-describedby={why()}>
              Compare
            </Button>
          )}
        </Gate>
      </div>
      <div class="g-row">
        <span data-parity-step="tooltip" data-parity-do="hover" data-parity-wait="700" data-parity-target=":document .gate.off"></span>
        <span data-parity-step="toggle" data-parity-target="button">
          <Button onclick={() => reason.set(reason.get() ? null : 'Server offline')}>Toggle the reason</Button>
        </span>
        <Gate reason={reason.get()}>
          {(why) => (
            <Button variant="primary" disabled={!!reason.get()} aria-describedby={why()}>
              Start training
            </Button>
          )}
        </Gate>
      </div>
    </div>
  );
}
