// Gallery twin of TextArea.svelte.
import { signal } from 'fw';
import TextArea from '$lib/ui/TextArea';

export default function TextAreaSection() {
  const notes = signal('');
  return (
    <div class="g-stack g-medium" data-parity-coarse>
      <TextArea placeholder="What should the labeler look for?" aria-label="Empty" rows={3} />
      <TextArea value={'Box every car and truck.\nSkip vehicles cut off by the edge.'} aria-label="With text" rows={3} />
      <TextArea value="Read only: written by the agent run." readonly aria-label="Read only" rows={2} />
      <TextArea value="Disabled text" disabled aria-label="Disabled" rows={2} />
      <TextArea value="Too short" invalid aria-label="Invalid" rows={2} aria-describedby="g-ta-note" class="g-extra" />
      <div data-parity-step="focus" data-parity-do="focus" data-parity-target="textarea">
        <TextArea placeholder="Focused" aria-label="Focus" rows={2} />
      </div>
      <div data-parity-step="type" data-parity-do="type:Label the dogs." data-parity-target="textarea">
        <TextArea bind:value={notes} aria-label="Type" rows={2} />
      </div>
      <p id="g-ta-note">{`Notes: ${notes.get() || 'none'}`}</p>
    </div>
  );
}
