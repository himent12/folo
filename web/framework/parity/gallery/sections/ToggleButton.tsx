// Gallery twin of ToggleButton.svelte.
import SquareDashed from '$icons/square-dashed';
import Tag from '$icons/tag';
import { signal } from 'fw';
import ToggleButton from '$lib/ui/ToggleButton';

export default function ToggleButtonSection() {
  const boxes = signal(true);
  const labels = signal(false);
  return (
    <div class="g-row" data-parity-coarse>
      <span data-parity-step="click" data-parity-target="button">
        <ToggleButton pressed={boxes.get()} label="Boxes" icon={SquareDashed} onchange={(on) => boxes.set(on)} />
      </span>
      <span data-parity-step="click" data-parity-target="button">
        <ToggleButton pressed={labels.get()} label="Labels" icon={Tag} onchange={(on) => labels.set(on)} />
      </span>
      <span data-parity-step="hover" data-parity-do="hover" data-parity-target="button">
        <ToggleButton pressed={false} label="Hover" icon={Tag} onchange={() => {}} />
      </span>
      <span data-parity-step="focus" data-parity-do="focus" data-parity-target="button">
        <ToggleButton pressed={true} label="Focus" icon={SquareDashed} onchange={() => {}} />
      </span>
    </div>
  );
}
