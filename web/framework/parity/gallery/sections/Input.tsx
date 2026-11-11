// Gallery twin of Input.svelte.
import Search from '$icons/search';
import { signal } from 'fw';
import Input from '$lib/ui/Input';

export default function InputSection() {
  const typed = signal('');
  return (
    <div class="g-stack g-narrow" data-parity-coarse>
      <Input value="coco-small" aria-label="Plain" />
      <Input placeholder="Search runs" aria-label="Placeholder" />
      <Input icon={Search} type="search" placeholder="Filter by name" aria-label="Icon" />
      <Input icon={Search} type="search" value="folo-s" aria-label="Icon with value" />
      <Input type="number" value={60} suffix="epochs" aria-label="Suffix" />
      <Input size="sm" value="small" aria-label="Small" />
      <Input size="sm" icon={Search} placeholder="Small with icon" aria-label="Small icon" suffix="px" />
      <Input value="disabled" disabled aria-label="Disabled" />
      <Input placeholder="Disabled placeholder" disabled aria-label="Disabled placeholder" />
      <Input value="not a number" invalid aria-label="Invalid" />
      <Input value="extra class" class="g-extra" aria-label="Extra class" aria-describedby="nothing" />
      <Input value="A very long value that does not fit in the field and has to be clipped at its right edge" aria-label="Long" />
      <div data-parity-step="focus" data-parity-do="focus" data-parity-target="input">
        <Input placeholder="Focused" aria-label="Focus" />
      </div>
      <div data-parity-step="focus" data-parity-do="focus" data-parity-target="input">
        <Input value="bad" invalid aria-label="Focus invalid" />
      </div>
      <div data-parity-step="type" data-parity-do="type:folo-n" data-parity-target="input">
        <Input bind:value={typed} placeholder="Type here" aria-label="Type" />
        <p>{`Typed: ${typed.get()}`}</p>
      </div>
      <div data-parity-step="hover" data-parity-do="hover" data-parity-target="input">
        <Input value="hovered" aria-label="Hover" />
      </div>
    </div>
  );
}
