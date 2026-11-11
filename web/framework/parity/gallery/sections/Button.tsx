// Gallery twin of Button.svelte.
import Plus from '$icons/plus';
import Copy from '$icons/copy';
import Trash2 from '$icons/trash-2';
import ChevronDown from '$icons/chevron-down';
import Button from '$lib/ui/Button';

export default function ButtonSection() {
  return (
    <div class="g-stack" data-parity-coarse data-parity-phone>
      <div class="g-row">
        <Button variant="primary">Primary</Button>
        <Button>Secondary</Button>
        <Button variant="ghost">Ghost</Button>
        <Button variant="danger" icon={Trash2}>Delete</Button>
        <Button variant="destructive">Destroy</Button>
      </div>
      <div class="g-row">
        <Button size="sm" icon={Plus}>New run</Button>
        <Button size="md" icon={Copy}>Copy</Button>
        <Button size="lg" iconRight={ChevronDown}>More</Button>
        <Button icon={Plus} aria-label="Add" />
        <Button variant="ghost" size="sm" icon={Copy} aria-label="Copy" />
        <Button size="sm" icon={Plus} aria-label="Add small" />
      </div>
      <div class="g-row">
        <Button variant="primary" loading>Saving</Button>
        <Button variant="primary" disabled>Disabled</Button>
        <Button disabled>Disabled</Button>
        <Button variant="ghost" disabled>Ghost off</Button>
        <Button variant="danger" disabled icon={Trash2}>Delete off</Button>
        <Button href="#/runs">A link</Button>
        <Button href="#/runs" disabled>A link off</Button>
        <Button type="submit" class="extra">Submit</Button>
      </div>
      <div class="g-row g-narrow">
        <Button variant="primary" block>Block</Button>
      </div>
      <div class="g-row cta-row">
        <Button variant="primary">Fix it</Button>
        <Button icon={Copy} aria-label="Copy command" />
      </div>
      <div class="g-row cta-stack">
        <Button variant="ghost">Connection settings</Button>
        <Button>Retry</Button>
        <Button variant="primary">Use CPU</Button>
      </div>
      <div class="g-row">
        <span data-parity-step="hover" data-parity-do="hover" data-parity-target="button"><Button variant="primary">Hover me</Button></span>
        <span data-parity-step="focus" data-parity-do="focus" data-parity-target="button"><Button>Focus me</Button></span>
      </div>
    </div>
  );
}
