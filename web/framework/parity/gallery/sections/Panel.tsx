// Gallery twin of Panel.svelte.
import Panel from '$lib/ui/Panel';
import Button from '$lib/ui/Button';

export default function PanelSection() {
  return (
    <div class="g-stack g-medium" data-parity-phone>
      <Panel
        title="Training loss"
        meta="60 epochs"
        actions={() => (
          <>
            <Button size="sm" variant="ghost">Last 10</Button>
            <Button size="sm" variant="ghost">All</Button>
          </>
        )}
      >
        <p>The body, with the default md padding.</p>
      </Panel>
      <Panel title="Divided, small padding" divided pad="sm" id="gallery-panel" as="h3" class="extra">
        <p>A hairline under the header; sm padding.</p>
      </Panel>
      <Panel
        title="A very long panel title that does not fit in its header and is cut with an ellipsis at its end"
        meta="1,234"
        actions={() => <Button size="sm">Export</Button>}
      >
        <p>Long titles truncate on desktop and keep their controls on a line of their own on phones.</p>
      </Panel>
      <Panel pad="none">
        <p>No header and no padding.</p>
      </Panel>
      <Panel pad="md">
        <p>No header, md padding (the body is the first child).</p>
      </Panel>
      <Panel pad="sm">
        <p>No header, sm padding.</p>
      </Panel>
      <Panel title="Header only" />
      <Panel actions={() => <Button size="sm">Actions alone</Button>} />
    </div>
  );
}
