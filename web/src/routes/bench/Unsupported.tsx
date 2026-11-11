// fw port of Unsupported.svelte.
/** The benchmark screens on a server without them (./support): what they need, in the page's own frame. */
import ArrowLeft from '$icons/arrow-left';
import Target from '$icons/target';
import { href } from '$lib/router';
import Button from '$lib/ui/Button';
import EmptyState from '$lib/ui/EmptyState';
import Panel from '$lib/ui/Panel';

export default function Unsupported() {
  return (
    <Panel pad="none">
      <EmptyState
        icon={Target}
        title="This server has no vision benchmarks"
        description="Benchmarks need a current `folo serve` started without `--mock`: it keeps their images, runs the entries and scores them."
        actions={() => (
          <Button icon={ArrowLeft} href={href('/models')}>
            All models
          </Button>
        )}
      />
    </Panel>
  );
}
