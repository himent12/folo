/** The labeling screens on a server that cannot label (support.ts): what it needs, in the page's own frame. */
import ArrowLeft from '$icons/arrow-left';
import Tags from '$icons/tags';
import { href } from '$lib/router';
import Button from '$lib/ui/Button';
import EmptyState from '$lib/ui/EmptyState';
import Panel from '$lib/ui/Panel';

export default function Unsupported() {
  return (
    <Panel pad="none">
      <EmptyState
        icon={Tags}
        title="This server cannot label images"
        description="Labeling asks Claude Code, Codex or Antigravity on the server's machine, so it needs a current `folo serve` started without `--mock`."
        actions={() => (
          <Button icon={ArrowLeft} href={href('/datasets')}>
            All datasets
          </Button>
        )}
      />
    </Panel>
  );
}
