// Gallery twin of EmptyState.svelte.
import Database from '$icons/database';
import SearchX from '$icons/search-x';
import Plus from '$icons/plus';
import Button from '$lib/ui/Button';
import EmptyState from '$lib/ui/EmptyState';

export default function EmptyStateSection() {
  return (
    <div class="g-stack g-medium" data-parity-phone>
      <EmptyState
        icon={Database}
        title="No datasets yet"
        description="Add a folder of images, or run `folo serve --demo` to try the sample data."
        actions={() => (
          <>
            <Button variant="primary" icon={Plus}>
              Add dataset
            </Button>
            <Button>Learn more</Button>
          </>
        )}
      />
      <EmptyState icon={SearchX} title="Not found" description="This run does not exist, or it was deleted." />
      <EmptyState title="Nothing to compare" />
      <EmptyState title="No runs match" description="Clear the filter to see every run." actions={() => <Button size="sm">Clear filter</Button>} />
    </div>
  );
}
