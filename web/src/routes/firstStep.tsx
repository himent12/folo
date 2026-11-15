// What an empty Runs, Models or Playground page asks for first, as Desktop's screens/first_step.rs: with no dataset
// at all, Add dataset (a run trains on one, and New run would only land on "Add a dataset first"); else New run.
import Plus from '$icons/plus';
import { effect, untrack } from 'fw';
import type { Dataset } from '$lib/api/types';
import { href } from '$lib/router';
import Button from '$lib/ui/Button';
import { datasetChanges } from './datasets/changes';

/** The server answered with no dataset (still loading or failed: not known, so New run as before). */
export function noDatasetsYet(datasets: readonly Dataset[] | undefined): boolean {
  return datasets?.length === 0;
}

/** Where the first step leads and what it says: Add dataset while the server has none, else New run. */
export const firstStepHref = (noDatasets: boolean) => href(noDatasets ? '/datasets?add=folder' : '/runs/new');
export const firstStepLabel = (noDatasets: boolean) => (noDatasets ? 'Add dataset' : 'New run');

/**
 * Keeps a page's dataset list (read for the first step) current, as the Datasets page does: read again after a dataset
 * action in this tab, and when the tab is shown again (the first dataset added in another tab meanwhile).
 */
export function refreshOnDatasetChange(datasets: { refresh(): Promise<void> }): void {
  const seen = datasetChanges.peek();
  effect(() => {
    if (datasetChanges.get() !== seen) untrack(() => void datasets.refresh());
  });
  effect(() => {
    const back = () => {
      // Also while the first read is still out: this read supersedes it (an answer from before the tab was hidden
      // would otherwise land last, and nothing would read the list again).
      if (document.visibilityState === 'visible') void datasets.refresh();
    };
    document.addEventListener('visibilitychange', back);
    return () => document.removeEventListener('visibilitychange', back);
  });
}

// One Button whose link and words follow `noDatasets` (a component runs once: a branch picked here would stay the one
// it was while the datasets were still loading).
export default function FirstStepAction(props: { noDatasets: boolean; primary?: boolean }) {
  return (
    <Button variant={props.primary ? 'primary' : 'secondary'} icon={Plus} href={firstStepHref(props.noDatasets)}>
      {firstStepLabel(props.noDatasets)}
    </Button>
  );
}
