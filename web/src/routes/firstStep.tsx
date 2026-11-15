// What an empty Runs, Models or Playground page asks for first, as Desktop's screens/first_step.rs: with no dataset
// at all, Add dataset (a run trains on one, and New run would only land on "Add a dataset first"); else New run.
import Plus from '$icons/plus';
import type { Dataset } from '$lib/api/types';
import { href } from '$lib/router';
import Button from '$lib/ui/Button';

/** The server answered with no dataset (still loading or failed: not known, so New run as before). */
export function noDatasetsYet(datasets: readonly Dataset[] | undefined): boolean {
  return datasets?.length === 0;
}

export default function FirstStepAction(props: { noDatasets: boolean; primary?: boolean }) {
  const variant = props.primary ? 'primary' : 'secondary';
  return props.noDatasets ? (
    <Button variant={variant} icon={Plus} href={href('/datasets?add=folder')}>
      Add dataset
    </Button>
  ) : (
    <Button variant={variant} icon={Plus} href={href('/runs/new')}>
      New run
    </Button>
  );
}
