// fw port of NotFound.svelte.
import Crosshair from '$icons/crosshair';
import { effect } from 'fw';
import { router, href } from '$lib/router';
import Page from '$lib/shell/Page';
import { setCrumbs } from '$lib/shell/crumbs';
import Button from '$lib/ui/Button';
import EmptyState from '$lib/ui/EmptyState';

export default function NotFound() {
  effect(() => setCrumbs({ label: 'Not found' }));

  // The same page frame and inline empty state as every other screen with nothing to frame (DESIGN_DIRECTION 8,
  // Compare with no runs): title, then the message frameless on the content edge, never a card.
  return (
    <Page title="Not found">
      <EmptyState
        icon={Crosshair}
        title="Nothing in frame"
        description={`There is no page at ${router.route.path}. It may have moved, or the link is wrong.`}
        actions={() => <Button href={href('/')}>Go to Overview</Button>}
      />
    </Page>
  );
}
