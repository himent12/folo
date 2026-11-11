// The screen of a route whose chunk could not be loaded (the connection dropped, a deploy replaced the chunk): the
// page frame of its section and the error, with Retry (the import again). Whichever later try brings the chunk in
// (Retry, a link to the page hovered), App shows the screen itself.
import { computed, effect } from 'fw';
import { router, section } from '$lib/router';
import { setCrumbs } from '$lib/shell/crumbs';
import { mainNav, settingsNav } from '$lib/shell/nav';
import Page from '$lib/shell/Page';
import ErrorState from '$lib/ui/ErrorState';
import type { ScreenName, Screens } from './routes';

export default function ScreenError(props: { screens: Screens }) {
  const name = computed(() => router.route.name as ScreenName);
  const nav = computed(() => {
    const s = section(name.get());
    return [...mainNav, settingsNav].find((n) => n.id === s);
  });
  const failure = computed(() => props.screens.failure(name.get()));
  effect(() => {
    const n = nav.get();
    setCrumbs(n ? { label: n.label, path: n.path } : { label: 'Error' });
  });

  return (
    <Page title={nav.get()?.label ?? 'FOLO Studio'}>
      <ErrorState
        error={failure.get()?.error}
        title="Could not load this screen"
        hint="Check the connection to the FOLO server, then retry."
        retry={() => void props.screens.preload(name.get()).catch(() => {})}
      />
    </Page>
  );
}
