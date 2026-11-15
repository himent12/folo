// The screen of a route whose chunk could not be loaded (the connection dropped, a deploy replaced the chunk): the
// page frame of its section and the error, with Retry (the import again). Whichever later try brings the chunk in
// (Retry, a link to the page hovered), App shows the screen itself. When the server now serves another build of Studio
// than the one this page runs (an upgrade of folo serve under an open tab), the old build's chunks are gone for good and
// no Retry can help: the screen says Studio was updated and offers Reload instead (shell F11).
import RotateCw from '$icons/rotate-cw';
import { computed, effect, signal } from 'fw';
import { router, section } from '$lib/router';
import { setCrumbs } from '$lib/shell/crumbs';
import { mainNav, settingsNav } from '$lib/shell/nav';
import Page from '$lib/shell/Page';
import Button from '$lib/ui/Button';
import ErrorState from '$lib/ui/ErrorState';
import type { ScreenName, Screens } from './routes';

/**
 * The screen of a route whose screen threw while it was created (data of a shape it did not expect, a bug): the page
 * frame of its section and the error, with Reload. Without it the screen shown before stayed on display under the new
 * URL, the sidebar on the new section, and nothing said it failed (s6-shell-2).
 */
export function ScreenCrash(props: { error: unknown }) {
  const nav = computed(() => {
    const s = section(router.route.name);
    return [...mainNav, settingsNav].find((n) => n.id === s);
  });
  effect(() => {
    const n = nav.get();
    setCrumbs(n ? { label: n.label, path: n.path } : { label: 'Error' });
  });
  return (
    <Page title={nav.get()?.label ?? 'FOLO Studio'}>
      <ErrorState
        error={props.error instanceof Error ? props.error : new Error(String(props.error))}
        title="This page could not be shown"
        hint="Reload the page. If it happens again, the error below says what went wrong."
        actions={
          <Button size="sm" variant="primary" icon={RotateCw} onclick={() => location.reload()}>
            Reload
          </Button>
        }
      />
    </Page>
  );
}

/** The entry script of a built Studio's index.html (`<script type="module" … src="/assets/index-….js">`), or null. */
export function entryScript(html: string): string | null {
  const tag = /<script\b[^>]*\btype="module"[^>]*>/.exec(html)?.[0];
  return tag ? (/\bsrc="([^"]+)"/.exec(tag)?.[1] ?? null) : null;
}

/**
 * Whether the server now serves another build of Studio than this page's: its index.html names another entry script.
 * False when that cannot be told (no answer, a dev server whose entry never changes).
 */
export async function studioReplaced(): Promise<boolean> {
  const mine = document.querySelector('script[type="module"][src]')?.getAttribute('src') ?? null;
  if (!mine) return false;
  try {
    const res = await fetch(location.pathname, { cache: 'no-store' });
    const theirs = res.ok ? entryScript(await res.text()) : null;
    return theirs !== null && theirs !== mine;
  } catch {
    return false;
  }
}

export default function ScreenError(props: { screens: Screens }) {
  const name = computed(() => router.route.name as ScreenName);
  const nav = computed(() => {
    const s = section(name.get());
    return [...mainNav, settingsNav].find((n) => n.id === s);
  });
  const failure = computed(() => props.screens.failure(name.get()));
  const replaced = signal(false);
  effect(() => {
    const n = nav.get();
    setCrumbs(n ? { label: n.label, path: n.path } : { label: 'Error' });
  });
  effect(() => {
    if (!failure.get() || replaced.peek()) return;
    void studioReplaced().then((v) => v && replaced.set(true));
  });

  return (
    <Page title={nav.get()?.label ?? 'FOLO Studio'}>
      {replaced.get() ? (
        <ErrorState
          error={new Error('This page runs an older version of Studio than the one the server now serves.')}
          title="Studio was updated"
          hint="Reload the page to use the new version."
          actions={
            <Button size="sm" variant="primary" icon={RotateCw} onclick={() => location.reload()}>
              Reload
            </Button>
          }
        />
      ) : (
        <ErrorState
          error={failure.get()?.error}
          title="Could not load this screen"
          hint="Check the connection to the FOLO server, then retry."
          retry={() => void props.screens.preload(name.get()).catch(() => {})}
        />
      )}
    </Page>
  );
}
