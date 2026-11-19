// @vitest-environment happy-dom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { retryFailedStyles, watchFailedStyles } from './failedStyles';

// happy-dom loads no stylesheet: a link fires `error` when appended, or `load` with handleDisabledFileLoadingAsSuccess.
// That setting is the network here.
const settings = (window as unknown as { happyDOM: { settings: { handleDisabledFileLoadingAsSuccess: boolean } } }).happyDOM.settings;
const network = (up: boolean) => void (settings.handleDisabledFileLoadingAsSuccess = up);

/** A stylesheet link as the preload helper adds it (at the end of the head). */
function link(href: string) {
  const l = document.createElement('link');
  l.rel = 'stylesheet';
  l.crossOrigin = '';
  l.setAttribute('nonce', 'n0');
  l.href = href;
  document.head.appendChild(l);
  return l;
}
/** The head's stylesheet links in document order, each with whether its last load succeeded. */
const sheets = () =>
  [...document.head.querySelectorAll<HTMLLinkElement>('link[rel=stylesheet]')].map((l) => `${l.href.replace('http://x/assets/', '')}${loaded.has(l) ? '' : ' (failed)'}`);
const loaded = new WeakSet<Element>();
document.addEventListener('load', (e) => void loaded.add(e.target as Element), true);

describe('failed stylesheets', () => {
  let stop = () => {};
  // happy-dom logs every stylesheet it does not load, on the console it was given before the test ran (Node's).
  beforeEach(() => void vi.spyOn(process.stderr, 'write').mockImplementation(() => true));
  afterEach(() => {
    vi.restoreAllMocks();
    stop();
    network(false);
    document.head.replaceChildren();
  });

  it('asks again for a stylesheet whose load failed, in the place of its link and with its attributes', async () => {
    stop = watchFailedStyles();
    network(true);
    link('http://x/assets/index-1.css');
    network(false);
    const dead = link('http://x/assets/shared-1.css');
    network(true);
    link('http://x/assets/Overview-1.css');
    expect(sheets()).toEqual(['index-1.css', 'shared-1.css (failed)', 'Overview-1.css']);
    await retryFailedStyles();
    // Where the failed one was, not after Overview's: the order a clean load gives them.
    expect(sheets()).toEqual(['index-1.css', 'shared-1.css', 'Overview-1.css']);
    const again = document.head.querySelectorAll('link')[1]!;
    expect(again).not.toBe(dead);
    expect(again.isConnected && !dead.isConnected).toBe(true);
    expect(again.getAttributeNames().map((n) => `${n}=${again.getAttribute(n)}`).sort()).toEqual(
      dead.getAttributeNames().map((n) => `${n}=${dead.getAttribute(n)}`).sort(),
    );
    // Loaded: nothing left to ask for.
    await retryFailedStyles();
    expect(document.head.querySelectorAll('link')[1]).toBe(again);
  });

  it('never rejects: a retry that fails stays in place and is tried again next time', async () => {
    stop = watchFailedStyles();
    network(true);
    link('http://x/assets/index-2.css');
    network(false);
    link('http://x/assets/shared-2.css');
    network(true);
    link('http://x/assets/Settings-2.css');
    network(false);
    await expect(retryFailedStyles()).resolves.toBeUndefined();
    expect(sheets()).toEqual(['index-2.css', 'shared-2.css (failed)', 'Settings-2.css']);
    network(true);
    await expect(retryFailedStyles()).resolves.toBeUndefined();
    expect(sheets()).toEqual(['index-2.css', 'shared-2.css', 'Settings-2.css']);
  });

  it('asks once while a retry is out', async () => {
    stop = watchFailedStyles();
    link('http://x/assets/shared-3.css');
    network(true);
    const a = retryFailedStyles();
    const b = retryFailedStyles();
    await Promise.all([a, b]);
    expect(sheets()).toEqual(['shared-3.css']);
  });

  it('leaves other links alone', async () => {
    stop = watchFailedStyles();
    const icon = document.createElement('link');
    icon.rel = 'icon';
    icon.href = 'http://x/favicon.svg';
    document.head.appendChild(icon);
    icon.dispatchEvent(new Event('error'));
    await expect(retryFailedStyles()).resolves.toBeUndefined();
    expect([...document.head.children]).toEqual([icon]);
  });
});
