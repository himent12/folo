// @vitest-environment happy-dom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { refetchFailedStyles, watchFailedStyles } from './failedStyles';

// happy-dom loads no stylesheet: a link fires `error` when appended, or `load` with handleDisabledFileLoadingAsSuccess.
// That setting is the network here.
const settings = (window as unknown as { happyDOM: { settings: { handleDisabledFileLoadingAsSuccess: boolean } } }).happyDOM.settings;
const network = (up: boolean) => void (settings.handleDisabledFileLoadingAsSuccess = up);

/** A stylesheet link as the preload helper adds it. */
function link(href: string) {
  const l = document.createElement('link');
  l.rel = 'stylesheet';
  l.href = href;
  document.head.appendChild(l);
  return l;
}
const sheets = () => [...document.head.querySelectorAll<HTMLLinkElement>('link[rel=stylesheet]')].map((l) => l.href);

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

  it('takes a stylesheet whose load failed out, and loads it again before the next chunk', async () => {
    stop = watchFailedStyles();
    network(true);
    link('http://x/assets/Overview-1.css');
    network(false);
    link('http://x/assets/shared-1.css');
    expect(sheets()).toEqual(['http://x/assets/Overview-1.css']);
    network(true);
    await refetchFailedStyles();
    expect(sheets()).toEqual(['http://x/assets/Overview-1.css', 'http://x/assets/shared-1.css']);
    // Loaded: nothing left to load again.
    await refetchFailedStyles();
    expect(sheets()).toHaveLength(2);
  });

  it('rejects when it fails again, and tries once more next time', async () => {
    stop = watchFailedStyles();
    link('http://x/assets/shared-2.css');
    await expect(refetchFailedStyles()).rejects.toThrow('shared-2.css');
    expect(sheets()).toEqual([]);
    network(true);
    await expect(refetchFailedStyles()).resolves.toBeUndefined();
    expect(sheets()).toEqual(['http://x/assets/shared-2.css']);
  });

  it('leaves other links alone', async () => {
    stop = watchFailedStyles();
    const icon = document.createElement('link');
    icon.rel = 'icon';
    icon.href = 'http://x/favicon.svg';
    document.head.appendChild(icon);
    icon.dispatchEvent(new Event('error'));
    expect(document.head.children).toHaveLength(1);
    await expect(refetchFailedStyles()).resolves.toBeUndefined();
    expect(document.head.children).toHaveLength(1);
  });
});
