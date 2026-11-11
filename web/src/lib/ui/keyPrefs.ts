// fw port of keyPrefs.svelte.ts (the same store, on an fw signal).
// Whether single-character shortcuts are on: N, G then a letter, [, ?, and J / K / X in the lists (WCAG 2.1.4: a
// keyboard or speech-input user must be able to turn them off). Ctrl / Cmd-K and the arrow keys, Home, End and Enter of
// a focused component always work. Settings → Keyboard shortcuts switches it; the choice stays in this browser.

import { signal } from 'fw';

const KEY = 'folo.singleKeys';

function load(): boolean {
  try {
    return localStorage.getItem(KEY) !== 'off';
  } catch {
    return true;
  }
}

class KeyPrefs {
  #on = signal(typeof localStorage === 'undefined' ? true : load());

  get singleKeys() {
    return this.#on.get();
  }
  set singleKeys(v: boolean) {
    this.#on.set(v);
    try {
      localStorage.setItem(KEY, v ? 'on' : 'off');
    } catch {
      /* storage may be unavailable (private mode); the choice then lasts for the session */
    }
  }
}

export const keyPrefs = new KeyPrefs();
