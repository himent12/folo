// // User preferences, persisted to localStorage.

import { computed, signal, type Signal } from 'fw';
import type { LabelAgentId } from './api/types';

export type ThemePref = 'dark' | 'light' | 'system';
export type Units = 'metric' | 'binary';

interface Prefs {
  theme: ThemePref;
  serverUrl: string;
  units: Units;
  sidebarCollapsed: boolean;
  /** The agent the labeling screens asked last. */
  labelAgent: LabelAgentId | null;
}

const KEY = 'folo.prefs';
const defaults: Prefs = { theme: 'dark', serverUrl: '', units: 'binary', sidebarCollapsed: false, labelAgent: null };

/** What is stored now (keys of a later version included), or null when storage cannot be read. */
function stored(): Prefs | null {
  try {
    const raw = localStorage.getItem(KEY);
    const theme = localStorage.getItem('folo.theme');
    const p = { ...defaults, ...(raw ? (JSON.parse(raw) as Partial<Prefs>) : {}) };
    if (theme === 'dark' || theme === 'light' || theme === 'system') p.theme = theme;
    return p;
  } catch {
    return null;
  }
}

function load(): Prefs {
  return stored() ?? { ...defaults };
}

/** Exported for tests; the app uses `prefs`. */
export class PrefsStore {
  #p: { [K in keyof Prefs]: Signal<Prefs[K]> };
  /** The stored object as last read or written (keys of a later version included), written back with the values over it. */
  #rest: Prefs;
  #systemDark = signal(true);
  #resolved = computed<'dark' | 'light'>(() => {
    const t = this.#p.theme.get();
    return t === 'system' ? (this.#systemDark.get() ? 'dark' : 'light') : t;
  });

  constructor() {
    const p = typeof localStorage === 'undefined' ? { ...defaults } : load();
    this.#rest = p;
    this.#p = {
      theme: signal(p.theme),
      serverUrl: signal(p.serverUrl),
      units: signal(p.units),
      sidebarCollapsed: signal(p.sidebarCollapsed),
      labelAgent: signal(p.labelAgent),
    };
    if (typeof window === 'undefined') return;
    const mq = matchMedia('(prefers-color-scheme: dark)');
    this.#systemDark.set(mq.matches);
    mq.addEventListener('change', (e) => this.#systemDark.set(e.matches));
    // Another tab of this Studio changed a preference: show it here too (shell F2: a tab kept its theme until reloaded).
    // The server address is the exception: this tab stays on the server it is connected to until it reloads, since a
    // reconnect would swap the data under its open pages; its own saves no longer put the old address back (#save).
    window.addEventListener('storage', (e) => {
      if (e.key !== null && e.key !== KEY && e.key !== 'folo.theme') return;
      const p = stored();
      if (!p) return;
      this.#rest = p;
      this.#p.theme.set(p.theme);
      this.#p.units.set(p.units);
      this.#p.sidebarCollapsed.set(p.sidebarCollapsed);
      this.#p.labelAgent.set(p.labelAgent);
    });
  }

  get resolvedTheme(): 'dark' | 'light' {
    return this.#resolved.get();
  }

  get theme() {
    return this.#p.theme.get();
  }
  set theme(v: ThemePref) {
    this.#p.theme.set(v);
    this.#save('theme');
  }
  get serverUrl() {
    return this.#p.serverUrl.get();
  }
  set serverUrl(v: string) {
    this.#p.serverUrl.set(v.trim().replace(/\/+$/, ''));
    this.#save('serverUrl');
  }
  get units() {
    return this.#p.units.get();
  }
  set units(v: Units) {
    this.#p.units.set(v);
    this.#save('units');
  }
  get sidebarCollapsed() {
    return this.#p.sidebarCollapsed.get();
  }
  set sidebarCollapsed(v: boolean) {
    this.#p.sidebarCollapsed.set(v);
    this.#save('sidebarCollapsed');
  }

  get labelAgent() {
    return this.#p.labelAgent.get();
  }
  set labelAgent(v: LabelAgentId | null) {
    this.#p.labelAgent.set(v);
    this.#save('labelAgent');
  }

  /** Cycle dark → light → system. */
  cycleTheme() {
    this.theme = this.theme === 'dark' ? 'light' : this.theme === 'light' ? 'system' : 'dark';
  }

  /**
   * Writes the one preference that changed over what is stored now (in its key order, as the Svelte store wrote it),
   * never this tab's copy of the others: another tab may have changed them since this one loaded (shell F2: a tab
   * opened earlier put back the old theme, units and server address the next time it saved anything).
   */
  #save<K extends keyof Prefs>(key: K) {
    const out: Prefs = { ...(stored() ?? this.#rest), [key]: this.#p[key].peek() };
    this.#rest = out;
    try {
      localStorage.setItem(KEY, JSON.stringify(out));
      localStorage.setItem('folo.theme', out.theme);
    } catch {
      /* storage may be unavailable (private mode); preferences then last for the session */
    }
  }
}

export const prefs = new PrefsStore();
