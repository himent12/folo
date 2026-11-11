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

function load(): Prefs {
  try {
    const raw = localStorage.getItem(KEY);
    const theme = localStorage.getItem('folo.theme');
    const p = { ...defaults, ...(raw ? (JSON.parse(raw) as Partial<Prefs>) : {}) };
    if (theme === 'dark' || theme === 'light' || theme === 'system') p.theme = theme;
    return p;
  } catch {
    return { ...defaults };
  }
}

class PrefsStore {
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
  }

  get resolvedTheme(): 'dark' | 'light' {
    return this.#resolved.get();
  }

  get theme() {
    return this.#p.theme.get();
  }
  set theme(v: ThemePref) {
    this.#p.theme.set(v);
    this.#save();
  }
  get serverUrl() {
    return this.#p.serverUrl.get();
  }
  set serverUrl(v: string) {
    this.#p.serverUrl.set(v.trim().replace(/\/+$/, ''));
    this.#save();
  }
  get units() {
    return this.#p.units.get();
  }
  set units(v: Units) {
    this.#p.units.set(v);
    this.#save();
  }
  get sidebarCollapsed() {
    return this.#p.sidebarCollapsed.get();
  }
  set sidebarCollapsed(v: boolean) {
    this.#p.sidebarCollapsed.set(v);
    this.#save();
  }

  get labelAgent() {
    return this.#p.labelAgent.get();
  }
  set labelAgent(v: LabelAgentId | null) {
    this.#p.labelAgent.set(v);
    this.#save();
  }

  /** Cycle dark → light → system. */
  cycleTheme() {
    this.theme = this.theme === 'dark' ? 'light' : this.theme === 'light' ? 'system' : 'dark';
  }

  #save() {
    const p = this.#p;
    // The same object the Svelte store writes: whatever it was loaded with, the current values over it, in its key order.
    const out: Prefs = {
      ...this.#rest,
      theme: p.theme.peek(),
      serverUrl: p.serverUrl.peek(),
      units: p.units.peek(),
      sidebarCollapsed: p.sidebarCollapsed.peek(),
      labelAgent: p.labelAgent.peek(),
    };
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
