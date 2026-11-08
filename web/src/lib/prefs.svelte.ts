// User preferences, persisted to localStorage.

export type ThemePref = 'dark' | 'light' | 'system';
export type Units = 'metric' | 'binary';

interface Prefs {
  theme: ThemePref;
  serverUrl: string;
  units: Units;
  sidebarCollapsed: boolean;
}

const KEY = 'folo.prefs';
const defaults: Prefs = { theme: 'dark', serverUrl: '', units: 'binary', sidebarCollapsed: false };

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
  #p = $state<Prefs>(typeof localStorage === 'undefined' ? { ...defaults } : load());
  #systemDark = $state(true);
  resolvedTheme = $derived<'dark' | 'light'>(
    this.#p.theme === 'system' ? (this.#systemDark ? 'dark' : 'light') : this.#p.theme,
  );

  constructor() {
    if (typeof window === 'undefined') return;
    const mq = matchMedia('(prefers-color-scheme: dark)');
    this.#systemDark = mq.matches;
    mq.addEventListener('change', (e) => (this.#systemDark = e.matches));
  }

  get theme() {
    return this.#p.theme;
  }
  set theme(v: ThemePref) {
    this.#p.theme = v;
    this.#save();
  }
  get serverUrl() {
    return this.#p.serverUrl;
  }
  set serverUrl(v: string) {
    this.#p.serverUrl = v.trim().replace(/\/+$/, '');
    this.#save();
  }
  get units() {
    return this.#p.units;
  }
  set units(v: Units) {
    this.#p.units = v;
    this.#save();
  }
  get sidebarCollapsed() {
    return this.#p.sidebarCollapsed;
  }
  set sidebarCollapsed(v: boolean) {
    this.#p.sidebarCollapsed = v;
    this.#save();
  }

  /** Cycle dark → light → system. */
  cycleTheme() {
    this.theme = this.theme === 'dark' ? 'light' : this.theme === 'light' ? 'system' : 'dark';
  }

  #save() {
    try {
      localStorage.setItem(KEY, JSON.stringify(this.#p));
      localStorage.setItem('folo.theme', this.#p.theme);
    } catch {
      /* storage may be unavailable (private mode); preferences then last for the session */
    }
  }
}

export const prefs = new PrefsStore();
