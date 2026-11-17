// User preferences, persisted to localStorage, and the custom themes installed in this browser (`folo.themes`).

import { computed, signal, type Signal } from 'fw';
import type { LabelAgentId } from './api/types';
import { findTheme, importThemes, readStoredThemes, resolveTheme, themesFor, type CustomThemeFile, type ImportResult, type Mode, type Theme } from './design/themes';

export type ThemePref = 'dark' | 'light' | 'system';
export type Units = 'metric' | 'binary';

interface Prefs {
  /** The mode: dark, light, or the operating system's. */
  theme: ThemePref;
  /** The theme each mode uses (design/THEMES.md); an id that names no theme of that mode shows the mode's default. */
  darkTheme: string;
  lightTheme: string;
  serverUrl: string;
  units: Units;
  sidebarCollapsed: boolean;
  /** The agent the labeling screens asked last. */
  labelAgent: LabelAgentId | null;
}

const KEY = 'folo.prefs';
/** The installed custom themes: an array of checked theme files (design/THEMES.md, Custom themes). */
export const THEMES_KEY = 'folo.themes';
const defaults: Prefs = { theme: 'dark', darkTheme: 'dark', lightTheme: 'light', serverUrl: '', units: 'binary', sidebarCollapsed: false, labelAgent: null };
/** A stored theme id: any non-empty string is kept, even one that names no theme now (it may be installed again). */
const isThemeId = (v: unknown): v is string => typeof v === 'string' && v.length > 0 && v.length <= 80;

/**
 * The stored object with every value of the wrong type (another writer, an older version, a hand edit) put back to its
 * default: a number or null for the server address made every request fail and Settings throw, so nothing could fix it
 * (s6-shell-1). Keys this version does not know stay, for a later version to read.
 */
export function sanitize(raw: unknown): Prefs {
  const o: Record<string, unknown> = raw !== null && typeof raw === 'object' && !Array.isArray(raw) ? { ...(raw as Record<string, unknown>) } : {};
  const pick = <T>(v: unknown, ok: (v: unknown) => v is T, d: T): T => (ok(v) ? v : d);
  return {
    ...o,
    theme: pick(o.theme, (v): v is ThemePref => v === 'dark' || v === 'light' || v === 'system', defaults.theme),
    darkTheme: pick(o.darkTheme, isThemeId, defaults.darkTheme),
    lightTheme: pick(o.lightTheme, isThemeId, defaults.lightTheme),
    serverUrl: pick(o.serverUrl, (v): v is string => typeof v === 'string', defaults.serverUrl).trim(),
    units: pick(o.units, (v): v is Units => v === 'metric' || v === 'binary', defaults.units),
    sidebarCollapsed: pick(o.sidebarCollapsed, (v): v is boolean => typeof v === 'boolean', defaults.sidebarCollapsed),
    labelAgent: pick(o.labelAgent, (v): v is LabelAgentId | null => v === null || v === 'claude' || v === 'codex', defaults.labelAgent),
  } as Prefs;
}

/** What is stored now (keys of a later version included), or null when storage cannot be read. */
function stored(): Prefs | null {
  try {
    const raw = localStorage.getItem(KEY);
    const theme = localStorage.getItem('folo.theme');
    let parsed: unknown = {};
    try {
      parsed = raw ? JSON.parse(raw) : {};
    } catch {
      // Not JSON (a hand edit, a truncated write): the defaults, as for nothing stored.
    }
    const p = sanitize(parsed);
    if (theme === 'dark' || theme === 'light' || theme === 'system') p.theme = theme;
    return p;
  } catch {
    return null;
  }
}

function load(): Prefs {
  return stored() ?? { ...defaults };
}

/** The installed custom themes as stored (entries that fail the check left out), or null when storage cannot be read. */
function storedThemes(): CustomThemeFile[] | null {
  let raw: string | null;
  try {
    raw = localStorage.getItem(THEMES_KEY);
  } catch {
    return null;
  }
  try {
    return readStoredThemes(raw ? JSON.parse(raw) : []);
  } catch {
    // Not JSON (a hand edit): none, as for nothing stored.
    return [];
  }
}

/** Exported for tests; the app uses `prefs`. */
export class PrefsStore {
  #p: { [K in keyof Prefs]: Signal<Prefs[K]> };
  /** The stored object as last read or written (keys of a later version included), written back with the values over it. */
  #rest: Prefs;
  /** Preferences changed in this tab whose last write failed. */
  #dirty = new Set<keyof Prefs>();
  #systemDark = signal(true);
  #resolved = computed<'dark' | 'light'>(() => {
    const t = this.#p.theme.get();
    return t === 'system' ? (this.#systemDark.get() ? 'dark' : 'light') : t;
  });
  #customThemes = signal<CustomThemeFile[]>([]);
  /** The theme on screen: the resolved mode's choice. */
  #active = computed<Theme>(() => this.slotTheme(this.#resolved.get()));

  constructor() {
    const p = typeof localStorage === 'undefined' ? { ...defaults } : load();
    this.#rest = p;
    this.#p = {
      theme: signal(p.theme),
      darkTheme: signal(p.darkTheme),
      lightTheme: signal(p.lightTheme),
      serverUrl: signal(p.serverUrl),
      units: signal(p.units),
      sidebarCollapsed: signal(p.sidebarCollapsed),
      labelAgent: signal(p.labelAgent),
    };
    if (typeof localStorage !== 'undefined') this.#customThemes.set(storedThemes() ?? []);
    if (typeof window === 'undefined') return;
    const mq = matchMedia('(prefers-color-scheme: dark)');
    this.#systemDark.set(mq.matches);
    mq.addEventListener('change', (e) => this.#systemDark.set(e.matches));
    // Another tab of this Studio changed a preference: show it here too (shell F2: a tab kept its theme until reloaded).
    // The server address is the exception: this tab stays on the server it is connected to until it reloads, since a
    // reconnect would swap the data under its open pages; its own saves no longer put the old address back (#save).
    window.addEventListener('storage', (e) => {
      // Custom themes installed or removed in another tab: this one lists and shows the same.
      if (e.key === null || e.key === THEMES_KEY) {
        const t = storedThemes();
        if (t) this.#customThemes.set(t);
      }
      if (e.key !== null && e.key !== KEY && e.key !== 'folo.theme') return;
      const p = stored();
      if (!p) return;
      this.#rest = p;
      // A change of this tab's that is not written yet (#dirty) stays: it is the newer one, and the next save writes it.
      const follow = <K extends 'theme' | 'darkTheme' | 'lightTheme' | 'units' | 'sidebarCollapsed' | 'labelAgent'>(k: K) => {
        if (!this.#dirty.has(k)) this.#p[k].set(p[k]);
      };
      follow('theme');
      follow('darkTheme');
      follow('lightTheme');
      follow('units');
      follow('sidebarCollapsed');
      follow('labelAgent');
    });
  }

  /** The mode on screen (`data-theme`): the preference, or the operating system's under System. */
  get resolvedTheme(): 'dark' | 'light' {
    return this.#resolved.get();
  }

  /** The theme on screen, resolved (its palette, its `data-theme-id`, a custom theme's inline colours). */
  get activeTheme(): Theme {
    return this.#active.get();
  }

  /** The stored theme id of a mode (it may name no theme now: `slotTheme` says what shows). */
  get darkTheme() {
    return this.#p.darkTheme.get();
  }
  get lightTheme() {
    return this.#p.lightTheme.get();
  }

  /** The theme a mode shows: its stored choice, or the mode's default when that names no theme of the mode. */
  slotTheme(mode: Mode): Theme {
    return resolveTheme(this.#p[mode === 'dark' ? 'darkTheme' : 'lightTheme'].get(), mode, this.#customThemes.get());
  }

  /** The themes of a mode, as Settings and the palette list them: built-ins in order, then custom ones by name. */
  themesFor(mode: Mode): Theme[] {
    return themesFor(mode, this.#customThemes.get());
  }

  /** The custom themes installed in this browser, as stored. */
  get customThemes(): readonly CustomThemeFile[] {
    return this.#customThemes.get();
  }

  /**
   * Picks theme `id` for its mode (design/THEMES.md, Choosing a theme). Under Dark or Light, a theme of the other mode
   * switches the mode too; under System the mode stays and the choice shows whenever that mode is in effect. False
   * when no theme has that id.
   */
  useTheme(id: string): boolean {
    const t = findTheme(id, this.#customThemes.peek());
    if (!t) return false;
    const slot = t.mode === 'dark' ? 'darkTheme' : 'lightTheme';
    this.#p[slot].set(t.id);
    this.#save(slot);
    const mode = this.#p.theme.peek();
    if (mode !== 'system' && mode !== t.mode) this.theme = t.mode;
    return true;
  }

  /** Installs theme files (their names and text): each one that passes the check, replacing one with its id. */
  installThemes(files: readonly { name: string; text: string }[]): ImportResult[] {
    // Over what is stored now, so themes another tab installed since are kept.
    const { themes, results } = importThemes(storedThemes() ?? this.#customThemes.peek(), files);
    this.#writeThemes(themes);
    return results;
  }

  /** Removes the custom theme `id`. A mode that used it shows its default; its stored choice stays. */
  removeTheme(id: string) {
    this.#writeThemes((storedThemes() ?? this.#customThemes.peek()).filter((t) => t.id !== id));
  }

  #writeThemes(themes: CustomThemeFile[]) {
    this.#customThemes.set(themes);
    try {
      localStorage.setItem(THEMES_KEY, JSON.stringify(themes));
    } catch {
      /* storage may be unavailable or full: the themes then last for the session */
    }
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
    // Changes not written yet (a write that failed, e.g. storage full) go with the next save instead of being lost.
    this.#dirty.add(key);
    const out = { ...(stored() ?? this.#rest) } as Record<keyof Prefs, unknown>;
    for (const k of this.#dirty) out[k] = this.#p[k].peek();
    this.#rest = out as Prefs;
    try {
      localStorage.setItem(KEY, JSON.stringify(out));
      localStorage.setItem('folo.theme', this.#rest.theme);
      this.#dirty.clear();
    } catch {
      /* storage may be unavailable (private mode); preferences then last for the session */
    }
  }
}

export const prefs = new PrefsStore();
