// @vitest-environment happy-dom
// Themes in the preferences (design/THEMES.md): each mode's slot, the mode/slot rule of Choosing a theme, the custom
// themes store (`folo.themes`) across tabs, and what is stored.
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { parseThemeFile, type CustomThemeFile, type ParsedFile } from './design/themes';
import { PrefsStore, sanitize, THEMES_KEY } from './prefs';

/** A picked file as read and checked (readThemeFile does this to a File). */
const file = (name: string, text: string): ParsedFile => ({ file: name, ...parseThemeFile(text) });
const prefsOf = () => JSON.parse(localStorage.getItem('folo.prefs')!) as Record<string, unknown>;
const night: CustomThemeFile = { folo_theme: 1, id: 'night', name: 'Night', mode: 'dark', base: 'midnight', color: { bg: { app: '#0B0D12' } } };
/** The browser's storage itself (another tab writes it directly, even while this tab's writes are refused). */
const realStorage = localStorage;
const fromOtherTab = (key: string, value: string) => {
  realStorage.setItem(key, value);
  window.dispatchEvent(new StorageEvent('storage', { key, newValue: value }));
};

/** A localStorage whose writes of `key` throw while `refuse()` (storage full); reads and other keys go through. */
function refusing(real: Storage, key: string, refuse: () => boolean): Storage {
  return {
    getItem: (k: string) => real.getItem(k),
    setItem: (k: string, v: string) => {
      if (refuse() && k === key) throw new DOMException('full', 'QuotaExceededError');
      real.setItem(k, v);
    },
    removeItem: (k: string) => real.removeItem(k),
    clear: () => real.clear(),
    key: (i: number) => real.key(i),
    get length() {
      return real.length;
    },
  };
}

describe('theme preferences', () => {
  beforeEach(() => localStorage.clear());
  afterEach(() => localStorage.clear());

  it('defaults each mode to its own theme', () => {
    const p = new PrefsStore();
    expect([p.darkTheme, p.lightTheme, p.activeTheme.id]).toEqual(['dark', 'light', 'dark']);
    p.theme = 'light';
    expect(p.activeTheme).toMatchObject({ id: 'light', base: 'light', mode: 'light', vars: [] });
  });

  it('keeps a stored id that names no theme, and shows the default for it', () => {
    expect(sanitize({ darkTheme: 'gone', lightTheme: 7 })).toMatchObject({ darkTheme: 'gone', lightTheme: 'light' });
    localStorage.setItem('folo.prefs', JSON.stringify({ theme: 'dark', darkTheme: 'gone' }));
    const p = new PrefsStore();
    expect([p.darkTheme, p.activeTheme.id, p.slotTheme('dark').id]).toEqual(['gone', 'dark', 'dark']);
    // Installing a theme with that id brings it back.
    p.installThemes([file('gone.json', JSON.stringify({ ...night, id: 'gone' }))]);
    expect(p.activeTheme).toMatchObject({ id: 'gone', custom: true });
  });

  it("picks a theme of the other mode: switches the mode under Dark or Light, keeps System", () => {
    const p = new PrefsStore();
    expect(p.useTheme('paper')).toBe(true);
    expect([p.theme, p.lightTheme, p.darkTheme, p.activeTheme.id]).toEqual(['light', 'paper', 'dark', 'paper']);
    p.useTheme('midnight');
    expect([p.theme, p.darkTheme, p.lightTheme]).toEqual(['dark', 'midnight', 'paper']);
    p.theme = 'system';
    p.useTheme('contrast-light');
    expect([p.theme, p.lightTheme]).toEqual(['system', 'contrast-light']);
    expect(p.useTheme('nothing')).toBe(false);
    expect(prefsOf()).toMatchObject({ theme: 'system', darkTheme: 'midnight', lightTheme: 'contrast-light' });
    // A new tab reads them back.
    const q = new PrefsStore();
    expect([q.theme, q.darkTheme, q.lightTheme]).toEqual(['system', 'midnight', 'contrast-light']);
  });

  it('installs, stores, removes custom themes; a mode that used a removed one shows its default', () => {
    const p = new PrefsStore();
    const r = p.installThemes([
      file('night.json', JSON.stringify(night)),
      file('bad.json', '{'),
    ]);
    expect(r.map((x) => x.ok)).toEqual([true, false]);
    expect(JSON.parse(localStorage.getItem(THEMES_KEY)!)).toEqual([night]);
    expect(p.themesFor('dark').map((t) => t.id)).toEqual(['dark', 'midnight', 'dim', 'contrast-dark', 'night']);
    p.useTheme('night');
    expect(p.activeTheme).toMatchObject({ id: 'night', base: 'midnight', vars: [['--bg-app', '#0b0d12']] });
    p.removeTheme('night');
    expect([p.darkTheme, p.activeTheme.id, p.customThemes.length]).toEqual(['night', 'dark', 0]);
    expect(JSON.parse(localStorage.getItem(THEMES_KEY)!)).toEqual([]);
  });

  it("follows another tab's custom themes and theme choices", () => {
    const p = new PrefsStore();
    fromOtherTab(THEMES_KEY, JSON.stringify([night]));
    fromOtherTab('folo.prefs', JSON.stringify({ theme: 'dark', darkTheme: 'night' }));
    expect(p.activeTheme.id).toBe('night');
    // Installing here keeps what the other tab installed.
    const q = new PrefsStore();
    q.installThemes([file('b.json', JSON.stringify({ ...night, id: 'b', name: 'B' }))]);
    fromOtherTab(THEMES_KEY, localStorage.getItem(THEMES_KEY)!);
    expect(p.customThemes.map((t) => t.id)).toEqual(['night', 'b']);
    localStorage.clear();
    window.dispatchEvent(new StorageEvent('storage', { key: null }));
    expect([p.customThemes.length, p.activeTheme.id]).toEqual([0, 'dark']);
  });

  describe('when the browser refuses to write folo.themes (storage full), reads still working', () => {
    const theme = (id: string) => file(`${id}.json`, JSON.stringify({ ...night, id, name: id.toUpperCase() }));
    let refuse = true;
    beforeEach(() => {
      refuse = true;
      vi.stubGlobal('localStorage', refusing(realStorage, THEMES_KEY, () => refuse));
    });
    afterEach(() => vi.unstubAllGlobals());

    it('keeps an unsaved import over the stored list: import A, select A, import B keeps both', () => {
      const p = new PrefsStore();
      expect(p.installThemes([theme('a')])[0]).toMatchObject({ ok: true, replaced: false });
      expect(p.themesUnsaved).toBe(true);
      expect(localStorage.getItem(THEMES_KEY)).toBeNull();
      expect(p.useTheme('a')).toBe(true);
      expect(p.installThemes([theme('b')])[0]).toMatchObject({ ok: true, replaced: false });
      expect(p.customThemes.map((t) => t.id)).toEqual(['a', 'b']);
      expect(p.activeTheme.id).toBe('a');
      // Replacing an unsaved theme says Replaced: the overlay is what is installed.
      expect(p.installThemes([theme('a')])[0]).toMatchObject({ ok: true, replaced: true });
      // Another tab's change keeps this tab's unsaved ones over it.
      fromOtherTab(THEMES_KEY, JSON.stringify([{ ...night, id: 'c', name: 'C' }]));
      expect(p.customThemes.map((t) => t.id)).toEqual(['c', 'a', 'b']);
      // The next write that works saves everything, and the note goes.
      refuse = false;
      p.removeTheme('c');
      expect(p.themesUnsaved).toBe(false);
      expect((JSON.parse(localStorage.getItem(THEMES_KEY)!) as { id: string }[]).map((t) => t.id)).toEqual(['a', 'b']);
    });

    it('a failed removal is not undone by the next operation', () => {
      refuse = false;
      const p = new PrefsStore();
      p.installThemes([theme('a'), theme('b')]);
      refuse = true;
      p.removeTheme('a');
      expect([p.customThemes.map((t) => t.id), p.themesUnsaved]).toEqual([['b'], true]);
      // Still stored: the write failed.
      expect((JSON.parse(localStorage.getItem(THEMES_KEY)!) as { id: string }[]).map((t) => t.id)).toEqual(['a', 'b']);
      p.installThemes([theme('c')]);
      expect(p.customThemes.map((t) => t.id)).toEqual(['b', 'c']);
      fromOtherTab('folo.prefs', JSON.stringify({ theme: 'dark' }));
      window.dispatchEvent(new StorageEvent('storage', { key: THEMES_KEY }));
      expect(p.customThemes.map((t) => t.id)).toEqual(['b', 'c']);
    });
  });

  describe("when storage can't be read or written (blocked, or localStorage null)", () => {
    const theme = (id: string, name = id.toUpperCase()) => file(`${id}.json`, JSON.stringify({ ...night, id, name }));
    const ids = (p: PrefsStore) => p.customThemes.map((t) => t.id);
    afterEach(() => vi.unstubAllGlobals());

    it('localStorage null: an import installs for this tab, usable in the selects and removable', () => {
      vi.stubGlobal('localStorage', null);
      const p = new PrefsStore();
      expect(p.installThemes([theme('aaa')])).toEqual([{ file: 'aaa.json', ok: true, id: 'aaa', name: 'AAA', replaced: false }]);
      expect([ids(p), p.themesUnsaved]).toEqual([['aaa'], true]);
      expect(p.themesFor('dark').map((t) => t.id)).toContain('aaa');
      expect(p.useTheme('aaa')).toBe(true);
      expect(p.activeTheme.id).toBe('aaa');
      expect(p.installThemes([theme('bbb')])[0]).toMatchObject({ ok: true, replaced: false });
      expect(p.installThemes([theme('aaa', 'Again')])[0]).toMatchObject({ ok: true, replaced: true });
      expect(ids(p)).toEqual(['aaa', 'bbb']);
      p.removeTheme('aaa');
      expect([ids(p), p.activeTheme.id]).toEqual([['bbb'], 'dark']);
    });

    it('reads and writes both throwing: the same, then reconciled over what storage holds once it is back', () => {
      realStorage.setItem(THEMES_KEY, JSON.stringify([{ ...night, id: 'old', name: 'Old' }]));
      let blocked = false;
      const fail = () => {
        if (blocked) throw new DOMException('blocked', 'SecurityError');
      };
      vi.stubGlobal('localStorage', {
        getItem: (k: string) => (fail(), realStorage.getItem(k)),
        setItem: (k: string, v: string) => (fail(), realStorage.setItem(k, v)),
        removeItem: (k: string) => (fail(), realStorage.removeItem(k)),
        clear: () => realStorage.clear(),
        key: (i: number) => realStorage.key(i),
        get length() {
          return realStorage.length;
        },
      });
      const p = new PrefsStore();
      expect(ids(p)).toEqual(['old']);
      blocked = true;
      p.installThemes([theme('a'), theme('b')]);
      p.removeTheme('old');
      expect([ids(p), p.themesUnsaved]).toEqual([['a', 'b'], true]);
      p.removeTheme('a');
      expect(ids(p)).toEqual(['b']);
      expect(p.useTheme('b')).toBe(true);
      // A storage event while blocked changes nothing.
      window.dispatchEvent(new StorageEvent('storage', { key: THEMES_KEY }));
      expect(ids(p)).toEqual(['b']);
      // Storage comes back holding another tab's change: the pending ones go over it, and the next write saves them.
      blocked = false;
      realStorage.setItem(THEMES_KEY, JSON.stringify([{ ...night, id: 'old', name: 'Old' }, { ...night, id: 'other', name: 'Other' }]));
      window.dispatchEvent(new StorageEvent('storage', { key: THEMES_KEY }));
      expect([ids(p), p.themesUnsaved]).toEqual([['other', 'b'], true]);
      p.installThemes([theme('c')]);
      expect([ids(p), p.themesUnsaved]).toEqual([['other', 'b', 'c'], false]);
      expect((JSON.parse(realStorage.getItem(THEMES_KEY)!) as { id: string }[]).map((t) => t.id)).toEqual(['other', 'b', 'c']);
    });
  });

  it('lists the installed themes in the catalog order (by name), not the order installed', () => {
    const p = new PrefsStore();
    p.installThemes([file('z.json', JSON.stringify({ ...night, id: 'zulu', name: 'Zulu' })), file('a.json', JSON.stringify({ ...night, id: 'amber', name: 'Amber', mode: 'light', base: 'paper' }))]);
    expect(p.customThemes.map((t) => t.id)).toEqual(['zulu', 'amber']);
    expect(p.installedThemes.map((t) => t.id)).toEqual(['amber', 'zulu']);
  });

  it('resolves each theme once: the lists and lookups are one catalog until the themes change', () => {
    const p = new PrefsStore();
    p.installThemes([file('n.json', JSON.stringify(night))]);
    const list = p.themesFor('dark');
    expect(p.themesFor('dark')).toBe(list);
    expect(p.installedThemes[0]).toBe(list.at(-1));
    expect(p.slotTheme('dark')).toBe(list[0]);
    p.installThemes([file('m.json', JSON.stringify({ ...night, id: 'm', name: 'M' }))]);
    expect(p.themesFor('dark')).not.toBe(list);
  });

  it('ignores a stored list that is not JSON or holds broken entries', () => {
    localStorage.setItem(THEMES_KEY, '[{"folo_theme":1');
    expect(new PrefsStore().customThemes).toEqual([]);
    localStorage.setItem(THEMES_KEY, JSON.stringify([night, { ...night, id: 'x', mode: 'grey' }]));
    expect(new PrefsStore().customThemes.map((t) => t.id)).toEqual(['night']);
  });
});
