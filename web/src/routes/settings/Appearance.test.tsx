// @vitest-environment happy-dom
// Settings › Appearance (design/THEMES.md, Choosing a theme): the four rows, the theme selects and the mode/slot rule,
// importing theme files (several at once, each one's result said) and removing one.
import { flush } from 'fw';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { prefs } from '$lib/prefs';
import { rank } from '$lib/shell/commands';
import { themeCommands } from '$lib/shell/themeCommands';
import { render } from '../../../framework/testing';
import { parseThemeFile, THEME_FILE_MAX, type ParsedFile } from '$lib/design/themes';
import Appearance, { importPicked, readPicked, resultText } from './Appearance';

const night = { folo_theme: 1, id: 'night', name: 'Night', mode: 'dark', base: 'midnight' };
/** A picked file as read and checked. */
const parsed = (name: string, v: unknown): ParsedFile => ({ file: name, ...parseThemeFile(JSON.stringify(v)) });
let target: HTMLElement;
let view: { unmount(): void } | undefined;

function reset() {
  for (const t of [...prefs.customThemes]) prefs.removeTheme(t.id);
  prefs.useTheme('light');
  prefs.useTheme('dark');
  prefs.theme = 'dark';
}

beforeEach(() => {
  reset();
  target = document.createElement('div');
  document.body.appendChild(target);
});
afterEach(() => {
  view?.unmount();
  view = undefined;
  target.remove();
  reset();
});

const show = () => {
  view = render(() => <Appearance />, target);
  flush();
};
const select = (mode: 'dark' | 'light') => target.querySelector<HTMLSelectElement>(`#appearance-${mode}`)!;
const choose = (mode: 'dark' | 'light', id: string) => {
  const s = select(mode);
  s.querySelector<HTMLOptionElement>(`option[value="${id}"]`)!.selected = true;
  s.dispatchEvent(new Event('change', { bubbles: true }));
  flush();
};
/** Picks files in the import's input and waits for the first file's result line. */
async function importFiles(files: [string, string][]) {
  const input = target.querySelector<HTMLInputElement>('#appearance input[type="file"]')!;
  Object.defineProperty(input, 'files', { configurable: true, value: files.map(([name, body]) => new File([body], name, { type: 'application/json' })) });
  input.dispatchEvent(new Event('change', { bubbles: true }));
  await vi.waitFor(() => {
    flush();
    if (!target.querySelector('#appearance .result')?.textContent?.includes(files[0]![0])) throw new Error('no results yet');
  });
}

describe('Settings › Appearance', () => {
  it('has four rows: the mode, the dark and light themes, custom themes', () => {
    show();
    expect([...target.querySelectorAll('#appearance .s-name')].map((n) => n.textContent)).toEqual(['Theme', 'Dark theme', 'Light theme', 'Custom themes']);
    expect([...select('dark').options].map((o) => o.textContent)).toEqual(['Dark', 'Midnight', 'Dim', 'High contrast dark']);
    expect([...select('light').options].map((o) => o.value)).toEqual(['light', 'paper', 'contrast-light']);
    expect(target.querySelector('#appearance')!.textContent).toContain('Used when the theme is Dark, or System while your system is dark.');
    expect(select('dark').getAttribute('aria-describedby')).toBe('appearance-dark-desc');
    expect(target.querySelector('label[for="appearance-light"]')?.textContent).toBe('Light theme');
  });

  it('picking a light theme under Dark switches the mode; under System it only fills the slot', () => {
    show();
    choose('light', 'paper');
    expect([prefs.theme, prefs.lightTheme, prefs.activeTheme.id]).toEqual(['light', 'paper', 'paper']);
    const radios = () => [...target.querySelectorAll('[role="radio"]')].map((r) => r.getAttribute('aria-checked'));
    expect(radios()).toEqual(['false', 'true', 'false']);
    prefs.theme = 'system';
    flush();
    choose('dark', 'dim');
    expect([prefs.theme, prefs.darkTheme]).toEqual(['system', 'dim']);
  });

  it('re-picking the theme a slot shows does nothing; another one applies', () => {
    show();
    // A browser fires no change for the value already chosen, but the rule doesn't lean on that: a change for the
    // slot's own theme (the same value dispatched) changes nothing. Light in the Light select, under Dark:
    choose('light', 'light');
    expect([prefs.theme, prefs.lightTheme, prefs.activeTheme.id]).toEqual(['dark', 'light', 'dark']);
    // The default a slot shows for a stored id that names no theme counts as its theme: re-picking it keeps the id.
    prefs.installThemes([parsed('n.json', night)]);
    prefs.useTheme('night');
    prefs.removeTheme('night');
    prefs.theme = 'light';
    flush();
    choose('dark', 'dark');
    expect([prefs.theme, prefs.darkTheme, prefs.activeTheme.id]).toEqual(['light', 'night', 'light']);
    // A different theme in the slot applies it and switches the explicit mode to the slot's.
    choose('dark', 'midnight');
    expect([prefs.theme, prefs.darkTheme, prefs.activeTheme.id]).toEqual(['dark', 'midnight', 'midnight']);
  });

  it('shows the default as selected while the stored id names no theme', () => {
    prefs.installThemes([parsed('n.json', night)]);
    prefs.useTheme('night');
    show();
    expect(select('dark').value).toBe('night');
    prefs.removeTheme('night');
    flush();
    expect([prefs.darkTheme, select('dark').value]).toEqual(['night', 'dark']);
  });

  it('imports several files, says what each did, lists them with Remove', async () => {
    show();
    expect(target.querySelector('#appearance .themes')).toBeNull();
    await importFiles([
      ['night.json', JSON.stringify(night)],
      ['typo.json', JSON.stringify({ ...night, id: 'typo', color: { bg: { ap: '#000000' } } })],
      ['notes.txt', 'hello'],
    ]);
    expect([...target.querySelectorAll('#appearance .result')].map((r) => [r.textContent, r.classList.contains('bad')])).toEqual([
      ['Installed Night (night.json)', false],
      ['typo.json: color.bg.ap: not a colour token', true],
      ['notes.txt: not valid JSON (' + (() => { try { JSON.parse('hello'); } catch (e) { return (e as Error).message; } })() + ')', true],
    ]);
    const item = target.querySelector('#appearance .themes li')!;
    expect(item.querySelector('.t-name')?.textContent).toBe('Night');
    expect(item.textContent).toContain('based on Midnight');
    expect([...select('dark').options].map((o) => o.value)).toEqual(['dark', 'midnight', 'dim', 'contrast-dark', 'night']);
    expect(select('dark').querySelector('optgroup')?.label).toBe('Custom');
    await importFiles([['night-2.json', JSON.stringify({ ...night, name: 'Night two' })]]);
    expect(target.querySelector('#appearance .result')?.textContent).toBe('Replaced Night two (night-2.json)');
    item.ownerDocument.querySelector<HTMLButtonElement>('#appearance button[aria-label="Remove Night two"]')!.click();
    flush();
    expect(prefs.customThemes).toEqual([]);
    expect(target.querySelector('#appearance .themes')).toBeNull();
  });

  it('lists the installed themes by name, as the selects and Desktop do, not in the order imported', async () => {
    show();
    await importFiles([['zulu.json', JSON.stringify({ ...night, id: 'zulu', name: 'Zulu' })]]);
    await importFiles([['amber.json', JSON.stringify({ ...night, id: 'amber', name: 'Amber', mode: 'light', base: 'paper' })]]);
    expect([...target.querySelectorAll('#appearance .themes .t-name')].map((n) => n.textContent)).toEqual(['Amber', 'Zulu']);
  });
});

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

describe('importing large picks and failed saves', () => {
  /** What the stub files below did, in order: "read <name>" when a read starts. */
  let log: string[] = [];
  beforeEach(() => {
    log = [];
  });
  /** A picked file whose read is logged (and waits for `gate`, when given); one over the limit must never be read. */
  const file = (name: string, size: number, text: string, gate?: Promise<void>) =>
    ({
      name,
      size,
      arrayBuffer: async () => {
        if (size > THEME_FILE_MAX) throw new Error('a refused file was read');
        log.push(`read ${name}`);
        await gate;
        return new TextEncoder().encode(text).buffer;
      },
    }) as unknown as File;

  it('refuses a file over 1 MB by its size, unread, and checks each file before the next is read', async () => {
    const files = readPicked([file('a.json', 10, JSON.stringify(night)), file('huge.json', THEME_FILE_MAX + 1, ''), file('b.json', 10, '{')]);
    // File 1 is read and checked (its result is out, its text gone) before file 2 is read.
    expect((await files.next()).value).toMatchObject({ file: 'a.json', ok: true, theme: { id: 'night' } });
    expect(log).toEqual(['read a.json']);
    expect((await files.next()).value).toEqual({ file: 'huge.json', ok: false, error: '(file): larger than 1 MB; not a theme' });
    expect(log).toEqual(['read a.json']);
    expect((await files.next()).value).toMatchObject({ file: 'b.json', ok: false });
    expect([log, (await files.next()).done]).toEqual([['read a.json', 'read b.json'], true]);
    const results = await importPicked([file('a.json', 10, JSON.stringify(night)), file('huge.json', THEME_FILE_MAX + 1, ''), file('b.json', 10, '{')]);
    expect(results.map(resultText)).toEqual(['Installed Night (a.json)', 'huge.json: larger than 1 MB; not a theme', expect.stringMatching(/^b\.json: not valid JSON/)]);
    expect((await importPicked([file('edge.json', THEME_FILE_MAX, JSON.stringify(night).padEnd(THEME_FILE_MAX, ' '))]))[0]).toMatchObject({ ok: true, replaced: true });
  });

  it('imports picks in the order made: a second pick waits for the first, so the later one wins', async () => {
    let release!: () => void;
    const gate = new Promise<void>((r) => (release = r));
    const one = importPicked([file('first.json', 10, JSON.stringify({ ...night, name: 'First' }), gate)]);
    const two = importPicked([file('second.json', 10, JSON.stringify({ ...night, name: 'Second' }))]);
    await new Promise((r) => setTimeout(r, 10));
    // The first pick is still reading: the second is not read, and nothing is installed.
    expect([log, prefs.customThemes.length]).toEqual([['read first.json'], 0]);
    release();
    expect((await one).map(resultText)).toEqual(['Installed First (first.json)']);
    expect((await two).map(resultText)).toEqual(['Replaced Second (second.json)']);
    expect(prefs.customThemes.map((t) => t.name)).toEqual(['Second']);
  });

  it('keeps one bad file to its own result: a pick of valid, deeply nested, unreadable, valid installs both valid ones', async () => {
    const n = 400_000;
    const deep = `{"folo_theme": 1, "id": "deep", "name": "Deep", "mode": "dark", "$c": ${'['.repeat(n)}${']'.repeat(n)}}`;
    // Even its size can't be had (a file gone since it was picked).
    const gone = {
      name: 'gone.json',
      get size(): number {
        throw new Error('gone');
      },
      arrayBuffer: () => Promise.reject(new Error('never read')),
    } as unknown as File;
    const dusk = { ...night, id: 'dusk', name: 'Dusk' };
    const results = await importPicked([file('a.json', 10, JSON.stringify(night)), file('deep.json', deep.length, deep), gone, file('b.json', 10, JSON.stringify(dusk))]);
    expect(results.map(resultText)).toEqual(['Installed Night (a.json)', 'deep.json: not valid JSON (nested too deep)', "gone.json: couldn't read (gone)", 'Installed Dusk (b.json)']);
    expect(prefs.customThemes.map((t) => t.id)).toEqual(['night', 'dusk']);
  });

  it('says per file when installing a pick fails, and the next pick still works', async () => {
    const spy = vi.spyOn(prefs, 'installThemes').mockImplementationOnce(() => {
      throw new Error('boom');
    });
    try {
      show();
      await importFiles([['night.json', JSON.stringify(night)]]);
      expect(target.querySelector('#appearance .result')?.textContent).toBe("night.json: couldn't import (boom)");
      await importFiles([['dusk.json', JSON.stringify({ ...night, id: 'dusk', name: 'Dusk' })]]);
      expect(target.querySelector('#appearance .result')?.textContent).toBe('Installed Dusk (dusk.json)');
    } finally {
      spy.mockRestore();
    }
  });

  it("says when the browser couldn't save the custom themes, and stops once a save works", async () => {
    let refuse = true;
    vi.stubGlobal('localStorage', refusing(localStorage, 'folo.themes', () => refuse));
    try {
      show();
      expect(target.querySelector('#appearance .unsaved')).toBeNull();
      await importFiles([['night.json', JSON.stringify(night)]]);
      expect(target.querySelector('#appearance .unsaved')?.textContent).toBe("Couldn't save to this browser's storage; your custom themes last until you close this tab.");
      expect(target.querySelector('#appearance .themes .t-name')?.textContent).toBe('Night');
      refuse = false;
      await importFiles([['dusk.json', JSON.stringify({ ...night, id: 'dusk', name: 'Dusk' })]]);
      expect(target.querySelector('#appearance .unsaved')).toBeNull();
    } finally {
      vi.unstubAllGlobals();
    }
  });
});

describe('the palette\'s theme commands', () => {
  it('offers one per theme for a query, custom ones too, and applies the mode/slot rule', () => {
    prefs.installThemes([parsed('n.json', night)]);
    const items = themeCommands();
    expect(items.map((i) => i.label)).toEqual([
      'Theme: Dark',
      'Theme: Midnight',
      'Theme: Dim',
      'Theme: High contrast dark',
      'Theme: Night',
      'Theme: Light',
      'Theme: Paper',
      'Theme: High contrast light',
    ]);
    expect(items.every((i) => i.scoped)).toBe(true);
    expect(rank(items, '')).toEqual([]);
    expect(rank(items, 'paper')[0]?.label).toBe('Theme: Paper');
    expect(rank(items, 'theme').length).toBe(items.length);
    expect(items[0]!.hint).toEqual(['Dark', 'in use']);
    items.find((i) => i.label === 'Theme: Paper')!.run();
    expect([prefs.theme, prefs.lightTheme]).toEqual(['light', 'paper']);
    themeCommands().find((i) => i.label === 'Theme: Night')!.run();
    expect([prefs.theme, prefs.darkTheme, prefs.activeTheme.base]).toEqual(['dark', 'night', 'midnight']);
  });
});
