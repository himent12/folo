// @vitest-environment happy-dom
// Settings › Appearance (design/THEMES.md, Choosing a theme): the four rows, the theme selects and the mode/slot rule,
// importing theme files (several at once, each one's result said) and removing one.
import { flush } from 'fw';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { prefs } from '$lib/prefs';
import { rank } from '$lib/shell/commands';
import { themeCommands } from '$lib/shell/themeCommands';
import { render } from '../../../framework/testing';
import Appearance from './Appearance';

const night = { folo_theme: 1, id: 'night', name: 'Night', mode: 'dark', base: 'midnight' };
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

  it('shows the default as selected while the stored id names no theme', () => {
    prefs.installThemes([{ name: 'n.json', text: JSON.stringify(night) }]);
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
});

describe('the palette\'s theme commands', () => {
  it('offers one per theme for a query, custom ones too, and applies the mode/slot rule', () => {
    prefs.installThemes([{ name: 'n.json', text: JSON.stringify(night) }]);
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
