/**
 * Settings › Appearance (design/THEMES.md, Choosing a theme; UNIFIED.md §4): the mode (Dark / Light / System), the
 * theme each mode uses, and the custom themes installed in this browser: Import theme… reads `.json` files (several at
 * once), says per file what it did, and lists what is installed with Remove.
 */
import CircleAlert from '$icons/circle-alert';
import CircleCheck from '$icons/circle-check';
import Monitor from '$icons/monitor';
import Moon from '$icons/moon';
import Sun from '$icons/sun';
import Upload from '$icons/upload';
import { computed, css, effect, For, signal } from 'fw';
import { readThemeFile, type ImportResult, type Mode, type ParsedFile, type Theme } from '$lib/design/themes';
import { themes as builtins } from '$lib/design/tokens';
import { prefs, type ThemePref } from '$lib/prefs';
import Button from '$lib/ui/Button';
import Ident from '$lib/ui/Ident';
import Meta from '$lib/ui/Meta';
import Panel from '$lib/ui/Panel';
import Segmented from '$lib/ui/Segmented';
import Select from '$lib/ui/Select';
import type { IconType } from '$lib/ui/types';
import Row from './Row';

/** The mode is one row's segmented control (design/UNIFIED.md §4: the preview cards went). */
const MODES: { value: ThemePref; label: string; icon: IconType }[] = [
  { value: 'dark', label: 'Dark', icon: Moon },
  { value: 'light', label: 'Light', icon: Sun },
  { value: 'system', label: 'System', icon: Monitor },
];

const SLOT: Record<Mode, { name: string; desc: string }> = {
  dark: { name: 'Dark theme', desc: 'Used when the theme is Dark, or System while your system is dark.' },
  light: { name: 'Light theme', desc: 'Used when the theme is Light, or System while your system is light.' },
};

/** What an import said about one file, as one line: "night.json: color.bg.ap: not a colour token". */
export function resultText(r: ImportResult): string {
  // A whole-file error's path is "(file)": the file's own name says that better.
  return r.ok ? `${r.replaced ? 'Replaced' : 'Installed'} ${r.name} (${r.file})` : `${r.file}: ${r.error.replace(/^\(file\): /, '')}`;
}

/** "Dark · based on Midnight": what a custom theme is, for its line in the list (its id follows). */
function themeFacts(t: Theme): string[] {
  return [t.mode === 'dark' ? 'Dark' : 'Light', ...(t.base !== t.mode ? [`based on ${builtins[t.base].name}`] : [])];
}

/**
 * Reads picked files one at a time, in the order picked, and checks each as soon as it is read (`readThemeFile`: one
 * over 1 MiB is refused by its size, unread): only its result is kept, so a large pick never holds more than one
 * file's bytes and text, and the next file is not read before the last one's result is out.
 */
export async function* readPicked(files: readonly File[]): AsyncGenerator<ParsedFile> {
  for (const f of files) yield await readThemeFile(f);
}

/** The pick being imported: the next one waits for it. */
let importing: Promise<unknown> = Promise.resolve();

/**
 * Imports one pick of files and says per file, in the order picked, what happened. Picks go one after another in the
 * order made (a second pick waits until the first is installed), so of two picks with one id the later wins.
 */
export function importPicked(files: readonly File[]): Promise<ImportResult[]> {
  const run = importing.then(async () => {
    const parsed: ParsedFile[] = [];
    for await (const r of readPicked(files)) parsed.push(r);
    return prefs.installThemes(parsed);
  });
  importing = run.catch(() => undefined);
  return run;
}

export default function Appearance() {
  // The mode, cycled from the top bar or the palette too while the page is open.
  const mode = signal<ThemePref>(prefs.theme);
  effect(() => mode.set(prefs.theme));
  const results = signal<ImportResult[]>([]);
  let input: HTMLInputElement | undefined;

  async function imported() {
    const files = [...(input?.files ?? [])];
    if (input) input.value = '';
    if (!files.length) return;
    results.set(await importPicked(files));
  }

  const slot = (m: Mode) => {
    const id = `appearance-${m}`;
    const value = signal(prefs.slotTheme(m).id);
    // What the mode shows: the default too when the stored id names no theme now (design/THEMES.md).
    effect(() => value.set(prefs.slotTheme(m).id));
    const options = computed(() =>
      prefs.themesFor(m).map((t) => ({ value: t.id, label: t.name, title: t.description || undefined, group: t.custom ? 'Custom' : undefined })),
    );
    return (
      <Row
        name={SLOT[m].name}
        for={id}
        desc={SLOT[m].desc}
        descId={`${id}-desc`}
        control={
          <span class="pick">
            <Select id={id} bind:value={value} options={options.get()} describedby={`${id}-desc`} onchange={(v) => prefs.useTheme(v)} />
          </span>
        }
      />
    );
  };

  // The resolved catalog's custom themes, in its order (by name, as Desktop lists them): nothing is rebuilt per row.
  const custom = computed(() => prefs.installedThemes);

  return (
    <Panel title="Appearance" meta="saved in this browser" id="appearance" divided pad="none">
      <Row
        name="Theme"
        desc={prefs.theme === 'system' ? <>Follows your operating system, currently {prefs.resolvedTheme}.</> : 'Dark or light. The top bar and the command palette switch it too.'}
        control={<Segmented label="Theme" bind:value={mode} onchange={(v) => (prefs.theme = v)} options={MODES} />}
      />
      {slot('dark')}
      {slot('light')}
      <Row
        name="Custom themes"
        desc={<>Theme files (.json) you import, kept in this browser.</>}
        control={
          <>
            <input ref={(e) => (input = e)} type="file" accept=".json,application/json" multiple hidden onchange={() => void imported()} />
            <Button icon={Upload} onclick={() => input?.click()}>
              Import theme…
            </Button>
          </>
        }
        below={
          results.get().length || custom.get().length || prefs.themesUnsaved ? (
            <div class="custom">
              {prefs.themesUnsaved ? (
                <p class="unsaved" role="status">
                  Couldn't save to this browser's storage; your custom themes last until you close this tab.
                </p>
              ) : null}
              {results.get().length ? (
                <ul class="results" aria-label="Import results" role="status">
                  <For each={results.get()} key={(r) => `${r.file}\u0000${resultText(r)}`}>
                    {(r) => (
                      <li class="result" class:bad={!r().ok}>
                        {r().ok ? <CircleCheck size={14} strokeWidth={1.75} aria-hidden="true" /> : <CircleAlert size={14} strokeWidth={1.75} aria-hidden="true" />}
                        <span>{resultText(r())}</span>
                      </li>
                    )}
                  </For>
                </ul>
              ) : null}
              {custom.get().length ? (
                <ul class="themes" aria-label="Custom themes">
                  <For each={custom.get()} key={(t) => t.id}>
                    {(t) => (
                      <li class="theme">
                        <div class="t-main">
                          <p class="t-name">{t().name}</p>
                          <p class="t-meta">
                            <Meta>
                              {themeFacts(t()).map((f) => (
                                <span>{f}</span>
                              ))}
                              <span>
                                <Ident value={t().id} kind="id" />
                              </span>
                            </Meta>
                          </p>
                        </div>
                        <Button size="sm" aria-label={`Remove ${t().name}`} onclick={() => prefs.removeTheme(t().id)}>
                          Remove
                        </Button>
                      </li>
                    )}
                  </For>
                </ul>
              ) : null}
            </div>
          ) : undefined
        }
      />
    </Panel>
  );
}

css`
  /* One width for both theme selects, so the two rows' controls line up. */
  .pick {
    display: flex;
    width: 200px;
    max-width: 100%;
  }
  .pick > :global(.select) {
    flex: 1;
  }
  .custom {
    display: grid;
    gap: var(--space-2);
    padding-bottom: var(--space-2);
  }
  .unsaved {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
  }
  .results {
    display: grid;
    gap: var(--space-1);
  }
  .result {
    display: flex;
    align-items: flex-start;
    gap: var(--space-2);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
  }
  .result > :global(svg) {
    flex-shrink: 0;
    /* The 14 px mark centred on the first 16 px line. */
    margin-top: 1px;
    color: var(--success-solid);
  }
  .result.bad {
    color: var(--danger-text);
  }
  .result.bad > :global(svg) {
    color: var(--danger-solid);
  }
  .result > span {
    min-width: 0;
    overflow-wrap: anywhere;
  }
  .themes {
    border: 1px solid var(--line-subtle);
    border-radius: var(--radius-md);
  }
  .theme {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-4);
    min-height: 48px;
    padding: var(--space-2) var(--space-3);
  }
  .theme + .theme {
    border-top: 1px solid var(--line-subtle);
  }
  .t-main {
    min-width: 0;
    display: grid;
    gap: 2px;
  }
  .t-name {
    color: var(--text-primary);
  }
  .t-meta {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-tertiary);
  }
`;
