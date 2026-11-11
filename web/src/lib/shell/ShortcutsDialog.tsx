// fw port of ShortcutsDialog.svelte.
import { bindable, computed, css, For } from 'fw';
import { isApple } from '$lib/keys';
import Dialog from '$lib/ui/Dialog';
import Kbd from '$lib/ui/Kbd';
import { keyPrefs } from '$lib/ui/keyPrefs';
import { shortcutList } from './shortcuts.fw';

export default function ShortcutsDialog(props: { open?: boolean }) {
  const open = bindable(props, 'open', false);

  /**
   * Off in Settings: the single keys leave their groups for one section of their own, marked as such; a row whose
   * action also has keys that always work (`also`: J K and ↓ ↑) stays in its group with those.
   */
  const working = computed(() =>
    keyPrefs.singleKeys
      ? shortcutList
      : shortcutList.filter((s) => !s.single || s.also).map((s) => (s.single && s.also ? { ...s, keys: s.also, also: undefined } : s)),
  );
  const off = computed(() => (keyPrefs.singleKeys ? [] : shortcutList.filter((s) => s.single).map((s) => ({ ...s, also: undefined }))));
  const groups = computed(() => [...new Set(working.get().map((s) => s.group))]);
  const paletteKeys = isApple ? '⌘ K' : 'Ctrl K';

  // Every group (General to Datasets): two columns where the dialog is wide, one on phones; the body scrolls with a
  // shadow at whichever edge has more (ui/Dialog).
  return (
    <Dialog
      bind:open={open}
      title="Keyboard shortcuts"
      description={
        keyPrefs.singleKeys
          ? 'Everything in Studio works from the keyboard.'
          : `Single-key shortcuts are off in Settings. ${paletteKeys}, the arrow keys and the image inspector's keys still work.`
      }
      size="lg"
      class="shortcuts"
    >
      <div class="groups">
        <For each={groups.get()} key={(g) => g}>
          {(g) => (
            <section>
              <h3 class="overline">{g()}</h3>
              <ul>
                <For each={working.get().filter((s) => s.group === g())} key={(s) => s.keys + s.label}>
                  {(s) => (
                    <li>
                      <span>{s().label}</span>
                      <span class="keys">
                        <Kbd keys={s().keys} />
                        {s().also ? (
                          <>
                            <span class="or">or</span>
                            <Kbd keys={s().also ?? ''} />
                          </>
                        ) : null}
                      </span>
                    </li>
                  )}
                </For>
              </ul>
            </section>
          )}
        </For>
        {off.get().length ? (
          <section class="off">
            <h3 class="overline">Disabled in Settings</h3>
            <ul>
              <For each={off.get()} key={(s) => s.keys + s.label}>
                {(s) => (
                  <li>
                    <span>
                      {s().label}
                      <span class="sr-only">, disabled in Settings</span>
                    </span>
                    <Kbd keys={s().keys} off />
                  </li>
                )}
              </For>
            </ul>
          </section>
        ) : null}
      </div>
    </Dialog>
  );
}

css`
  /* A reference card, not a page: 640 px, two columns of 13 px rows, a group never split across them. */
  /* \`.shortcuts\` twice: fw puts a file's CSS before that of the files it imports (Svelte after), so ui/Dialog's
     \`dialog.lg\` (same specificity) would come later and win; the extra class keeps this rule on top in either order. */
  :global(dialog.shortcuts.shortcuts.lg) {
    width: min(640px, calc(100vw - 32px));
  }
  .groups {
    columns: 260px 2;
    column-gap: var(--space-6);
    /* Each group's own bottom space would add to the dialog's padding after the last one. */
    margin-bottom: calc(-1 * var(--space-5));
  }
  section {
    break-inside: avoid;
    padding-bottom: var(--space-5);
  }
  h3 {
    margin-bottom: var(--space-2);
  }
  ul {
    display: grid;
  }
  li {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-4);
    min-height: 32px;
    border-bottom: 1px solid var(--line-subtle);
    font-size: var(--type-base-size);
    line-height: var(--type-base-line);
    color: var(--text-primary);
  }
  .keys {
    display: inline-flex;
    align-items: center;
    gap: var(--space-2);
    flex-shrink: 0;
  }
  .or {
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
  }
  li:last-child {
    border-bottom: 0;
  }
  /* Keys that do nothing now: tertiary labels and outlined caps, a step back from the working ones in both themes. */
  .off li {
    color: var(--text-tertiary);
  }
  @media (max-width: 767px) {
    :global(dialog.shortcuts.shortcuts.lg) {
      width: 100%;
    }
  }
`;
