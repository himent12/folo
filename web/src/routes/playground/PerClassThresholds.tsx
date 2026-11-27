/**
 * Collapsible per-class confidence threshold controls for Playground: inline per-class overrides (range + numeric
 * input + per-row reset), Reset all, and JSON import/export supporting both `folo-vision`'s `ThresholdCalibration`
 * format (`{ global, per_class: [{ class, best_conf, ... }] }`) and simple `{ "0": 0.4 }` / `{ "person": 0.4 }` maps.
 */
import ArrowDownToLine from '$icons/arrow-down-to-line';
import ChevronRight from '$icons/chevron-right';
import RotateCcw from '$icons/rotate-ccw';
import Upload from '$icons/upload';
import X from '$icons/x';
import { computed, createUniqueId, css, For, signal } from 'fw';
import { classColor } from '$lib/design/tokens';
import { downloadText, safeFilename, toJson } from '$lib/tableExport';
import Button from '$lib/ui/Button';
import { toast } from '$lib/ui/toast';
import Swatch from './Swatch';
import {
  exportThresholdsJson,
  parseThresholdsJson,
  perClassConfCount,
  roundConf,
  type PerClassConfMap,
} from './perClassConf';

export interface PerClassThresholdsProps {
  classes: string[];
  globalConf: number;
  perClassConf: PerClassConfMap;
  modelName?: string | null;
  inline?: boolean;
  onSetClassConf: (classId: number, value: number) => void;
  onClearClassConf: (classId: number) => void;
  onResetAll: () => void;
  onImport: (res: { globalConf: number | null; perClassConf: PerClassConfMap }) => void;
}

export default function PerClassThresholds(props: PerClassThresholdsProps) {
  const open = signal(false);
  const importOpen = signal(false);
  const importText = signal('');
  const importError = signal<string | null>(null);
  const importNote = signal<string | null>(null);
  const filterQuery = signal('');
  const jsonInputId = `pc-json-${createUniqueId()}`;
  let fileInput!: HTMLInputElement;

  const overrideCount = computed(() => perClassConfCount(props.perClassConf));

  const rows = computed(() => {
    const list: { id: number; name: string }[] = props.classes.map((name, id) => ({ id, name }));
    for (const k of Object.keys(props.perClassConf)) {
      const id = Number(k);
      if (Number.isInteger(id) && id >= 0 && id >= props.classes.length) {
        list.push({ id, name: `class ${id}` });
      }
    }
    list.sort((a, b) => a.id - b.id);
    const q = filterQuery.get().trim().toLowerCase();
    return q ? list.filter((r) => r.name.toLowerCase().includes(q) || String(r.id) === q) : list;
  });

  function applyImportText(raw: string) {
    const parsed = parseThresholdsJson(raw, props.classes);
    if (!parsed.ok) {
      importError.set(parsed.error);
      importNote.set(null);
      return;
    }
    importError.set(null);
    props.onImport({ globalConf: parsed.globalConf, perClassConf: parsed.perClassConf });
    importOpen.set(false);
    importText.set('');
    const parts: string[] = [];
    if (parsed.globalConf !== null) parts.push(`global ${parsed.globalConf.toFixed(2)}`);
    parts.push(`${parsed.appliedCount} ${parsed.appliedCount === 1 ? 'class override' : 'class overrides'}`);
    const note = parsed.warnings.length ? `${parts.join(', ')} (${parsed.warnings[0]})` : parts.join(', ');
    importNote.set(`Applied ${note}.`);
    toast.success('Thresholds imported', `Applied ${note}.`, undefined, { key: 'playground-thresholds' });
  }

  async function onPickFile(e: Event) {
    const input = e.currentTarget as HTMLInputElement;
    const f = input.files?.[0];
    input.value = '';
    if (!f) return;
    try {
      const text = await f.text();
      importText.set(text);
      applyImportText(text);
    } catch {
      importError.set(`Could not read "${f.name}".`);
    }
  }

  function onExportJson() {
    const payload = exportThresholdsJson(props.globalConf, props.perClassConf, props.classes);
    const base = `thresholds-${props.modelName || 'playground'}`;
    const ok = downloadText(
      toJson(payload),
      safeFilename(base, 'json'),
      'application/json;charset=utf-8',
      'playground-thresholds',
    );
    if (ok) {
      const n = payload.per_class.length;
      toast.success(
        'Thresholds exported',
        `Saved global ${payload.global.best_conf.toFixed(2)} and ${n} ${n === 1 ? 'class override' : 'class overrides'}.`,
        undefined,
        { key: 'playground-thresholds' },
      );
    }
  }

  function onNumCommit(classId: number, raw: string) {
    const trimmed = raw.trim();
    if (!trimmed) {
      props.onClearClassConf(classId);
      return;
    }
    const v = Number(trimmed);
    if (!Number.isFinite(v)) return;
    props.onSetClassConf(classId, roundConf(v));
  }

  return (
    <div class="pc-root" class:inline={!!props.inline} data-per-class-thresholds>
      <input
        ref={(el) => (fileInput = el)}
        type="file"
        accept=".json,application/json"
        class="sr-only"
        aria-label="Upload threshold calibration JSON"
        onchange={(e) => void onPickFile(e)}
      />
      <button
        type="button"
        class="pc-toggle"
        aria-expanded={open.get()}
        onclick={() => open.set(!open.peek())}
      >
        <ChevronRight size={14} class={open.get() ? 'chev open' : 'chev'} aria-hidden="true" />
        <span class="pc-title">Per-class thresholds</span>
        <span class="pc-badge num" class:active={overrideCount.get() > 0} data-per-class-badge>
          {overrideCount.get() > 0
            ? `${overrideCount.get()} ${overrideCount.get() === 1 ? 'override' : 'overrides'}`
            : 'Default'}
        </span>
      </button>

      {open.get() ? (
        <div class="pc-body">
          <div class="pc-actions">
            <Button
              size="sm"
              variant="ghost"
              icon={Upload}
              data-per-class-import-btn
              onclick={() => {
                importOpen.set(!importOpen.peek());
                importError.set(null);
              }}
            >
              Import JSON
            </Button>
            <Button size="sm" variant="ghost" icon={ArrowDownToLine} data-per-class-export-btn onclick={onExportJson}>
              Export JSON
            </Button>
            <Button
              size="sm"
              variant="ghost"
              icon={RotateCcw}
              data-per-class-reset-all
              disabled={overrideCount.get() === 0}
              onclick={() => {
                props.onResetAll();
                importNote.set(null);
              }}
            >
              Reset all
            </Button>
          </div>

          {importOpen.get() ? (
            <div class="pc-import" data-per-class-import>
              <label class="sr-only" for={jsonInputId}>
                Threshold calibration JSON
              </label>
              <textarea
                id={jsonInputId}
                class="pc-textarea mono"
                rows={4}
                placeholder={'{"global":{"best_conf":0.35},"per_class":[{"class":0,"best_conf":0.5}]}\nor {"person": 0.4, "0": 0.6}'}
                value={importText.get()}
                oninput={(e) => {
                  importText.set((e.currentTarget as HTMLTextAreaElement).value);
                  importError.set(null);
                }}
              />
              {importError.get() ? (
                <p class="pc-err" role="alert" data-per-class-error>
                  {importError.get()}
                </p>
              ) : null}
              <div class="pc-import-row">
                <Button size="sm" variant="primary" data-per-class-apply onclick={() => applyImportText(importText.peek())}>
                  Apply
                </Button>
                <Button size="sm" variant="ghost" onclick={() => fileInput.click()}>
                  Upload .json
                </Button>
                <Button
                  size="sm"
                  variant="ghost"
                  onclick={() => {
                    importOpen.set(false);
                    importError.set(null);
                  }}
                >
                  Cancel
                </Button>
              </div>
            </div>
          ) : null}

          {importNote.get() ? (
            <p class="pc-note" role="status" data-per-class-status>
              {importNote.get()}
            </p>
          ) : null}

          {props.classes.length > 8 ? (
            <input
              type="search"
              class="pc-search"
              placeholder={`Filter ${props.classes.length} classes…`}
              aria-label="Filter classes"
              value={filterQuery.get()}
              oninput={(e) => filterQuery.set((e.currentTarget as HTMLInputElement).value)}
            />
          ) : null}

          {rows.get().length === 0 ? (
            <p class="pc-empty">
              {props.classes.length === 0 ? 'Select a model to customize class thresholds.' : 'No matching classes.'}
            </p>
          ) : (
            <ul class="pc-list" aria-label="Per-class confidence overrides">
              <For each={rows.get()} key={(r) => r.id}>
                {(r) => {
                  const id = () => r().id;
                  const name = () => r().name;
                  const hasOv = () => Object.prototype.hasOwnProperty.call(props.perClassConf, id());
                  const eff = () => (hasOv() ? props.perClassConf[id()]! : props.globalConf);
                  return (
                    <li class="pc-row" class:overridden={hasOv()} data-class-threshold-row={id()}>
                      <span class="pc-class">
                        <Swatch color={classColor(id())} />
                        <span class="pc-name truncate" title={name()}>
                          {name()}
                        </span>
                      </span>
                      <input
                        type="range"
                        class="pc-range"
                        min={0}
                        max={1}
                        step={0.01}
                        aria-label={`${name()} confidence slider`}
                        value={String(eff())}
                        oninput={(e) => props.onSetClassConf(id(), roundConf(Number((e.currentTarget as HTMLInputElement).value)))}
                      />
                      <input
                        type="number"
                        class="pc-num num"
                        min={0}
                        max={1}
                        step="any"
                        aria-label={`${name()} confidence threshold`}
                        placeholder={props.globalConf.toFixed(2)}
                        value={hasOv() ? String(props.perClassConf[id()]!) : ''}
                        onchange={(e) => onNumCommit(id(), (e.currentTarget as HTMLInputElement).value)}
                      />
                      <button
                        type="button"
                        class="pc-clear"
                        disabled={!hasOv()}
                        aria-label={`Reset ${name()} threshold`}
                        title={hasOv() ? `Reset to global (${props.globalConf.toFixed(2)})` : 'Using global confidence'}
                        onclick={() => props.onClearClassConf(id())}
                      >
                        <X size={12} strokeWidth={2} aria-hidden="true" />
                      </button>
                    </li>
                  );
                }}
              </For>
            </ul>
          )}
        </div>
      ) : null}
    </div>
  );
}

css`
  .pc-root {
    display: grid;
    gap: var(--space-2);
    margin-top: var(--space-3);
    padding-top: var(--space-3);
    border-top: 1px solid var(--line-subtle);
    min-width: 0;
  }
  .pc-toggle {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    width: 100%;
    min-height: var(--control-sm);
    padding: 0;
    color: var(--text-primary);
    text-align: left;
  }
  .pc-toggle :global(svg.chev) {
    flex-shrink: 0;
    color: var(--text-tertiary);
    transition: transform var(--dur-fast) var(--ease-standard);
  }
  .pc-toggle :global(svg.chev.open) {
    transform: rotate(90deg);
  }
  .pc-title {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    font-weight: var(--weight-medium);
    color: var(--text-secondary);
  }
  .pc-toggle:hover .pc-title {
    color: var(--text-primary);
  }
  .pc-badge {
    margin-left: auto;
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
    white-space: nowrap;
  }
  .pc-badge.active {
    padding: 0 var(--space-2);
    border-radius: var(--radius-sm);
    background: var(--accent-muted);
    color: var(--accent-text);
    font-weight: var(--weight-medium);
  }
  .pc-body {
    display: grid;
    gap: var(--space-2);
    min-width: 0;
  }
  .pc-actions {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: var(--space-1);
    margin-left: calc(-1 * var(--space-2));
  }
  .pc-import {
    display: grid;
    gap: var(--space-2);
    padding: var(--space-2);
    border: 1px solid var(--line-subtle);
    border-radius: var(--radius-md);
    background: var(--bg-inset);
  }
  .pc-textarea {
    width: 100%;
    min-height: 76px;
    padding: var(--space-2);
    border: 1px solid var(--line-default);
    border-radius: var(--radius-sm);
    background: var(--bg-panel);
    color: var(--text-primary);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    resize: vertical;
  }
  .pc-import-row {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: var(--space-1);
  }
  .pc-err {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--danger-text);
  }
  .pc-note {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-tertiary);
  }
  .pc-search {
    width: 100%;
    height: var(--control-sm);
    padding: 0 var(--space-2);
    border: 1px solid var(--line-default);
    border-radius: var(--radius-sm);
    background: var(--bg-inset);
    color: var(--text-primary);
    font-size: var(--type-sm-size);
  }
  .pc-empty {
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
  }
  .pc-list {
    display: grid;
    gap: var(--space-1);
    max-height: 220px;
    overflow-y: auto;
    overscroll-behavior: contain;
    min-width: 0;
  }
  .pc-row {
    display: grid;
    grid-template-columns: minmax(0, 1fr) 64px 56px 24px;
    align-items: center;
    gap: var(--space-2);
    min-height: 28px;
  }
  .pc-class {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    min-width: 0;
  }
  .pc-name {
    font-size: var(--type-sm-size);
    color: var(--text-secondary);
  }
  .pc-row.overridden .pc-name {
    color: var(--text-primary);
    font-weight: var(--weight-medium);
  }
  .pc-range {
    width: 100%;
    margin: 0;
    accent-color: var(--accent-solid);
    cursor: pointer;
  }
  .pc-num {
    width: 100%;
    height: 24px;
    padding: 0 var(--space-1);
    border: 1px solid var(--line-default);
    border-radius: var(--radius-sm);
    background: var(--bg-inset);
    color: var(--text-primary);
    font-size: var(--type-sm-size);
    text-align: right;
  }
  .pc-row.overridden .pc-num {
    border-color: var(--accent-solid);
  }
  .pc-clear {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 24px;
    height: 24px;
    border-radius: var(--radius-sm);
    color: var(--text-tertiary);
  }
  .pc-clear:not(:disabled):hover {
    background: var(--bg-hover);
    color: var(--text-primary);
  }
  .pc-clear:disabled {
    opacity: 0.25;
    cursor: default;
  }
  @media (pointer: coarse) {
    .pc-textarea,
    .pc-search,
    .pc-num {
      font-size: var(--type-lg-size);
    }
    .pc-num,
    .pc-search {
      min-height: 44px;
    }
    .pc-clear {
      width: 44px;
      height: 44px;
    }
    .pc-row {
      grid-template-columns: minmax(0, 1fr) 64px 64px 44px;
      min-height: 44px;
    }
  }
`;
