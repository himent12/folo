/**
 * Boxes sidebar panel for the labeling editor (`Editor.tsx`): Undo/Redo toolbar with history depth, Add box and Copy
 * previous frame boxes actions, the scrollable box list, and the selected box controls (class select, Duplicate, Delete).
 */
import Copy from '$icons/copy';
import Plus from '$icons/plus';
import Redo from '$icons/redo-2';
import Undo from '$icons/undo-2';
import X from '$icons/x';
import { css, Index } from 'fw';
import type { Box, LabelImage } from '$lib/api/types';
import { classColor } from '$lib/design/tokens';
import Button from '$lib/ui/Button';
import Field from '$lib/ui/Field.tsx';
import Select from '$lib/ui/Select';
import { tooltip } from '$lib/ui/tooltip';
import { boxDimensions, copyPrevTooltip, historyTooltip } from './editorNav';

export interface EditorBoxesPanelProps {
  uid: string;
  classes: readonly string[];
  draft: readonly Box[];
  selected: number | null;
  locked: boolean;
  undoCount: number;
  redoCount: number;
  prevLabeled: LabelImage | null;
  onUndo: () => void;
  onRedo: () => void;
  onAddBox: () => void;
  onCopyPrev: () => void;
  onDuplicateBox: () => void;
  onRemoveBox: () => void;
  onSelect: (index: number | null) => void;
  onHover: (index: number | null) => void;
  onSetClass: (cls: number) => void;
}

export default function EditorBoxesPanel(props: EditorBoxesPanelProps) {
  const name = (c: number) => props.classes[c] ?? `class ${c + 1}`;
  const picked = () => (props.selected === null ? undefined : props.draft[props.selected]);
  const canUndo = () => props.undoCount > 0;
  const canRedo = () => props.redoCount > 0;
  const canCopyPrev = () => !props.locked && props.prevLabeled !== null;

  return (
    <section>
      <div class="head">
        <h3>
          Boxes <span class="meta num">{props.draft.length}</span>
        </h3>
        <span class="tools">
          <span class:off={!canUndo() || props.locked} use:tooltip={historyTooltip('undo', props.undoCount, props.locked)}>
            <Button variant="ghost" size="sm" icon={Undo} aria-label="Undo" disabled={!canUndo() || props.locked} onclick={props.onUndo} />
          </span>
          <span class:off={!canRedo() || props.locked} use:tooltip={historyTooltip('redo', props.redoCount, props.locked)}>
            <Button variant="ghost" size="sm" icon={Redo} aria-label="Redo" disabled={!canRedo() || props.locked} onclick={props.onRedo} />
          </span>
          <Button variant="ghost" size="sm" icon={Plus} disabled={props.locked} data-add-box onclick={props.onAddBox}>
            Add box
          </Button>
        </span>
      </div>

      <div class="seq-row">
        <span class="seq-wrap" class:off={!canCopyPrev()} use:tooltip={copyPrevTooltip(props.prevLabeled, props.locked)}>
          <Button
            variant="secondary"
            size="sm"
            icon={Copy}
            disabled={!canCopyPrev()}
            data-copy-prev
            aria-label={
              props.prevLabeled
                ? `Copy previous frame boxes (${props.prevLabeled.boxes.length} from ${props.prevLabeled.file})`
                : 'Copy previous frame boxes'
            }
            onclick={props.onCopyPrev}
          >
            Copy prev{props.prevLabeled ? ` (${props.prevLabeled.boxes.length})` : ''}
          </Button>
        </span>
      </div>

      {props.undoCount > 0 || props.redoCount > 0 ? (
        <p class="hist-depth num">
          History: {props.undoCount} undo · {props.redoCount} redo
        </p>
      ) : null}

      {props.draft.length ? (
        <ul class="boxes scroll-shade" onpointerleave={() => props.onHover(null)}>
          <Index each={props.draft}>
            {(b, i) => (
              <li>
                <button
                  type="button"
                  class:on={props.selected === i}
                  aria-current={props.selected === i ? 'true' : undefined}
                  aria-label={`${name(b().class)}, ${Math.round(b().x2 - b().x1)} by ${Math.round(b().y2 - b().y1)} pixels`}
                  onpointerenter={(e) => e.pointerType !== 'touch' && props.onHover(i)}
                  onclick={() => props.onSelect(props.selected === i ? null : i)}
                >
                  <span class="sw" style:background={classColor(b().class)}></span>
                  <span class="cn truncate">{name(b().class)}</span>
                  <span class="sz num">{boxDimensions(b())}</span>
                </button>
              </li>
            )}
          </Index>
        </ul>
      ) : (
        <p class="none">No boxes. Saved like this, the image is a background image: no objects.</p>
      )}

      {picked() ? (
        <div class="sel">
          <Field label="Class" for={`${props.uid}-class`} hint="Keys 1–9 set it too.">
            <Select
              id={`${props.uid}-class`}
              value={String(picked()!.class)}
              options={props.classes.map((c, i) => ({ value: String(i), label: `${i + 1} · ${c}` }))}
              disabled={props.locked}
              onchange={(v) => props.onSetClass(Number(v))}
            />
          </Field>
          <div class="sel-actions">
            <span use:tooltip={{ text: 'Duplicate box', keys: 'D' }}>
              <Button size="sm" icon={Copy} disabled={props.locked} data-duplicate-box onclick={props.onDuplicateBox}>
                Duplicate
              </Button>
            </span>
            <Button size="sm" icon={X} disabled={props.locked} onclick={props.onRemoveBox}>
              Delete box
            </Button>
          </div>
        </div>
      ) : null}
    </section>
  );
}

css`
  h3 {
    font-size: var(--type-base-size);
    font-weight: var(--weight-medium);
  }
  .head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-2);
  }
  .tools {
    display: flex;
    align-items: center;
    gap: var(--space-1);
  }
  .tools > span,
  .seq-wrap {
    display: inline-flex;
  }
  /* A disabled button takes no pointer events in some browsers: the wrapper takes the hover so contextual tips show. */
  .tools > span.off,
  .seq-wrap.off {
    cursor: not-allowed;
  }
  .tools > span.off > :global(:disabled),
  .seq-wrap.off > :global(:disabled) {
    pointer-events: none;
  }
  .seq-row {
    display: flex;
    align-items: center;
    gap: var(--space-2);
  }
  .hist-depth {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-tertiary);
  }
  .meta {
    margin-left: var(--space-1);
    color: var(--text-tertiary);
    font-weight: var(--weight-regular);
  }
  .boxes {
    max-height: 220px;
    overflow-y: auto;
  }
  .boxes button {
    display: grid;
    grid-template-columns: 8px minmax(0, 1fr) auto;
    align-items: center;
    gap: var(--space-2);
    width: 100%;
    height: 28px;
    padding: 0 var(--space-2);
    border-radius: var(--radius-sm);
    text-align: left;
  }
  @media (hover: hover) {
    .boxes button:hover {
      background: var(--bg-hover);
    }
  }
  .boxes button.on {
    background: var(--bg-active);
    box-shadow: var(--select-bar);
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
  }
  @media (pointer: coarse) {
    .boxes button {
      height: 44px;
    }
  }
  .sw {
    width: 8px;
    height: 8px;
    border-radius: 2px;
  }
  .sz {
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
  }
  .none {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
  }
  .sel {
    display: grid;
    gap: var(--space-2);
    justify-items: start;
  }
  .sel :global(.field) {
    width: 100%;
  }
  .sel-actions {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: var(--space-2);
  }
`;
