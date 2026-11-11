/**
 * Label + control + hint/error, laid out consistently. The control is the child snippet; pass `for` = its id, and
 * mark the control invalid (`invalid` on Input / Select / Segmented) while `error` is set. The error carries a danger
 * icon as well as its colour, so it is never red text alone. Where a valid value can be derived from what was typed,
 * `fix` offers it as one click after the message ("Try my-dataset"), an inline action in ink.
 *
 * The message shown is the control's accessible description, with no wiring by the caller: Field publishes its id
 * and the error state (`ui/field`), and Input, Select, Segmented, Switch and Slider take them into `aria-describedby`
 * and `aria-invalid` (DESIGN_DIRECTION §2.11). Without `for` the messages still get ids, from the Field's own.
 */
import CircleAlert from '$icons/circle-alert';
import { computed, createUniqueId, css } from 'fw';
import CodeText from './CodeText';
import { setField } from './field';

export interface FieldProps {
  label: string;
  for?: string;
  /** Helper text; identifiers in backticks are set as code (`folo serve`), never broken inside (ui/CodeText). */
  hint?: string;
  error?: string | null;
  children: JSX.Element;
  aside?: JSX.Element;
  /** A one-click correction shown after the error: `label` is the whole button text ("Try my-dataset"). */
  fix?: { label: string; apply: () => void } | null;
  /** The value may stay empty: a quiet "Optional" after the label, so it does not read as a peer of required ones. */
  optional?: boolean;
}

export default function Field(props: FieldProps) {
  const uid = createUniqueId();
  const base = computed(() => props.for ?? `field-${uid}`);
  setField({
    get describedby() {
      return props.error ? `${base.get()}-error` : props.hint ? `${base.get()}-hint` : undefined;
    },
    get invalid() {
      return !!props.error;
    },
  });

  return (
    <div class="field" class:invalid={!!props.error}>
      <div class="top">
        <label for={props.for}>
          {props.label}
          {props.optional ? <span class="optional"> Optional</span> : null}
        </label>
        {props.aside ? <span class="aside">{props.aside}</span> : null}
      </div>
      {props.children}
      {props.error ? (
        <p class="msg error" id={`${base.get()}-error`} role="alert">
          <CircleAlert size={14} strokeWidth={1.75} aria-hidden="true" />
          <span>
            {props.error}
            {props.fix ? (
              <>
                {' '}
                <button type="button" class="fix" onclick={() => props.fix?.apply()}>
                  {props.fix.label}
                </button>
              </>
            ) : null}
          </span>
        </p>
      ) : props.hint ? (
        <p class="msg" id={`${base.get()}-hint`}>
          <CodeText text={props.hint} />
        </p>
      ) : null}
    </div>
  );
}

css`
  .field {
    display: grid;
    gap: 6px;
    min-width: 0;
  }
  .top {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: var(--space-2);
  }
  label {
    color: var(--text-secondary);
    font-weight: var(--weight-medium);
  }
  .aside {
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
  }
  .optional {
    margin-left: var(--space-1);
    font-size: var(--type-sm-size);
    font-weight: var(--weight-regular);
    color: var(--text-tertiary);
  }
  /* Helper text keeps a readable measure (DESIGN_DIRECTION §2.3: ≤ 60 characters a line). */
  .msg {
    max-inline-size: var(--measure);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-tertiary);
  }
  .msg.error {
    display: flex;
    align-items: flex-start;
    gap: var(--space-1);
    color: var(--danger-text);
  }
  /* The offered correction: an inline action in ink (never a status hue), underlined so it reads as one. */
  .fix {
    display: inline;
    padding: 0;
    border: 0;
    background: none;
    font: inherit;
    color: var(--text-primary);
    text-decoration: underline;
    text-underline-offset: 3px;
    text-decoration-color: var(--line-strong);
    cursor: pointer;
    white-space: nowrap;
  }
  /* Touch: a 48 px tall target around the 16 px line (DESIGN_DIRECTION §2.9). */
  @media (pointer: coarse) {
    .fix {
      position: relative;
    }
    .fix::after {
      content: '';
      position: absolute;
      inset: -16px -4px;
    }
  }
  @media (hover: hover) {
    .fix:hover {
      text-decoration-color: currentColor;
    }
  }
  /* The icon sits on the first line's centre (16 px line, 14 px icon). */
  .msg.error :global(svg) {
    flex-shrink: 0;
    margin-top: 1px;
  }
`;
