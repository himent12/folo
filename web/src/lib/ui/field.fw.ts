/**
 * fw twin of field.ts (the Field → control contract, DESIGN_DIRECTION §2.11, §2.24) until the switch: the same
 * context, through fw's context instead of Svelte's. `describedBy` is the same function.
 */
import { getContext, setContext } from 'fw';

export interface FieldContext {
  /** The id of the message the Field shows now: its error while set, else its hint; none without either. */
  readonly describedby: string | undefined;
  readonly invalid: boolean;
}

const KEY = Symbol('field');

export function setField(ctx: FieldContext) {
  setContext(KEY, ctx);
}

/** The enclosing Field, if any. A control reads it once, at init; its getters stay live. */
export function getField(): FieldContext | undefined {
  return getContext<FieldContext | undefined>(KEY);
}

/** Space-separated ids, each once, empty parts dropped; `undefined` when none are left (no empty attribute). */
export function describedBy(...ids: (string | null | undefined | false)[]): string | undefined {
  const out = [...new Set(ids.flatMap((s) => (s ? s.split(/\s+/) : [])).filter(Boolean))];
  return out.length ? out.join(' ') : undefined;
}
