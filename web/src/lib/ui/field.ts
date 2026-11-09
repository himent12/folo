/**
 * The Field → control contract (DESIGN_DIRECTION §2.11, §2.24): a Field's visible hint or error is its control's
 * accessible description. Field publishes the id of the message it shows and whether it is in error; every control
 * (Input, Select, Segmented, Switch, Slider) reads it and merges it into its own `aria-describedby` and `aria-invalid`,
 * so no caller has to wire the ids by hand (`fields.svelte.test.ts`).
 */
import { getContext, setContext } from 'svelte';

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
