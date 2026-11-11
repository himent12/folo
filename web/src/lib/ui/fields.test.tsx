// @vitest-environment happy-dom
import { flush as flushSync } from 'fw';
import { afterEach, describe, expect, it } from 'vitest';
import { fromHTML, render as mountFw } from '../../../framework/testing';
import Field from './Field.tsx';
import FieldsFixture from './FieldsFixture.testing';
import { describedBy } from './field.fw';
import { undescribedFields } from './fieldA11y';
import Gate from './Gate';
import Segmented from './Segmented';
import { isOffline, OFFLINE_REASON } from './offline';
import { ApiError } from '$lib/api/client';

let target: HTMLElement;
let c: { unmount(): void } | undefined;

/** Port of `mount(component, { target, props })` + `flushSync()`. */
function render(code: () => JSX.Element) {
  target = document.createElement('div');
  document.body.appendChild(target);
  c = mountFw(code, target);
  flushSync();
  return target;
}

afterEach(() => {
  if (c) c.unmount();
  c = undefined;
  target?.remove();
});

describe('invalid fields', () => {
  it('Field: the error carries a danger icon and is announced, so it is not red text alone', () => {
    const t = render(() => (
      <Field label="Epochs" for="epochs" error="Use 1 to 10000.">
        {() => fromHTML('<input id="epochs" />')}
      </Field>
    ));
    const msg = t.querySelector('#epochs-error')!;
    expect(msg.getAttribute('role')).toBe('alert');
    expect(msg.textContent?.trim()).toBe('Use 1 to 10000.');
    expect(msg.querySelector('svg')?.getAttribute('aria-hidden')).toBe('true');
  });

  it('Segmented: invalid marks the radio group and links its message', () => {
    const t = render(() => (
      <Segmented
        label="Split"
        value="a"
        options={[
          { value: 'a', label: 'A' },
          { value: 'b', label: 'B' },
        ]}
        invalid={true}
        describedby="split-error"
      />
    ));
    const group = t.querySelector('[role="radiogroup"]')!;
    expect(group.getAttribute('aria-invalid')).toBe('true');
    expect(group.getAttribute('aria-describedby')).toBe('split-error');
  });
});

describe('a Field describes its control (the visible hint is the accessible description)', () => {
  it('forwards the hint id to every shared control, with no wiring by the caller', () => {
    const t = render(() => <FieldsFixture />);
    expect(undescribedFields(t)).toEqual([]);
    const d = (sel: string) => t.querySelector(sel)!.getAttribute('aria-describedby');
    expect(d('#f-input')).toBe('extra f-input-hint');
    expect(d('#f-seg')).toBe('f-seg-hint');
    expect(d('#f-select')).toBe('f-select-hint');
    // A Switch's own hint is its description too, and its name is the label alone.
    expect(d('#f-switch')).toBe('f-switch-desc f-switch-hint');
    expect(t.querySelector('#f-switch')!.getAttribute('aria-labelledby')).toBe('f-switch-label');
    // A Field without `for` still ties its hint (an id of its own).
    const mode = t.querySelectorAll('[role="radiogroup"]')[1]!.getAttribute('aria-describedby')!;
    expect(document.getElementById(mode)?.textContent).toBe('Without an id of its own.');
  });

  it('forwards the error instead while there is one, and marks the control invalid', () => {
    const t = render(() => <FieldsFixture error="Pick one." />);
    expect(undescribedFields(t)).toEqual([]);
    expect(t.querySelector('#f-input')!.getAttribute('aria-describedby')).toBe('extra f-input-error');
    for (const sel of ['#f-input', '#f-seg', '#f-select']) expect(t.querySelector(sel)!.getAttribute('aria-invalid')).toBe('true');
  });

  it('the guard names a hint no control is described by', () => {
    const t = render(() => (
      <Field label="Seed" for="seed" hint="Any whole number.">
        {() => fromHTML('<input id="seed" />')}
      </Field>
    ));
    expect(undescribedFields(t)).toEqual(['Seed: its input is not described by #seed-hint']);
  });

  it('describedBy: each id once, empty parts dropped, nothing when none', () => {
    expect(describedBy('a b', undefined, 'b', false, '', 'c')).toBe('a b c');
    expect(describedBy(undefined, null)).toBeUndefined();
  });
});

describe('disabled with a reason', () => {
  it('Gate: the reason is the control’s accessible description while set', () => {
    const t = render(() => (
      <Gate reason={OFFLINE_REASON}>{(why: () => string | undefined) => fromHTML(`<button disabled aria-describedby="${why() ?? ''}">Add dataset</button>`)}</Gate>
    ));
    const id = t.querySelector('button')!.getAttribute('aria-describedby')!;
    expect(id).not.toBe('');
    expect(document.getElementById(id)?.textContent).toBe('Server offline');
  });

  it('Gate: no reason, no description', () => {
    const t = render(() => <Gate reason={null}>{(why: () => string | undefined) => fromHTML(`<button data-why="${why() ?? 'none'}">Add</button>`)}</Gate>);
    expect(t.querySelector('button')!.dataset.why).toBe('none');
    expect(t.querySelector('.sr-only')).toBeNull();
  });

  it('isOffline: only a request nothing answered', () => {
    expect(isOffline(new ApiError(0, 'network', 'Cannot reach the FOLO server.'))).toBe(true);
    expect(isOffline(new ApiError(500, 'internal', 'boom'))).toBe(false);
    expect(isOffline(new Error('x'))).toBe(false);
    expect(isOffline(undefined)).toBe(false);
  });
});
