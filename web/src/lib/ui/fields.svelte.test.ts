// @vitest-environment happy-dom
import { createRawSnippet, flushSync, mount, unmount, type Component } from 'svelte';
import { afterEach, describe, expect, it } from 'vitest';
import Field from './Field.svelte';
import Gate from './Gate.svelte';
import Input from './Input.svelte';
import Segmented from './Segmented.svelte';
import { isOffline, OFFLINE_REASON } from './offline';
import { ApiError } from '$lib/api/client';

let target: HTMLElement;
let c: ReturnType<typeof mount> | undefined;

function render(component: Component<never>, props: Record<string, unknown>) {
  target = document.createElement('div');
  document.body.appendChild(target);
  c = mount(component, { target, props: props as never });
  flushSync();
  return target;
}

afterEach(() => {
  if (c) unmount(c);
  c = undefined;
  target?.remove();
});

describe('invalid fields', () => {
  it('Field: the error carries a danger icon and is announced, so it is not red text alone', () => {
    const t = render(Field as Component<never>, {
      label: 'Epochs',
      for: 'epochs',
      error: 'Use 1 to 10000.',
      children: createRawSnippet(() => ({ render: () => '<input id="epochs" />' })),
    });
    const msg = t.querySelector('#epochs-error')!;
    expect(msg.getAttribute('role')).toBe('alert');
    expect(msg.textContent?.trim()).toBe('Use 1 to 10000.');
    const icon = msg.querySelector('svg')!;
    expect(icon).not.toBeNull();
    expect(icon.getAttribute('aria-hidden')).toBe('true');
  });

  it('Field: a hint has no icon', () => {
    const t = render(Field as Component<never>, {
      label: 'Seed',
      for: 'seed',
      hint: 'Any whole number.',
      children: createRawSnippet(() => ({ render: () => '<input id="seed" />' })),
    });
    expect(t.querySelector('#seed-hint svg')).toBeNull();
  });

  it('Input: invalid marks the input (base.css draws the danger edge and ring from it)', () => {
    const t = render(Input as Component<never>, { invalid: true, 'aria-label': 'Name' });
    expect(t.querySelector('input')!.getAttribute('aria-invalid')).toBe('true');
  });

  it('Segmented: invalid marks the radio group and links its message', () => {
    const t = render(Segmented as Component<never>, {
      label: 'Split',
      value: 'a',
      options: [
        { value: 'a', label: 'A' },
        { value: 'b', label: 'B' },
      ],
      invalid: true,
      describedby: 'split-error',
    });
    const group = t.querySelector('[role="radiogroup"]')!;
    expect(group.getAttribute('aria-invalid')).toBe('true');
    expect(group.getAttribute('aria-describedby')).toBe('split-error');
  });

  it('Segmented: valid groups carry no aria-invalid', () => {
    const t = render(Segmented as Component<never>, { label: 'Split', value: 'a', options: [{ value: 'a', label: 'A' }] });
    expect(t.querySelector('[role="radiogroup"]')!.hasAttribute('aria-invalid')).toBe(false);
  });
});

describe('disabled with a reason', () => {
  it('Gate: the reason is the control’s accessible description while set', () => {
    const t = render(Gate as Component<never>, {
      reason: OFFLINE_REASON,
      children: createRawSnippet((why: () => string | undefined) => ({
        render: () => `<button disabled aria-describedby="${why() ?? ''}">Add dataset</button>`,
      })),
    });
    const id = t.querySelector('button')!.getAttribute('aria-describedby')!;
    expect(id).not.toBe('');
    expect(document.getElementById(id)?.textContent).toBe('Server offline');
  });

  it('Gate: no reason, no description', () => {
    const t = render(Gate as Component<never>, {
      reason: null,
      children: createRawSnippet((why: () => string | undefined) => ({ render: () => `<button data-why="${why() ?? 'none'}">Add</button>` })),
    });
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
