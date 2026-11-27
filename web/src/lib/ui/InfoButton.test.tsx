// @vitest-environment happy-dom
import { describe, expect, it } from 'vitest';
import { flush as flushSync } from 'fw';
import { render } from '../../../framework/testing';
import InfoButton, { type InfoButtonProps } from './InfoButton';
import InfoPanel from './InfoPanel.testing';

/** Port of `mount(C, { target, props })`: renders into `target`; `unmount(c)` is `c.unmount()`. */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const mount = (C: (p: any) => JSX.Element, o: { target: HTMLElement; props?: object }) => render(() => <C {...(o.props ?? {})} />, o.target);
const unmount = (c: { unmount(): void }) => c.unmount();

// happy-dom has no Popover API, so these run InfoButton's own open state (the same one a browser without popovers
// gets); in a browser the popover's light dismiss does the closing and the state follows its toggle event.
function setup(props: Record<string, unknown>) {
  const target = document.createElement('div');
  document.body.appendChild(target);
  const c = mount(InfoButton, { target, props: { label: 'What the loss terms mean', ...props } as InfoButtonProps });
  const button = target.querySelector<HTMLButtonElement>('button')!;
  return {
    target,
    button,
    done: () => {
      unmount(c);
      target.remove();
    },
  };
}

describe('InfoButton', () => {
  it('opens by tap, click, Enter or Space (a click), with aria-expanded and aria-controls', () => {
    const { target, button, done } = setup({ text: 'Box: how far predicted boxes are from the true ones.' });
    const pop = document.getElementById(button.getAttribute('aria-controls')!)!;
    expect(button.getAttribute('aria-label')).toBe('What the loss terms mean');
    expect(button.getAttribute('aria-expanded')).toBe('false');
    expect(pop.textContent).toContain('how far predicted boxes');
    expect(target.contains(pop)).toBe(true);
    button.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true }));
    button.click();
    flushSync();
    expect(button.getAttribute('aria-expanded')).toBe('true');
    expect(pop.classList.contains('open')).toBe(true);
    // A second press on the button closes it.
    button.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true }));
    button.click();
    flushSync();
    expect(button.getAttribute('aria-expanded')).toBe('false');
    done();
  });

  it('closes on Escape (focus stays on the button) and on a tap outside', () => {
    const { button, done } = setup({ text: 'Size = √(box area ÷ image area).' });
    button.focus();
    button.click();
    flushSync();
    expect(button.getAttribute('aria-expanded')).toBe('true');
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
    flushSync();
    expect(button.getAttribute('aria-expanded')).toBe('false');
    expect(document.activeElement).toBe(button);
    button.click();
    flushSync();
    document.body.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true }));
    flushSync();
    expect(button.getAttribute('aria-expanded')).toBe('false');
    done();
  });

  it('stays on screen when its fold opens: placed again, scrolling within the room it has', () => {
    const { button, done } = setup({ text: 'Loss: how wrong the model is.', how: 'A long method.' });
    const pop = document.getElementById(button.getAttribute('aria-controls')!)!;
    // A 600 px tall window, the button low on it; the popover 120 px tall, 500 px once its fold is open.
    const inner = window.innerHeight;
    Object.defineProperty(window, 'innerHeight', { configurable: true, value: 600 });
    let natural = 120;
    const rect = (x: number, y: number, w: number, h: number) => ({ x, y, left: x, top: y, width: w, height: h, right: x + w, bottom: y + h }) as DOMRect;
    button.getBoundingClientRect = () => rect(300, 400, 24, 24);
    pop.getBoundingClientRect = () => rect(0, 0, 320, pop.style.maxHeight ? parseFloat(pop.style.maxHeight) : natural);
    flushSync();
    button.click();
    flushSync();
    const y = () => Number(pop.style.translate.split(' ')[1]!.replace('px', ''));
    expect(y()).toBeGreaterThanOrEqual(424);
    expect(pop.style.maxHeight).toBe('');
    natural = 500;
    const fold = pop.querySelector('details')!;
    fold.open = true;
    fold.dispatchEvent(new Event('toggle'));
    const height = parseFloat(pop.style.maxHeight);
    expect(height).toBeLessThan(500);
    expect(y()).toBeGreaterThanOrEqual(0);
    expect(y() + height).toBeLessThanOrEqual(600);
    // Above the button (the side with more room), its foot by the button.
    expect(y() + height).toBeLessThanOrEqual(400);
    Object.defineProperty(window, 'innerHeight', { configurable: true, value: inner });
    done();
  });

  it('drives an inline disclosure the page renders (Model’s metric help)', () => {
    const region = document.createElement('dl');
    region.id = 'metric-help';
    document.body.appendChild(region);
    const { target, button, done } = setup({ label: 'What these metrics mean', controls: 'metric-help' });
    expect(button.getAttribute('aria-controls')).toBe('metric-help');
    expect(target.querySelector('[popover], .info-pop')).toBeNull();
    button.click();
    flushSync();
    expect(button.getAttribute('aria-expanded')).toBe('true');
    // A tap inside the region keeps it open; Escape closes it.
    region.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true }));
    flushSync();
    expect(button.getAttribute('aria-expanded')).toBe('true');
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
    flushSync();
    expect(button.getAttribute('aria-expanded')).toBe('false');
    done();
    region.remove();
  });

  it("opens with its detail folded under How it's computed", () => {
    const { target, button, done } = setup({ text: 'A short line.', how: 'Size = √(box area ÷ image area).' });
    button.click();
    flushSync();
    const fold = target.querySelector<HTMLDetailsElement>('details.info-how')!;
    expect(fold.open).toBe(false);
    expect(fold.textContent).toContain('Size = √(box area');
    done();
  });

  it('ignores Escape and pointerdown while an unrelated modal dialog is open above it', () => {
    const { button, done } = setup({ text: 'Background help.' });
    button.click();
    flushSync();
    expect(button.getAttribute('aria-expanded')).toBe('true');

    const dialog = document.createElement('dialog');
    dialog.open = true;
    const inner = document.createElement('button');
    dialog.appendChild(inner);
    document.body.appendChild(dialog);

    let dialogSawEsc = false;
    const onDialogKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') dialogSawEsc = true;
    };
    dialog.addEventListener('keydown', onDialogKey);
    try {
      inner.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true }));
      flushSync();
      expect(button.getAttribute('aria-expanded')).toBe('true');

      inner.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
      flushSync();
      expect(dialogSawEsc).toBe(true);
      expect(button.getAttribute('aria-expanded')).toBe('true');
    } finally {
      dialog.removeEventListener('keydown', onDialogKey);
      dialog.remove();
      done();
    }
  });
});
