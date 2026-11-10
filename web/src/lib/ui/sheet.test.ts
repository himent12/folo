// @vitest-environment happy-dom
// A bottom sheet (ui/Dialog on phones, or one built by hand) closes on a pull of its grab handle (DESIGN_DIRECTION 12).
import { afterEach, describe, expect, it, vi } from 'vitest';
import { SHEET_CLOSE_PX, sheetDrag } from './sheetDrag';

afterEach(() => {
  document.body.innerHTML = '';
});

function pointer(type: string, y: number, target: Element = document.body) {
  const e = new MouseEvent(type, { bubbles: true, clientY: y, button: 0 }) as MouseEvent & { pointerId: number };
  Object.defineProperty(e, 'pointerId', { value: 1 });
  target.dispatchEvent(e);
}

describe('bottom sheets', () => {
  it('a pull past the threshold closes; a shorter one springs back', () => {
    const sheet = document.createElement('div');
    const head = document.createElement('div');
    sheet.append(head);
    document.body.append(sheet);
    const close = vi.fn();
    const a = sheetDrag(head, { sheet: () => sheet, close });
    pointer('pointerdown', 100, head);
    pointer('pointermove', 120, head);
    expect(sheet.style.transform).toBe('translateY(20px)');
    pointer('pointerup', 120, head);
    expect(close).not.toHaveBeenCalled();
    expect(sheet.style.transform).toBe('');
    pointer('pointerdown', 100, head);
    pointer('pointermove', 100 + SHEET_CLOSE_PX + 8, head);
    pointer('pointerup', 100 + SHEET_CLOSE_PX + 8, head);
    expect(close).toHaveBeenCalledOnce();
    a.destroy();
  });

  it('a cancelled pull springs back and never closes, however far it went', () => {
    const sheet = document.createElement('div');
    const head = document.createElement('div');
    sheet.append(head);
    document.body.append(sheet);
    const close = vi.fn();
    const a = sheetDrag(head, { sheet: () => sheet, close });
    for (const type of ['pointercancel', 'lostpointercapture']) {
      pointer('pointerdown', 100, head);
      pointer('pointermove', 100 + SHEET_CLOSE_PX * 3, head);
      expect(sheet.style.transform).toBe(`translateY(${SHEET_CLOSE_PX * 3}px)`);
      pointer(type, 100 + SHEET_CLOSE_PX * 3, head);
      expect(close, type).not.toHaveBeenCalled();
      expect(sheet.style.transform, type).toBe('');
      expect(sheet.style.transition, type).toBe('');
      // The drag is over: a later move does nothing, and a release without a new press closes nothing.
      pointer('pointermove', 100 + SHEET_CLOSE_PX * 4, head);
      expect(sheet.style.transform, type).toBe('');
      pointer('pointerup', 100 + SHEET_CLOSE_PX * 4, head);
      expect(close, type).not.toHaveBeenCalled();
    }
    // A fresh pull after a cancel starts from nothing and still closes past the threshold.
    pointer('pointerdown', 100, head);
    pointer('pointermove', 100 + SHEET_CLOSE_PX + 8, head);
    pointer('pointerup', 100 + SHEET_CLOSE_PX + 8, head);
    expect(close).toHaveBeenCalledOnce();
    a.destroy();
  });

  it('captures the pointer for the drag and releases it at the end', () => {
    const sheet = document.createElement('div');
    const head = document.createElement('div');
    sheet.append(head);
    document.body.append(sheet);
    let captured: number | null = null;
    const set = vi.fn((id: number) => (captured = id));
    const release = vi.fn((id: number) => {
      if (captured === id) captured = null;
    });
    Object.assign(head, { setPointerCapture: set, releasePointerCapture: release, hasPointerCapture: (id: number) => captured === id });
    const close = vi.fn();
    const a = sheetDrag(head, { sheet: () => sheet, close });
    for (const type of ['pointerup', 'pointercancel']) {
      pointer('pointerdown', 100, head);
      expect(set, type).toHaveBeenLastCalledWith(1);
      expect(captured, type).toBe(1);
      pointer('pointermove', 110, head);
      pointer(type, 110, head);
      expect(release, type).toHaveBeenLastCalledWith(1);
      expect(captured, type).toBeNull();
    }
    expect(set).toHaveBeenCalledTimes(2);
    expect(close).not.toHaveBeenCalled();
    a.destroy();
  });

  it('a press on a control in the grab area stays a click', () => {
    const sheet = document.createElement('div');
    const head = document.createElement('div');
    const btn = document.createElement('button');
    head.append(btn);
    sheet.append(head);
    document.body.append(sheet);
    const close = vi.fn();
    sheetDrag(head, { sheet: () => sheet, close });
    pointer('pointerdown', 100, btn);
    pointer('pointermove', 200, head);
    pointer('pointerup', 200, head);
    expect(close).not.toHaveBeenCalled();
    expect(sheet.style.transform).toBe('');
  });
});
