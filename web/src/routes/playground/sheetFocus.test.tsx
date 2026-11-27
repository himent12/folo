// @vitest-environment happy-dom
// Playground tablet popover focusout (modal dialog preservation, Tab/Shift+Tab close) and non-stealing outside click.
import { flush, signal } from 'fw';
import { render } from '../../../framework/testing';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { BrowserEngine } from '$lib/inference/browser';
import ImagePanel from './ImagePanel';

let target: HTMLElement;

beforeEach(() => {
  target = document.createElement('div');
  document.body.appendChild(target);
});

afterEach(() => {
  target.remove();
  document.body.innerHTML = '';
});

describe('ImagePanel tablet popover focus handling', () => {
  it('stays open when focus moves into a modal dialog, and closes when Tab or Shift+Tab leaves both popover and trigger', () => {
    const moreOpen = signal(true);
    const sheetEl = signal<HTMLElement | undefined>(undefined);
    const moreBtn = signal<HTMLElement | undefined>(undefined);
    const outsideBtn = document.createElement('button');
    outsideBtn.textContent = 'Outside';
    const modal = document.createElement('dialog');
    modal.open = true;
    const modalInput = document.createElement('input');
    modal.appendChild(modalInput);
    document.body.append(outsideBtn, modal);

    const view = render(
      () => (
        <ImagePanel
          notice={null}
          badFile={null}
          model={null}
          chooseWeights={() => {}}
          browse={() => {}}
          error={null}
          oom={false}
          browserOff={null}
          smaller={null}
          localBackend="cpu"
          chooseModel={() => {}}
          ondragover={() => {}}
          ondragleave={() => {}}
          ondrop={() => {}}
          source={{ kind: 'file', file: new File([], 'a.jpg'), name: 'a.jpg', url: 'blob:a', width: 100, height: 100 }}
          onSeg={() => {}}
          runOnNote={null}
          sourceName="a.jpg"
          shortName="a.jpg"
          dims={{ width: 100, height: 100 }}
          clear={() => {}}
          dragging={false}
          samples={undefined}
          setSource={() => {}}
          src="blob:a"
          pred={null}
          kept={[]}
          truthBoxes={[]}
          classes={[]}
          labels={true}
          scores={true}
          only={null}
          activeIdx={null}
          selectedIdx={null}
          pickFromImage={() => {}}
          predicting={false}
          engine={{ phase: 'idle' } as unknown as BrowserEngine}
          sheetShown={moreOpen.get()}
          tablet={true}
          rail={false}
          result={null}
          rows={[]}
          hidden={new Set()}
          maxDet={100}
          weightsMissing={false}
          bind:moreBtn={moreBtn}
          closeSheet={() => moreOpen.set(false)}
          bind:moreOpen={moreOpen}
          sheetTitle="More settings"
          bind:sheetEl={sheetEl}
          resetButton={() => <button type="button">Reset</button>}
          moreBody={() => <button type="button" id="inside-pop">Inside</button>}
        />
      ),
      target,
    );
    try {
      flush();
      const pop = target.querySelector<HTMLElement>('#pg-more')!;
      const trigger = target.querySelector<HTMLButtonElement>('.more-btn button')!;
      expect(pop).not.toBeNull();

      // Moving focus into an open modal dialog (e.g. Ctrl+K CommandPalette) keeps the popover open.
      pop.dispatchEvent(new FocusEvent('focusout', { bubbles: true, relatedTarget: modalInput }));
      flush();
      expect(moreOpen.get()).toBe(true);

      // Moving focus from popover back to trigger (Shift+Tab) keeps it open.
      pop.dispatchEvent(new FocusEvent('focusout', { bubbles: true, relatedTarget: trigger }));
      flush();
      expect(moreOpen.get()).toBe(true);

      // Continuing Shift+Tab from trigger to an outside control closes the popover without stealing focus.
      const focusSpy = vi.spyOn(trigger, 'focus');
      trigger.dispatchEvent(new FocusEvent('focusout', { bubbles: true, relatedTarget: outsideBtn }));
      flush();
      expect(moreOpen.get()).toBe(false);
      expect(focusSpy).not.toHaveBeenCalled();

      // Re-open and Tab out of the popover to an outside control: closes cleanly.
      moreOpen.set(true);
      flush();
      const pop2 = target.querySelector<HTMLElement>('#pg-more')!;
      pop2.dispatchEvent(new FocusEvent('focusout', { bubbles: true, relatedTarget: outsideBtn }));
      flush();
      expect(moreOpen.get()).toBe(false);
    } finally {
      view.unmount();
    }
  });

  it('dismisses the phone Thresholds sheet when Tab leaves it for controls beneath and omits Tap hint when empty or comparing', () => {
    const moreOpen = signal(true);
    const sheetEl = signal<HTMLElement | undefined>(undefined);
    const moreBtn = signal<HTMLElement | undefined>(undefined);
    const classesSummary = document.createElement('button');
    classesSummary.textContent = 'Classes';
    const modal = document.createElement('dialog');
    modal.open = true;
    const modalInput = document.createElement('input');
    modal.appendChild(modalInput);
    document.body.append(classesSummary, modal);

    const view = render(
      () => (
        <ImagePanel
          notice={null}
          badFile={null}
          model={null}
          chooseWeights={() => {}}
          browse={() => {}}
          error={null}
          oom={false}
          browserOff={null}
          smaller={null}
          localBackend="cpu"
          chooseModel={() => {}}
          ondragover={() => {}}
          ondragleave={() => {}}
          ondrop={() => {}}
          source={{ kind: 'file', file: new File([], 'a.jpg'), name: 'a.jpg', url: 'blob:a', width: 100, height: 100 }}
          onSeg={() => {}}
          runOnNote={null}
          sourceName="a.jpg"
          shortName="a.jpg"
          dims={{ width: 100, height: 100 }}
          clear={() => {}}
          dragging={false}
          samples={undefined}
          setSource={() => {}}
          src="blob:a"
          pred={{ model_id: 'm1', image: { width: 100, height: 100 }, boxes: [], timing: { preprocess_ms: 1, inference_ms: 2, postprocess_ms: 1, total_ms: 4 }, device: 'cpu' }}
          kept={[]}
          truthBoxes={[]}
          classes={[]}
          labels={true}
          scores={true}
          only={null}
          activeIdx={null}
          selectedIdx={null}
          pickFromImage={() => {}}
          predicting={false}
          engine={{ phase: 'idle' } as unknown as BrowserEngine}
          sheetShown={moreOpen.get()}
          tablet={false}
          phone={true}
          rail={false}
          iou={0.65}
          result={{ boxes: [], ms: 0.1 }}
          rows={[]}
          hidden={new Set()}
          maxDet={1}
          weightsMissing={false}
          bind:moreBtn={moreBtn}
          closeSheet={() => moreOpen.set(false)}
          bind:moreOpen={moreOpen}
          sheetTitle="Thresholds"
          bind:sheetEl={sheetEl}
          resetButton={() => <button type="button">Reset</button>}
          moreBody={() => <button type="button" id="inside-sheet">Inside</button>}
        />
      ),
      target,
    );
    try {
      flush();
      const sheet = target.querySelector<HTMLElement>('#pg-more.sheet')!;
      const trigger = target.querySelector<HTMLButtonElement>('.more-btn button')!;
      expect(sheet).not.toBeNull();
      // With zero detections shown, the footer omits "Tap a box to inspect" so custom thresholds fit cleanly on mobile.
      expect(target.querySelector('.vfoot .tap')).toBeNull();

      // Modal dialog keeps the sheet open.
      sheet.dispatchEvent(new FocusEvent('focusout', { bubbles: true, relatedTarget: modalInput }));
      flush();
      expect(moreOpen.get()).toBe(true);

      // Shift+Tab back to the Thresholds trigger keeps the sheet open.
      sheet.dispatchEvent(new FocusEvent('focusout', { bubbles: true, relatedTarget: trigger }));
      flush();
      expect(moreOpen.get()).toBe(true);

      // Tabbing out of the sheet onto Classes beneath closes the sheet without stealing focus.
      const focusSpy = vi.spyOn(trigger, 'focus');
      sheet.dispatchEvent(new FocusEvent('focusout', { bubbles: true, relatedTarget: classesSummary }));
      flush();
      expect(moreOpen.get()).toBe(false);
      expect(focusSpy).not.toHaveBeenCalled();
    } finally {
      view.unmount();
    }
  });
});
