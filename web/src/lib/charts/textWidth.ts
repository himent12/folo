import { labelWidth } from './endLabels';

/**
 * An end label's width as drawn (12 px medium Geist, from the type tokens), so a label beside its marker really sits
 * beside it; `labelWidth`'s estimate where there is no canvas (tests).
 */
export function textMeasurer(): (t: string) => number {
  let measurer: CanvasRenderingContext2D | null | undefined;
  function textWidth(t: string): number {
    if (measurer === undefined) {
      try {
        const cs = getComputedStyle(document.documentElement);
        measurer = document.createElement('canvas').getContext('2d');
        if (measurer) measurer.font = `${cs.getPropertyValue('--weight-semibold').trim() || 600} ${cs.getPropertyValue('--type-sm-size').trim()} ${cs.getPropertyValue('--font-sans')}`;
      } catch {
        measurer = null;
      }
    }
    const w = measurer?.measureText(t).width;
    // Measured at the highlighted (semibold) weight, the widest a label is drawn, plus the halo.
    return w ? Math.ceil(w) + 4 : labelWidth(t);
  }
  return textWidth;
}
