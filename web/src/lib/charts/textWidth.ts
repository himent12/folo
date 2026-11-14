// Chart label widths as drawn, measured on a canvas (an estimate where there is none, as in tests).
import { labelWidth } from './endLabels';

/**
 * An end label's width as drawn (12 px medium Geist, from the type tokens), so a label beside its marker really sits
 * beside it; `labelWidth`'s estimate where there is no canvas (tests). `regular`: 12 px text at the regular weight, as
 * the legend draws it, without the halo.
 */
export function textMeasurer(weight: 'semibold' | 'regular' = 'semibold'): (t: string) => number {
  let measurer: CanvasRenderingContext2D | null | undefined;
  function textWidth(t: string): number {
    if (measurer === undefined) {
      try {
        const cs = getComputedStyle(document.documentElement);
        measurer = document.createElement('canvas').getContext('2d');
        const wt = weight === 'regular' ? cs.getPropertyValue('--weight-regular').trim() || 400 : cs.getPropertyValue('--weight-semibold').trim() || 600;
        if (measurer) measurer.font = `${wt} ${cs.getPropertyValue('--type-sm-size').trim()} ${cs.getPropertyValue('--font-sans')}`;
      } catch {
        measurer = null;
      }
    }
    const w = measurer?.measureText(t).width;
    // An end label: measured at the highlighted (semibold) weight, the widest it is drawn, plus the halo.
    if (!w) return labelWidth(t);
    return weight === 'regular' ? Math.ceil(w) : Math.ceil(w) + 4;
  }
  return textWidth;
}
