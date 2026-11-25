// Chart label widths as drawn, measured on a single shared canvas with a bounded width cache (falling back to an
// estimate where there is no 2D canvas, as in tests).
import { labelWidth } from './endLabels';

const MAX_CACHE = 512;

let sharedCtx: CanvasRenderingContext2D | null | undefined;
let cachedTheme: string | undefined;
let fontsBound = false;
const fontByWeight: Partial<Record<'semibold' | 'regular', string>> = {};
const widthsByFont = new Map<string, Map<string, number>>();

const clearWidths = () => widthsByFont.clear();

function bindFontLoad() {
  if (fontsBound || typeof document === 'undefined') return;
  fontsBound = true;
  const fonts = document.fonts;
  if (!fonts) return;
  fonts.addEventListener?.('loadingdone', clearWidths);
  void fonts.ready?.then(clearWidths, () => {});
}

function getCtx(): CanvasRenderingContext2D | null {
  if (sharedCtx === undefined) {
    try {
      sharedCtx = document.createElement('canvas').getContext('2d');
    } catch {
      sharedCtx = null;
    }
  }
  return sharedCtx;
}

function resolveFont(weight: 'semibold' | 'regular'): string {
  const root = document.documentElement;
  const theme = `${root.dataset.theme ?? ''}|${root.dataset.themeId ?? ''}|${root.style.cssText}`;
  if (theme !== cachedTheme) {
    cachedTheme = theme;
    fontByWeight.semibold = undefined;
    fontByWeight.regular = undefined;
  }
  let font = fontByWeight[weight];
  if (!font) {
    const cs = getComputedStyle(document.documentElement);
    const wt = weight === 'regular' ? cs.getPropertyValue('--weight-regular').trim() || 400 : cs.getPropertyValue('--weight-semibold').trim() || 600;
    font = `${wt} ${cs.getPropertyValue('--type-sm-size').trim()} ${cs.getPropertyValue('--font-sans')}`;
    fontByWeight[weight] = font;
  }
  return font;
}

/** Resets shared canvas/font/width caches (tests). */
export function resetTextMeasurerCache() {
  if (fontsBound && typeof document !== 'undefined') document.fonts?.removeEventListener?.('loadingdone', clearWidths);
  fontsBound = false;
  sharedCtx = undefined;
  cachedTheme = undefined;
  fontByWeight.semibold = undefined;
  fontByWeight.regular = undefined;
  widthsByFont.clear();
}

/**
 * An end label's width as drawn (12 px medium Geist, from the type tokens), so a label beside its marker really sits
 * beside it; `labelWidth`'s estimate where there is no canvas (tests). `regular`: 12 px text at the regular weight, as
 * the legend draws it, without the halo.
 */
export function textMeasurer(weight: 'semibold' | 'regular' = 'semibold'): (t: string) => number {
  function textWidth(t: string): number {
    const ctx = getCtx();
    if (!ctx) return labelWidth(t);
    bindFontLoad();
    let font: string;
    try {
      font = resolveFont(weight);
    } catch {
      return labelWidth(t);
    }
    const key = `${weight}|${font}`;
    let cache = widthsByFont.get(key);
    if (!cache) {
      cache = new Map();
      widthsByFont.set(key, cache);
    }
    const hit = cache.get(t);
    if (hit !== undefined) return hit;
    if (ctx.font !== font) ctx.font = font;
    const w = ctx.measureText(t).width;
    // An end label: measured at the highlighted (semibold) weight, the widest it is drawn, plus the halo.
    if (!w) return labelWidth(t);
    const measured = weight === 'regular' ? Math.ceil(w) : Math.ceil(w) + 4;
    if (cache.size >= MAX_CACHE) cache.clear();
    cache.set(t, measured);
    return measured;
  }
  return textWidth;
}
