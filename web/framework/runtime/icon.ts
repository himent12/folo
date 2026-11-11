// Lucide icons as components. `import X from '$icons/x'` (the compiler's Vite plugin builds the module from
// @lucide/svelte's own icon data, so the version and the shapes are the ones the Svelte app draws) renders the same
// `<svg>` as @lucide/svelte's Icon: same attributes, same order, same classes.

import { renderEffect } from './core';
import { setAttribute } from './dom';
import { setDelegated, DELEGATED, on } from './events';

export type IconNode = [tag: string, attrs: Record<string, string>][];
export interface IconData {
  name: string;
  size?: number;
  node: IconNode;
  aliases?: string[];
}

export interface IconProps {
  size?: number | string;
  color?: string;
  strokeWidth?: number | string;
  absoluteStrokeWidth?: boolean;
  nonScalingStroke?: boolean;
  class?: string;
  title?: string;
  role?: string;
  [attr: `aria-${string}` | `data-${string}`]: unknown;
  [attr: string]: unknown;
}

/** An icon component (lucide or our own), rendered at the size the parent picks. */
export type IconComponent = (props: IconProps) => JSX.Element;

const NS = 'http://www.w3.org/2000/svg';
const OWN = new Set(['color', 'size', 'width', 'height', 'strokeWidth', 'absoluteStrokeWidth', 'nonScalingStroke', 'class', 'children', 'iconNode', 'icon']);

function mergeClasses(...classes: (string | undefined)[]): string {
  return classes.filter((c, i, a) => Boolean(c) && c!.trim() !== '' && a.indexOf(c) === i).join(' ').trim();
}

/** Internal: the component for one lucide icon. */
export function createIcon(icon: IconData): IconComponent {
  const box = icon.size ?? 24;
  const classes = mergeClasses('lucide', `lucide-${icon.name}`, ...(icon.aliases ?? []).filter((a) => a.trim() !== '').map((a) => `lucide-${a}`), 'lucide-icon');
  let body: SVGElement | undefined;
  return (props) => {
    if (body === undefined) {
      body = document.createElementNS(NS, 'svg');
      for (const [tag, attrs] of icon.node) {
        const child = document.createElementNS(NS, tag);
        for (const k in attrs) child.setAttribute(k, attrs[k]!);
        body.appendChild(child);
      }
    }
    const svg = body.cloneNode(false) as SVGElement;
    let prev: Record<string, unknown> = {};
    renderEffect(() => {
      const p = props as Record<string, unknown>;
      const size = p.size ?? 24;
      const width = p.width ?? size;
      const height = p.height ?? size;
      const strokeWidth = p.strokeWidth ?? 2;
      const rest: Record<string, unknown> = {};
      let a11y = false;
      for (const k in p) {
        if (OWN.has(k)) continue;
        rest[k] = p[k];
        if (k.startsWith('aria-') || k === 'role' || k === 'title') a11y = true;
      }
      const next: Record<string, unknown> = {
        xmlns: NS,
        width,
        height,
        viewBox: `0 0 ${box} ${box}`,
        fill: 'none',
        stroke: p.color || 'currentColor',
        'stroke-width': p.absoluteStrokeWidth ? (Number(strokeWidth) * box) / Number(size) : strokeWidth,
        'stroke-linecap': 'round',
        'stroke-linejoin': 'round',
        class: p.class ? `${classes} ${p.class}` : classes,
        ...(a11y || 'children' in p ? {} : { 'aria-hidden': 'true' }),
        ...rest,
      };
      for (const k in prev) if (!(k in next)) svg.removeAttribute(k);
      for (const k in next) {
        const v = next[k];
        if (v === prev[k]) continue;
        if (k[0] === 'o' && k[1] === 'n' && typeof v === 'function') {
          const type = k.slice(2);
          if (DELEGATED.has(type)) setDelegated(svg, type, v);
          else on(svg, type, v as (e: Event) => void);
        } else setAttribute(svg, k, v);
      }
      prev = next;
    });
    const nonScaling = (props as IconProps).nonScalingStroke;
    for (let c = body.firstChild; c !== null; c = c.nextSibling) {
      const clone = c.cloneNode(false) as SVGElement;
      if (nonScaling) clone.setAttribute('vector-effect', 'non-scaling-stroke');
      svg.appendChild(clone);
    }
    return svg;
  };
}
