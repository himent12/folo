// Lucide icons as components. `import X from '$icons/x'` (the compiler's Vite plugin builds the module from the
// icon data vendored in framework/icons/lucide.json, lucide 1.70.0's) renders the same
// `<svg>` as @lucide/svelte's Icon: same attributes, same order, same classes.

import { renderEffect } from './core';
import { PROXY } from './component';
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

const DEEP = Symbol.for('fw.deep');

function hasReactiveProps(p: Record<PropertyKey, unknown>): boolean {
  if (p[PROXY] || p[DEEP]) return true;
  const proto = Object.getPrototypeOf(p);
  if (proto !== Object.prototype && proto !== null) return true;
  for (const k of Object.getOwnPropertyNames(p)) {
    if (Object.getOwnPropertyDescriptor(p, k)?.get !== undefined) return true;
  }
  return false;
}

function setEventOrAttr(svg: SVGElement, k: string, v: unknown): void {
  if (k[0] === 'o' && k[1] === 'n' && typeof v === 'function') {
    const type = k.slice(2);
    if (DELEGATED.has(type)) setDelegated(svg, type, v);
    else on(svg, type, v as (e: Event) => void);
  } else setAttribute(svg, k, v);
}

/** Internal: the component for one lucide icon. */
export function createIcon(icon: IconData): IconComponent {
  const box = icon.size ?? 24;
  const viewBox = `0 0 ${box} ${box}`;
  const classes = mergeClasses('lucide', `lucide-${icon.name}`, ...(icon.aliases ?? []).filter((a) => a.trim() !== '').map((a) => `lucide-${a}`), 'lucide-icon');
  const defaults: Record<string, unknown> = {
    xmlns: NS,
    width: 24,
    height: 24,
    viewBox,
    fill: 'none',
    stroke: 'currentColor',
    'stroke-width': 2,
    'stroke-linecap': 'round',
    'stroke-linejoin': 'round',
    class: classes,
    'aria-hidden': 'true',
  };
  let body: SVGElement | undefined;
  return (props) => {
    if (body === undefined) {
      body = document.createElementNS(NS, 'svg');
      body.setAttribute('xmlns', NS);
      body.setAttribute('width', '24');
      body.setAttribute('height', '24');
      body.setAttribute('viewBox', viewBox);
      body.setAttribute('fill', 'none');
      body.setAttribute('stroke', 'currentColor');
      body.setAttribute('stroke-width', '2');
      body.setAttribute('stroke-linecap', 'round');
      body.setAttribute('stroke-linejoin', 'round');
      body.setAttribute('class', classes);
      body.setAttribute('aria-hidden', 'true');
      for (const [tag, attrs] of icon.node) {
        const child = document.createElementNS(NS, tag);
        for (const k in attrs) child.setAttribute(k, attrs[k]!);
        body.appendChild(child);
      }
    }
    const p = props as Record<string, unknown>;
    const nonScaling = Boolean(p.nonScalingStroke);
    const svg = body.cloneNode(!nonScaling) as SVGElement;
    if (nonScaling) {
      for (let c = body.firstChild; c !== null; c = c.nextSibling) {
        const clone = c.cloneNode(false) as SVGElement;
        clone.setAttribute('vector-effect', 'non-scaling-stroke');
        svg.appendChild(clone);
      }
    }
    if (!hasReactiveProps(p)) {
      const size = p.size ?? 24;
      const width = p.width ?? size;
      const height = p.height ?? size;
      const strokeWidth = p.strokeWidth ?? 2;
      const sw = p.absoluteStrokeWidth ? (Number(strokeWidth) * box) / Number(size) : strokeWidth;
      const stroke = p.color || 'currentColor';
      const cls = p.class ? `${classes} ${p.class}` : classes;
      if (width !== 24) setAttribute(svg, 'width', width);
      if (height !== 24) setAttribute(svg, 'height', height);
      if (stroke !== 'currentColor') setAttribute(svg, 'stroke', stroke);
      if (sw !== 2) setAttribute(svg, 'stroke-width', sw);
      if (cls !== classes) setAttribute(svg, 'class', cls);
      let a11y = 'children' in p;
      const hasAriaHidden = 'aria-hidden' in p;
      if (hasAriaHidden) svg.removeAttribute('aria-hidden');
      for (const k in p) {
        if (OWN.has(k)) continue;
        if (k.startsWith('aria-') || k === 'role' || k === 'title') a11y = true;
        setEventOrAttr(svg, k, p[k]);
      }
      if (a11y && !hasAriaHidden) svg.removeAttribute('aria-hidden');
      return svg;
    }
    let prev: Record<string, unknown> = defaults;
    renderEffect(() => {
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
        viewBox,
        fill: 'none',
        stroke: p.color || 'currentColor',
        'stroke-width': p.absoluteStrokeWidth ? (Number(strokeWidth) * box) / Number(size) : strokeWidth,
        'stroke-linecap': 'round',
        'stroke-linejoin': 'round',
        class: p.class ? `${classes} ${p.class}` : classes,
        ...(a11y || 'children' in p ? {} : { 'aria-hidden': 'true' }),
        ...rest,
      };
      const hadExplicitAria = prev === defaults && 'aria-hidden' in rest;
      if (hadExplicitAria) svg.removeAttribute('aria-hidden');
      for (const k in prev) if (!(k in next)) svg.removeAttribute(k);
      for (const k in next) {
        const v = next[k];
        if (v === prev[k] && !(hadExplicitAria && k === 'aria-hidden')) continue;
        setEventOrAttr(svg, k, v);
      }
      prev = next;
    });
    return svg;
  };
}
