<script lang="ts">
  /**
   * Full-size image inspector: zoom (wheel / pinch / + - / double-click) and pan (drag) over the image with its
   * labels, a label list that highlights boxes, a class legend and image meta. J/K or arrows step through the
   * current filter; Esc closes. The open image lives in the URL (`?image=`), so it is linkable.
   */
  import { untrack } from 'svelte';
  import ChevronLeft from '@lucide/svelte/icons/chevron-left';
  import ChevronRight from '@lucide/svelte/icons/chevron-right';
  import Maximize from '@lucide/svelte/icons/maximize';
  import Minus from '@lucide/svelte/icons/minus';
  import Plus from '@lucide/svelte/icons/plus';
  import X from '@lucide/svelte/icons/x';
  import { hub } from '$lib/api/index.svelte';
  import type { Box } from '$lib/api/types';
  import DatasetImage from '$lib/detect/DatasetImage.svelte';
  import { classColor } from '$lib/design/tokens';
  import { num } from '$lib/format';
  import Badge from '$lib/ui/Badge.svelte';
  import Button from '$lib/ui/Button.svelte';
  import Kbd from '$lib/ui/Kbd.svelte';
  import Spinner from '$lib/ui/Spinner.svelte';
  import { tooltip } from '$lib/ui/tooltip';
  import type { ImagePager } from './pager.svelte';

  let {
    pager,
    classes,
    imageId,
    only,
    onnavigate,
    onclose,
    onmissing,
  }: {
    pager: ImagePager;
    classes: readonly string[];
    imageId: string | null;
    only: ReadonlySet<number> | null;
    onnavigate: (id: string) => void;
    onclose: () => void;
    /** The linked image is not in this filter (or does not exist). */
    onmissing: (id: string) => void;
  } = $props();

  const id = $derived(imageId);
  let dialog: HTMLDialogElement;
  let viewport = $state<HTMLDivElement>();
  let vw = $state(0);
  let vh = $state(0);
  let active = $state<number | null>(null);

  const idx = $derived(id ? pager.indexOf(id) : -1);
  const img = $derived(idx >= 0 ? pager.get(idx) : undefined);
  const total = $derived(pager.total ?? 0);

  // ---------- open / close ----------
  $effect(() => {
    if (id && !dialog.open) dialog.showModal();
    else if (!id && dialog.open) dialog.close();
  });

  // Resolve a linked id that is not loaded yet.
  $effect(() => {
    const want = id;
    if (!want || untrack(() => pager.indexOf(want)) >= 0) return;
    let live = true;
    void pager.find(want).then(
      (i) => {
        if (live && i < 0) onmissing(want);
      },
      () => {},
    );
    return () => {
      live = false;
    };
  });

  // Prefetch neighbours (metadata and the full-size image) so J/K feel instant.
  $effect(() => {
    if (idx < 0) return;
    pager.ensure(idx - 1, idx + 2);
    const ds = pager.datasetId;
    for (const j of [idx + 1, idx - 1]) {
      const n = untrack(() => pager.get(j));
      if (n) void hub.api.imageUrl(ds, n.id).then((u) => void (new Image().src = u), () => {});
    }
  });

  async function step(d: number) {
    if (idx < 0) return;
    const j = idx + d;
    if (j < 0 || j >= total) return;
    const n = await pager.at(j);
    if (n) onnavigate(n.id);
  }

  // ---------- zoom & pan ----------
  /** z = 1 is "fit"; tx/ty is the image's top-left in viewport px (clamped on use). */
  let z = $state(1);
  let tx = $state(0);
  let ty = $state(0);

  const s0 = $derived(img && vw && vh ? Math.min(vw / img.width, vh / img.height) : 1);
  const maxZ = $derived(img ? Math.max(2, (8 / s0) | 0) : 8);
  const dw = $derived(img ? img.width * s0 * z : 0);
  const dh = $derived(img ? img.height * s0 * z : 0);
  const clampAxis = (t: number, d: number, v: number) => (d <= v ? (v - d) / 2 : Math.min(0, Math.max(v - d, t)));
  const x = $derived(clampAxis(tx, dw, vw));
  const y = $derived(clampAxis(ty, dh, vh));
  const zoomPct = $derived(Math.round(s0 * z * 100));
  /** The zoom readout: "Fit" at the fitted view (where it opens), the scale in % once zoomed. */
  const zoomLabel = $derived(z <= 1.001 ? 'Fit' : `${zoomPct}%`);

  // Fit whenever the image changes.
  $effect(() => {
    void id;
    untrack(() => {
      z = 1;
      active = null;
    });
  });

  function zoomAt(f: number, px = vw / 2, py = vh / 2) {
    const nz = Math.min(maxZ, Math.max(1, z * f));
    const ux = (px - x) / (s0 * z);
    const uy = (py - y) / (s0 * z);
    z = nz;
    tx = px - ux * s0 * nz;
    ty = py - uy * s0 * nz;
  }

  function fit() {
    z = 1;
  }

  function zoomToBox(b: Box) {
    if (!img) return;
    const bw = Math.max(8, b.x2 - b.x1);
    const bh = Math.max(8, b.y2 - b.y1);
    const nz = Math.min(maxZ, Math.max(1, Math.min((vw * 0.45) / (bw * s0), (vh * 0.45) / (bh * s0))));
    z = nz;
    tx = vw / 2 - ((b.x1 + b.x2) / 2) * s0 * nz;
    ty = vh / 2 - ((b.y1 + b.y2) / 2) * s0 * nz;
  }

  function local(e: { clientX: number; clientY: number }) {
    const r = viewport!.getBoundingClientRect();
    return [e.clientX - r.left, e.clientY - r.top] as const;
  }

  // Wheel needs a non-passive listener to stop the page (and the browser's pinch-zoom) from scrolling.
  $effect(() => {
    const v = viewport;
    if (!v) return;
    const onwheel = (e: WheelEvent) => {
      e.preventDefault();
      const unit = e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? vh : 1;
      const [px, py] = local(e);
      zoomAt(Math.exp(-e.deltaY * unit * (e.ctrlKey ? 0.01 : 0.002)), px, py);
    };
    v.addEventListener('wheel', onwheel, { passive: false });
    return () => v.removeEventListener('wheel', onwheel);
  });

  const pointers = new Map<number, { x: number; y: number }>();
  let pinch: { d: number; mx: number; my: number } | null = null;
  let dragging = $state(false);
  /** The touch gesture hint fades once someone has touched the image. */
  let touched = $state(false);

  function pinchState() {
    const [a, b] = [...pointers.values()];
    if (!a || !b) return null;
    return { d: Math.hypot(a.x - b.x, a.y - b.y), mx: (a.x + b.x) / 2, my: (a.y + b.y) / 2 };
  }

  function onpointerdown(e: PointerEvent) {
    if (e.button !== 0) return;
    if (e.pointerType === 'touch') touched = true;
    viewport!.setPointerCapture(e.pointerId);
    const [px, py] = local(e);
    pointers.set(e.pointerId, { x: px, y: py });
    pinch = pointers.size === 2 ? pinchState() : null;
    dragging = true;
  }

  function onpointermove(e: PointerEvent) {
    const prev = pointers.get(e.pointerId);
    if (!prev) return;
    const [px, py] = local(e);
    pointers.set(e.pointerId, { x: px, y: py });
    if (pointers.size >= 2) {
      const now = pinchState();
      if (now && pinch && pinch.d > 0) {
        tx = x + (now.mx - pinch.mx);
        ty = y + (now.my - pinch.my);
        zoomAt(now.d / pinch.d, now.mx, now.my);
      }
      pinch = now;
    } else {
      tx = x + (px - prev.x);
      ty = y + (py - prev.y);
    }
  }

  function onpointerup(e: PointerEvent) {
    pointers.delete(e.pointerId);
    pinch = pointers.size === 2 ? pinchState() : null;
    if (pointers.size === 0) dragging = false;
  }

  function ondblclick(e: MouseEvent) {
    const [px, py] = local(e);
    if (z >= maxZ * 0.99) fit();
    else zoomAt(2.5, px, py);
  }

  function onkeydown(e: KeyboardEvent) {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    const k = e.key;
    if (k === 'j' || k === 'J' || k === 'ArrowRight') step(1);
    else if (k === 'k' || k === 'K' || k === 'ArrowLeft') step(-1);
    else if (k === '+' || k === '=') zoomAt(1.5);
    else if (k === '-' || k === '_') zoomAt(1 / 1.5);
    else if (k === '0') fit();
    else return;
    e.preventDefault();
  }

  // ---------- labels ----------
  const counts = $derived.by(() => {
    const c = new Map<number, number>();
    for (const b of img?.boxes ?? []) c.set(b.class, (c.get(b.class) ?? 0) + 1);
    return c;
  });
  const legend = $derived(classes.map((name, i) => ({ i, name, n: counts.get(i) ?? 0 })));
  const name = (c: number) => classes[c] ?? `class ${c}`;
  const stem = (f: string) => f.split('/').pop() ?? f;
</script>

{#snippet zoomTools()}
  <div class="group" role="group" aria-label="Zoom">
    <span use:tooltip={{ text: 'Zoom out', keys: '-' }}><Button variant="ghost" size="sm" icon={Minus} aria-label="Zoom out" disabled={z <= 1} onclick={() => zoomAt(1 / 1.5)} /></span>
    <button type="button" class="zoom num" onclick={fit} use:tooltip={{ text: 'Fit to view', keys: '0' }} aria-label="Zoom {zoomPct}%. Fit to view">{zoomLabel}</button>
    <span use:tooltip={{ text: 'Zoom in', keys: '+' }}><Button variant="ghost" size="sm" icon={Plus} aria-label="Zoom in" disabled={z >= maxZ} onclick={() => zoomAt(1.5)} /></span>
    <span use:tooltip={{ text: 'Fit', keys: '0' }}><Button variant="ghost" size="sm" icon={Maximize} aria-label="Fit to view" onclick={fit} /></span>
  </div>
{/snippet}

<!-- svelte-ignore a11y_no_noninteractive_element_interactions -->
<dialog
  bind:this={dialog}
  class="insp"
  aria-label={img ? `Image ${stem(img.file)}` : 'Image'}
  onclose={() => id && onclose()}
  {onkeydown}
>
  <header>
    <div class="title">
      {#if img}
        <span class="file mono truncate" title={img.file}>{stem(img.file)}</span>
        <Badge tone="outline" mono>{img.split}</Badge>
      {:else}
        <span class="file faint">Loading…</span>
      {/if}
      {#if idx >= 0}<span class="pos num" aria-label="Image {num(idx + 1)} of {num(total)}">{num(idx + 1)} <i>/</i> {num(total)}</span>{/if}
    </div>
    <div class="tools">
      <div class="head-zoom">{@render zoomTools()}</div>
      <span class="sep head-zoom" aria-hidden="true"></span>
      <div class="group" role="group" aria-label="Navigate">
        <span use:tooltip={{ text: 'Previous', keys: 'K' }}><Button variant="ghost" size="sm" icon={ChevronLeft} aria-label="Previous image" disabled={idx <= 0} onclick={() => step(-1)} /></span>
        <span use:tooltip={{ text: 'Next', keys: 'J' }}><Button variant="ghost" size="sm" icon={ChevronRight} aria-label="Next image" disabled={idx < 0 || idx >= total - 1} onclick={() => step(1)} /></span>
      </div>
      <span class="sep" aria-hidden="true"></span>
      <span use:tooltip={{ text: 'Close', keys: 'Escape' }}><Button variant="ghost" size="sm" icon={X} aria-label="Close" onclick={() => dialog.close()} /></span>
    </div>
  </header>

  <div class="body">
    <div
      class="viewport"
      class:zoomed={z > 1}
      class:dragging
      bind:this={viewport}
      bind:clientWidth={vw}
      bind:clientHeight={vh}
      style:--ar={img ? `${img.width} / ${img.height}` : undefined}
      role="img"
      aria-label={img ? `${stem(img.file)} with ${img.boxes.length} labelled boxes` : 'Loading image'}
      {onpointerdown}
      {onpointermove}
      {onpointerup}
      onpointercancel={onpointerup}
      {ondblclick}
    >
      {#if img}
        {#key img.id}
          <div class="stage" style:width="{dw}px" style:height="{dh}px" style:transform="translate3d({x}px, {y}px, 0)">
            <DatasetImage
              datasetId={pager.datasetId}
              imageId={img.id}
              width={img.width}
              height={img.height}
              boxes={img.boxes}
              {classes}
              {only}
              {active}
              onhover={(i) => (active = i)}
              alt={stem(img.file)}
            />
          </div>
        {/key}
      {:else}
        <div class="loading"><Spinner /></div>
      {/if}
      <div class="hint" class:touched aria-hidden="true">
        <span class="fine"><Kbd keys="J" subtle /><Kbd keys="K" subtle /> step</span>
        <span class="fine">scroll to zoom · drag to pan</span>
        <span class="touch">Pinch to zoom · drag to pan</span>
      </div>
    </div>
    <!-- Narrow screens: the zoom controls get their own 44 px bar under the image (never over the labels), with the
         gesture hint beneath it. -->
    <div class="zoom-bar">
      {@render zoomTools()}
      <p class="touch-hint">Pinch to zoom · drag to pan · double-tap to zoom in</p>
    </div>

    <aside>
      <section>
        <h3>Labels <span class="meta num">{img?.boxes.length ?? 0}</span></h3>
        {#if img && img.boxes.length}
          <ul class="labels" onpointerleave={() => (active = null)}>
            {#each img.boxes as b, i (i)}
              <li>
                <button
                  type="button"
                  class:on={active === i}
                  class:dim={only !== null && !only.has(b.class)}
                  onpointerenter={() => (active = i)}
                  onfocus={() => (active = i)}
                  onblur={() => (active = null)}
                  onclick={() => zoomToBox(b)}
                  aria-label="{name(b.class)}, {Math.round(b.x2 - b.x1)} by {Math.round(b.y2 - b.y1)} pixels. Zoom to box"
                >
                  <span class="sw" style:background={classColor(b.class)}></span>
                  <span class="cn truncate">{name(b.class)}</span>
                  <span class="sz num">{Math.round(b.x2 - b.x1)}×{Math.round(b.y2 - b.y1)}</span>
                </button>
              </li>
            {/each}
          </ul>
        {:else if img}
          <p class="none">No boxes. This image is a background example.</p>
        {/if}
      </section>

      <section>
        <h3>Classes</h3>
        <ul class="legend">
          {#each legend as l (l.i)}
            <li class:absent={l.n === 0}>
              <span class="sw" style:background={classColor(l.i)}></span>
              <span class="truncate">{l.name}</span>
              {#if l.n}<span class="num faint">{l.n}</span>{/if}
            </li>
          {/each}
        </ul>
      </section>

      {#if img}
        <section>
          <h3>Image</h3>
          <dl>
            <dt>File</dt>
            <dd class="mono" title={img.file}>{img.file}</dd>
            <dt>Split</dt>
            <dd>{img.split}</dd>
            <dt>Size</dt>
            <dd class="num">{img.width} × {img.height} px</dd>
            <dt>ID</dt>
            <dd class="mono">{img.id}</dd>
          </dl>
        </section>
      {/if}
    </aside>
  </div>
</dialog>

<style>
  .insp {
    width: min(1480px, calc(100vw - 48px));
    height: min(960px, calc(100dvh - 48px));
    max-width: none;
    max-height: none;
    margin: auto;
    padding: 0;
    border: 0;
    border-radius: var(--radius-xl);
    background: var(--bg-overlay);
    color: var(--text-primary);
    box-shadow: var(--shadow-lg);
    overflow: hidden;
  }
  .insp[open] {
    display: flex;
    flex-direction: column;
    animation: enter var(--dur-slow) var(--ease-out);
  }
  .insp::backdrop {
    background: var(--bg-scrim);
    backdrop-filter: blur(2px);
  }
  header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-3);
    height: 48px;
    padding: 0 var(--space-2) 0 var(--space-4);
    border-bottom: 1px solid var(--line-subtle);
    flex-shrink: 0;
  }
  .title {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    min-width: 0;
  }
  .file {
    font-weight: var(--weight-medium);
  }
  .pos {
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
    white-space: nowrap;
  }
  .pos i {
    font-style: normal;
    color: var(--text-disabled);
  }
  .tools {
    display: flex;
    align-items: center;
    gap: var(--space-1);
  }
  .group {
    display: flex;
    align-items: center;
    gap: 2px;
  }
  .tools span,
  .zoom-bar span {
    display: inline-flex;
  }
  .head-zoom {
    display: contents;
  }
  .zoom-bar {
    display: none;
  }
  .sep {
    width: 1px;
    height: 16px;
    margin: 0 var(--space-1);
    background: var(--line-default);
  }
  .zoom {
    min-width: 48px;
    height: var(--control-sm);
    padding: 0 6px;
    border-radius: var(--radius-sm);
    font-size: var(--type-sm-size);
    color: var(--text-secondary);
  }
  .zoom:hover {
    background: var(--bg-hover);
    color: var(--text-primary);
  }
  .body {
    flex: 1;
    min-height: 0;
    display: grid;
    grid-template-columns: minmax(0, 1fr) 288px;
  }
  .viewport {
    position: relative;
    overflow: hidden;
    background: var(--bg-inset);
    touch-action: none;
    cursor: zoom-in;
  }
  .viewport.zoomed {
    cursor: grab;
  }
  .viewport.dragging.zoomed {
    cursor: grabbing;
  }
  .stage {
    position: absolute;
    left: 0;
    top: 0;
    will-change: transform;
  }
  .loading {
    position: absolute;
    inset: 0;
    display: grid;
    place-items: center;
    color: var(--text-tertiary);
  }
  .hint {
    position: absolute;
    left: var(--space-3);
    bottom: var(--space-3);
    display: flex;
    gap: var(--space-3);
    padding: 4px 8px;
    border-radius: var(--radius-md);
    background: color-mix(in srgb, var(--bg-overlay) 82%, transparent);
    backdrop-filter: blur(6px);
    font-size: var(--type-xs-size);
    color: var(--text-tertiary);
    pointer-events: none;
  }
  .hint span {
    display: inline-flex;
    align-items: center;
    gap: 4px;
  }
  .hint .touch {
    display: none;
  }
  @media (pointer: coarse) {
    .hint .fine {
      display: none;
    }
    .hint .touch {
      display: inline-flex;
    }
    .hint {
      transition: opacity var(--dur-slow) var(--ease-in);
    }
    .hint.touched {
      opacity: 0;
    }
  }
  aside {
    border-left: 1px solid var(--line-subtle);
    overflow-y: auto;
    padding: var(--space-2) 0 var(--space-4);
  }
  section {
    padding: var(--space-3) var(--space-4);
  }
  section + section {
    border-top: 1px solid var(--line-subtle);
  }
  h3 {
    display: flex;
    align-items: baseline;
    gap: 6px;
    margin-bottom: var(--space-2);
    font-size: var(--type-micro-size);
    line-height: var(--type-micro-line);
    letter-spacing: var(--type-micro-tracking);
    font-weight: var(--type-micro-weight);
    text-transform: uppercase;
    color: var(--text-tertiary);
  }
  h3 .meta {
    color: var(--text-secondary);
    letter-spacing: 0;
  }
  .labels {
    margin: 0 calc(var(--space-2) * -1);
    display: grid;
  }
  .labels button {
    width: 100%;
    display: grid;
    grid-template-columns: 10px minmax(0, 1fr) auto;
    align-items: center;
    gap: var(--space-2);
    height: 28px;
    padding: 0 var(--space-2);
    border-radius: var(--radius-sm);
    text-align: left;
    transition: background-color var(--dur-fast) var(--ease-standard);
  }
  .labels button.on {
    background: var(--bg-active);
  }
  .labels button.dim {
    opacity: 0.5;
  }
  .labels button:focus-visible {
    outline-offset: -2px;
  }
  .sw {
    width: 10px;
    height: 10px;
    border-radius: 2px;
    flex-shrink: 0;
  }
  /* Class swatches keep the palette hue in both themes; a hairline edge holds pale hues on light surfaces. */
  :global([data-theme='light']) .sw {
    box-shadow: inset 0 0 0 1px #0a0e1433;
  }
  .sz {
    font-size: var(--type-xs-size);
    color: var(--text-tertiary);
  }
  .none {
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
  }
  .legend {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 6px var(--space-3);
  }
  .legend li {
    display: flex;
    align-items: center;
    gap: 6px;
    min-width: 0;
    font-size: var(--type-sm-size);
  }
  .legend li.absent {
    color: var(--text-tertiary);
  }
  .legend li.absent .sw {
    opacity: 0.4;
  }
  dl {
    display: grid;
    grid-template-columns: 48px minmax(0, 1fr);
    gap: 6px var(--space-3);
    font-size: var(--type-sm-size);
  }
  dt {
    color: var(--text-tertiary);
  }
  dd {
    overflow-wrap: anywhere;
  }
  @keyframes enter {
    from {
      opacity: 0;
      transform: scale(0.985);
    }
  }
  @media (max-width: 1023px) {
    .body {
      grid-template-columns: minmax(0, 1fr) 248px;
    }
  }
  @media (max-width: 767px) {
    .insp {
      width: 100vw;
      height: 100dvh;
      border-radius: 0;
    }
    header {
      padding-left: var(--space-3);
    }
    /* The zoom bar under the image carries the controls and the gesture hint. */
    .head-zoom,
    .hint {
      display: none;
    }
    /* The image sets the stage height (no empty bands); the label list takes the rest. */
    .body {
      position: relative;
      display: flex;
      flex-direction: column;
    }
    .viewport {
      flex: none;
      width: 100%;
      aspect-ratio: var(--ar, 4 / 3);
      max-height: 70vh;
    }
    .zoom-bar {
      flex: none;
      display: grid;
      justify-items: center;
      padding: 0 var(--space-3) var(--space-2);
      border-top: 1px solid var(--line-subtle);
    }
    .zoom-bar .group {
      height: 44px;
    }
    .zoom-bar .zoom {
      min-width: 56px;
      height: 44px;
    }
    .touch-hint {
      font-size: var(--type-xs-size);
      line-height: var(--type-xs-line);
      color: var(--text-tertiary);
    }
    aside {
      flex: 1;
      min-height: 0;
      border-left: 0;
      border-top: 1px solid var(--line-subtle);
      padding-bottom: calc(var(--space-4) + env(safe-area-inset-bottom));
    }
  }
  /* After the base .labels rule so it wins on touch screens. */
  @media (pointer: coarse) {
    .labels button {
      height: 44px;
    }
  }
</style>
