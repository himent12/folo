/**
 * The inspector's rail: the image's labels (hovering one marks its box, a click zooms to it), the class legend with
 * this image's counts, and the image's facts with its path, shortened in the middle and one click from the clipboard.
 */
import Copy from '$icons/copy';
import { bindable, computed, css, For, Index, signal } from 'fw';
import type { DatasetImage as Img } from '$lib/api/types';
import { classColor } from '$lib/design/tokens';
import DialogStatusRegion from '$lib/ui/DialogStatus';
import type { DialogStatus } from '$lib/ui/feedback';
import { tooltip } from '$lib/ui/tooltip';
import { swatchStyle } from './swatch';
import { shortDir, stem } from './imagePath';

export interface InspectorRailProps {
  img: Img | undefined;
  classes: readonly string[];
  only: ReadonlySet<number> | null;
  hovered?: number | null;
  selected: number | null;
  /** A label's click: select it and zoom to its box, or let go of it. */
  pick: (i: number) => void;
  /** Where a copy's result is said (the dialog's own status line). */
  status: DialogStatus;
}

export default function InspectorRail(props: InspectorRailProps) {
  const hovered = bindable(props, 'hovered');

  // ---------- labels ----------
  const counts = computed(() => {
    const c = new Map<number, number>();
    for (const b of props.img?.boxes ?? []) c.set(b.class, (c.get(b.class) ?? 0) + 1);
    return c;
  });
  const legend = computed(() => props.classes.map((name, i) => ({ i, name, n: counts.get().get(i) ?? 0 })));
  const name = (c: number) => props.classes[c] ?? `class ${c}`;
  /** The path's room, and one mono cell, so the folders shorten to whole characters with "…" set tight against "/". */
  const pathW = signal(0);
  const ch10 = signal(0);
  const chW = computed(() => ch10.get() / 10);

  async function copyFile(f: string) {
    try {
      await navigator.clipboard.writeText(f);
      props.status.success('Path copied', f);
    } catch {
      props.status.error(new Error('The browser blocked clipboard access; select the path and copy it by hand.'), 'Could not copy the path');
    }
  }

  const img = () => props.img!;
  const sd = computed(() => (props.img ? shortDir(props.img.file, pathW.get(), chW.get()) : { text: '', cut: false }));

  return (
    <aside>
      <section class="labels-sec">
        <h3>
          Labels <span class="meta num">{props.img?.boxes.length ?? 0}</span>
        </h3>
        {props.img && props.img.boxes.length ? (
          <div class="labels-scroll scroll-shade">
            <ul class="labels" onpointerleave={() => hovered.set(null)}>
              <Index each={img().boxes}>
                {(b, i) => (
                  <li>
                    <button
                      type="button"
                      class:hot={hovered.get() === i}
                      class:on={props.selected === i}
                      class:dim={props.only !== null && !props.only.has(b().class)}
                      aria-current={props.selected === i ? 'true' : undefined}
                      onpointerenter={(e) => {
                        // A tap is not a hover: on touch the selection alone marks the row.
                        if (e.pointerType !== 'touch') hovered.set(i);
                      }}
                      onpointerleave={() => {
                        if (hovered.get() === i) hovered.set(null);
                      }}
                      onfocus={(e) => {
                        // Keyboard focus previews its box; a click or tap's focus does not, so it never leaves a hover behind.
                        if (e.currentTarget.matches(':focus-visible')) hovered.set(i);
                      }}
                      onblur={() => hovered.set(null)}
                      onclick={() => props.pick(i)}
                      aria-label={`${name(b().class)}, ${Math.round(b().x2 - b().x1)} by ${Math.round(b().y2 - b().y1)} pixels. ${props.selected === i ? 'Selected; show the whole image' : 'Zoom to box'}`}
                    >
                      <span class="sw" style:background={classColor(b().class)}></span>
                      <span class="cn truncate">{name(b().class)}</span>
                      <span class="sz num">
                        {Math.round(b().x2 - b().x1)}×{Math.round(b().y2 - b().y1)}
                      </span>
                    </button>
                  </li>
                )}
              </Index>
            </ul>
          </div>
        ) : props.img ? (
          <p class="none">No boxes. This image is a background example.</p>
        ) : null}
      </section>

      <section>
        <h3>Classes</h3>
        <ul class="legend">
          <For each={legend.get()} key={(l) => l.i}>
            {(l) => (
              <li class:absent={l().n === 0}>
                <span class="sw" style={swatchStyle(l().i, l().n === 0)}></span>
                <span class="truncate">{l().name}</span>
                {l().n ? <span class="num faint">{l().n}</span> : null}
              </li>
            )}
          </For>
        </ul>
      </section>

      {props.img ? (
        <section>
          <h3>Image</h3>
          <dl>
            <dt>File</dt>
            {/* One line with the middle shortened (images/…/000001.jpg): the folders give way, the file name stays
                whole; the full path is in the tooltip and one click from the clipboard. */}
            <dd class="path" bind:clientWidth={pathW}>
              <span class="mono ch" aria-hidden="true" bind:clientWidth={ch10}>
                0000000000
              </span>
              <span class="mono mid" aria-label={img().file} use:tooltip={{ text: img().file }}>
                {sd.get().text ? (
                  <>
                    <span class="dir">{sd.get().text}</span>
                    {sd.get().cut ? <span class="ell">…</span> : null}/
                  </>
                ) : null}
                <span class="base">{stem(img().file)}</span>
              </span>
              <button type="button" class="copy" aria-label={`Copy the path ${img().file}`} use:tooltip={{ text: 'Copy path' }} onclick={() => copyFile(img().file)}>
                <Copy size={12} strokeWidth={2} aria-hidden="true" />
              </button>
            </dd>
            <dt>Split</dt>
            <dd>{img().split}</dd>
            <dt>Size</dt>
            <dd class="num">
              {img().width} × {img().height} px
            </dd>
            <dt>ID</dt>
            <dd class="mono">{img().id}</dd>
          </dl>
          <DialogStatusRegion status={props.status} class="copy-status" />
        </section>
      ) : null}
    </aside>
  );
}

css`
  /* The rail: the label list scrolls inside it (with the shade at an edge that has more), while Classes and the image's
     facts stay pinned under it, so the legend is never cut at the dialog's foot however many labels an image has. */
  aside {
    display: flex;
    flex-direction: column;
    min-height: 0;
    overflow-y: auto; /* only when even the pinned parts outgrow a short window */
    border-left: 1px solid var(--line-subtle);
    padding: var(--space-2) 0 var(--space-4);
  }
  aside > section {
    flex: none;
  }
  aside > section.labels-sec {
    flex: 0 1 auto;
    display: flex;
    flex-direction: column;
    min-height: 84px; /* grid-ok: the heading and two label rows */
  }
  .labels-scroll {
    min-height: 0;
    overflow-y: auto;
    margin: 0 calc(var(--space-4) * -1);
    padding: 0 var(--space-4);
  }
  section {
    padding: var(--space-2) var(--space-4);
  }
  section + section {
    border-top: 1px solid var(--line-subtle);
  }
  h3 {
    display: flex;
    align-items: center;
    gap: 6px;
    height: 28px;
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    font-weight: var(--weight-medium);
    color: var(--text-secondary);
  }
  h3 .meta {
    color: var(--text-tertiary);
    font-weight: var(--weight-regular);
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
  /* Hover is a light neutral step (only where a pointer can hover, cleared when the pointer leaves the row, so a tap
     never leaves one behind); the selection takes the one rule (bg.active and the 2 px ink bar, \`--select-bar\`, on the leading edge),
     which hover never changes. */
  @media (hover: hover) {
    .labels button.hot:not(.on) {
      background: var(--bg-hover);
    }
  }
  .labels button.on {
    background: var(--bg-active);
    box-shadow: var(--select-bar);
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
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
  .sz {
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
  }
  .none {
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
  }
  .legend {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    column-gap: var(--space-3);
  }
  .legend li {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    height: 28px;
    min-width: 0;
    font-size: var(--type-sm-size);
  }
  .legend li.absent {
    color: var(--text-tertiary);
  }
  dl {
    display: grid;
    grid-template-columns: 48px minmax(0, 1fr);
    grid-auto-rows: minmax(28px, auto);
    align-items: center;
    column-gap: var(--space-3);
    font-size: var(--type-sm-size);
  }
  dt {
    color: var(--text-tertiary);
  }
  dd {
    overflow-wrap: anywhere;
  }
  /* The copy's result, under the image's facts and beside the button that made it. */
  section :global(.copy-status.on) {
    margin-top: var(--space-2);
  }
  .path {
    position: relative;
    display: flex;
    align-items: center;
    gap: var(--space-1);
    min-width: 0;
  }
  .mid {
    display: flex;
    min-width: 0;
    white-space: nowrap;
  }
  .dir {
    min-width: 0;
    overflow: hidden;
    text-overflow: clip;
  }
  /* The cut folders end in a sans "…" (the mono one sits in a full cell and reads as "images… /"). */
  .ell {
    font-family: var(--font-sans);
  }
  /* One mono cell, measured (font and size as the path), out of the flow. */
  .ch {
    position: absolute;
    visibility: hidden;
    pointer-events: none;
  }
  .base {
    flex-shrink: 0;
  }
  .copy {
    display: grid;
    place-items: center;
    flex-shrink: 0;
    width: 24px;
    height: 24px;
    border-radius: var(--radius-sm);
    color: var(--text-tertiary);
  }
  .copy:hover {
    background: var(--bg-hover);
    color: var(--text-primary);
  }
  @media (max-width: 767px) {
    .labels-scroll {
      overflow: visible;
      background: none;
    }
    /* Phones: the rail under the image is short, so it scrolls as one list (labels, classes, facts) rather than
       pinning Classes under a two-row label list. */
    aside {
      display: block;
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
    .copy {
      width: 44px;
      height: 44px;
      margin-block: calc(var(--space-2) * -1);
    }
  }
`;
