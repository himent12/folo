/**
 * A file system path cut in the middle when it does not fit: its start gives way with an ellipsis, and its last two
 * parts (the folder and the one holding it) stay whole, so `…/place-a/cells` and `…/place-b/cells` tell two datasets
 * apart where an ellipsis at the end left both reading `D:/data/w…`. The full path is its title and accessible name.
 * Mono, in the colour of the text around it. A tail that is too long by itself is cut at its end, as a last resort.
 */
import { computed, css } from 'fw';

export interface PathTextProps {
  path: string;
  class?: string;
}

/** `path` split into what may give way and the last two parts kept whole (`/` separators kept as written). */
export function splitPath(path: string): { head: string; tail: string } {
  // A trailing separator goes, but not a root's (`/`, `D:/`).
  const trimmed = path.replace(/(?<=[^\\/:])[\\/]+$/, '');
  const cuts = [...trimmed.matchAll(/[\\/]/g)].map((m) => m.index!);
  // A path of three parts or fewer (D:/data/coco) is all tail: there is nothing worth cutting first.
  const at = cuts.length >= 3 ? cuts[cuts.length - 2]! : 0;
  return { head: trimmed.slice(0, at), tail: trimmed.slice(at) || path };
}

/**
 * Marks the path `cut` while its start is clipped, measured on every resize: the start then ends in a sans "…" of its
 * own (a mono one sits in a full cell and reads as "D:/fo… /data").
 */
function cutWhenNarrow(node: HTMLElement) {
  let raf = 0;
  let destroyed = false;
  const update = () => {
    const head = node.querySelector<HTMLElement>('.head');
    node.classList.toggle('cut', !!head && head.scrollWidth > head.clientWidth + 1);
  };
  // Defer ResizeObserver DOM writes to the next animation frame so toggling `.cut` (which reveals `.ell`)
  // never triggers Chromium's "ResizeObserver loop completed with undelivered notifications." error.
  const schedule = () => {
    if (raf || destroyed) return;
    if (typeof requestAnimationFrame === 'undefined') {
      update();
      return;
    }
    raf = requestAnimationFrame(() => {
      raf = 0;
      if (!destroyed && node.isConnected) update();
    });
  };
  update();
  const ro = typeof ResizeObserver === 'undefined' ? null : new ResizeObserver(schedule);
  ro?.observe(node);
  return {
    destroy() {
      destroyed = true;
      if (raf && typeof cancelAnimationFrame !== 'undefined') cancelAnimationFrame(raf);
      raf = 0;
      ro?.disconnect();
    },
  };
}

export default function PathText(props: PathTextProps) {
  const parts = computed(() => splitPath(props.path));
  return (
    <span class={`ui-path mono ${props.class ?? ''}`} title={props.path} aria-label={props.path} use:cutWhenNarrow>
      {parts.get().head ? (
        <>
          <span class="head" aria-hidden="true">
            {parts.get().head}
          </span>
          <span class="ell" aria-hidden="true">
            …
          </span>
        </>
      ) : null}
      <span class="tail" aria-hidden="true">
        {parts.get().tail}
      </span>
    </span>
  );
}

css`
  .ui-path {
    display: inline-flex;
    min-width: 0;
    max-width: 100%;
    white-space: nowrap;
  }
  .head,
  .tail {
    overflow: hidden;
  }
  /* The start gives way first, down to nothing, clipped; its "…" shows only while it is cut. The tail gives way only
     when it alone is wider than the room. */
  .head {
    flex: 0 1 auto;
    min-width: 0;
    text-overflow: clip;
  }
  .ell {
    display: none;
    flex: none;
    font-family: var(--font-sans);
  }
  .cut .ell {
    display: inline;
  }
  .tail {
    flex: none;
    max-width: 100%;
    text-overflow: ellipsis;
  }
`;
