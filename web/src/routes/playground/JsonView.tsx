/**
 * Pretty, syntax-tinted JSON with Copy and Download actions. From 1024 px every line stays whole (one detection per
 * line, as ./json lays them out): the block scrolls sideways (`data-scrolls`) and fades whichever edge has more behind
 * it. Narrower, long lines wrap at a space and hang deeper than their own indent.
 * Highlighted spans are mounted via a DocumentFragment in one effect so large detection payloads avoid per-token
 * reactive overhead.
 */
import Check from '$icons/check';
import Copy from '$icons/copy';
import Download from '$icons/download';
import { computed, css, defaults, effect, onCleanup, signal } from 'fw';
import Button from '$lib/ui/Button';
import { toast } from '$lib/ui/toast';
import { tooltip } from '$lib/ui/tooltip';
import { jsonLines, jsonText, jsonTokens } from './json';

/** A key stays with its value when a line wraps: the space after the colon does not break (Copy keeps plain JSON). */
const COLON = ':\u00a0';

function syncEdges(node: HTMLElement) {
  const setFlag = (key: 'moreX' | 'moreXStart' | 'moreY', on: boolean) => {
    if (on) node.dataset[key] = '';
    else delete node.dataset[key];
  };
  setFlag('moreXStart', node.scrollLeft > 1);
  setFlag('moreX', node.scrollWidth - node.clientWidth - node.scrollLeft > 1);
  setFlag('moreY', node.scrollHeight - node.clientHeight - node.scrollTop > 1);
}

export default function JsonView(props: { value: unknown; label?: string; filename?: string }) {
  const p = defaults(props, { label: 'JSON', filename: 'prediction.json' });
  const lines = computed(() => jsonLines(jsonTokens(p.value)));
  const copied = signal(false);
  const preEl = signal<HTMLPreElement | undefined>(undefined);
  const codeEl = signal<HTMLElement | undefined>(undefined);
  let reset: ReturnType<typeof setTimeout> | undefined;
  let alive = true;
  onCleanup(() => {
    alive = false;
    clearTimeout(reset);
  });

  // Track overflow edges on scroll and resize.
  effect(() => {
    const node = preEl.get();
    if (!node) return;
    const update = () => syncEdges(node);
    update();
    node.addEventListener('scroll', update, { passive: true });
    const ro = typeof ResizeObserver === 'undefined' ? null : new ResizeObserver(update);
    ro?.observe(node);
    return () => {
      node.removeEventListener('scroll', update);
      ro?.disconnect();
    };
  });

  // Render highlighted lines into <code> via DocumentFragment (O(1) reactive effects regardless of payload size).
  effect(() => {
    const code = codeEl.get();
    const list = lines.get();
    if (!code) return;
    const frag = document.createDocumentFragment();
    for (const ln of list) {
      const row = document.createElement('span');
      row.className = 'ln';
      row.style.setProperty('--ind', `${ln.indent}ch`);
      if (ln.indent) row.append(' '.repeat(ln.indent));
      for (const tk of ln.tokens) {
        if (tk.t === 'ws') row.append(tk.v);
        else {
          const span = document.createElement('span');
          span.className = tk.t;
          span.textContent = tk.v === ': ' ? COLON : tk.v;
          row.append(span);
        }
      }
      frag.append(row);
    }
    code.replaceChildren(frag);
    const node = preEl.peek();
    if (node) syncEdges(node);
  });

  async function copy() {
    try {
      await navigator.clipboard.writeText(jsonText(p.value));
      if (!alive) return;
      copied.set(true);
      clearTimeout(reset);
      reset = setTimeout(() => copied.set(false), 1600);
    } catch {
      if (alive) toast.info('Copy failed', 'Your browser blocked clipboard access.');
    }
  }

  function download() {
    let url: string | undefined;
    try {
      const blob = new Blob([jsonText(p.value)], { type: 'application/json' });
      url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = p.filename;
      a.style.display = 'none';
      document.body.appendChild(a);
      a.click();
      a.remove();
      const revoke = url;
      setTimeout(() => URL.revokeObjectURL(revoke), 1000);
    } catch {
      if (url) URL.revokeObjectURL(url);
      toast.info('Download failed', 'Your browser blocked the file download.');
    }
  }

  // From 1024 px a line never wraps (the block scrolls sideways); narrower, long lines wrap at a space and hang 4
  // characters deeper than their own indent.
  return (
    <div class="json">
      <div class="copy">
        <span class="tip" use:tooltip={'Download JSON'}>
          <Button size="sm" icon={Download} aria-label="Download JSON" onclick={download} />
        </span>
        <Button size="sm" icon={copied.get() ? Check : Copy} onclick={copy}>
          {copied.get() ? 'Copied' : 'Copy'}
        </Button>
      </div>
      <pre ref={(e) => preEl.set(e)} tabindex={0} aria-label={p.label} data-scrolls>
        <code ref={(e) => codeEl.set(e)}></code>
      </pre>
    </div>
  );
}

css`
  /* The frame is the wrapper, so the scroller's edge fades never fade the frame's own edge. */
  .json {
    position: relative;
    min-width: 0;
    border-radius: var(--radius-md);
    background: var(--bg-inset);
    border: 1px solid var(--line-subtle);
    overflow: hidden;
  }
  .copy {
    position: absolute;
    top: var(--space-2);
    right: var(--space-2);
    z-index: 1;
    display: flex;
    align-items: center;
    gap: var(--space-1);
  }
  .tip {
    display: inline-flex;
  }
  pre {
    --fade-l: 0px;
    --fade-r: 0px;
    --fade-b: 0px;
    --fade-l-end: #000;
    --fade-r-end: #000;
    margin: 0;
    max-height: 420px;
    overflow: auto;
    padding: var(--space-3) var(--space-4);
    padding-right: 132px; /* grid-ok: room for the Download and Copy buttons */
    font-family: var(--font-mono);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
    tab-size: 2;
    white-space: pre;
  }
  .json :global(.ln) {
    display: block;
  }
  /* More below, more to the left/right, or both: each edge with more behind it fades cleanly. */
  pre:global([data-more-x-start]) {
    --fade-l: 24px;
    --fade-l-end: transparent;
  }
  pre:global([data-more-x]) {
    --fade-r: 32px;
    --fade-r-end: transparent;
  }
  pre:global([data-more-y]) {
    --fade-b: 32px;
  }
  pre:global([data-more-y]) {
    mask-image: linear-gradient(#000 calc(100% - var(--fade-b)), transparent);
  }
  pre:global([data-more-x]),
  pre:global([data-more-x-start]) {
    mask-image: linear-gradient(90deg, var(--fade-l-end), #000 var(--fade-l), #000 calc(100% - var(--fade-r)), var(--fade-r-end));
  }
  pre:global([data-more-y][data-more-x]),
  pre:global([data-more-y][data-more-x-start]) {
    mask-image:
      linear-gradient(#000 calc(100% - var(--fade-b)), transparent),
      linear-gradient(90deg, var(--fade-l-end), #000 var(--fade-l), #000 calc(100% - var(--fade-r)), var(--fade-r-end));
    mask-composite: intersect;
  }
  /* Under 1024 px: long lines wrap at a space, and a wrapped line's continuation starts two steps (4 ch) right of the
     line's own indent. */
  @media (max-width: 1023px) {
    pre {
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }
    .json :global(.ln) {
      padding-left: calc(var(--ind) + 4ch);
      text-indent: calc(-1 * (var(--ind) + 4ch));
    }
  }
  pre:focus-visible {
    outline-offset: -2px;
  }
  .json :global(.key) {
    color: var(--text-primary);
  }
  .json :global(.str) {
    color: var(--chart-text-2);
  }
  .json :global(.num) {
    color: var(--chart-text-4);
  }
  .json :global(.lit) {
    color: var(--chart-text-3);
  }
  .json :global(.punct) {
    color: var(--text-tertiary);
  }
`;
