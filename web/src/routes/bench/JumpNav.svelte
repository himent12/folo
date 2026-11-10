<script lang="ts">
  /**
   * A long result page's sections as one sticky strip under the top bar (Settings' jump strip, at every width here: the
   * page is one column of wide charts, so there is no rail to hold them): a tap or Enter scrolls the section's heading
   * clear of the chrome and moves focus to it, and the strip marks the section being read (./settings/spy's rule: the
   * one under the chrome, never the last because the page bottomed out). Where the links do not fit (a phone) it
   * scrolls sideways, the clipped edge fading with a chevron, the current link kept whole in view.
   */
  import ChevronRight from '@lucide/svelte/icons/chevron-right';
  import { fadeScroll } from '$lib/ui/fadeScroll';
  import { SCROLL_KEYS, sectionAt } from '../settings/spy';

  let { sections, label }: { sections: { id: string; label: string }[]; label: string } = $props();

  let current = $state<string | undefined>(undefined);
  let chosen: string | null = null;
  let el = $state<HTMLElement>();
  let spy = () => {};
  $effect(() => {
    const list = sections;
    let raf = 0;
    const update = () => {
      raf = 0;
      if (chosen) {
        current = chosen;
        return;
      }
      const line = el ? el.getBoundingClientRect().bottom : 100;
      const tops = list.flatMap((x) => {
        const s = document.getElementById(x.id);
        return s ? [{ id: x.id, top: s.getBoundingClientRect().top }] : [];
      });
      current = sectionAt(tops, line + 24);
    };
    spy = update;
    const onscroll = () => {
      if (!raf) raf = requestAnimationFrame(update);
    };
    const release = () => {
      if (chosen === null) return;
      chosen = null;
      onscroll();
    };
    const onkey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      if (SCROLL_KEYS.has(e.key) && !t?.closest('input, textarea, select, [role=slider], [role=radiogroup], [role=group], [role=grid]')) release();
    };
    const onpointer = (e: PointerEvent) => e.target === document.documentElement && release();
    update();
    window.addEventListener('scroll', onscroll, { passive: true });
    window.addEventListener('resize', onscroll);
    window.addEventListener('wheel', release, { passive: true });
    window.addEventListener('touchmove', release, { passive: true });
    window.addEventListener('keydown', onkey);
    window.addEventListener('pointerdown', onpointer);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener('scroll', onscroll);
      window.removeEventListener('resize', onscroll);
      window.removeEventListener('wheel', release);
      window.removeEventListener('touchmove', release);
      window.removeEventListener('keydown', onkey);
      window.removeEventListener('pointerdown', onpointer);
    };
  });

  // The current link stays whole in view, clear of the faded edges.
  $effect(() => {
    void current;
    const on = el?.querySelector<HTMLElement>('a.on');
    if (!el || !on) return;
    const left = on.offsetLeft - el.offsetLeft;
    const fade = 48;
    if (left - fade < el.scrollLeft || left + on.offsetWidth + fade > el.scrollLeft + el.clientWidth)
      el.scrollTo({ left: Math.max(0, left - (el.clientWidth - on.offsetWidth) / 2), behavior: 'instant' });
  });

  function goTo(id: string, keyboard: boolean) {
    chosen = id;
    current = id;
    const target = document.getElementById(id);
    if (!target) return;
    target.scrollIntoView({ block: 'start' });
    const heading = target.querySelector<HTMLElement>('h2, h3') ?? target;
    if (!heading.hasAttribute('tabindex')) heading.setAttribute('tabindex', '-1');
    heading.focus({ preventScroll: true, ...(keyboard ? {} : { focusVisible: false }) });
    spy();
  }
</script>

<div class="jump-bar" data-tip-chrome>
  <nav class="jump" aria-label={label} data-scrolls use:fadeScroll bind:this={el}>
    {#each sections as s (s.id)}<a
        href="#{s.id}"
        class:on={current === s.id}
        aria-current={current === s.id ? 'location' : undefined}
        onclick={(e) => {
          e.preventDefault();
          goTo(s.id, e.detail === 0);
        }}>{s.label}</a
      >{/each}
  </nav>
  <span class="jump-more" aria-hidden="true"><ChevronRight size={16} strokeWidth={1.75} /></span>
</div>

<style>
  /* Full bleed on the page's background just under the top bar, a hairline under it; its links neutral, the current
     one bg.active (the rail's and Settings' strip's selection). */
  .jump-bar {
    position: sticky;
    top: var(--size-topbar);
    z-index: var(--z-sticky);
    margin: calc(-1 * var(--space-2)) calc(-1 * var(--space-6)) 0;
    background: var(--bg-app);
    border-bottom: 1px solid var(--line-subtle);
  }
  .jump {
    display: flex;
    align-items: center;
    gap: var(--space-1);
    height: 44px;
    padding: 0 var(--space-5);
    overflow-x: auto;
    scrollbar-width: none;
  }
  .jump::-webkit-scrollbar {
    display: none;
  }
  .jump:global([data-fade='end']) {
    mask-image: linear-gradient(to right, #000 calc(100% - 80px), transparent calc(100% - 32px));
  }
  .jump:global([data-fade='start']) {
    mask-image: linear-gradient(to left, #000 calc(100% - 48px), transparent);
  }
  .jump:global([data-fade='both']) {
    mask-image: linear-gradient(to right, transparent, #000 48px, #000 calc(100% - 80px), transparent calc(100% - 32px));
  }
  .jump-more {
    position: absolute;
    top: 0;
    right: var(--space-2);
    bottom: 0;
    display: none;
    align-items: center;
    color: var(--text-tertiary);
    pointer-events: none;
  }
  .jump:global([data-fade='end']) + .jump-more,
  .jump:global([data-fade='both']) + .jump-more {
    display: flex;
  }
  .jump a {
    flex-shrink: 0;
    display: flex;
    align-items: center;
    height: 32px;
    padding: 0 var(--space-3);
    border-radius: var(--radius-md);
    color: var(--text-secondary);
    white-space: nowrap;
  }
  .jump a.on {
    background: var(--bg-active);
    color: var(--text-primary);
  }
  @media (hover: hover) {
    .jump a:hover {
      background: var(--bg-hover);
      color: var(--text-primary);
    }
  }
  @media (max-width: 767px) {
    .jump-bar {
      margin-inline: calc(-1 * var(--space-4));
    }
    .jump {
      padding: 0 var(--space-3);
    }
  }
  /* Touch: 44 px targets, so the strip is 52 px. */
  @media (pointer: coarse) {
    .jump {
      height: 52px;
    }
    .jump a {
      height: 44px;
    }
  }
</style>
