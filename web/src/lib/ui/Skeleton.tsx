import { css, defaults } from 'fw';

export interface SkeletonProps {
  w?: string;
  h?: string;
  r?: string;
  /** Sits in a line of text (after a zero-width space), so the line keeps its real height. */
  inline?: boolean;
  /**
   * Holds the place of a photo (a thumbnail, an inspector canvas): a raised fill with faint viewfinder corners, so it
   * reads as an image loading rather than an empty dark photo, and a slow sweep (none under reduced motion).
   */
  image?: boolean;
  class?: string;
}

export default function Skeleton(props: SkeletonProps) {
  const p = defaults(props, { w: '100%', h: '12px', r: 'var(--radius-sm)', inline: false, image: false, class: '' });
  return (
    <span
      class={`sk ${p.class}`}
      class:inline={p.inline}
      class:image={p.image}
      style:width={p.w}
      style:height={p.h}
      style:border-radius={p.r}
      aria-hidden="true"
    ></span>
  );
}

css`
  .sk {
    display: block;
    background: linear-gradient(90deg, var(--bg-active) 0%, var(--bg-hover) 50%, var(--bg-active) 100%);
    background-size: 200% 100%;
    animation: shimmer 1.4s linear infinite;
  }
  .inline {
    display: inline-block;
    vertical-align: middle;
  }
  .image {
    position: relative;
    overflow: hidden;
    background: var(--bg-raised);
    animation: none;
  }
  /* Light: bg.raised is a hair off white, so a grid of them reads as a blank page. The step the dark raised fill takes
     from the app background (about 1.1:1), taken the other way: bg.active over bg.raised, a grey a step below the app
     background, framed by its corners. */
  :global([data-theme='light']) .image {
    background: linear-gradient(var(--bg-active), var(--bg-active)) var(--bg-raised);
  }
  /* The viewfinder: four 12 px corners of a frame 8 px in from the edges (a mask keeps only the corners). */
  .image::before {
    content: '';
    position: absolute;
    inset: var(--space-2);
    border: 1.5px solid var(--line-strong);
    border-radius: 2px;
    mask:
      linear-gradient(#000 0 0) top left / 12px 12px no-repeat,
      linear-gradient(#000 0 0) top right / 12px 12px no-repeat,
      linear-gradient(#000 0 0) bottom left / 12px 12px no-repeat,
      linear-gradient(#000 0 0) bottom right / 12px 12px no-repeat;
  }
  /* A soft band that sweeps across, so loading reads as loading (still under reduced motion). */
  .image::after {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(100deg, transparent 30%, var(--bg-hover) 50%, transparent 70%);
    background-size: 250% 100%;
    background-position: 125% 0;
    animation: sweep 1.6s var(--ease-standard) infinite;
  }
  @media (prefers-reduced-motion: reduce) {
    .image::after {
      display: none;
    }
  }
  @keyframes shimmer {
    from {
      background-position: 100% 0;
    }
    to {
      background-position: -100% 0;
    }
  }
  @keyframes sweep {
    to {
      background-position: -25% 0;
    }
  }
`;
