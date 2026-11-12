/**
 * One image of a labeling job in the grid, in each of its states: waiting (the image alone), with the agent (a live
 * mark), labeled (its boxes, or "No objects" at rest, so a background image never reads as a tile that failed to
 * load), failed (dimmed, with the word). An image corrected by hand says "Edited", one a person checked (marked right,
 * or saved by hand) "Reviewed": an edited image says both, and "Mark not reviewed" takes the second away.
 */
import Check from '$icons/check';
import CircleAlert from '$icons/circle-alert';
import Pencil from '$icons/pencil';
import { computed, css } from 'fw';
import type { Box, LabelImage } from '$lib/api/types';
import DatasetImage from '$lib/detect/DatasetImage';
import type { Density } from '$lib/ui/TileGrid';

export interface LabelTileProps {
  img: LabelImage;
  jobId: string;
  classes: readonly string[];
  boxes: boolean;
  density: Density;
}

/** One empty list for every tile without boxes shown, so a tile's boxes prop only changes when its boxes do. */
const NO_BOXES: Box[] = [];

export default function LabelTile(props: LabelTileProps) {
  const labeled = () => props.img.status === 'labeled';
  // The image's id on its own: DatasetImage loads the image again whenever what its `imageId` reads changes, and a new
  // copy of the same image (the agent took it, labeled it) is not another image to load.
  const imageId = computed(() => props.img.id);

  return (
    <span class="tile" class:failed={props.img.status === 'failed'}>
      <DatasetImage
        jobId={props.jobId}
        imageId={imageId.get()}
        width={props.img.width}
        height={props.img.height}
        thumb={256}
        boxes={props.boxes && labeled() ? props.img.boxes : NO_BOXES}
        classes={props.classes}
        compact
        fit="contain"
        alt=""
      />
      <span class="marks">
        {props.img.status === 'labeling' ? (
          <span class="mark live">
            <span class="dot" aria-hidden="true"></span>Labeling
          </span>
        ) : props.img.status === 'failed' ? (
          <span class="mark">
            <CircleAlert size={12} strokeWidth={2} class="bad" aria-hidden="true" />
            Failed
          </span>
        ) : labeled() && props.img.boxes.length === 0 ? (
          <span class="mark">No objects</span>
        ) : null}
        {props.img.source === 'manual' && props.img.status !== 'labeling' ? (
          <span class="mark">
            <Pencil size={12} strokeWidth={2} aria-hidden="true" />
            Edited
          </span>
        ) : null}
        {labeled() && props.img.reviewed ? (
          <span class="mark">
            <Check size={12} strokeWidth={2} aria-hidden="true" />
            Reviewed
          </span>
        ) : null}
      </span>
      {props.density !== 's' ? (
        <span class="file">
          <span class="mono truncate">{props.img.file}</span>
        </span>
      ) : null}
    </span>
  );
}

css`
  .tile {
    position: absolute;
    inset: 0;
  }
  /* A failed image steps back, and says so in its own frame. */
  .failed :global(.dsimg) {
    opacity: 0.55;
  }
  .marks,
  .file {
    position: absolute;
    left: 4px;
    right: 4px;
    display: flex;
    gap: var(--space-1);
    pointer-events: none;
  }
  .marks {
    top: 4px;
    flex-wrap: wrap;
  }
  .file {
    bottom: 4px;
    opacity: 0;
    transition: opacity var(--dur-fast) var(--ease-standard);
  }
  :global(.cell:hover) .file,
  :global(.cell:focus-visible) .file {
    opacity: 1;
  }
  .mark,
  .file > span {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    height: 18px;
    padding: 0 4px;
    border-radius: var(--radius-xs);
    background: var(--bg-scrim);
    color: var(--text-primary);
    font-size: var(--type-micro-size); /* small: image chip (a thumbnail's state) */
    font-weight: var(--weight-medium);
    backdrop-filter: blur(4px);
    white-space: nowrap;
  }
  .file > span {
    display: block;
    line-height: 18px;
    font-weight: var(--weight-regular);
  }
  :global([data-theme='light']) .mark,
  :global([data-theme='light']) .file > span {
    background: color-mix(in srgb, var(--bg-overlay) 88%, transparent);
  }
  .mark :global(.bad) {
    color: var(--danger-text);
  }
  /* The agent has the image now: the live dot, with the heartbeat that lasts exactly as long as the call. */
  .dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--accent-solid);
    --pulse: var(--accent-line);
    animation: live-pulse 2.4s var(--ease-out) infinite;
  }
  :global([data-theme='light']) .dot {
    background: var(--accent-line);
  }
  @media (prefers-reduced-motion: reduce) {
    .dot {
      animation: none;
    }
  }
`;
