/**
 * A dataset image, or a labeling job's (`jobId`), resolved through the API so it works with the mock too, with
 * optional boxes on top. The tile
 * keeps its geometry throughout: a skeleton until the image has loaded, the boxes only over a loaded image (never
 * floating on an empty frame), and a failed URL or image says so, with Retry.
 */
import ImageOff from '$icons/image-off';
import { computed, css, effect, signal, splitProps } from 'fw';
import { hub } from '$lib/api/index';
import Skeleton from '$lib/ui/Skeleton';
import DetectionView, { type DetectionViewProps, type ViewBox } from './DetectionView';

export interface DatasetImageProps {
  /** The dataset the image is in; or `jobId`, the labeling job's. */
  datasetId?: string;
  jobId?: string;
  imageId: string;
  width: number;
  height: number;
  thumb?: number;
  boxes?: ViewBox[];
  truth?: ViewBox[];
  classes: readonly string[];
  labels?: boolean;
  scores?: boolean;
  only?: ReadonlySet<number> | null;
  active?: number | null;
  onhover?: (i: number | null) => void;
  /** Passed through to DetectionView: the selected box (accent stroke, the others recede). */
  selected?: number | null;
  onselect?: (i: number) => void;
  fit?: 'contain' | 'cover';
  compact?: boolean;
  alt?: string;
}

const NO_BOXES: ViewBox[] = [];

export default function DatasetImage(props: DatasetImageProps) {
  const [, rest] = splitProps(props, ['datasetId', 'jobId', 'imageId', 'thumb', 'boxes', 'truth']);

  const src = signal<string | null>(null);
  const phase = signal<'loading' | 'ready' | 'error'>('loading');
  const attempt = signal(0);
  const el = signal<HTMLElement>();
  // Inside a button or link (a picker thumbnail) Retry cannot be a button of its own. While the image is unavailable
  // the tile itself retries instead of picking (click, Enter or Space alike; an unavailable image cannot be picked),
  // and "Retry" is part of the tile's accessible name, so keyboard and screen-reader users get the same action.
  const host = computed(() => el.get()?.parentElement?.closest<HTMLElement>('button, a[href]') ?? null);
  const nested = computed(() => !!host.get());
  effect(() => {
    const h = host.get();
    if (!h || phase.get() !== 'error') return;
    h.addEventListener('click', retry, { capture: true });
    // A tile with its own aria-label ("Use …", "Try …") is named by that, not by the text inside: say it there.
    const label = h.getAttribute('aria-label');
    if (label !== null) h.setAttribute('aria-label', `${label}: image unavailable, activate to retry`);
    return () => {
      h.removeEventListener('click', retry, { capture: true });
      if (label !== null) h.setAttribute('aria-label', label);
    };
  });

  effect(() => {
    void attempt.get();
    let live = true;
    src.set(null);
    phase.set('loading');
    const jobId = props.jobId;
    const imageId = props.imageId;
    const thumb = props.thumb;
    (jobId ? hub.api.labelImageUrl(jobId, imageId, thumb) : hub.api.imageUrl(props.datasetId ?? '', imageId, thumb)).then(
      (u: string) => {
        if (live) src.set(u);
      },
      () => {
        if (live) phase.set('error');
      },
    );
    return () => {
      live = false;
    };
  });

  /** Native (not delegated) capture listeners, so the tile around it never sees the click. */
  function retry(e: Event) {
    e.stopPropagation();
    e.preventDefault();
    attempt.set((a) => a + 1);
  }

  return (
    <div class="dsimg" ref={el} aria-busy={phase.get() === 'loading'}>
      <DetectionView
        src={phase.get() === 'error' ? null : src.get()}
        boxes={phase.get() === 'ready' ? (props.boxes ?? NO_BOXES) : NO_BOXES}
        truth={phase.get() === 'ready' ? (props.truth ?? NO_BOXES) : NO_BOXES}
        onload={(ok) => phase.set(ok ? 'ready' : 'error')}
        {...(rest as Omit<DetectionViewProps, 'src'>)}
      />
      {phase.get() === 'loading' ? (
        <span class="cover">
          <Skeleton w="100%" h="100%" r="0" image />
        </span>
      ) : phase.get() === 'error' ? (
        <span class="cover failed">
          <ImageOff size={16} strokeWidth={1.75} aria-hidden="true" />
          <span class="msg">Image unavailable</span>
          {nested.get() ? (
            /* The tile is the control (see `host` above): this is its label. */
            <span class="retry">Retry</span>
          ) : (
            <button type="button" class="retry" onclickcapture={retry} onkeydowncapture={(e) => e.stopPropagation()}>
              Retry
            </button>
          )}
        </span>
      ) : null}
    </div>
  );
}

css`
  .dsimg {
    position: relative;
    width: 100%;
    height: 100%;
  }
  .cover {
    position: absolute;
    inset: 0;
    display: flex;
  }
  .cover > :global(.sk) {
    flex: 1;
  }
  .failed {
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: var(--space-1);
    padding: var(--space-2);
    background: var(--bg-inset);
    color: var(--text-tertiary);
    text-align: center;
  }
  .msg {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
  }
  .retry {
    display: inline-flex;
    align-items: center;
    height: var(--control-sm);
    padding: 0 var(--space-2);
    border-radius: var(--radius-sm);
    font-size: var(--type-sm-size);
    font-weight: var(--weight-medium);
    color: var(--text-primary);
    text-decoration: underline;
    text-decoration-color: var(--line-strong);
    text-underline-offset: 3px;
    cursor: pointer;
  }
  .retry:hover {
    background: var(--bg-hover);
    text-decoration-color: currentColor;
  }
  /* Touch: a 44 px target. Inside a button the whole tile is the target, so the pointer-only Retry stays small. */
  @media (pointer: coarse) {
    button.retry {
      min-height: 44px;
    }
  }
`;
