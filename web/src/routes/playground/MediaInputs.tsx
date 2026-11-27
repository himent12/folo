/**
 * Hidden file pickers (.folo weights, image/video files, video-only picker) and the offscreen
 * HTMLVideoElement decoder kept mounted for Playground's local video scrubber and live webcam stream.
 */
import { css } from 'fw';
import { PLAYGROUND_ACCEPT, VIDEO_ACCEPT } from './media';
import type { MediaSession } from './useMediaSession';

export interface MediaInputsProps {
  media: MediaSession;
  onWeightsRef: (el: HTMLInputElement) => void;
  onFileRef: (el: HTMLInputElement) => void;
  onVideoRef: (el: HTMLInputElement) => void;
  onPickWeights: (e: Event & { currentTarget: HTMLInputElement }) => void;
  onPickFile: (f: File) => void;
  onPickVideo: (f: File) => void;
}

export default function MediaInputs(props: MediaInputsProps) {
  const m = props.media;
  return (
    <>
      <input
        ref={props.onWeightsRef}
        type="file"
        accept=".folo"
        class="sr-only"
        tabindex={-1}
        aria-hidden="true"
        onchange={props.onPickWeights}
      />
      <input
        ref={props.onFileRef}
        type="file"
        accept={PLAYGROUND_ACCEPT}
        class="sr-only"
        tabindex={-1}
        aria-hidden="true"
        onchange={(e) => {
          const f = e.currentTarget.files?.[0];
          if (f) props.onPickFile(f);
          e.currentTarget.value = '';
        }}
      />
      <input
        ref={props.onVideoRef}
        type="file"
        accept={VIDEO_ACCEPT}
        class="sr-only"
        tabindex={-1}
        aria-hidden="true"
        onchange={(e) => {
          const f = e.currentTarget.files?.[0];
          if (f) props.onPickVideo(f);
          e.currentTarget.value = '';
        }}
      />
      <video
        ref={m.videoEl}
        src={m.videoUrl.get() ?? undefined}
        muted
        playsinline
        preload="auto"
        class="media-decoder"
        aria-hidden="true"
        {...m.videoEvents}
      ></video>
    </>
  );
}

css`
  .media-decoder {
    position: fixed;
    left: -9999px; /* grid-ok: offscreen video decoder kept active for frame extraction */
    width: 1px;
    height: 1px;
    opacity: 0;
    pointer-events: none;
  }
`;
