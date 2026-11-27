// Image, video, and webcam source lifecycle for Playground: URL resolution, retries, file intake,
// drag-and-drop (routing local video files into the inline frame scrubber), clipboard paste, live
// frame updates without wiping hidden classes/zoom, and object-URL cleanup.
import { effect, onCleanup, signal, untrack, type Signal } from 'fw';
import { hub } from '$lib/api/index';
import type { BrowserEngine } from '$lib/inference/browser';
import { toast } from '$lib/ui/toast';
import { droppedImages, takeImage } from './intake';
import { isVideoFile } from './media';
import type { Source } from './source';

export type Seg = 'file' | 'dataset' | 'video' | 'webcam';

export interface UseSourceInput {
  engine: BrowserEngine;
  hasModels: () => boolean;
  onResetState: () => void;
  onClearExtra: () => void;
  onDropVideo: (file: File) => void;
  onOpenVideo: () => void;
  onStartWebcam: () => void;
  onSwitchAwayFromMedia: () => void;
  openFilePicker: () => void;
  pickerOpen: Signal<boolean>;
}

export function usePlaygroundSource(input: UseSourceInput) {
  const source = signal<Source | null>(null);
  const src = signal<string | null>(null);
  const seg = signal<Seg>('file');
  const dragging = signal(false);
  const imageError = signal(false);
  const imageRetry = signal(0);
  const retryImage = () => imageRetry.set((n) => n + 1);

  effect(() => {
    const s = source.get();
    const retry = imageRetry.get();
    untrack(() => imageError.set(false));
    if (!s) {
      src.set(null);
      return;
    }
    if (s.kind === 'file') {
      src.set(s.url);
      return;
    }
    let live = true;
    src.set(null);
    hub.api.imageUrl(s.ds.id, s.img.id).then(
      (u) => {
        if (!live) return;
        src.set(retry > 0 && !/^(blob|data):/.test(u) ? `${u}${u.includes('?') ? '&' : '?'}retry=${retry}` : u);
      },
      () => {
        if (live) imageError.set(true);
      },
    );
    return () => {
      live = false;
    };
  });

  let decodeSeq = 0;
  let destroyed = false;
  function choose() {
    const seq = ++decodeSeq;
    return () => seq === decodeSeq && !destroyed;
  }

  function setSource(s: Source) {
    decodeSeq++;
    toast.clear('image');
    input.onSwitchAwayFromMedia();
    const old = source.peek();
    if (old?.kind === 'file') URL.revokeObjectURL(old.url);
    input.engine.clearImageCache();
    source.set(s);
    imageRetry.set(0);
    seg.set(s.kind);
    input.onResetState();
  }

  /**
   * Clears any previous still/dataset image and invalidates pending image decodes or URL lookups before
   * entering local video or live webcam mode so stale frames/metadata never leak onto the new media stage.
   */
  function prepareMediaSource(segKind: 'video' | 'webcam') {
    decodeSeq++;
    toast.clear('image');
    const old = source.peek();
    if (old?.kind === 'file') URL.revokeObjectURL(old.url);
    input.engine.clearImageCache();
    source.set(null);
    imageRetry.set(0);
    seg.set(segKind);
    input.onResetState();
  }

  /**
   * Updates Playground's active frame from a local video or live webcam without invalidating an explicit
   * user still-image pick/drop (`decodeSeq`), resetting hidden classes/zoom, or tearing down the decode canvas.
   */
  function setLiveFrame(blob: Blob, name: string, width: number, height: number, segKind: 'video' | 'webcam') {
    toast.clear('image');
    const old = source.peek();
    if (old?.kind === 'file') URL.revokeObjectURL(old.url);
    source.set({ kind: 'file', file: blob, name, url: URL.createObjectURL(blob), width, height });
    imageRetry.set(0);
    seg.set(segKind);
  }

  async function setFile(file: File | Blob, name = file instanceof File ? file.name : 'pasted image') {
    if (file instanceof File && isVideoFile(file)) {
      input.onDropVideo(file);
      return;
    }
    const live = choose();
    const t = await takeImage(file, name);
    if (!live()) return;
    if ('error' in t) {
      toast.error(new Error(t.error), 'Unsupported image', { key: 'image' });
      return;
    }
    setSource({ kind: 'file', file: t.file, name, url: URL.createObjectURL(t.file), width: t.width, height: t.height });
  }

  function clear() {
    decodeSeq++;
    toast.clear('image');
    input.onSwitchAwayFromMedia();
    const old = source.peek();
    if (old?.kind === 'file') URL.revokeObjectURL(old.url);
    input.engine.clearImageCache();
    source.set(null);
    imageRetry.set(0);
    seg.set('file');
    input.onResetState();
    input.onClearExtra();
  }

  onCleanup(() => {
    destroyed = true;
    decodeSeq++;
    const old = source.peek();
    if (old?.kind === 'file') URL.revokeObjectURL(old.url);
  });

  function onSeg(v: Seg, activeMedia: 'none' | 'video' | 'webcam' = 'none') {
    if (v === 'file') {
      input.openFilePicker();
      seg.set(activeMedia !== 'none' ? activeMedia : (source.peek()?.kind ?? 'file'));
    } else if (v === 'dataset') {
      input.pickerOpen.set(true);
      seg.set(activeMedia !== 'none' ? activeMedia : (source.peek()?.kind ?? 'file'));
    } else if (v === 'video') {
      input.onOpenVideo();
      seg.set(activeMedia !== 'none' ? activeMedia : (source.peek()?.kind ?? 'file'));
    } else if (v === 'webcam') {
      seg.set('webcam');
      input.onStartWebcam();
    }
  }

  function onpaste(e: ClipboardEvent) {
    if (document.querySelector('dialog[open]') || !input.hasModels()) return;
    const item = [...(e.clipboardData?.items ?? [])].find((i) => i.kind === 'file' && i.type.startsWith('image/'));
    const file = item?.getAsFile() ?? [...(e.clipboardData?.files ?? [])].find((f) => f.type.startsWith('image/'));
    if (!file) return;
    e.preventDefault();
    void setFile(file, 'pasted image');
  }

  function ondrop(e: DragEvent) {
    e.preventDefault();
    dragging.set(false);
    const allFiles = [...(e.dataTransfer?.files ?? [])];
    const videoFile = allFiles.find((f) => isVideoFile(f));
    if (videoFile && !allFiles.some((f) => f.type.startsWith('image/'))) {
      input.onDropVideo(videoFile);
      return;
    }
    const [file, ...rest] = droppedImages(e.dataTransfer?.files);
    if (!file) return;
    void setFile(file);
    if (rest.length) {
      toast.info(`Using ${file.name}`, `The Playground runs one image at a time, so the other ${rest.length === 1 ? 'image was' : `${rest.length} images were`} left out.`);
    }
  }

  function ondragover(e: DragEvent) {
    if (!e.dataTransfer?.types.includes('Files')) return;
    e.preventDefault();
    e.dataTransfer.dropEffect = 'copy';
    dragging.set(true);
  }

  function ondragleave(e: DragEvent) {
    const next = e.relatedTarget as Node | null;
    if (!next || !(e.currentTarget as HTMLElement).contains(next)) dragging.set(false);
  }

  return {
    source,
    src,
    seg,
    dragging,
    imageError,
    retryImage,
    choose,
    setSource,
    prepareMediaSource,
    setLiveFrame,
    setFile,
    clear,
    onSeg,
    onpaste,
    ondrop,
    ondragover,
    ondragleave,
    isDestroyed: () => destroyed,
  };
}
