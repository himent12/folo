// Image source lifecycle for Playground: URL resolution, retries, file intake, drag-and-drop (including routing
// dropped video files to the frame capture dialog), clipboard paste, and object-URL cleanup.
import { effect, onCleanup, signal, untrack, type Signal } from 'fw';
import { hub } from '$lib/api/index';
import type { BrowserEngine } from '$lib/inference/browser';
import { toast } from '$lib/ui/toast';
import { droppedImages, takeImage } from './intake';
import type { Source } from './source';

export type Seg = 'file' | 'dataset' | 'webcam';

export interface UseSourceInput {
  engine: BrowserEngine;
  hasModels: () => boolean;
  onResetState: () => void;
  onClearExtra: () => void;
  onDropVideo: (file: File) => void;
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
    const old = source.peek();
    if (old?.kind === 'file') URL.revokeObjectURL(old.url);
    input.engine.clearImageCache();
    source.set(s);
    imageRetry.set(0);
    seg.set(s.kind);
    input.onResetState();
  }

  async function setFile(file: File | Blob, name = file instanceof File ? file.name : 'pasted image') {
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

  function onSeg(v: Seg) {
    if (v === 'file') input.openFilePicker();
    else if (v === 'dataset') input.pickerOpen.set(true);
    seg.set(source.peek()?.kind ?? 'file');
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
    const videoFile = allFiles.find((f) => f.type.startsWith('video/') || /\.(mp4|webm|mov|ogv)$/i.test(f.name));
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
