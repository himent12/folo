// Where an action's result is said. The toaster is for page-level events (a run that finished, a page action's
// outcome); an action taken inside an open dialog shows its result in that dialog, in its status region
// (`ui/DialogStatus.svelte`), since the toaster sits outside the modal: behind its scrim and outside its focus trap.
// `ui/Dialog` keeps one `DialogStatus` and provides it to what it renders; a dialog of its own (the dataset inspector)
// makes one and renders the region itself. A component that may sit in either place calls `feedback()` once, at init.

import { getContext, setContext, signal } from 'fw';
import { errorDetail, toast, type ToastTone } from './toast';

export interface Notice {
  id: number;
  tone: ToastTone;
  title: string;
  detail?: string;
}

/** What `feedback()` hands out: the toaster, or a dialog's status region, with the same three calls. */
export interface Notifier {
  success(title: string, detail?: string): unknown;
  info(title: string, detail?: string): unknown;
  error(err: unknown, title?: string): unknown;
}

/** One dialog's latest result. It stays until the next result, Dismiss or the dialog closing: no timer. */
export class DialogStatus implements Notifier {
  #notice = signal<Notice | null>(null);
  get notice(): Notice | null {
    return this.#notice.get();
  }
  set notice(v: Notice | null) {
    this.#notice.set(v);
  }
  #id = 0;

  show(tone: ToastTone, title: string, detail?: string) {
    this.notice = { id: ++this.#id, tone, title, detail };
  }
  success(title: string, detail?: string) {
    this.show('success', title, detail);
  }
  info(title: string, detail?: string) {
    this.show('info', title, detail);
  }
  error(err: unknown, title = 'Something went wrong') {
    this.show('error', title, errorDetail(err));
  }
  clear() {
    this.notice = null;
  }
}

const KEY = Symbol('dialog-status');

/** Makes `status` the place where components rendered inside this one report (ui/Dialog calls it). */
export function provideStatus(status: DialogStatus): DialogStatus {
  setContext(KEY, status);
  return status;
}

/** The status region of the dialog this component renders in, else the toaster. Call at component init. */
export function feedback(): Notifier {
  return getContext<DialogStatus | undefined>(KEY) ?? toast;
}
