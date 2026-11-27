// Local .folo weights picker state and validation for Playground (Model A and Compare Model B).
import { signal, type Signal } from 'fw';
import { sentence } from '$lib/format';
import { looksLikeFolo } from './folo';

export interface BadWeightsFile {
  name: string;
  why: string | null;
}

export interface UseLocalWeightsInput {
  modelAId: () => string | null;
  modelBId: () => string | null;
  isDestroyed: () => boolean;
}

export function useLocalWeights(input: UseLocalWeightsInput) {
  const localWeights = signal(new Map<string, File>());
  const weightsMissing = signal(false);
  const weightsTarget = signal<'a' | 'b'>('a');
  const badFile = signal<BadWeightsFile | null>(null);
  const badFileB = signal<BadWeightsFile | null>(null);
  const weightPickSeq = new Map<string, number>();

  async function pickWeights(e: Event & { currentTarget: HTMLInputElement }) {
    const inputEl = e.currentTarget;
    const f = inputEl.files?.[0];
    inputEl.value = '';
    const side = weightsTarget.peek();
    const id = side === 'b' ? input.modelBId() : input.modelAId();
    if (!f || !id) return;
    const seq = (weightPickSeq.get(id) ?? 0) + 1;
    weightPickSeq.set(id, seq);
    if (!(await looksLikeFolo(f))) {
      if (!input.isDestroyed() && weightPickSeq.get(id) === seq) {
        if (side === 'b' && input.modelBId() === id) badFileB.set({ name: f.name, why: null });
        else if (side === 'a' && input.modelAId() === id) badFile.set({ name: f.name, why: null });
      }
      return;
    }
    if (input.isDestroyed() || weightPickSeq.get(id) !== seq) return;
    if (side === 'b' && input.modelBId() === id) badFileB.set(null);
    else if (side === 'a' && input.modelAId() === id) badFile.set(null);
    localWeights.set(new Map(localWeights.peek()).set(id, f));
  }

  function dropLocal(id: string, f: File, e: unknown, targetBad: Signal<BadWeightsFile | null> = badFile) {
    const cur = localWeights.peek();
    if (cur.get(id) === f) {
      const next = new Map(cur);
      next.delete(id);
      localWeights.set(next);
    }
    targetBad.set({ name: f.name, why: e instanceof Error ? sentence(e.message) : null });
  }

  return {
    localWeights,
    weightsMissing,
    weightsTarget,
    badFile,
    badFileB,
    pickWeights,
    dropLocal,
  };
}
