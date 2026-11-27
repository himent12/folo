// Model catalog, ranking, and URL/vanished-model selection state for Playground.
import { computed, signal } from 'fw';
import { hub } from '$lib/api/index';
import { modelsResource, refreshOnModelsChange } from '$lib/api/models';
import { Resource } from '$lib/api/resource';
import type { Dataset, Model, Run } from '$lib/api/types';
import { router } from '$lib/router';
import { betterModel, modelChoices, scoreContext } from '$lib/scoreSummary';
import { watched } from '$lib/shell/runList';
import { noDatasetsYet, refreshOnDatasetChange } from '../firstStep';
import { followVanished } from './vanished';

export function useModelCatalog(onSyncUrl: (id: string) => void) {
  const models = modelsResource();
  const runs = new Resource((s) => hub.api.runs(s), { startFrom: () => watched.source });
  refreshOnModelsChange(runs);
  const datasets = new Resource((s) => hub.api.datasets(s));
  const noDatasets = computed(() => noDatasetsYet(datasets.data));
  refreshOnDatasetChange(datasets);

  const choices = computed(() => {
    if (!models.data || (!runs.data && !runs.error) || (!datasets.data && !datasets.error)) return [];
    const runById = new Map<string, Run>((runs.data ?? []).map((r) => [r.id, r]));
    const dsById = new Map<string, Dataset>((datasets.data ?? []).map((d) => [d.id, d]));
    return modelChoices(models.data, (m) => scoreContext(m, runById, dsById));
  });
  const ranked = computed(() => choices.get().map((c) => c.model));
  const best = computed(() => [...ranked.get()].sort(betterModel)[0] ?? null);
  const queryModel = computed(() => router.route.query.get('model'));
  const pickedModel = signal<string | null>(null);
  const wanted = computed(() => pickedModel.get() ?? queryModel.get());
  const model = computed((): Model | null => ranked.get().find((m) => m.id === wanted.get()) ?? best.get());
  const modelId = computed(() => model.get()?.id ?? null);
  const o2o = computed(() => model.get()?.head === 'o2o');
  const pickedMeta = computed(() => choices.get().find((c) => c.model.id === modelId.get())?.meta ?? null);

  function chooseModel(id: string) {
    pickedModel.set(id);
    onSyncUrl(id);
  }
  followVanished(() => model.get(), () => ranked.get(), () => wanted.get(), chooseModel);

  return {
    models,
    noDatasets,
    choices,
    ranked,
    queryModel,
    pickedModel,
    model,
    modelId,
    o2o,
    pickedMeta,
    chooseModel,
  };
}
