// New run (`#/runs/new`): the training form (dataset, model, training, augmentation), its estimate, the matching
// `folo train` command, and Start. The parts are in ./runnew/.
import Images from '$icons/images';
import Plus from '$icons/plus';
import { computed, css, Key } from 'fw';
import { href, router } from '$lib/router';
import { ownCreateAction, setCrumbs } from '$lib/shell/crumbs';
import Page from '$lib/shell/Page';
import Button from '$lib/ui/Button';
import EmptyState from '$lib/ui/EmptyState';
import ErrorState from '$lib/ui/ErrorState';
import Panel from '$lib/ui/Panel';
import AugmentSection from './runnew/AugmentSection';
import DatasetSection from './runnew/DatasetSection';
import ModelSection from './runnew/ModelSection';
import { DEFAULT_AUGMENT, IMGSZ_OPTIONS } from './runnew/newPrefill';
import OomNote from './runnew/OomNote';
import StartDock from './runnew/StartDock';
import { runNewState } from './runnew/state';
import { SummaryRail, SummarySheet } from './runnew/SummaryRail';
import TrainingSection from './runnew/TrainingSection';
import WhyLine from './runnew/WhyLine';

/**
 * New run follows its own URL while open (v-data-runs N1): a link that changes it (the palette's "New run on X", a
 * pasted link, Back / Forward) opens the form again under the rules of a first open, the draft included (that same
 * link's draft whole, else the link's fields over the draft's). The form never writes its own URL, so this cannot loop.
 */
export default function RunNew() {
  let shown = '';
  // Leaving (a start goes to the run), the route is another screen's: the form stays as it is until it is gone.
  const link = computed(() => (router.route.name === 'run-new' ? (shown = String(router.route.query)) : shown));
  return <Key when={link.get()}>{() => <RunNewForm />}</Key>;
}

function RunNewForm() {
  setCrumbs({ label: 'Runs', path: '/runs' }, { label: 'New run' });
  // This page is the create action: the top bar's New run steps aside instead of showing a pressed, disabled copy.
  ownCreateAction();

  const s = runNewState();

  return (
    <Page title="New training run">
      {s.datasets.error && !s.datasets.data ? (
        <ErrorState error={s.datasets.error} retry={() => s.datasets.refresh()} title="Could not load datasets" />
      ) : s.datasets.data && s.datasets.data.length === 0 ? (
        <Panel>
          <EmptyState
            icon={Images}
            title="Add a dataset first"
            description="Runs train on a registered dataset: add a folder or generate a synthetic one."
            actions={() => (
              // Straight to the Add dataset dialog, as Overview's first step (Datasets reads `?add=`).
              <>
                <Button variant="primary" icon={Plus} href={href('/datasets?add=folder')}>
                  Add dataset
                </Button>
                <Button href={href('/datasets?add=synthetic')}>Generate sample set</Button>
              </>
            )}
          />
        </Panel>
      ) : (
        <form class="layout" ref={s.formEl} onsubmit={s.submit} novalidate>
          {s.oomRun.get() ? (
            <OomNote
              oomRun={s.oomRun.get()!}
              oomChanges={s.oomChanges.get()}
              vramFit={s.vramFit.get()}
              vram={s.vram.get()}
              gpuMem={s.gpuMem.get()}
              vramLevel={s.vramLevel.get()}
            />
          ) : null}
          <div class="form">
            <DatasetSection datasets={s.datasets} bind:datasetId={s.datasetId} dataset={s.dataset.get()} errors={s.errors.get()} />

            <ModelSection
              bind:arch={s.arch}
              bind:weights={s.weights}
              folo2Supported={s.folo2Supported.get()}
              tp={s.tp}
              oomRun={s.oomRun.get()}
              was={s.was}
              pickedPublished={s.pickedPublished.get()}
              selArch={s.selArch.get()}
              startHint={s.startHint.get()}
              pickedAbout={s.pickedAbout.get()}
              errors={s.errors.get()}
              weightChoices={s.weightChoices.get()}
              otherChoices={s.otherChoices.get()}
              publishedOptions={s.publishedOptions.get()}
              scoreCtx={s.scoreCtx.get()}
              pickWeights={s.pickWeights}
            />

            <TrainingSection
              bind:epochs={s.epochs}
              bind:batch={s.batch}
              bind:imgsz={s.imgsz}
              bind:imgszChosen={s.imgszChosen}
              bind:device={s.device}
              bind:optimizer={s.optimizer}
              bind:amp={s.amp}
              bind:ampRefused={s.ampRefused}
              bind:e2e={s.e2e}
              bind:lr0={s.lr0}
              bind:seed={s.seed}
              bind:advancedOpen={s.advancedOpen}
              arch={s.arch.get()}
              errors={s.errors.get()}
              was={s.was}
              oomRun={s.oomRun.get()}
              dataset={s.dataset.get()}
              datasetImgsz={s.datasetImgsz.get()}
              IMGSZ={IMGSZ_OPTIONS}
              cloudOn={s.cloudOn.get()}
              onCloud={s.onCloud.get()}
              cloudPick={s.cloudPick}
              system={s.system}
              gpu={s.gpu.get()}
              deviceMsg={s.deviceMsg.get()}
              deviceErr={s.deviceErr.get()}
              caps={s.caps.get()}
              ampBlock={s.ampBlock.get()}
              ampErr={s.ampErr.get()}
              ampOn={s.ampOn.get()}
            />

            <AugmentSection
              bind:augment={s.augment}
              bind:augOpen={s.augOpen}
              bind:closeEdited={s.closeEdited}
              defaults={DEFAULT_AUGMENT}
              errors={s.errors.get()}
              closeEff={s.closeEff.get()}
              epochs={s.epochs.get()}
              dataset={s.dataset.get()}
              sampleImages={s.sampleImages}
            />
          </div>

          <SummaryRail s={s} />

          <StartDock
            startVisible={s.startVisible.get()}
            showWhy={s.showWhy.get()}
            why={s.why.get()}
            whyLine={(id, cls) => <WhyLine id={id} cls={cls} why={s.showWhy.get() ? s.why.get() : null} jump={s.jump} />}
            bind:sumOpen={s.sumOpen}
            shown={s.shown.get()}
            estReady={s.estReady.get()}
            estStale={s.estStale.get()}
            cloudCost={s.cloudCost.get()}
            memShown={s.memShown.get()}
            noFit={s.noFit.get()}
            gpuMem={s.gpuMem.get()}
            vram={s.vram.get()}
            short={s.short.get()}
            dataset={s.dataset.get()}
            arch={s.arch.get()}
            fix={s.fix.get()}
            fixLoading={s.fixLoading.get()}
            applyFix={s.applyFix}
            submitting={s.submitting.get()}
            valid={s.valid.get()}
            startLabel={s.startLabel.get()}
          />

          {/* Inside the form, so its Start button submits it. */}
          <SummarySheet s={s} />
        </form>
      )}
    </Page>
  );
}

css`
  .layout {
    display: grid;
    grid-template-columns: minmax(0, 1fr) 340px;
    gap: var(--space-4);
    align-items: start;
  }
  .form {
    display: grid;
    gap: var(--space-4);
    min-width: 0;
  }
  .layout :global([data-jumped]) {
    outline: 2px solid var(--focus);
    outline-offset: 4px;
    border-radius: var(--radius-md);
  }
  @media (max-width: 1279px) {
    /* Laptops keep the sticky rail beside the form, 40 px slimmer. */
    .layout {
      grid-template-columns: minmax(0, 1fr) 300px;
    }
  }
  @media (max-width: 1023px) {
    .layout {
      padding-bottom: 64px;
      grid-template-columns: minmax(0, 1fr);
    }
  }
`;
