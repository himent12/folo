// Sticky Summary rail and narrow-screen Summary confirmation sheet for New run (`../RunNew.tsx`).
import Play from '$icons/play';
import { css } from 'fw';
import { useLabel } from '$lib/oom';
import Button from '$lib/ui/Button';
import Dialog from '$lib/ui/Dialog';
import Field from '$lib/ui/Field.tsx';
import Input from '$lib/ui/Input';
import Panel from '$lib/ui/Panel';
import { tooltip } from '$lib/ui/tooltip';
import CliCommand from './CliCommand';
import EstimateFigures from './EstimateFigures';
import { nameHint, runName } from './name';
import { aheadLine } from './queue';
import ReviewList from './ReviewList';
import { START_KEYS } from './startKeys';
import type { RunNewState } from './state';
import { roughBytes } from './summary';
import WhyLine from './WhyLine';

export interface SummaryProps {
  s: RunNewState;
}

/** Shared estimate figures block wired to `RunNewState`. */
function FiguresBlock(props: { s: RunNewState; where: string }) {
  const s = props.s;
  return (
    <EstimateFigures
      where={props.where}
      blocked={s.noFigures.get()}
      estReady={s.estReady.get()}
      sched={s.sched.get()}
      batchN={s.batchN.get()}
      est={s.est.get()}
      bind:stepsOpen={s.stepsOpen}
      shown={s.shown.get()}
      estStale={s.estStale.get()}
      estimating={s.estimating.get()}
      memShown={s.memShown.get()}
      vram={s.vram.get()}
      onCloud={s.onCloud.get()}
      cloudPick={s.cloudPick}
      cloudCost={s.cloudCost.get()}
      gpuMem={s.gpuMem.get()}
      scale={s.scale.get()}
      onCpu={s.onCpu.get()}
      vramLevel={s.vramLevel.get()}
      vramPct={s.vramPct.get()}
      noFit={s.noFit.get()}
      fix={s.fix.get()}
      fixLoading={s.fixLoading.get()}
      short={s.short.get()}
      bind:shortWhy={s.shortWhy}
      bind:epochs={s.epochs}
    />
  );
}

/** Desktop/laptop sticky Summary rail (full-width two-column summary below 1024 px). */
export function SummaryRail(props: SummaryProps) {
  const s = props.s;
  return (
    <aside class="summary">
      <Panel title="Summary">
        {/* Two groups: what the run is (name, settings) and what it costs (estimates, Start). A rail stacks them; a
            full-width summary (below 1024 px) sets them side by side. */}
        <div class="sum">
          <div class="sum-col">
            <Field label="Run name" for="name" error={s.errors.get().name} hint={nameHint(s.name.get(), s.taken.get(), s.nameEdited.get()) ?? undefined}>
              <Input
                id="name"
                bind:value={s.name}
                oninput={() => s.nameEdited.set(true)}
                onblur={() => {
                  const n = runName(s.name.get());
                  if (n && n !== s.name.get()) s.name.set(n);
                }}
                invalid={!!s.errors.get().name}
                spellcheck="false"
                autocomplete="off"
              />
            </Field>
            <ReviewList withName={false} name={runName(s.name.get())} errors={s.errors.get()} review={s.review.get()} jump={s.jump} />
          </div>
          <div class="sum-col">
            <FiguresBlock s={s} where="rail" />
            {s.ahead.get() ? <p class="ahead">{aheadLine(s.ahead.get()!)}</p> : null}
            <div class="start" ref={s.startEl}>
              {s.noFit.get() && (s.fix.get() || s.fixLoading.get()) ? (
                <Button variant="primary" size="lg" block loading={s.fixLoading.get()} onclick={() => s.fix.get() && s.applyFix(s.fix.get()!)}>
                  {s.fix.get() ? `${useLabel(s.fix.get()!)}${s.fix.get()!.vram !== undefined ? ` · ${roughBytes(s.fix.get()!.vram!)}` : ''}` : 'Finding settings that fit'}
                </Button>
              ) : null}
              <span class="start-tip" use:tooltip={{ text: s.startLabel.get(), keys: START_KEYS }}>
                <Button
                  type="submit"
                  variant={s.noFit.get() ? 'secondary' : 'primary'}
                  size="lg"
                  icon={Play}
                  block
                  loading={s.submitting.get()}
                  disabled={!s.valid.get()}
                  aria-describedby={[s.showWhy.get() ? 'start-why' : '', s.noFit.get() ? 'rail-nofit' : ''].filter(Boolean).join(' ') || undefined}
                >
                  {s.startLabel.get()}
                </Button>
              </span>
              <WhyLine id="start-why" why={s.showWhy.get() ? s.why.get() : null} jump={s.jump} />
            </div>
            <CliCommand
              cliParts={s.cli.parts.get()}
              cli={s.cli.line.get()}
              cliPlaceholder={s.cli.placeholder.get()}
              pending={!s.prefilled.get() ? 'Loading run settings…' : s.pendingWeights.get() ? 'Waiting for linked weights to load…' : null}
              seedMissing={!!s.errors.get().seed}
            />
          </div>
        </div>
      </Panel>
    </aside>
  );
}

/** Narrow-screen Summary dialog inside `<form>` so its Start button submits the form cleanly. */
export function SummarySheet(props: SummaryProps) {
  const s = props.s;
  return (
    <Dialog
      bind:open={s.sumOpen}
      title="Summary"
      size="sm"
      footer={() => (
        <>
          <Button variant="ghost" onclick={() => s.sumOpen.set(false)}>
            Keep editing
          </Button>
          {s.noFit.get() && (s.fix.get() || s.fixLoading.get()) ? (
            <>
              <Button
                type="submit"
                icon={Play}
                loading={s.submitting.get()}
                disabled={!s.valid.get()}
                aria-describedby={[s.showWhy.get() ? 'sheet-why' : '', 'sheet-nofit'].filter(Boolean).join(' ')}
              >
                {s.startLabel.get()}
              </Button>
              <Button variant="primary" loading={s.fixLoading.get()} onclick={() => s.fix.get() && s.applyFix(s.fix.get()!)}>
                {s.fix.get() ? useLabel(s.fix.get()!) : 'Finding a fit'}
              </Button>
            </>
          ) : (
            <Button
              type="submit"
              variant="primary"
              icon={Play}
              loading={s.submitting.get()}
              disabled={!s.valid.get()}
              aria-describedby={s.showWhy.get() ? 'sheet-why' : undefined}
            >
              {s.startLabel.get()}
            </Button>
          )}
        </>
      )}
    >
      <div class="sum">
        <ReviewList withName name={runName(s.name.get())} errors={s.errors.get()} review={s.review.get()} jump={s.jump} />
        <FiguresBlock s={s} where="sheet" />
        {s.ahead.get() ? <p class="ahead">{aheadLine(s.ahead.get()!)}</p> : null}
        <WhyLine id="sheet-why" why={s.showWhy.get() ? s.why.get() : null} jump={s.jump} />
      </div>
    </Dialog>
  );
}

css`
  .summary {
    position: sticky;
    top: calc(var(--size-topbar) + var(--space-4));
  }
  .sum,
  .sum-col {
    display: grid;
    align-content: start;
    gap: var(--space-4);
    min-width: 0;
  }
  .start-tip {
    display: block;
  }
  /* What the run waits for: a plain line in the Summary's secondary ink, not a warning (waiting is normal). */
  .ahead {
    max-inline-size: var(--measure);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
  }
  @media (max-width: 1023px) {
    .summary {
      position: static;
    }
    /* Full width under the form: the settings on the left, the estimates and Start on the right. */
    .summary .sum {
      grid-template-columns: repeat(2, minmax(0, 1fr));
      column-gap: var(--space-6);
    }
  }
  @media (max-width: 767px) {
    .summary .sum {
      grid-template-columns: minmax(0, 1fr);
    }
  }
`;
