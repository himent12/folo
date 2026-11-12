/**
 * The model to run (its score and what it is on under the select) and where it runs: on the title row from 1024 px
 * up, or (`toolbar`) a toolbar row of their own under the title below that.
 */
import ArrowUpRight from '$icons/arrow-up-right';
import { bindable, css } from 'fw';
import type { Model } from '$lib/api/types';
import type { Backend } from '$lib/inference/browser';
import type { ModelChoice } from '$lib/scoreSummary';
import { href } from '$lib/router';
import Button from '$lib/ui/Button';
import Segmented from '$lib/ui/Segmented';
import Select from '$lib/ui/Select';

export interface ModelPickersProps {
  toolbar?: boolean;
  ranked: Model[];
  modelId: string | null;
  choices: ModelChoice<Model>[];
  chooseModel: (id: string) => void;
  model: Model | null;
  pickedMeta: string | null;
  runOn?: 'server' | Backend;
  webgpu: boolean;
  browserOff: string | null;
  runOnNote: string | null;
}

export default function ModelPickers(props: ModelPickersProps) {
  const runOn = bindable(props, 'runOn', 'server');

  // The model and where it runs: on the title row from 1024 px up, a toolbar row under the title below that.
  const pickers = () => (
    <>
      {props.ranked.length ? (
        <>
          {/* Closed, the select shows the model's name alone; its score and everything the score is on wrap in the meta
              line under it (never cut at the select's edge), which is also its description. The open list keeps the
              groups and each option's context. */}
          <div class="model-pick">
            <span class="pick-label">Model</span>
            <Select
              label="Model"
              value={props.modelId ?? ''}
              options={props.choices.map((c) => ({ value: c.model.id, label: c.label, group: c.group, short: c.short }))}
              onchange={props.chooseModel}
              describedby={props.model ? 'model-meta' : undefined}
            />
            {props.model ? <Button icon={ArrowUpRight} href={href(`/models/${props.model.id}`)} aria-label={`Open ${props.model.name}`} /> : null}
            {/* Always a line ("Not evaluated" for a model without a score), so switching models never moves the page. */}
            {props.model ? (
              <p class="pick-meta num" id="model-meta">
                {props.pickedMeta ?? 'Not evaluated'}
              </p>
            ) : null}
          </div>
          {/* A build of Studio without the in-browser runtime runs on the server only: no choice to show. */}
          {props.browserOff ? null : (
            <div class="runon" role="group" aria-label="Run on" aria-describedby={props.runOnNote ? 'runon-note' : undefined}>
              <span class="pick-label" aria-hidden="true">
                Run on
              </span>
              {/* A choice that is off says why in its own tip, and the image panel's status line says it once ("No WebGPU
                  in this browser"), so the toolbar stays one row at every tier: never a note wrapping under the group
                  (round 24's tablet). The group's description is the same words, for assistive technology. */}
              <span class="runon-seg">
                <Segmented
                  label="Run the model on"
                  size="sm"
                  bind:value={runOn}
                  options={[
                    { value: 'server', label: 'Server' },
                    { value: 'webgpu', label: 'WebGPU', disabled: !props.webgpu, tip: 'No WebGPU in this browser' },
                    { value: 'cpu', label: 'CPU' },
                  ]}
                />
                {props.runOnNote ? (
                  <span class="sr-only" id="runon-note">
                    {props.runOnNote}
                  </span>
                ) : null}
              </span>
            </div>
          )}
        </>
      ) : null}
    </>
  );

  return <>{props.toolbar ? <div class="pg-tools">{pickers()}</div> : pickers()}</>;
}

css`
  /* Label, select and open button on one row; the score's meta line under the select, wrapping, never cut. */
  .model-pick {
    display: grid;
    grid-template-columns: auto minmax(0, 1fr) auto;
    align-items: center;
    gap: var(--space-1) var(--space-2);
    width: 400px;
  }
  .pick-meta {
    grid-column: 2 / -1;
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-tertiary);
    text-wrap: pretty;
  }
  /* Below 1024 px: one full-width toolbar row under the title; the model select takes what Run on leaves (a choice
     that is off says why in its tip and on the image panel's status line, never in a note that wraps the row). */
  .pg-tools {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: var(--space-2) var(--space-6);
  }
  .pg-tools .model-pick {
    flex: 1 1 320px;
    width: auto;
    min-width: 0;
  }
  .pg-tools .runon {
    flex-wrap: wrap;
    row-gap: var(--space-1);
  }
  .pg-tools .runon-seg {
    display: contents;
  }
  .model-pick :global(.select) {
    min-width: 0;
  }
  .runon {
    display: flex;
    align-items: center;
    gap: var(--space-2);
  }
  /* On the title row, Run on sits on the model select's row, not centred on the select and its meta line together. */
  .model-pick + .runon {
    align-self: flex-start;
    min-height: var(--control-md);
  }
  .pick-label {
    color: var(--text-tertiary);
    font-size: var(--type-sm-size);
  }
  .runon-seg {
    display: flex;
  }
  @media (max-width: 1023px) {
    /* The toolbar row takes the title row's place for the controls; no extra gap under the title. */
    .pg-tools {
      margin-top: calc(-1 * var(--space-1));
    }
  }
  @media (max-width: 767px) {
    /* Phones: two rows, the model, then Run on with its control on the right; the note a row of its own under them. */
    .pg-tools {
      flex-direction: column;
      align-items: stretch;
      gap: var(--space-2);
      margin-top: 0;
    }
    .pg-tools .model-pick {
      flex: none;
    }
    .pg-tools .runon {
      justify-content: space-between;
    }
  }
`;
