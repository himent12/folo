/**
 * New run, section 3: epochs, batch and image size, where the run trains (this machine or a cloud connection, its
 * GPU and time limit), the device, mixed precision, and the advanced settings folded under them.
 */
import ChevronRight from '$icons/chevron-right';
import RotateCcw from '$icons/rotate-ccw';
import CircleAlert from '$icons/circle-alert';
import { bindable, computed, css, field, type Signal } from 'fw';
import type { Resource } from '$lib/api/resource';
import type { Arch, Capabilities, Dataset, Optimizer, Run, SystemInfo } from '$lib/api/types';
import { deviceName, pricePerHour, sentence, usd } from '$lib/format';
import { availabilityNote, maxCost, menuOrder, offerLine } from '$lib/cloud';
import { prefs } from '$lib/prefs';
import { href } from '$lib/router';
import Button from '$lib/ui/Button';
import Field from '$lib/ui/Field.tsx';
import Input from '$lib/ui/Input';
import Panel from '$lib/ui/Panel';
import Remedy from '$lib/ui/Remedy';
import Segmented from '$lib/ui/Segmented';
import Select from '$lib/ui/Select';
import Switch from '$lib/ui/Switch';
import Meta from '$lib/ui/Meta';
import DriverReport from '$lib/ui/DriverReport';
import { optimizerLabel } from '../run/resolved';
import { CloudPick, LOCAL } from './cloudPick';
import { AMP_SPEEDUP } from './estimate';
import type { AmpBlock } from './amp';
import type { FormErrors } from './jump';

export interface TrainingSectionProps {
  epochs?: number;
  batch?: string;
  imgsz?: string;
  imgszChosen?: boolean;
  device?: string;
  optimizer?: Optimizer;
  amp?: boolean;
  ampRefused?: { device: string; message: string } | null;
  e2e?: boolean;
  lr0?: number;
  seed?: number;
  advancedOpen?: boolean;
  arch: Arch;
  errors: FormErrors;
  was: (key: string) => string | undefined;
  oomRun: Run | null;
  dataset: Dataset | undefined;
  datasetImgsz: number | null;
  IMGSZ: string[];
  cloudOn: boolean;
  onCloud: boolean;
  cloudPick: CloudPick;
  system: Resource<SystemInfo>;
  gpu: SystemInfo['gpus'][number] | undefined;
  deviceMsg: string | null;
  deviceErr: { device: string; gone: boolean; detail: string } | null;
  caps: Capabilities | undefined;
  ampBlock: AmpBlock | null;
  ampErr: { device: string; message: string } | null;
  ampOn: boolean;
}

type Bound = 'epochs' | 'batch' | 'imgsz' | 'imgszChosen' | 'device' | 'optimizer' | 'amp' | 'ampRefused' | 'e2e' | 'lr0' | 'seed' | 'advancedOpen';

const BATCHES = ['8', '16', '32', '64'];

/** The standard choices plus `v` when it is not one of them (a copied run's batch 4, 1024 px), in order. */
const withValue = (list: string[], v: string) => (list.includes(v) ? list : [...list, v].sort((a, b) => Number(a) - Number(b)));

export default function TrainingSection(props: TrainingSectionProps) {
  // The parent binds every one of these (`$bindable` in the Svelte original); they are never undefined.
  const b = <K extends Bound>(k: K) => bindable(props, k) as Signal<Exclude<TrainingSectionProps[K], undefined>>;
  const epochs = b('epochs');
  const batch = b('batch');
  const imgsz = b('imgsz');
  const imgszChosen = b('imgszChosen');
  const device = b('device');
  const optimizer = b('optimizer');
  const amp = b('amp');
  const ampRefused = b('ampRefused');
  const e2e = b('e2e');
  const lr0 = b('lr0');
  const seed = b('seed');
  const advancedOpen = b('advancedOpen');
  const cloudPick = props.cloudPick;

  /** What the switch does, with the speed measured for this size (docs/BENCHMARKS.md), where there is one. */
  const ampHint = computed(() => {
    if (props.onCloud) return 'Checked on the instance when it boots.';
    const x = AMP_SPEEDUP[props.arch];
    return x ? `f16 on tensor cores: about ${x.toFixed(1)}× faster, 40% less memory.` : 'f16 on tensor cores: faster, about 40% less memory.';
  });

  /** The way out of a device the server cannot use: Auto, then back to the field. */
  function useAuto() {
    device.set('auto');
    document.getElementById('device')?.focus();
  }

  // Run on's offers (null while they load, or for this machine), the chosen one, and the most the run can cost.
  const offers = () => cloudPick.offerList;
  const chosen = () => cloudPick.offer;
  const most = () => {
    const c = chosen();
    return c && cloudPick.hours !== null ? maxCost(c.price_per_hour, cloudPick.hours) : null;
  };

  return (
    <Panel title="3 · Training" id="sec-train">
      <div class="fields">
        <Field label="Epochs" for="epochs" error={props.errors.epochs}>
          <Input id="epochs" type="number" min="1" max="10000" step="1" bind:value={epochs} invalid={!!props.errors.epochs} suffix="epochs" />
        </Field>
        <Field label="Batch size" for="batch" hint="Images per batch. Lower it if GPU memory runs out.">
          {/* A retry's change sits beside the new value: "16 · was 32". */}
          <span class="with-was">
            <Segmented id="batch" label="Batch size" bind:value={batch} options={withValue(BATCHES, batch.get()).map((v) => ({ value: v, label: v }))} />
            {props.was('batch') ? <span class="was">was {props.oomRun?.config.batch}</span> : null}
          </span>
        </Field>
        <Field
          label="Image size"
          for="imgsz"
          hint={
            !imgszChosen.get() && props.datasetImgsz !== null && props.datasetImgsz < 640
              ? `Longer side in pixels. ${props.dataset?.name}'s images are all ${props.datasetImgsz} px.`
              : 'Longer side in pixels, letterboxed.'
          }
        >
          <span class="with-was">
            <Segmented
              id="imgsz"
              label="Image size"
              bind:value={imgsz}
              onchange={() => imgszChosen.set(true)}
              options={withValue(props.IMGSZ, imgsz.get()).map((v) => ({ value: v, label: v }))}
            />
            {props.was('imgsz') ? <span class="was">was {props.oomRun?.config.imgsz}</span> : null}
          </span>
        </Field>
        {props.cloudOn ? (
          <>
            {/* Where the run trains (docs/CLOUD.md): this machine, or a cloud connection; the GPU and the time limit
                follow it. Not a Field: its hint holds a link, and a refusal at Start is said here. */}
            <div class="field runon" class:invalid={!!props.errors.runon}>
              <div class="dev-top">
                <label for="runon">Run on</label>
              </div>
              <Select
                id="runon"
                bind:value={field(cloudPick, 'target')}
                onchange={(v) => cloudPick.choose(v)}
                invalid={!!props.errors.runon}
                describedby={props.errors.runon ? 'runon-error' : 'runon-hint'}
                options={[
                  { value: LOCAL, label: 'This machine' },
                  // A connection whose check failed cannot run anything: listed, off, its error the reason.
                  ...(cloudPick.connections.data ?? []).map((c) => ({
                    value: c.id,
                    label: c.status === 'error' ? `${c.name} · ${c.error ?? 'check failed'}` : c.account ? `${c.name} · ${c.account}` : c.name,
                    short: c.name,
                    disabled: c.status === 'error',
                    title: c.error ?? undefined,
                  })),
                ]}
              />
              {props.errors.runon ? (
                <p class="sw-err" id="runon-error" role="alert">
                  <CircleAlert size={14} strokeWidth={1.75} aria-hidden="true" />
                  <span>{props.errors.runon}</span>
                </p>
              ) : cloudPick.lost ? (
                <p class="dev-hint" id="runon-hint">
                  {cloudPick.lost}
                </p>
              ) : cloudPick.connection ? (
                <p class="dev-hint num" id="runon-hint">
                  <Meta>
                    {cloudPick.connection.name !== cloudPick.provider ? <span>{cloudPick.provider}</span> : null}
                    {cloudPick.connection.account ? <span>{cloudPick.connection.account}</span> : null}
                    {cloudPick.connection.balance_usd !== null ? <span>Balance {usd(cloudPick.connection.balance_usd)}</span> : null}
                  </Meta>
                </p>
              ) : cloudPick.connections.data && !cloudPick.connections.data.length ? (
                <p class="dev-hint" id="runon-hint">
                  Connect a cloud in{' '}
                  <a class="hint-link" href={href('/settings?section=cloud')}>
                    Settings
                  </a>{' '}
                  to train on a rented GPU.
                </p>
              ) : (
                <p class="dev-hint" id="runon-hint">
                  This server’s own GPU or CPU.
                </p>
              )}
            </div>
            {props.onCloud ? (
              <>
                {/* The connection's offers, each with its price; the cheapest one with the memory these settings need is
                    preselected. Loading, the select waits in place; an error is said in place with Retry. */}
                <div class="field offer" class:invalid={!!props.errors.offer}>
                  <div class="dev-top">
                    <label for="offer">GPU</label>
                  </div>
                  {cloudPick.offers.error && !offers() ? (
                    <div class="dev-line">
                      <p class="sw-err" id="offer-error" role="alert">
                        <CircleAlert size={14} strokeWidth={1.75} aria-hidden="true" />
                        <span>
                          {sentence(cloudPick.offers.error instanceof Error ? cloudPick.offers.error.message : String(cloudPick.offers.error), true)}
                        </span>
                      </p>
                      <Button size="sm" icon={RotateCcw} onclick={() => cloudPick.retryOffers()}>
                        Retry
                      </Button>
                    </div>
                  ) : (
                    <>
                      <Select
                        id="offer"
                        value={cloudPick.offerId ?? ''}
                        onchange={(v) => ((cloudPick.offerId = v), (cloudPick.offerChosen = true))}
                        disabled={!offers()}
                        invalid={!!props.errors.offer}
                        describedby={props.errors.offer ? 'offer-error' : 'offer-hint'}
                        options={
                          offers()
                            ? menuOrder(offers()!).map((o) => {
                                const note = availabilityNote(o.availability);
                                // Closed, the GPU and its memory; the price and the rest are the line under it.
                                return {
                                  value: o.id,
                                  label: [offerLine(o, prefs.units), pricePerHour(o.price_per_hour), note].filter(Boolean).join(' · '),
                                  short: offerLine({ ...o, detail: null }, prefs.units),
                                  disabled: o.availability === 'none',
                                };
                              })
                            : [{ value: cloudPick.offerId ?? '', label: 'Loading offers…' }]
                        }
                      />
                      {props.errors.offer ? (
                        <p class="sw-err" id="offer-error" role="alert">
                          <CircleAlert size={14} strokeWidth={1.75} aria-hidden="true" />
                          <span>{props.errors.offer}</span>
                        </p>
                      ) : chosen() ? (
                        <p class="dev-hint num" id="offer-hint">
                          <Meta>
                            <span class="price">{chosen()!.price_per_hour === null ? 'Not billed' : pricePerHour(chosen()!.price_per_hour)}</span>
                            {chosen()!.detail ? <span>{chosen()!.detail}</span> : null}
                            {chosen()!.region ? <span>{chosen()!.region}</span> : null}
                            {availabilityNote(chosen()!.availability) ? <span>{availabilityNote(chosen()!.availability)}</span> : null}
                          </Meta>
                        </p>
                      ) : (
                        <p class="dev-hint" id="offer-hint">
                          {offers() ? 'Nothing is available on this connection now.' : 'Asking the provider what it can run on.'}
                        </p>
                      )}
                    </>
                  )}
                </div>
                <Field
                  label="Stop after"
                  for="hours"
                  error={props.errors.hours}
                  hint={
                    cloudPick.provisions
                      ? `The run is stopped and the instance terminated after this long at the latest.${most() !== null ? ` It cannot cost more than ${usd(most()!)}.` : ''}`
                      : 'The run is stopped after this long at the latest.'
                  }
                >
                  <Input
                    id="hours"
                    type="number"
                    min="0.25"
                    max="720"
                    step="0.25"
                    bind:value={field(cloudPick, 'hours')}
                    oninput={() => (cloudPick.hoursEdited = true)}
                    invalid={!!props.errors.hours}
                    suffix="hours"
                  />
                </Field>
              </>
            ) : null}
          </>
        ) : null}
        {props.onCloud ? (
          // The worker picks its GPU (the config sends `auto`): a fact in the device's place, not a control.
          <div class="field dev">
            <div class="dev-top">
              <span class="dev-label">Device</span>
            </div>
            <p class="dev-fact">{cloudPick.provisions ? 'The instance’s GPU' : 'The server’s GPU'}</p>
            <p class="dev-hint">The worker picks it when it trains.</p>
          </div>
        ) : (
          // Not a Field: a device the server cannot use gets a line with a way out and folded details.
          <div class="field dev" class:invalid={!!props.deviceMsg}>
            <div class="dev-top">
              <label for="device">Device</label>
              {props.was('device') ? <span class="was">was {props.oomRun?.config.device}</span> : null}
            </div>
            <Select
              id="device"
              bind:value={device}
              invalid={!!props.deviceMsg}
              describedby={props.deviceMsg ? 'device-error' : 'device-hint'}
              options={[
                { value: 'auto', label: `Auto${props.gpu ? ` · ${deviceName(props.gpu.name)}` : ''}` },
                ...(props.system.data?.gpus ?? []).map((g) => ({ value: `gpu:${g.ordinal}`, label: `gpu:${g.ordinal} · ${deviceName(g.name)}` })),
                // A device the run asked for that the machine no longer lists stays selectable, so the form keeps it.
                ...(device.get().startsWith('gpu:') && props.system.data && !props.system.data.gpus.some((g) => `gpu:${g.ordinal}` === device.get())
                  ? [{ value: device.get(), label: `${device.get()} · not found` }]
                  : []),
                { value: 'cpu', label: `CPU${props.system.data ? ` · ${props.system.data.cpu.threads} threads` : ''}` },
              ]}
            />
            {props.deviceMsg && props.deviceErr ? (
              <div class="dev-err">
                <div class="dev-line">
                  <p id="device-error" role="alert">
                    {props.deviceMsg}
                  </p>
                  <Button size="sm" onclick={useAuto}>
                    Use Auto
                  </Button>
                </div>
                {props.deviceErr.detail ? (
                  <details class="dev-detail">
                    <summary>
                      <ChevronRight size={12} aria-hidden="true" />
                      Technical details
                    </summary>
                    <p>{props.deviceErr.detail}</p>
                  </details>
                ) : null}
              </div>
            ) : (
              <p class="dev-hint" id="device-hint">
                Auto prefers the GPU over the CPU.
              </p>
            )}
          </div>
        )}
        {props.caps || props.onCloud ? (
          // A field on the same grid: the label's line carries the switch at the field's right edge, the hint (or, where
          // the server cannot train in mixed precision, why not and what to do) under it. Only the control is disabled;
          // the label keeps the field labels' colour. Hidden when the server does not say.
          <div class="field sw-field amp" class:off={!!props.ampBlock}>
            <Switch
              id="amp"
              label="Mixed precision"
              checked={props.ampOn}
              onchange={(v) => (amp.set(v), ampRefused.set(null))}
              disabled={!!props.ampBlock}
              describedby={props.ampErr && props.ampOn ? 'amp-error' : props.ampBlock ? 'amp-why' : 'amp-hint'}
            />
            {props.ampErr && props.ampOn ? (
              // A refusal, where the setting is (never a toast), while it still applies: the switch is on and Start
              // waits. One short cause and the way out (the switch beside it: off, Start trains in full precision), then
              // the server's whole message folded and copyable. Once the switch is off, the refusal is answered and
              // steps down to the quiet note below (DESIGN_DIRECTION §2.11).
              <>
                <p class="sw-err" id="amp-error" role="alert">
                  <CircleAlert size={14} strokeWidth={1.75} aria-hidden="true" />
                  <span>Mixed precision is unavailable: the server cannot use it now. Turn it off to train in full precision.</span>
                </p>
                <div class="sw-why">
                  <DriverReport report={props.ampErr.message} />
                </div>
              </>
            ) : props.ampErr && !props.ampBlock ? (
              <p class="sw-hint" id="amp-hint">
                The server refused mixed precision; this run trains in full precision.
              </p>
            ) : props.ampBlock ? (
              // After a refusal the capabilities were read again: the switch is off and held off, and the field says why
              // and how to turn it back on, quietly (no alert: nothing is wrong with the run now).
              <div class="sw-why">
                <Remedy
                  compact
                  id="amp-why"
                  cause={props.ampBlock.why}
                  shell={props.ampBlock.shell}
                  command={props.ampBlock.command}
                  report={props.ampErr?.message ?? props.ampBlock.raw}
                />
              </div>
            ) : (
              <p class="sw-hint" id="amp-hint">
                {ampHint.get()}
              </p>
            )}
          </div>
        ) : null}
      </div>
      <details class="advanced" open={advancedOpen.get()} ontoggle={(e) => advancedOpen.set((e.currentTarget as HTMLDetailsElement).open)}>
        <summary>
          <ChevronRight size={14} /> Advanced <span class="faint">optimizer, learning rate, seed, head</span>
        </summary>
        <div class="fields">
          <Field label="Optimizer" for="opt">
            <Select id="opt" bind:value={optimizer} options={(['auto', 'sgd', 'adamw', 'musgd'] as const).map((value) => ({ value, label: optimizerLabel(value) }))} />
          </Field>
          <Field label="Initial learning rate" for="lr0" error={props.errors.lr0} hint="Warmup 3 epochs, then linear to 1%.">
            <Input id="lr0" type="number" step="0.001" min="0" bind:value={lr0} invalid={!!props.errors.lr0} />
          </Field>
          <Field label="Seed" for="seed" hint="Same seed, same run." error={props.errors.seed}>
            <Input id="seed" type="number" min="0" step="1" bind:value={seed} invalid={!!props.errors.seed} />
          </Field>
          {/* TrainConfig.head `o2o` (docs/MODEL.md §6.6); the cost measured on the shapes gate (docs/BENCHMARKS.md). */}
          <div class="field sw-field">
            <Switch id="head" label="End-to-end (no NMS)" checked={e2e.get()} onchange={(v) => e2e.set(v)} describedby="head-hint" />
            <p class="sw-hint" id="head-hint">
              No NMS step at inference; about 0.015 lower mAP50-95 than NMS on the same model.
            </p>
          </div>
        </div>
      </details>
    </Panel>
  );
}

css`
  /* Two fields a row on one grid: a label row, a control row of one height, then help. Each field spans the three rows
     (subgrid), so labels, controls and help line up across the pair whatever each one holds. */
  .fields {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    grid-auto-rows: auto minmax(var(--control-md), auto) auto;
    gap: 0 var(--space-5);
  }
  /* Every field of a row starts on the row's shared tracks at their top (align-items: start): a two-line hint under
     Batch and none under Epochs never shifts one field's label or control against its neighbour's. */
  .fields > :global(div.field) {
    grid-row: span 3;
    display: grid;
    grid-template-rows: subgrid;
    row-gap: 6px;
    align-items: start;
    padding-bottom: var(--space-4);
  }
  .fields > :global(div.field > :first-child) {
    align-self: end;
  }
  .fields > :global(div.field > p),
  .fields > :global(div.field > .dev-err) {
    align-self: start;
  }
  /* The Device field: a Field's shape, with room for its error's way out. */
  .dev-top {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: var(--space-2);
  }
  .dev-top label,
  .dev-label {
    color: var(--text-secondary);
    font-weight: var(--weight-medium);
  }
  /* A cloud run's device: a fact on the control's line, at its height, so the row keeps its neighbour's geometry. */
  .dev-fact {
    display: flex;
    align-items: center;
    min-height: var(--control-md);
    color: var(--text-primary);
  }
  /* The price of the chosen GPU, read first on its line. */
  .dev-hint .price {
    color: var(--text-secondary);
  }
  /* "Settings" in Run on's hint: an ink link to Settings › Cloud. */
  .hint-link {
    color: var(--text-primary);
    text-decoration: underline;
    text-decoration-color: var(--line-strong);
    text-underline-offset: 3px;
  }
  .hint-link:hover {
    text-decoration-color: var(--text-secondary);
  }
  .dev-top .was {
    font-size: var(--type-sm-size);
  }
  .dev-hint {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-tertiary);
  }
  .dev-err {
    display: grid;
    gap: var(--space-1);
  }
  .dev-line {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: var(--space-3);
  }
  .dev-line p {
    padding-top: var(--space-1);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--danger-text);
  }
  .dev-line :global(.btn) {
    flex-shrink: 0;
  }
  .dev-detail summary {
    display: inline-flex;
    align-items: center;
    gap: var(--space-1);
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
    cursor: pointer;
    list-style: none;
    border-radius: var(--radius-sm);
  }
  .dev-detail summary::-webkit-details-marker {
    display: none;
  }
  .dev-detail summary:hover {
    color: var(--text-primary);
  }
  .dev-detail summary :global(svg) {
    transition: transform var(--dur-fast) var(--ease-standard);
  }
  .dev-detail[open] summary :global(svg) {
    transform: rotate(90deg);
  }
  .dev-detail p {
    margin-top: var(--space-1);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-tertiary);
    overflow-wrap: anywhere;
  }
  @media (pointer: coarse) {
    .dev-detail summary {
      min-height: 44px;
    }
  }
  /* A switch field (Mixed precision, End-to-end): the Switch is the label's line, its track at the field's right edge,
     and the label reads as every field label (text.secondary, medium), also while only the control is disabled. */
  .fields > .sw-field :global(.switch .label) {
    color: var(--text-secondary);
    font-weight: var(--weight-medium);
  }
  /* Mixed precision is the fifth field of a two-column grid: a settings row across the full width (label and hint on
     the left, the track at the section's right edge), one row of its own height, never a lone half-row beside a hole.
     Where the server cannot train in mixed precision, its remedy (ui/Remedy: a sentence, the command on one line) has
     the same full width, so the command fits whole. */
  .fields > .sw-field.amp {
    grid-column: 1 / -1;
    grid-row: auto;
    grid-template-rows: none;
  }
  .fields > .sw-field > .sw-why {
    align-self: start;
    min-width: 0;
  }
  /* Start's refusal at the field (\`501 not_supported\`): danger ink with its icon, never red text alone (§2.11). */
  .sw-err {
    display: flex;
    gap: var(--space-1);
    align-items: flex-start;
    max-inline-size: var(--measure);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--danger-text);
    text-wrap: pretty;
    overflow-wrap: anywhere;
  }
  /* As a Field's error: the icon on the first line's centre (16 px line, 14 px icon). */
  .sw-err :global(svg) {
    flex: none;
    margin-top: 1px;
  }
  .sw-hint {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-tertiary);
    text-wrap: pretty;
  }
  @media (pointer: coarse) {
    .sw-field :global(.switch) {
      min-height: 44px;
    }
  }
  /* A retry's change beside the control it changed. */
  .with-was {
    display: flex;
    align-items: center;
    gap: var(--space-3);
    min-width: 0;
  }
  .with-was .was {
    font-size: var(--type-sm-size);
    white-space: nowrap;
  }
  /* A field the retry changed: its earlier value beside the label. */
  .was {
    color: var(--text-secondary);
    font-variant-numeric: tabular-nums;
  }
  .advanced {
    margin-top: var(--space-4);
    border-top: 1px solid var(--line-subtle);
    padding-top: var(--space-3);
  }
  .advanced summary {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    height: var(--control-md);
    color: var(--text-secondary);
    font-weight: var(--weight-medium);
    cursor: pointer;
    list-style: none;
    border-radius: var(--radius-sm);
  }
  .advanced summary::-webkit-details-marker {
    display: none;
  }
  .advanced summary :global(svg) {
    transition: transform var(--dur-fast) var(--ease-standard);
  }
  .advanced[open] summary :global(svg) {
    transform: rotate(90deg);
  }
  .advanced summary:hover {
    color: var(--text-primary);
  }
  .advanced summary .faint {
    font-weight: var(--weight-regular);
  }
  .advanced[open] .fields {
    margin-top: var(--space-3);
  }
  @media (pointer: coarse) {
    .advanced summary {
      height: 44px;
    }
  }
  @media (max-width: 767px) {
    .fields {
      grid-template-columns: minmax(0, 1fr);
    }
    /* One column: no neighbour to line up with, so a switch field is one row of its own height. */
    .fields > .sw-field {
      grid-row: auto;
      grid-template-rows: none;
    }
  }
`;
