<script lang="ts">
  import Check from '@lucide/svelte/icons/check';
  import ChevronRight from '@lucide/svelte/icons/chevron-right';
  import ChevronUp from '@lucide/svelte/icons/chevron-up';
  import Copy from '@lucide/svelte/icons/copy';
  import Images from '@lucide/svelte/icons/images';
  import Play from '@lucide/svelte/icons/play';
  import Shuffle from '@lucide/svelte/icons/shuffle';
  import RotateCcw from '@lucide/svelte/icons/rotate-ccw';
  import TriangleAlert from '@lucide/svelte/icons/triangle-alert';
  import { tick, untrack } from 'svelte';
  import { hub } from '$lib/api/index.svelte';
  import { Resource } from '$lib/api/resource.svelte';
  import type { Arch, Augment, DatasetImage, Device, Optimizer, TrainConfig } from '$lib/api/types';
  import ClassStrip from '$lib/detect/ClassStrip.svelte';
  import { bytes, compact, duration, num } from '$lib/format';
  import { prefs } from '$lib/prefs.svelte';
  import { href, router } from '$lib/router.svelte';
  import { setCrumbs } from '$lib/shell/crumbs.svelte';
  import Page from '$lib/shell/Page.svelte';
  import Button from '$lib/ui/Button.svelte';
  import Dialog from '$lib/ui/Dialog.svelte';
  import EmptyState from '$lib/ui/EmptyState.svelte';
  import ErrorState from '$lib/ui/ErrorState.svelte';
  import Field from '$lib/ui/Field.svelte';
  import Input from '$lib/ui/Input.svelte';
  import Panel from '$lib/ui/Panel.svelte';
  import Segmented from '$lib/ui/Segmented.svelte';
  import Select from '$lib/ui/Select.svelte';
  import Skeleton from '$lib/ui/Skeleton.svelte';
  import Slider from '$lib/ui/Slider.svelte';
  import { toast } from '$lib/ui/toast.svelte';
  import AugPreview, { type AugKey } from './runnew/AugPreview.svelte';
  import { arches, estimate } from './runnew/estimate';

  setCrumbs({ label: 'Runs', path: '/runs' }, { label: 'New run' });

  const q = router.route.query;
  const defaults: Augment = { mosaic: 1, mixup: 0, hsv: [0.015, 0.7, 0.4], fliplr: 0.5, scale: 0.5, translate: 0.1, close_mosaic: 10 };

  let datasetId = $state<string>(q.get('dataset') ?? '');
  // The dock repeats the summary's Start button for narrow screens; it steps aside while that button is on screen.
  let startEl = $state<HTMLElement>();
  let startVisible = $state(false);
  $effect(() => {
    const el = startEl;
    if (!el || typeof IntersectionObserver === 'undefined') return;
    const io = new IntersectionObserver(([e]) => (startVisible = !!e?.isIntersecting));
    io.observe(el);
    return () => io.disconnect();
  });
  let arch = $state<Arch>('folo-s');
  let epochs = $state(100);
  let batch = $state('16');
  const IMGSZ = ['320', '480', '640', '960', '1280'];
  // ?imgsz= (e.g. from a dataset's "Train at 960 px" health finding) preselects a supported size.
  let imgsz = $state(IMGSZ.includes(q.get('imgsz') ?? '') ? q.get('imgsz')! : '640');
  let device = $state<string>('auto');
  let optimizer = $state<Optimizer>('auto');
  let lr0 = $state(0.01);
  let seed = $state(0);
  let weights = $state<string>('none');
  let name = $state('');
  let nameEdited = $state(false);
  let augment = $state<Augment>(structuredClone(defaults));
  let focus = $state<AugKey | null>(null);
  let shuffle = $state(1);
  let submitting = $state(false);
  let advancedOpen = $state(false);
  /** Up to 1199 px the augmentation settings fold behind a one-line summary (wide screens always show them). */
  let augOpen = $state(false);
  /** Up to 1199 px the dock opens the summary as a sheet, so the run can be reviewed without scrolling down. */
  let sumOpen = $state(false);

  const datasets = new Resource(async (s) => {
    const list = await hub.api.datasets(s);
    return Promise.all(list.map((d) => hub.api.dataset(d.id, s).catch(() => d)));
  });
  const system = new Resource((s) => hub.api.system(s));
  const models = new Resource((s) => hub.api.models(s));

  // Prefill from an earlier run (?from=run_x, e.g. "Retry with batch 32"). Loaded through the hub, so it waits for
  // the real server instead of asking the mock that answers before the connection settles.
  const fromId = q.get('from');
  const source = new Resource((s) => (fromId ? hub.api.run(fromId, s) : Promise.resolve(null)));
  let prefilled = $state(!fromId);
  $effect(() => {
    const r = source.data;
    if (!r || untrack(() => prefilled)) return;
    const c = r.config;
    datasetId = c.dataset_id;
    arch = c.arch;
    epochs = c.epochs;
    batch = q.get('batch') ?? String(c.batch);
    imgsz = String(c.imgsz);
    device = c.device;
    optimizer = c.optimizer;
    lr0 = c.lr0;
    seed = c.seed;
    weights = c.weights ?? 'none';
    augment = structuredClone(c.augment);
    name = `${c.name}-retry`;
    nameEdited = true;
    prefilled = true;
  });
  $effect(() => {
    if (source.error && !untrack(() => prefilled)) {
      toast.error(source.error, 'Could not load the run to copy');
      prefilled = true;
    }
  });

  $effect(() => {
    const list = datasets.data;
    if (list?.length && !untrack(() => datasetId)) datasetId = list[0]!.id;
  });

  const dataset = $derived(datasets.data?.find((d) => d.id === datasetId));

  // Pretrained weights must match the architecture; switching size drops an incompatible choice.
  const weightOptions = $derived((models.data ?? []).filter((m) => m.arch === arch));
  $effect(() => {
    const opts = weightOptions;
    if (!models.data) return;
    if (untrack(() => weights) !== 'none' && !opts.some((m) => m.id === untrack(() => weights))) weights = 'none';
  });
  const gpu = $derived(system.data?.gpus[0]);

  $effect(() => {
    const auto = `${dataset?.name ?? 'run'}-${arch.replace('folo-', '')}`;
    if (!untrack(() => nameEdited)) name = auto;
  });

  // A few training images for the augmentation preview.
  const sampleImages = new Resource<DatasetImage[]>(async (s) => {
    const id = datasetId;
    if (!id) return [];
    return (await hub.api.datasetImages(id, { split: 'train', offset: 0, limit: 12 }, s)).images.filter((i) => i.boxes.length > 0);
  });

  // Estimates are rough (architecture constants, a reference GPU): show them with matching precision.
  const roughDuration = (s: number) =>
    s < 3600 ? `~${Math.max(1, Math.round(s / 60))} min` : `~${(s / 3600).toFixed(s < 36000 ? 1 : 0)} h`;
  const roughBytes = (b: number) =>
    prefs.units === 'binary' ? `~${(b / 2 ** 30).toFixed(1)} GiB` : `~${(b / 1e9).toFixed(1)} GB`;

  /** Number inputs bind `null` when cleared; accept only finite whole numbers in range. */
  const isInt = (v: unknown, lo: number, hi: number) => typeof v === 'number' && Number.isInteger(v) && v >= lo && v <= hi;

  const batchN = $derived(Math.max(1, Math.round(Number(batch) || 16)));
  const imgszN = $derived(Number(imgsz) || 640);
  const onCpu = $derived(device === 'cpu' || (device === 'auto' && !gpu));
  const est = $derived(estimate(arch, dataset?.splits.train ?? 0, epochs, batchN, imgszN, onCpu));
  // wgpu cannot report adapter memory (real servers send 0): then there is no meter and no out-of-memory guess.
  const gpuMem = $derived(gpu && gpu.memory_bytes > 0 ? gpu.memory_bytes : null);
  const vramPct = $derived(gpuMem ? est.vramBytes / gpuMem : 0);

  const errors = $derived({
    name: !name.trim() ? 'Give the run a name.' : /[^\w.-]/.test(name.trim()) ? 'Letters, digits, dot, dash and underscore only.' : null,
    epochs: !isInt(epochs, 1, 10000) ? 'A whole number from 1 to 10,000.' : null,
    lr0: !(typeof lr0 === 'number' && lr0 > 0 && lr0 < 10) ? 'Greater than 0 and below 10.' : null,
    seed: !isInt(seed, 0, 2 ** 32 - 1) ? 'A whole number, 0 or more.' : null,
    close: !isInt(augment.close_mosaic, 0, Number.isFinite(epochs) ? epochs : 0) ? 'A whole number up to the epoch count.' : null,
    dataset: !dataset ? 'Pick a dataset.' : null,
    weights: weights !== 'none' && !weightOptions.some((m) => m.id === weights) ? 'Pick weights of the same model size.' : null,
  });
  const valid = $derived(prefilled && Object.values(errors).every((e) => e === null));
  const firstError = $derived(Object.values(errors).find((e) => e !== null) ?? null);
  // A folded section must not hide its own error: while one is shown the section stays open whatever was toggled.
  const augShown = $derived(augOpen || errors.close !== null);

  const trim = (v: number) => String(+v.toFixed(3));
  /** The augmentation settings as one line; values that differ from the defaults are emphasised. */
  const augLine = $derived([
    { k: 'mosaic', v: trim(augment.mosaic), changed: augment.mosaic !== defaults.mosaic },
    { k: 'mixup', v: trim(augment.mixup), changed: augment.mixup !== defaults.mixup },
    { k: 'flip', v: trim(augment.fliplr), changed: augment.fliplr !== defaults.fliplr },
    { k: 'scale', v: `±${trim(augment.scale)}`, changed: augment.scale !== defaults.scale },
    { k: 'translate', v: `±${trim(augment.translate)}`, changed: augment.translate !== defaults.translate },
    { k: 'colour', v: trim(augment.hsv[1]), changed: augment.hsv[1] !== defaults.hsv[1] },
    { k: 'close mosaic', v: String(augment.close_mosaic), changed: augment.close_mosaic !== defaults.close_mosaic },
  ]);

  const config = $derived<TrainConfig>({
    name: name.trim(),
    dataset_id: datasetId,
    arch,
    epochs,
    batch: batchN,
    imgsz: imgszN,
    device: device as Device,
    optimizer,
    lr0,
    weights: weights === 'none' ? null : weights,
    augment: { ...augment, hsv: [...augment.hsv] as [number, number, number], close_mosaic: Math.min(augment.close_mosaic, epochs) },
    seed,
  });

  const cliParts = $derived(
    [
      'folo train',
      `--data ${dataset?.path ?? datasetId}`,
      `--model ${arch}`,
      `--epochs ${epochs}`,
      `--batch ${batchN}`,
      `--imgsz ${imgszN}`,
      device !== 'auto' ? `--device ${device}` : '',
      weights !== 'none' ? `--weights ${weights}` : '',
      `--name ${name.trim()}`,
    ].filter(Boolean),
  );
  const cli = $derived(cliParts.join(' '));
  /** Shell line continuation, for the multi-line display of the command. */
  const CONT = ' \\\n  ';

  async function submit(e: SubmitEvent) {
    e.preventDefault();
    if (!valid || submitting) return;
    sumOpen = false;
    submitting = true;
    try {
      const run = await hub.api.createRun(config);
      toast.success(`Started ${run.name}`, `${run.epochs} epochs of ${run.config.arch} on ${dataset?.name}.`);
      router.go(`/runs/${run.id}`);
    } catch (err) {
      toast.error(err, 'Could not start the run');
    } finally {
      submitting = false;
    }
  }

  async function copyCli() {
    try {
      await navigator.clipboard.writeText(cli);
      toast.success('Command copied');
    } catch (err) {
      toast.error(err, 'Could not copy');
    }
  }

  const pct2 = (v: number) => `${Math.round(v * 100)}%`;

  // ---------- review ----------
  const shortGpu = (n: string) => n.replace('NVIDIA GeForce ', '');
  const deviceLabel = $derived.by(() => {
    if (device === 'auto') return `Auto${gpu ? ` · ${shortGpu(gpu.name)}` : ''}`;
    if (device === 'cpu') return 'CPU';
    const g = system.data?.gpus.find((x) => `gpu:${x.ordinal}` === device);
    return g ? `${device} · ${shortGpu(g.name)}` : device;
  });
  type Jump = 'name' | 'data' | 'model' | 'epochs' | 'batch' | 'imgsz' | 'device' | 'weights' | 'opt';
  /** The settings that shape the run, each a way back to its control. Optimizer only once it is not the default. */
  const review = $derived<{ key: Jump; label: string; value: string; mono?: boolean; warn?: boolean }[]>([
    { key: 'data', label: 'Data', value: dataset ? `${dataset.name} · ${compact(dataset.splits.train)} images` : '—', warn: !!errors.dataset },
    { key: 'model', label: 'Model', value: arch, mono: true },
    { key: 'epochs', label: 'Epochs', value: errors.epochs ? '—' : num(epochs), warn: !!errors.epochs },
    { key: 'batch', label: 'Batch', value: String(batchN) },
    { key: 'imgsz', label: 'Image size', value: `${imgszN} px` },
    { key: 'device', label: 'Device', value: deviceLabel },
    {
      key: 'weights',
      label: 'Start from',
      value: weights === 'none' ? 'Random init' : (weightOptions.find((m) => m.id === weights)?.name ?? weights),
      mono: weights !== 'none',
      warn: !!errors.weights,
    },
    ...(optimizer !== 'auto' ? [{ key: 'opt' as const, label: 'Optimizer', value: optimizer === 'sgd' ? 'SGD + momentum' : 'AdamW' }] : []),
  ]);

  /** The control a review row returns to; the picker that is on screen (cards or the phone select). */
  function jumpTarget(key: Jump): HTMLElement | null {
    const visible = (el: HTMLElement | null) => (el && el.getClientRects().length ? el : null);
    switch (key) {
      case 'data':
        return visible(document.getElementById('dataset-pick')) ?? document.querySelector<HTMLElement>('input[name="dataset"]:checked');
      case 'model':
        return document.querySelector<HTMLElement>('input[name="arch"]:checked');
      case 'batch':
      case 'imgsz':
        return document.querySelector<HTMLElement>(`#${key} [aria-checked="true"]`);
      default:
        return document.getElementById(key);
    }
  }

  async function jump(key: Jump) {
    sumOpen = false;
    if (key === 'opt') advancedOpen = true;
    await tick();
    // A closing dialog hands focus back to its opener; move on after that.
    requestAnimationFrame(() => {
      const el = jumpTarget(key);
      if (!el) return;
      const box = el.closest<HTMLElement>('.field, .card, .arch, .ds-pick') ?? el;
      const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
      box.scrollIntoView({ block: 'center', behavior: reduce ? 'auto' : 'smooth' });
      el.focus({ preventScroll: true, focusVisible: true } as FocusOptions);
      // Where the browser keeps the focus ring hidden after a mouse click, mark the landing spot until focus leaves.
      if (el.matches(':focus-visible')) return;
      box.dataset.jumped = '';
      el.addEventListener('blur', () => delete box.dataset.jumped, { once: true });
    });
  }
</script>

{#snippet reviewList(withName: boolean)}
  <ul class="review" aria-label="Settings">
    {#if withName}
      <li><button type="button" class="rv" class:warn={!!errors.name} onclick={() => jump('name')}><span class="rv-k">Run name</span><span class="rv-v mono">{name.trim() || '—'}</span></button></li>
    {/if}
    {#each review as r (r.key)}
      <li>
        <button type="button" class="rv" class:warn={r.warn} onclick={() => jump(r.key)} aria-label="{r.label}: {r.value}. Edit">
          <span class="rv-k">{r.label}</span><span class="rv-v" class:mono={r.mono}>{r.value}</span>
        </button>
      </li>
    {/each}
  </ul>
{/snippet}

{#snippet estimates(withName: boolean)}
  {@render reviewList(withName)}
  <dl class="est">
    <dt>Steps / epoch</dt><dd class="num">{num(est.stepsPerEpoch)}</dd>
    <dt>Est. time</dt><dd class="num">{roughDuration(est.totalSeconds)} <span class="faint">(~{num(Math.round(est.imgsPerS / 10) * 10, 0)} img/s)</span></dd>
    {#if !onCpu && gpu}
      <dt>Est. GPU memory</dt>
      <dd class="num" class:warn={vramPct > 0.9}>{roughBytes(est.vramBytes)} <span class="faint">/ {gpuMem ? bytes(gpuMem, prefs.units) : 'GPU memory unknown'}</span></dd>
    {/if}
  </dl>
  {#if !onCpu && gpuMem}
    <div class="vram" aria-hidden="true"><i class:warn={vramPct > 0.9} style:width={pct2(Math.min(1, vramPct))}></i></div>
  {/if}
  {#if vramPct > 1}
    <p class="warning" role="alert">
      <TriangleAlert size={14} /> This likely runs out of GPU memory. Try batch {Math.max(1, Math.floor(batchN / Math.ceil(vramPct)))} or a smaller image size.
    </p>
  {:else if onCpu}
    <p class="warning soft"><TriangleAlert size={14} /> Training on the CPU is roughly 14× slower than on a GPU.</p>
  {/if}
{/snippet}

<Page title="New training run">
  {#snippet sub()}<span>Pick data and a model size; the defaults are tuned to work well. Everything else is optional.</span>{/snippet}

  {#if datasets.error && !datasets.data}
    <ErrorState error={datasets.error} retry={() => datasets.refresh()} title="Could not load datasets" />
  {:else if datasets.data && datasets.data.length === 0}
    <Panel>
      <EmptyState icon={Images} title="Add a dataset first" description="Runs train on a registered dataset: add a folder or generate a synthetic one.">
        {#snippet actions()}<Button variant="primary" href={href('/datasets')}>Go to Datasets</Button>{/snippet}
      </EmptyState>
    </Panel>
  {:else}
    <form class="layout" onsubmit={submit} novalidate>
      <div class="form">
        <Panel title="1 · Dataset" id="sec-data">
          {#if !datasets.data}
            <div class="cards">{#each [0, 1, 2] as i (i)}<Skeleton h="88px" r="var(--radius-md)" />{/each}</div>
          {:else}
            <!-- Phones: a compact picker and a one-line summary; the cards are for wider screens. -->
            <div class="ds-pick">
              <Select id="dataset-pick" label="Dataset" bind:value={datasetId} options={datasets.data.map((d) => ({ value: d.id, label: d.name }))} />
              {#if dataset}
                <p class="ds-line num">
                  {num(dataset.splits.train)} train · {num(dataset.splits.val)} val · {dataset.classes.length} classes · <span class="mono">{dataset.format}</span>
                </p>
              {/if}
            </div>
            <div class="cards" role="radiogroup" aria-label="Dataset">
              {#each datasets.data as d (d.id)}
                {@const on = d.id === datasetId}
                <label class="card" class:on>
                  <input type="radio" name="dataset" value={d.id} bind:group={datasetId} class="sr-only" />
                  <span class="c-top"><span class="c-name">{d.name}</span><span class="c-fmt mono">{d.format}</span></span>
                  {#if on}<span class="tick" aria-hidden="true"><Check size={14} strokeWidth={2.25} /></span>{/if}
                  <span class="c-meta num">{compact(d.splits.train)} train · {compact(d.splits.val)} val · {d.classes.length} classes</span>
                  {#if d.stats}<ClassStrip counts={d.stats.class_counts} classes={d.classes} height={4} />{/if}
                </label>
              {/each}
            </div>
          {/if}
        </Panel>

        <Panel title="2 · Model" id="sec-model">
          <div class="arches" role="radiogroup" aria-label="Model size">
            <!-- Up to 1199 px the cards become single-line rows; this is their column header. -->
            <div class="a-head" aria-hidden="true"><span class="a-head-lead"></span><span>Params</span><span>GFLOPs</span><span>Latency</span></div>
            {#each arches as a (a.arch)}
              {@const on = a.arch === arch}
              <label class="arch" class:on>
                <input type="radio" name="arch" value={a.arch} bind:group={arch} class="sr-only" />
                <span class="a-top"><span class="a-size">{a.size}</span><span class="a-name mono">{a.arch}</span></span>
                {#if on}<span class="tick" aria-hidden="true"><Check size={14} strokeWidth={2.25} /></span>{/if}
                <span class="a-note">{a.note}</span>
                <span class="a-bars">
                  <span class="a-row"><span class="faint">Params</span><span class="num">{compact(a.params)}</span></span>
                  <span class="a-row"><span class="faint">GFLOPs</span><span class="num">{num(a.flops, 1)}</span></span>
                  <span class="a-row"><span class="faint">Latency</span><span class="num">{num(a.ms, 1)} ms</span></span>
                </span>
              </label>
            {/each}
          </div>
          <p class="ref">Latency is a reference figure: RTX 4080-class GPU, 640 px input, fp32, one image.</p>
          <div class="row2">
            <Field label="Start from" for="weights" hint="Fine-tuning from a trained model converges much faster.">
              <Select
                id="weights"
                bind:value={weights}
                options={[{ value: 'none', label: 'Random init (from scratch)' }, ...weightOptions.map((m) => ({ value: m.id, label: `${m.name} · mAP ${m.metrics?.map50_95.toFixed(3) ?? '—'}` }))]}
              />
            </Field>
          </div>
        </Panel>

        <Panel title="3 · Training" id="sec-train">
          <div class="fields">
            <Field label="Epochs" for="epochs" error={errors.epochs}>
              <Input id="epochs" type="number" min="1" max="10000" step="1" bind:value={epochs} invalid={!!errors.epochs} suffix="epochs" />
            </Field>
            <Field label="Batch size" for="batch" hint="Per step. Lower it if you run out of GPU memory.">
              <Segmented id="batch" label="Batch size" bind:value={batch} options={['8', '16', '32', '64'].map((v) => ({ value: v, label: v }))} />
            </Field>
            <Field label="Image size" for="imgsz" hint="Longer side in pixels, letterboxed.">
              <Segmented id="imgsz" label="Image size" bind:value={imgsz} options={IMGSZ.map((v) => ({ value: v, label: v }))} />
            </Field>
            <Field label="Device" for="device">
              <Select
                id="device"
                bind:value={device}
                options={[
                  { value: 'auto', label: `Auto${gpu ? ` (${gpu.name.replace('NVIDIA GeForce ', '')})` : ''}` },
                  ...(system.data?.gpus ?? []).map((g) => ({ value: `gpu:${g.ordinal}`, label: `gpu:${g.ordinal} · ${g.name.replace('NVIDIA GeForce ', '')}` })),
                  { value: 'cpu', label: `CPU${system.data ? ` · ${system.data.cpu.threads} threads` : ''}` },
                ]}
              />
            </Field>
          </div>
          <details class="advanced" open={advancedOpen} ontoggle={(e) => (advancedOpen = (e.currentTarget as HTMLDetailsElement).open)}>
            <summary><ChevronRight size={14} /> Advanced <span class="faint">optimizer, learning rate, seed</span></summary>
            <div class="fields">
            <Field label="Optimizer" for="opt">
              <Select id="opt" bind:value={optimizer} options={[{ value: 'auto', label: 'Auto (AdamW for short runs, else SGD)' }, { value: 'sgd', label: 'SGD + momentum' }, { value: 'adamw', label: 'AdamW' }]} />
            </Field>
            <Field label="Initial learning rate" for="lr0" error={errors.lr0} hint="Warmup 3 epochs, then cosine to 1%.">
              <Input id="lr0" type="number" step="0.001" min="0" bind:value={lr0} invalid={!!errors.lr0} />
            </Field>
            <Field label="Seed" for="seed" hint="Same seed, same run." error={errors.seed}>
              <Input id="seed" type="number" min="0" step="1" bind:value={seed} invalid={!!errors.seed} />
            </Field>
            </div>
          </details>
        </Panel>

        <Panel title="4 · Augmentation" id="sec-aug">
          {#snippet actions()}
            <span class="aug-actions" class:shut={!augShown}>
              <Button size="sm" variant="ghost" icon={Shuffle} onclick={() => shuffle++}>New samples</Button>
              <Button size="sm" variant="ghost" icon={RotateCcw} onclick={() => (augment = structuredClone(defaults))}>Defaults</Button>
            </span>
          {/snippet}
          <button type="button" class="aug-toggle" aria-expanded={augShown} aria-controls="aug-body" onclick={() => (augOpen = !augShown)}>
            <ChevronRight size={14} aria-hidden="true" />
            <span class="aug-line num">
              {#each augLine as it, i (it.k)}{#if i > 0}<span class="sep" aria-hidden="true">·</span>{/if}<span class:changed={it.changed}>{it.k} {it.v}</span>{/each}
            </span>
            <span class="aug-verb">{augShown ? 'Hide' : 'Edit'}</span>
          </button>
          <div class="aug" id="aug-body" class:shut={!augShown}>
            <div class="aug-controls">
              {#snippet ctl(k: AugKey, children: import('svelte').Snippet)}
                <!-- svelte-ignore a11y_no_static_element_interactions -->
                <div
                  class="aug-row"
                  class:focused={focus === k}
                  onpointerenter={() => (focus = k)}
                  onpointerleave={() => (focus = null)}
                  onfocusin={() => (focus = k)}
                  onfocusout={() => (focus = null)}
                >
                  {@render children()}
                </div>
              {/snippet}
              {#snippet mosaicCtl()}<Slider label="Mosaic" bind:value={augment.mosaic} min={0} max={1} step={0.05} hint="Stitch 4 images into one: more objects and context per sample." />{/snippet}
              {#snippet mixupCtl()}<Slider label="Mixup" bind:value={augment.mixup} min={0} max={1} step={0.05} hint="Blend two images and keep both sets of boxes." />{/snippet}
              {#snippet flipCtl()}<Slider label="Horizontal flip" bind:value={augment.fliplr} min={0} max={1} step={0.05} hint="Chance to mirror left-right." />{/snippet}
              {#snippet scaleCtl()}<Slider label="Scale" bind:value={augment.scale} min={0} max={0.9} step={0.05} format={(v) => `±${v.toFixed(2)}`} hint="Random zoom in and out." />{/snippet}
              {#snippet translateCtl()}<Slider label="Translate" bind:value={augment.translate} min={0} max={0.5} step={0.01} format={(v) => `±${v.toFixed(2)}`} hint="Random shift, as a fraction of the size." />{/snippet}
              {#snippet hsvCtl()}<Slider label="Colour jitter (saturation)" bind:value={augment.hsv[1]} min={0} max={1} step={0.05} hint="Hue ±{augment.hsv[0]}, value ±{augment.hsv[2]}: robustness to lighting." />{/snippet}
              {@render ctl('mosaic', mosaicCtl)}
              {@render ctl('mixup', mixupCtl)}
              {@render ctl('fliplr', flipCtl)}
              {@render ctl('scale', scaleCtl)}
              {@render ctl('translate', translateCtl)}
              {@render ctl('hsv', hsvCtl)}
              <div class="aug-row">
                <Field label="Close mosaic" for="close" hint="Train the last N epochs on plain images to settle the boxes." error={errors.close}>
                  <Input id="close" type="number" min="0" step="1" max={epochs} bind:value={augment.close_mosaic} suffix="epochs" invalid={!!errors.close} />
                </Field>
              </div>
            </div>
            <div class="aug-preview">
              <p class="overline">{focus ? `Preview · ${focus === 'fliplr' ? 'flip' : focus} only` : 'Preview · all augmentations'}</p>
              {#if dataset}
                <AugPreview datasetId={dataset.id} images={sampleImages.data ?? []} classes={dataset.classes} {augment} {focus} seed={shuffle} />
              {/if}
              <p class="aug-hint">Hover a setting to isolate it. Boxes move with the pixels, exactly as the trainer transforms them.</p>
            </div>
          </div>
        </Panel>
      </div>

      <aside class="summary">
        <Panel title="Summary">
          <div class="sum">
            <Field label="Run name" for="name" error={errors.name}>
              <Input id="name" bind:value={name} oninput={() => (nameEdited = true)} invalid={!!errors.name} spellcheck="false" autocomplete="off" />
            </Field>
            {@render estimates(false)}
            <div class="start" bind:this={startEl}>
              <Button type="submit" variant="primary" size="lg" icon={Play} block loading={submitting} disabled={!valid}>Start training</Button>
            </div>
            <details class="cli">
              <summary><ChevronRight size={14} aria-hidden="true" />CLI command <span class="faint">the same run from a terminal</span></summary>
              <div class="cli-body">
                <code>{cliParts.join(CONT)}</code>
                <span class="cli-copy"><Button size="sm" variant="ghost" icon={Copy} aria-label="Copy command" title="Copy command" onclick={copyCli} /></span>
              </div>
            </details>
          </div>
        </Panel>
      </aside>

      <!-- Narrow screens: the summary sits below the form, so keep the essentials and the action in reach. -->
      <div class="dock" class:away={startVisible} role="region" aria-label="Start training" inert={startVisible}>
        <button type="button" class="dock-est num" aria-haspopup="dialog" aria-label="Show the run summary" onclick={() => (sumOpen = true)}>
          <span>{roughDuration(est.totalSeconds)}</span>
          {#if !onCpu && gpu}<span class:warn={vramPct > 0.9}>{roughBytes(est.vramBytes)} VRAM</span>{/if}
          {#if dataset}<span class="faint">{dataset.name} · {arch}</span>{/if}
          <span class="dock-more"><ChevronUp size={14} strokeWidth={1.75} aria-hidden="true" /></span>
        </button>
        <Button type="submit" variant="primary" size="lg" icon={Play} loading={submitting} disabled={!valid}>Start training</Button>
      </div>

      <!-- Inside the form, so its Start button submits it. -->
      <Dialog bind:open={sumOpen} title="Summary" size="sm">
        <div class="sum">
          {@render estimates(true)}
          {#if firstError && prefilled}<p class="warning" role="alert"><TriangleAlert size={14} /> {firstError}</p>{/if}
        </div>
        {#snippet footer()}
          <Button variant="ghost" onclick={() => (sumOpen = false)}>Keep editing</Button>
          <Button type="submit" variant="primary" icon={Play} loading={submitting} disabled={!valid}>Start training</Button>
        {/snippet}
      </Dialog>
    </form>
  {/if}
</Page>

<style>
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
  .summary {
    position: sticky;
    top: calc(var(--size-topbar) + var(--space-4));
  }
  .cards {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: var(--space-2);
  }
  .card,
  .arch {
    display: grid;
    gap: 6px;
    align-content: start;
    padding: var(--space-3);
    border-radius: var(--radius-md);
    border: 1px solid var(--line-default);
    background: var(--bg-raised);
    cursor: pointer;
    transition:
      border-color var(--dur-fast) var(--ease-standard),
      background-color var(--dur-fast) var(--ease-standard),
      box-shadow var(--dur-fast) var(--ease-standard);
  }
  .card:hover,
  .arch:hover {
    border-color: var(--line-strong);
  }
  /* Selection: a strong edge on the selection tint and a small ink check; lime stays for the Start button and focus. */
  .card,
  .arch {
    position: relative;
  }
  .card.on,
  .arch.on {
    border-color: var(--line-strong);
    background: var(--bg-selected);
  }
  .tick {
    position: absolute;
    top: 12px;
    right: 12px;
    display: grid;
    color: var(--text-primary);
  }
  .card.on .c-fmt,
  .arch.on .a-name {
    visibility: hidden;
  }
  .card:has(:focus-visible),
  .arch:has(:focus-visible) {
    outline: 2px solid var(--focus);
    outline-offset: 2px;
  }
  .c-top,
  .a-top {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: var(--space-2);
  }
  .c-name {
    font-weight: var(--weight-medium);
  }
  .c-fmt,
  .a-name {
    font-size: var(--type-xs-size);
    color: var(--text-tertiary);
  }
  .c-meta {
    font-size: var(--type-sm-size);
    color: var(--text-secondary);
  }
  .arches {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: var(--space-2);
  }
  .a-size {
    font-size: var(--type-2xl-size);
    line-height: var(--type-2xl-line);
    font-weight: var(--weight-semibold);
    letter-spacing: var(--type-2xl-tracking);
  }
  .arch.on .a-size {
    color: var(--text-primary);
  }
  .a-note {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
    min-height: 32px;
  }
  .a-bars {
    display: grid;
    gap: 4px;
    margin-top: 2px;
  }
  .a-row {
    display: flex;
    justify-content: space-between;
    gap: var(--space-2);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
  }
  .a-row .num {
    color: var(--text-secondary);
  }
  .arch.on .a-row .num {
    color: var(--text-primary);
  }
  .ref {
    margin-top: var(--space-2);
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
  }
  .row2 {
    margin-top: var(--space-4);
    max-width: 420px;
  }
  .fields {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: var(--space-4) var(--space-5);
  }
  .aug {
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
    gap: var(--space-5);
  }
  .aug-controls {
    display: grid;
    gap: 2px;
  }
  .aug-row {
    padding: var(--space-2) var(--space-3);
    margin: 0 calc(-1 * var(--space-3));
    border-radius: var(--radius-md);
    transition: background-color var(--dur-fast) var(--ease-standard);
  }
  .aug-row.focused {
    background: var(--bg-hover);
  }
  .aug-preview {
    display: grid;
    gap: var(--space-2);
    align-content: start;
    align-self: start;
    position: sticky;
    top: calc(var(--size-topbar) + var(--space-4));
  }
  .aug-hint {
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
  }
  .sum {
    display: grid;
    gap: var(--space-4);
  }
  /* Settings to review: 28 px rows, each a way back to its control. */
  .review {
    display: grid;
    margin: calc(-1 * var(--space-1)) calc(-1 * var(--space-2)) 0;
  }
  /* Touch: each review row is a full 44 px target (after the base rule, so it wins). */
  @media (pointer: coarse) {
    .rv.rv {
      height: 44px;
    }
  }
  .rv {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-3);
    width: 100%;
    height: 28px;
    padding: 0 var(--space-2);
    border-radius: var(--radius-sm);
    font-size: var(--type-sm-size);
    text-align: left;
    transition: background-color var(--dur-fast) var(--ease-standard);
  }
  .rv:hover {
    background: var(--bg-hover);
  }
  .rv:focus-visible {
    outline-offset: -2px;
  }
  .rv-k {
    flex-shrink: 0;
    color: var(--text-tertiary);
  }
  .rv-v {
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    color: var(--text-primary);
  }
  .rv.warn .rv-v {
    color: var(--danger-text);
  }
  .layout :global([data-jumped]) {
    outline: 2px solid var(--focus);
    outline-offset: 4px;
    border-radius: var(--radius-md);
  }
  .est {
    padding-top: var(--space-3);
    border-top: 1px solid var(--line-subtle);
    display: grid;
    grid-template-columns: auto 1fr;
    gap: 6px var(--space-3);
    font-size: var(--type-sm-size);
  }
  .est dt {
    color: var(--text-tertiary);
  }
  .est dd {
    text-align: right;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .est dd.warn {
    color: var(--warning-text);
  }
  /* A neutral meter; the warning colour only above 90 % of the GPU's memory. */
  .vram {
    height: 4px;
    margin-top: calc(-1 * var(--space-2));
    border-radius: 2px;
    background: var(--bg-active);
    overflow: hidden;
  }
  .vram i {
    display: block;
    height: 100%;
    background: var(--text-secondary);
    transition: width var(--dur-chart) var(--ease-standard);
  }
  .vram i.warn {
    background: var(--warning-solid);
  }
  .ds-pick,
  .a-head,
  .aug-toggle {
    display: none;
  }
  .ds-line {
    margin-top: var(--space-2);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
  }
  .ds-line .mono {
    color: var(--text-tertiary);
  }
  .aug-toggle {
    align-items: center;
    gap: var(--space-2);
    min-height: var(--control-lg);
    margin-top: calc(-1 * var(--space-1));
    padding: 0 var(--space-2);
    margin-inline: calc(-1 * var(--space-2));
    width: calc(100% + 2 * var(--space-2));
    border-radius: var(--radius-md);
    color: var(--text-secondary);
    font-size: var(--type-sm-size);
    text-align: left;
  }
  .aug-toggle:hover {
    background: var(--bg-hover);
    color: var(--text-primary);
  }
  .aug-toggle > :global(svg) {
    flex-shrink: 0;
    transition: transform var(--dur-fast) var(--ease-standard);
  }
  .aug-toggle[aria-expanded='true'] > :global(svg) {
    transform: rotate(90deg);
  }
  .aug-line {
    flex: 1;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .aug-line .sep {
    margin-inline: var(--space-1-5);
    color: var(--text-tertiary);
  }
  .aug-line .changed {
    color: var(--text-primary);
    font-weight: var(--weight-medium);
  }
  .aug-verb {
    flex-shrink: 0;
    color: var(--text-tertiary);
  }
  .aug-actions {
    display: contents;
  }
  .warning {
    display: flex;
    gap: var(--space-2);
    padding: var(--space-2) var(--space-3);
    border-radius: var(--radius-md);
    background: var(--danger-muted);
    color: var(--danger-text);
    font-size: var(--type-sm-size);
  }
  .warning.soft {
    background: var(--warning-muted);
    color: var(--warning-text);
  }
  .warning :global(svg) {
    flex-shrink: 0;
    margin-top: 4px;
  }
  .cli {
    margin-top: calc(-1 * var(--space-1));
    padding-top: var(--space-2);
    border-top: 1px solid var(--line-subtle);
  }
  .cli summary {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    height: var(--control-md);
    color: var(--text-secondary);
    font-size: var(--type-sm-size);
    font-weight: var(--weight-medium);
    cursor: pointer;
    list-style: none;
    border-radius: var(--radius-sm);
  }
  .cli summary::-webkit-details-marker {
    display: none;
  }
  .cli summary :global(svg) {
    transition: transform var(--dur-fast) var(--ease-standard);
  }
  .cli[open] summary :global(svg) {
    transform: rotate(90deg);
  }
  .cli summary:hover {
    color: var(--text-primary);
  }
  .cli summary .faint {
    font-weight: var(--weight-regular);
  }
  .cli-body {
    position: relative;
    margin-top: var(--space-2);
  }
  .cli-copy {
    position: absolute;
    top: var(--space-1);
    right: var(--space-1);
  }
  code {
    display: block;
    padding: var(--space-2) var(--space-8) var(--space-2) var(--space-3);
    border-radius: var(--radius-md);
    background: var(--bg-inset);
    font-family: var(--font-mono);
    font-size: var(--type-sm-size);
    line-height: 18px;
    color: var(--text-secondary);
    white-space: pre-wrap;
    overflow-wrap: anywhere;
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
  .dock {
    display: none;
  }
  @media (max-width: 1199px) {
    .layout {
      padding-bottom: 64px;
    }
    .dock {
      position: fixed;
      left: 0;
      right: 0;
      bottom: 0;
      z-index: var(--z-sticky);
      display: flex;
      align-items: center;
      justify-content: flex-end;
      gap: var(--space-4);
      padding: var(--space-3) var(--space-4);
      background: color-mix(in srgb, var(--bg-app) 90%, transparent);
      backdrop-filter: blur(12px) saturate(1.4);
      -webkit-backdrop-filter: blur(12px) saturate(1.4);
      border-top: 1px solid var(--line-subtle);
      transition:
        transform var(--dur-base) var(--ease-out),
        opacity var(--dur-base) var(--ease-out);
    }
    .dock.away {
      transform: translateY(100%);
      opacity: 0;
    }
    .dock-est {
      display: flex;
      align-items: center;
      gap: var(--space-3);
      min-height: var(--control-lg);
      padding: 0 var(--space-2);
      border-radius: var(--radius-md);
      color: var(--text-secondary);
      font-size: var(--type-sm-size);
      text-align: left;
    }
    .dock-est:hover {
      background: var(--bg-hover);
      color: var(--text-primary);
    }
    .dock-more {
      display: grid;
      color: var(--text-tertiary);
    }
    .dock-est .warn {
      color: var(--warning-text);
    }
    .layout {
      grid-template-columns: minmax(0, 1fr);
    }
    .summary {
      position: static;
    }
    /* Stacked under the form, the summary keeps the rail's measure so labels stay next to their values. */
    .summary .sum {
      max-width: 480px;
    }
    /* Model sizes as single-line rows: letter, name, note, then the three specs in columns. */
    .arches {
      --a-cols: 16px auto minmax(0, 1fr) repeat(3, 72px) 14px;
      grid-template-columns: minmax(0, 1fr);
      gap: var(--space-1);
    }
    .a-head,
    .arch {
      grid-template-columns: var(--a-cols);
      align-items: center;
      column-gap: var(--space-3);
    }
    .a-head {
      display: grid;
      padding: 0 calc(var(--space-3) + 1px) var(--space-1);
      font-size: var(--type-sm-size);
      color: var(--text-tertiary);
      text-align: right;
    }
    .a-head-lead {
      grid-column: 1 / -5;
    }
    .arch {
      gap: 0 var(--space-3);
      align-content: center;
      min-height: 36px;
      padding: 0 var(--space-3);
    }
    .a-top,
    .a-bars {
      display: contents;
    }
    .a-size {
      font-size: var(--type-md-size);
      line-height: var(--type-md-line);
      letter-spacing: 0;
    }
    .arch.on .a-name {
      visibility: visible;
      color: var(--text-secondary);
    }
    .a-note {
      grid-column: 3;
      grid-row: 1;
      min-height: 0;
      color: var(--text-tertiary);
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .a-row {
      display: block;
      text-align: right;
      white-space: nowrap;
    }
    .a-row .faint {
      display: none;
    }
    .arch .tick {
      position: static;
      grid-column: -2;
      grid-row: 1;
    }
    .aug-toggle {
      display: flex;
    }
    .aug.shut,
    .aug-actions.shut {
      display: none;
    }
    .aug-toggle[aria-expanded='true'] {
      margin-bottom: var(--space-3);
    }
  }
  .dock-est .faint {
    display: none;
  }
  @media (max-width: 767px) {
    /* Two columns: the estimate wraps on the left, a 44 px Start button holds the right; 16 px gutters. */
    .dock {
      bottom: calc(var(--tabbar-h) + env(safe-area-inset-bottom));
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: var(--space-3);
      padding: var(--space-2) var(--space-4);
    }
    .dock-est {
      position: relative;
      flex-wrap: wrap;
      gap: 0 var(--space-3);
      min-width: 0;
      margin-left: calc(-1 * var(--space-2));
      padding: var(--space-1) var(--space-6) var(--space-1) var(--space-2);
      line-height: var(--type-sm-line);
    }
    .dock-more {
      position: absolute;
      right: var(--space-2);
      top: 50%;
      transform: translateY(-50%);
    }
    .dock-est .faint {
      display: block;
      flex-basis: 100%;
      font-size: var(--type-sm-size);
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .dock :global(.btn.lg) {
      --h: 44px;
    }
    .ds-pick {
      display: block;
    }
    .ds-pick :global(select) {
      height: 44px;
    }
    .cards {
      display: none;
    }
    /* Model rows: the note gives way so the three specs keep their columns; 44 px touch rows. */
    .arches {
      --a-cols: 16px minmax(0, 1fr) repeat(3, 52px) 14px;
    }
    .a-head,
    .arch {
      column-gap: var(--space-2);
    }
    .a-head {
      padding-inline: calc(var(--space-3) + 1px);
    }
    .arch {
      min-height: 44px;
    }
    .a-note {
      display: none;
    }
    .fields,
    .aug {
      grid-template-columns: minmax(0, 1fr);
    }
    .aug-preview {
      position: static;
      order: -1;
    }
  }
</style>
