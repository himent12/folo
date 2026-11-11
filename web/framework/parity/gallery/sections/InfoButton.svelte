<!-- Gallery twin of InfoButton.tsx. -->
<script lang="ts">
  import InfoButton from '$lib/ui/InfoButton.svelte';

  let open = $state(true);
</script>

<div class="g-stack" data-parity-coarse data-parity-phone>
  <div class="g-row">
    <span data-parity-step="open" data-parity-target="button"
      ><InfoButton label="What the loss terms mean" text="Box: how far predicted boxes are from the true ones." /></span
    >
    <span data-parity-step="how" data-parity-target="button"
      ><InfoButton label="How mAP is computed" text="Mean average precision over IoU 0.50 to 0.95." how="Averaged over ten IoU thresholds, 0.50 to 0.95 in steps of 0.05, then over every class." /></span
    >
    <span data-parity-step="list" data-parity-target="button"
      ><InfoButton label="What the classes mean" side="right" class="extra"
        ><b>Box</b> where it is; <b>Cls</b> what it is.</InfoButton
      ></span
    >
    <InfoButton label="Already open by its parent" text="Shown from the start." bind:open />
    <span data-parity-step="how" data-parity-target=":document .info-how summary"></span>
  </div>
  <p class="g-narrow">
    Each epoch takes <span data-parity-step="words" data-parity-target="button"
      ><InfoButton label="Steps per epoch" text="Images ÷ batch size, rounded up.">{#snippet trigger()}30 steps / epoch{/snippet}</InfoButton></span
    > at batch 16, so a run of 60 epochs is 1,800 steps.
  </p>
  <div class="g-col">
    <span data-parity-step="controls" data-parity-target="button"
      ><InfoButton label="What these metrics mean" controls="gallery-metric-help" /></span
    >
    <dl id="gallery-metric-help" class="g-note">
      <dt>mAP50</dt>
      <dd>Precision at IoU 0.50.</dd>
    </dl>
  </div>
  <div class="g-row">
    <span data-parity-step="howlabel" data-parity-target="button"
      ><InfoButton label="Where the weights come from" text="COCO 80 classes." howLabel="Where it's from">{#snippet how()}<b>COCO</b> 2017 train, 118k images.{/snippet}</InfoButton></span
    >
  </div>
</div>
