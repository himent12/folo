<!-- Gallery twin of Meta.tsx. -->
<script lang="ts">
  import Meta from '$lib/ui/Meta.svelte';
</script>

<div class="g-stack" data-parity-phone>
  <Meta><span>60 epochs</span><span>folo-s</span><span>best mAP50-95 0.535</span></Meta>
  <div class="g-narrow">
    <Meta title="All facts"><span>12,480 images</span><span>80 classes</span><span>3 splits</span><span>updated 2 hours ago</span><span>4.2 GB</span></Meta>
  </div>
  <div class="g-narrow">
    <Meta line class="extra"><span>A very long free-text dataset name that will not fit</span><span>12,480 images</span><span>80 classes</span></Meta>
  </div>
  <Meta><span>outer</span><span><Meta line><span>inner a</span><span>inner b</span></Meta></span></Meta>
</div>
