<!-- Gallery twin of DetectDatasetImage.tsx: a dataset image loading, loaded, unavailable, and inside a picker button. -->
<script lang="ts">
  import { hub } from '$lib/api/index.svelte';
  import DatasetImage from '$lib/detect/DatasetImage.svelte';
  import { galleryApi, sceneBoxes, sceneClasses } from '../chartData';

  // `ok` resolves to the scene, `slow` never resolves (the skeleton), anything else fails (Image unavailable).
  hub.api = galleryApi(hub.api);
</script>

<div class="g-row">
  <div style="width: 320px; height: 240px">
    <DatasetImage datasetId="ds" imageId="ok" width={640} height={480} boxes={sceneBoxes} classes={sceneClasses} />
  </div>
  <div style="width: 160px; height: 120px">
    <DatasetImage datasetId="ds" imageId="slow" width={640} height={480} classes={sceneClasses} compact />
  </div>
  <div style="width: 160px; height: 120px">
    <DatasetImage datasetId="ds" imageId="gone" width={640} height={480} boxes={sceneBoxes} classes={sceneClasses} />
  </div>
  <button type="button" aria-label="Use img_0004.jpg" style="width: 160px; height: 120px; padding: 0">
    <DatasetImage datasetId="ds" imageId="gone" width={640} height={480} classes={sceneClasses} compact />
  </button>
</div>
