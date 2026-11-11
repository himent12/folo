<!-- Gallery twin of Progress.tsx. -->
<script lang="ts">
  import Progress from '$lib/ui/Progress.svelte';
</script>

<div class="g-stack g-medium">
  <Progress value={0.42} label="Epoch 25 of 60" />
  <Progress value={0} tone="success" />
  <Progress value={1} tone="success" size="md" />
  <Progress value={0.7} tone="danger" size="xs" />
  <Progress value={0.33} tone="warning" />
  <Progress value={0.9} tone="neutral" size="md" />
  <Progress value={1.7} />
  <Progress value={NaN} />
  <Progress value={-1} tone="neutral" />
</div>
