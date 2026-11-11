<!-- Gallery twin of Spinner.tsx. -->
<script lang="ts">
  import Spinner from '$lib/ui/Spinner.svelte';
</script>

<div class="g-row">
  <Spinner />
  <Spinner size={16} />
  <Spinner size={24} label="Loading runs" />
  <span style="color: var(--accent-text)"><Spinner size={32} /></span>
</div>
