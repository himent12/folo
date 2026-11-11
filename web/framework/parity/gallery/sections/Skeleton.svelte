<!-- Gallery twin of Skeleton.tsx. -->
<script lang="ts">
  import Skeleton from '$lib/ui/Skeleton.svelte';
</script>

<div class="g-stack g-medium">
  <Skeleton />
  <Skeleton w="200px" h="20px" />
  <Skeleton w="48px" h="48px" r="var(--radius-full)" class="extra" />
  <p>A line with a value loading: &#8203;<Skeleton inline w="64px" h="12px" /> and more text after it.</p>
  <div class="g-row">
    <Skeleton image w="160px" h="120px" />
    <Skeleton image w="96px" h="96px" r="var(--radius-md)" />
  </div>
</div>
