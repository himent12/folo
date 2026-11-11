// Gallery twin of Skeleton.svelte.
import Skeleton from '$lib/ui/Skeleton';

export default function SkeletonSection() {
  return (
    <div class="g-stack g-medium">
      <Skeleton />
      <Skeleton w="200px" h="20px" />
      <Skeleton w="48px" h="48px" r="var(--radius-full)" class="extra" />
      <p>
        A line with a value loading: {'\u200b'}
        <Skeleton inline w="64px" h="12px" /> and more text after it.
      </p>
      <div class="g-row">
        <Skeleton image w="160px" h="120px" />
        <Skeleton image w="96px" h="96px" r="var(--radius-md)" />
      </div>
    </div>
  );
}
