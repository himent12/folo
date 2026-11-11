// Gallery twin of DetectClassStrip.svelte: the class-balance strip and its breakdown (detect/ClassStrip).
import ClassStrip from '$lib/detect/ClassStrip';

export default function DetectClassStripSection() {
  const many = Array.from({ length: 14 }, (_, i) => `class-${i}`);
  return (
    <div class="g-stack g-medium" data-parity-phone>
      <div data-parity-step="open" data-parity-target="button">
        <ClassStrip counts={[300, 0, 100, 600]} classes={['bolt', 'nut', 'washer', 'screw']} named />
      </div>
      <ClassStrip counts={[300, 0, 100, 600]} classes={['bolt', 'nut', 'washer', 'screw']} height={4} />
      <div data-parity-step="many" data-parity-target="button">
        <ClassStrip counts={many.map((_, i) => ((i * 37) % 90) + 3)} classes={many} />
      </div>
      <ClassStrip counts={[0, 0]} classes={['a', 'b']} named />
    </div>
  );
}
