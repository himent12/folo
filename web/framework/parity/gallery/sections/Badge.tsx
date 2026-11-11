// Gallery twin of Badge.svelte.
import Badge from '$lib/ui/Badge';

export default function BadgeSection() {
  return (
    <div class="g-row">
      <Badge>neutral</Badge>
      <Badge tone="accent">accent</Badge>
      <Badge tone="success">success</Badge>
      <Badge tone="warning">warning</Badge>
      <Badge tone="danger">danger</Badge>
      <Badge tone="info">info</Badge>
      <Badge tone="outline">outline</Badge>
      <Badge mono>run_7d1e</Badge>
      <Badge tone="accent" mono>0.535</Badge>
      <Badge tone="danger" mono>OOM</Badge>
    </div>
  );
}
