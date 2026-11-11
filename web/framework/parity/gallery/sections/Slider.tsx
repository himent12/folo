// Gallery twin of Slider.svelte.
import { signal } from 'fw';
import Slider from '$lib/ui/Slider';

export default function SliderSection() {
  const conf = signal(0.25);
  const iou = signal(0.7);
  const epochs = signal(60);
  const size = signal(640);
  const moved = signal(0);
  return (
    <div class="g-stack g-medium" data-parity-coarse>
      <Slider label="Confidence" bind:value={conf} />
      <Slider label="IoU threshold" bind:value={iou} hint="Boxes overlapping more than this are merged." />
      <Slider label="Epochs" bind:value={epochs} min={1} max={300} step={1} format={(v) => `${v}`} />
      <Slider label="Image size" bind:value={size} min={320} max={1280} step={32} format={(v) => `${v} px`} inline />
      <Slider label="At zero" value={0} />
      <Slider label="At max" value={1} id="g-sl-max" />
      <Slider label="Degenerate range" value={3} min={3} max={3} />
      <div data-parity-step="key" data-parity-do="press:ArrowRight" data-parity-target="input">
        <Slider label="Keyboard" bind:value={conf} oninput={(v) => moved.set(v)} />
      </div>
      <div data-parity-step="focus" data-parity-do="focus" data-parity-target="input">
        <Slider label="Focused" value={0.5} hint="The thumb takes the ring." />
      </div>
      <p>{`Moved to ${moved.get()}.`}</p>
    </div>
  );
}
