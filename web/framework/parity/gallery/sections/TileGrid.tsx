// Gallery twin of TileGrid.svelte.
import { signal } from 'fw';
import TileGrid from '$lib/ui/TileGrid';

export default function TileGridSection() {
  const opened = signal(-1);
  return (
    <div class="g-stack" data-parity-coarse data-parity-phone>
      <p class="g-note">{`Opened: ${opened.get()}`}</p>
      <div class="g-wide" data-parity-step="keys" data-parity-do="press:ArrowRight" data-parity-target="[data-i='0']">
        <span data-parity-step="open" data-parity-target=":document [data-i='3']"></span>
        <TileGrid total={23} density="m" aspect={4 / 3} label={(i) => `Image ${i + 1}`} onopen={(i) => opened.set(i)} toolbar=".none" tile={(i) => <span class="g-tile">{`#${i()}`}</span>} />
      </div>
      <div class="g-medium">
        <TileGrid total={9} density="s" aspect={1} label={(i) => `Square ${i + 1}`} onopen={(i) => opened.set(i)} toolbar=".none" tile={(i) => <span class="g-tile">{i()}</span>} />
      </div>
      <div class="g-wide">
        <TileGrid total={5} density="l" aspect={16 / 9} label={(i) => `Wide ${i + 1}`} onopen={() => {}} toolbar=".none" tile={(i) => <span class="g-tile">{i()}</span>} />
      </div>
      <div class="g-wide">
        <TileGrid total={0} density="m" aspect={4 / 3} label={() => ''} onopen={() => {}} toolbar=".none" tile={(i) => <span class="g-tile">{i()}</span>} />
      </div>
    </div>
  );
}
