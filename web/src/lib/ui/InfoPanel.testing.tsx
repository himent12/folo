// A chart panel with its ⓘ in the header, for InfoButton.test.tsx.
import InfoButton from './InfoButton';
import Panel from './Panel';

export default function InfoPanel() {
  return (
    <Panel title="Box size" actions={() => <InfoButton chart label="How box size is measured" text="Size = √(box area ÷ image area)." />}>
      <p class="readout">44,433 boxes</p>
    </Panel>
  );
}
