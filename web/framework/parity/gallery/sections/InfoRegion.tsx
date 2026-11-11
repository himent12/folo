// Gallery twin of InfoRegion.svelte: a chart's ⓘ in a Panel's header, in place under the heading on phones.
import InfoButton from '$lib/ui/InfoButton';
import Panel from '$lib/ui/Panel';

export default function InfoRegionSection() {
  return (
    <div class="g-stack g-medium" data-parity-phone data-parity-coarse>
      <Panel
        title="Box size"
        actions={() => (
          <span data-parity-step="text" data-parity-target="button">
            <InfoButton chart label="How box size is measured" text="Size = √(box area ÷ image area)." />
          </span>
        )}
      >
        <p class="readout">44,433 boxes</p>
      </Panel>
      <Panel
        title="Loss terms"
        meta="3"
        actions={() => (
          <span data-parity-step="list" data-parity-target="button">
            <InfoButton chart label="What the loss terms mean" how="Each term is averaged over the epoch's batches.">
              <dl>
                <dt>Box</dt>
                <dd>how far predicted boxes are from the true ones</dd>
              </dl>
            </InfoButton>
          </span>
        )}
      >
        <p class="readout">Box 1.21 · Cls 0.84 · DFL 1.02</p>
      </Panel>
    </div>
  );
}
