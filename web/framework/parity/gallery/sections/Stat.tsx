// Gallery twin of Stat.svelte.
import Stat from '$lib/ui/Stat';

export default function StatSection() {
  return (
    <div class="g-stack">
      <div class="g-row">
        <Stat label="mAP50-95" value="0.535" />
        <Stat label="Throughput" value="412" unit="img/s" sub="on the RTX 4080" />
        <Stat label="Learning rate" value="8.3×10⁻⁴" size="sm" sub="AdamW (auto)" />
        <Stat
          label="Best epoch"
          value="57"
          size="lg"
          unit="of 60"
          sub={() => (
            <>
              <b>+0.012</b> over the last
            </>
          )}
        />
      </div>
      <div class="g-row">
        <Stat label="Loading md" value="—" loading />
        <Stat label="Loading sm" value="—" size="sm" loading sub="x" />
        <Stat label="Loading lg" value="—" size="lg" loading sub="x" />
      </div>
      <div class="g-narrow">
        <Stat label="A long value" value="A value much too long to fit in its box, cut with an ellipsis" sub="And a sub line that is also much too long to fit" />
      </div>
    </div>
  );
}
