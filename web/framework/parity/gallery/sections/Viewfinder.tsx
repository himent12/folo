// Gallery twin of Viewfinder.svelte.
import Button from '$lib/ui/Button';
import Viewfinder from '$lib/ui/Viewfinder';

export default function ViewfinderSection() {
  return (
    <div class="g-stack g-medium" data-parity-phone>
      <Viewfinder
        title="Drop an image to detect objects"
        description={() => (
          <>
            PNG or JPEG, up to 20 MB. Or <b>paste</b> one.
          </>
        )}
        actions={() => (
          <>
            <Button variant="primary">Choose image</Button>
            <Button>Use a sample</Button>
          </>
        )}
      />
      <Viewfinder dragging title="Drop to add" description={() => 'Images are added to this job.'} />
      <div style="--vf-pad: 8px">
        <Viewfinder title="Small room" description={() => 'A tight frame.'} />
      </div>
    </div>
  );
}
