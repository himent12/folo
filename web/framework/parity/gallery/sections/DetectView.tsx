// Gallery twin of DetectView.svelte: detections over an image (detect/DetectionView).
import DetectionView from '$lib/detect/DetectionView';
import { sceneBoxes, sceneClasses, sceneSrc, sceneTruth } from '../chartData';

export default function DetectViewSection() {
  return (
    <div class="g-stack" data-parity-phone>
      <div style="width: 100%; max-width: 900px; height: 520px">
        <DetectionView src={sceneSrc} width={640} height={480} boxes={sceneBoxes} truth={sceneTruth} classes={sceneClasses} active={0} />
      </div>
      <div class="g-row">
        <div style="width: 360px; height: 270px">
          <DetectionView src={sceneSrc} width={640} height={480} boxes={sceneBoxes} truth={sceneTruth} classes={sceneClasses} selected={2} />
        </div>
        <div style="width: 160px; height: 120px">
          <DetectionView src={sceneSrc} width={640} height={480} boxes={sceneBoxes} truth={sceneTruth} classes={sceneClasses} compact />
        </div>
        <div style="width: 160px; height: 160px">
          <DetectionView src={sceneSrc} width={640} height={480} boxes={sceneBoxes} classes={sceneClasses} fit="cover" scores={false} selected={1} />
        </div>
      </div>
    </div>
  );
}
