// Type-checks the documented JS API against the generated declarations (docs/WASM.md):
//   sh scripts/build-wasm.sh && web/node_modules/.bin/tsc -p crates/folo-wasm/demo/tsconfig.json
import init, {
  FoloDetector,
  hasWebGpu,
  version,
  type FoloDetectResult,
  type FoloDetectorInfo,
  type FoloTileDetectOptions,
  type FoloTtaDetectOptions,
} from "../../../web/public/folo-wasm/folo_wasm.js";

export async function check(bytes: Uint8Array, buffer: ArrayBuffer, img: ImageData, rgba: Uint8ClampedArray): Promise<number> {
  await init();
  const a: FoloDetector = await FoloDetector.load(bytes);
  const b = await FoloDetector.load(buffer, { backend: "cpu", modelId: "m_1", imgsz: 320, precision: "f16" });
  const r1: FoloDetectResult = await a.detect(img);
  const r2 = await b.detect(rgba, 64, 48, { conf: 0.5, iou: 0.6, maxDet: 10 });
  const r3 = await a.detect(img, undefined, undefined, { conf: 0.3 });
  const tileOpts: FoloTileDetectOptions = {
    tileSize: 320,
    overlap: 0.2,
    fullImage: true,
    tileBatch: 4,
    mergeMetric: "ios",
    mergeThreshold: 0.5,
    mergeFusion: "vote",
    classAgnostic: false,
    conf: 0.25,
    iou: 0.7,
    maxDet: 100,
  };
  const r4: FoloDetectResult = await a.detectTiled(img, undefined, undefined, tileOpts);
  const r5: FoloDetectResult = await b.detectTiled(rgba, 64, 48, { tileWidth: 32, tileHeight: 32, mergeIou: 0.5 });
  const ttaOpts: FoloTtaDetectOptions = {
    hflip: true,
    scales: [0.83, 1.0, 1.17],
    mergeIou: 0.6,
    mergeFusion: "sebf",
    classAgnostic: false,
    conf: 0.25,
    iou: 0.7,
    maxDet: 100,
  };
  const r6: FoloDetectResult = await a.detectTta(img, undefined, undefined, ttaOpts);
  const r7: FoloDetectResult = await b.detectTta(rgba, 64, 48);
  const info: FoloDetectorInfo = a.info();
  const backend: "webgpu" | "cpu" = r1.backend;
  const precision: "f32" | "f16" = info.precision;
  const why: string | null = info.precisionFallback;
  a.free();
  b.free();
  return r1.boxes.length + r2.timing.total_ms + r3.image.width + r4.boxes.length + r5.boxes.length
    + r6.boxes.length + r7.boxes.length + info.classes.length + backend.length
    + precision.length + (why ?? "").length + (hasWebGpu() ? 1 : 0) + version().length + (a.backend === "cpu" ? 0 : 1);
}
