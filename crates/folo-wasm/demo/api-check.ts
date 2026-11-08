// Type-checks the documented JS API against the generated declarations (docs/WASM.md):
//   sh scripts/build-wasm.sh && web/node_modules/.bin/tsc -p crates/folo-wasm/demo/tsconfig.json
import init, { FoloDetector, hasWebGpu, version, type FoloDetectResult, type FoloDetectorInfo } from "../../../web/public/folo-wasm/folo_wasm.js";

export async function check(bytes: Uint8Array, buffer: ArrayBuffer, img: ImageData, rgba: Uint8ClampedArray): Promise<number> {
  await init();
  const a: FoloDetector = await FoloDetector.load(bytes);
  const b = await FoloDetector.load(buffer, { backend: "cpu", modelId: "m_1", imgsz: 320, precision: "f16" });
  const r1: FoloDetectResult = await a.detect(img);
  const r2 = await b.detect(rgba, 64, 48, { conf: 0.5, iou: 0.6, maxDet: 10 });
  const r3 = await a.detect(img, undefined, undefined, { conf: 0.3 });
  const info: FoloDetectorInfo = a.info();
  const backend: "webgpu" | "cpu" = r1.backend;
  const precision: "f32" | "f16" = info.precision;
  const why: string | null = info.precisionFallback;
  a.free();
  b.free();
  return r1.boxes.length + r2.timing.total_ms + r3.image.width + info.classes.length + backend.length
    + precision.length + (why ?? "").length + (hasWebGpu() ? 1 : 0) + version().length + (a.backend === "cpu" ? 0 : 1);
}
