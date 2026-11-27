import { describe, expect, it, vi } from 'vitest';
import {
  clampMediaTime,
  createFpsTracker,
  createFrameThrottle,
  DEFAULT_FRAME_STEP_SEC,
  describeCameraError,
  formatMediaTime,
  frameIndexAt,
  isVideoFile,
  stepFrameTime,
  stopMediaStream,
  totalFramesFor,
  videoFrameName,
} from './media';

describe('Playground media helpers', () => {
  it('detects local video files by MIME type or extension', () => {
    expect(isVideoFile({ name: 'clip.mp4', type: 'video/mp4' })).toBe(true);
    expect(isVideoFile({ name: 'clip.webm', type: '' })).toBe(true);
    expect(isVideoFile({ name: 'clip.MOV', type: '' })).toBe(true);
    expect(isVideoFile({ name: 'photo.jpg', type: 'image/jpeg' })).toBe(false);
    expect(isVideoFile({ name: 'weights.folo', type: '' })).toBe(false);
  });

  it('clamps, steps, and formats timestamps and frame indices deterministically', () => {
    expect(clampMediaTime(-2, 10)).toBe(0);
    expect(clampMediaTime(15, 10)).toBe(10);
    expect(clampMediaTime(NaN, 10)).toBe(0);
    expect(clampMediaTime(3, NaN)).toBe(0);

    // 30 fps stepping snaps to frame boundary + small sub-frame PTS nudge.
    const next = stepFrameTime(1.0, 1, 10, DEFAULT_FRAME_STEP_SEC);
    expect(next).toBeGreaterThan(1.033);
    expect(next).toBeLessThan(1.04);
    expect(stepFrameTime(0.01, -1, 10, DEFAULT_FRAME_STEP_SEC)).toBe(0);
    // Rounded 1 ms WebM timestamp at 30 fps (0.033 s is frame 2): +1f advances to frame 3 (~0.0687 s), never repeats frame 2.
    expect(frameIndexAt(0.033, DEFAULT_FRAME_STEP_SEC, 300)).toBe(2);
    const from33ms = stepFrameTime(0.033, 1, 10, DEFAULT_FRAME_STEP_SEC);
    expect(from33ms).toBeGreaterThan(0.067);
    expect(from33ms).toBeLessThan(0.075);
    expect(stepFrameTime(0.033, -1, 10, DEFAULT_FRAME_STEP_SEC)).toBe(0);
    // Already inside the last frame: +1f never moves backward.
    expect(stepFrameTime(9.99, 1, 10, DEFAULT_FRAME_STEP_SEC)).toBe(9.99);

    // 10 fps clip (1.1 s, 11 frames at 0.0..1.0): every +1f / -1f moves directly to the adjacent frame.
    const f1 = stepFrameTime(0, 1, 1.1, 0.1);
    expect(f1).toBeCloseTo(0.102, 3);
    const f2 = stepFrameTime(f1, 1, 1.1, 0.1);
    expect(f2).toBeCloseTo(0.202, 3);
    // Stepping back from mid-frame (0.165 s, inside frame 1) lands cleanly on frame 0.
    expect(stepFrameTime(0.165, -1, 1.1, 0.1)).toBe(0);
    // Stepping back from the clip end (1.1 s) lands on frame 9 (not frame 10 again).
    expect(stepFrameTime(1.1, -1, 1.1, 0.1)).toBeCloseTo(0.902, 3);

    expect(formatMediaTime(0)).toBe('0:00.0');
    expect(formatMediaTime(65.34)).toBe('1:05.3');
    expect(formatMediaTime(-5)).toBe('0:00.0');
    expect(formatMediaTime(NaN)).toBe('0:00.0');

    expect(frameIndexAt(0, 0.1)).toBe(1);
    expect(frameIndexAt(0.25, 0.1)).toBe(3);
    expect(frameIndexAt(1.1, 0.1, 11)).toBe(11);
    expect(totalFramesFor(3, 0.1)).toBe(30);
    expect(totalFramesFor(1.1, 0.1)).toBe(11);
    expect(totalFramesFor(0)).toBe(0);

    expect(videoFrameName('traffic.mp4', 4.25)).toBe('traffic@4.25s.jpg');
    expect(videoFrameName('clip', -1)).toBe('clip@0.00s.jpg');
  });

  it('coalesces rapid scrub events and enforces single-flight frame dispatch without stale queues', () => {
    vi.useFakeTimers();
    try {
      let now = 1000;
      const fired: number[] = [];
      const throttle = createFrameThrottle({
        minIntervalMs: 120,
        now: () => now,
        onTrigger: () => {
          fired.push(now);
          throttle.beginFlight();
        },
      });

      // First scrub triggers immediately.
      throttle.schedule(false);
      expect(fired).toEqual([1000]);
      expect(throttle.inFlight()).toBe(true);

      // Ten rapid scrubs while inference is in flight collapse into ONE pending trailing trigger.
      for (let i = 0; i < 10; i++) {
        now += 10;
        throttle.schedule(false);
      }
      expect(fired).toEqual([1000]);

      // Inference settles at 1050ms (< 120ms since 1000ms): schedules trailing timer for remaining 70ms.
      now = 1050;
      throttle.endFlight();
      expect(fired).toEqual([1000]);

      now = 1120;
      vi.advanceTimersByTime(70);
      expect(fired).toEqual([1000, 1120]);

      // Cancel clears flight and any pending work cleanly.
      throttle.schedule(false);
      throttle.cancel();
      expect(throttle.inFlight()).toBe(false);
      now = 1500;
      vi.advanceTimersByTime(500);
      expect(fired).toEqual([1000, 1120]);
    } finally {
      vi.useRealTimers();
    }
  });

  it('tracks rolling FPS and smoothed inference latency', () => {
    const tracker = createFpsTracker(4);
    expect(tracker.snapshot()).toEqual({ fps: 0, latencyMs: 0 });
    // First frame estimates FPS from latency.
    expect(tracker.record(50, 1000)).toEqual({ fps: 20, latencyMs: 50 });
    // Subsequent frames measure wall-clock throughput over the rolling window.
    expect(tracker.record(40, 1100)).toEqual({ fps: 10, latencyMs: 45 });
    expect(tracker.record(60, 1200)).toEqual({ fps: 10, latencyMs: 50 });
    tracker.reset();
    expect(tracker.snapshot()).toEqual({ fps: 0, latencyMs: 0 });
  });

  it('stops every MediaStream track safely and maps camera errors cleanly', () => {
    const stopA = vi.fn();
    const stopB = vi.fn(() => {
      throw new Error('already ended');
    });
    const stream = { getTracks: () => [{ stop: stopA }, { stop: stopB }] } as unknown as MediaStream;
    stopMediaStream(stream);
    expect(stopA).toHaveBeenCalledTimes(1);
    expect(stopB).toHaveBeenCalledTimes(1);
    expect(() => stopMediaStream(null)).not.toThrow();

    expect(describeCameraError(null)).toContain('unavailable');
    const denied = Object.assign(new Error('denied'), { name: 'NotAllowedError' });
    expect(describeCameraError(denied)).toContain('permission was denied');
    const missing = Object.assign(new Error('none'), { name: 'NotFoundError' });
    expect(describeCameraError(missing)).toContain('No camera was found');
    const busy = Object.assign(new Error('in use'), { name: 'NotReadableError' });
    expect(describeCameraError(busy)).toContain('already in use');
  });
});
