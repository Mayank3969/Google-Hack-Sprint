// @vitest-environment jsdom
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { SignCapture } from './capture';
import { FaceLandmarker, HandLandmarker } from '@mediapipe/tasks-vision';
vi.mock('@mediapipe/tasks-vision', () => ({ FilesetResolver: { forVisionTasks: vi.fn(async () => ({})) }, HandLandmarker: { createFromOptions: vi.fn() }, FaceLandmarker: { createFromOptions: vi.fn() } }));
let video: HTMLVideoElement, stop: ReturnType<typeof vi.fn>, handClose: ReturnType<typeof vi.fn>, faceClose: ReturnType<typeof vi.fn>, getMedia: ReturnType<typeof vi.fn>;
beforeEach(() => {
  vi.useFakeTimers(); video = document.createElement('video');
  vi.spyOn(HTMLMediaElement.prototype, 'play').mockResolvedValue(); vi.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(() => {}); vi.spyOn(HTMLMediaElement.prototype, 'load').mockImplementation(() => {});
  Object.defineProperty(video, 'readyState', { value: 2 });
  stop = vi.fn(); getMedia = vi.fn(async () => ({ getTracks: () => [{ stop }] }));
  Object.defineProperty(navigator, 'mediaDevices', { configurable: true, value: { getUserMedia: getMedia } });
  handClose = vi.fn(); faceClose = vi.fn();
  vi.mocked(HandLandmarker.createFromOptions).mockResolvedValue({ close: handClose, detectForVideo: vi.fn(() => ({ landmarks: [Array.from({ length: 21 }, () => ({ x: 0.1, y: 0.2, z: -0.3 }))], handedness: [[{ categoryName: 'Left' }]] })) } as unknown as HandLandmarker);
  vi.mocked(FaceLandmarker.createFromOptions).mockResolvedValue({ close: faceClose, detectForVideo: vi.fn(() => ({ faceLandmarks: [] })) } as unknown as FaceLandmarker);
  vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) => setTimeout(() => callback(performance.now()), 10)); vi.stubGlobal('cancelAnimationFrame', clearTimeout);
  URL.createObjectURL = vi.fn(() => 'blob:local'); URL.revokeObjectURL = vi.fn();
});
afterEach(() => { vi.useRealTimers(); vi.restoreAllMocks(); vi.unstubAllGlobals(); vi.clearAllMocks(); });
it('samples at <=15fps, caps at60 and closes models/tracks before completion', async () => {
  const session = new SignCapture(video), done = vi.fn(), failed = vi.fn(); await session.prepare();
  expect(getMedia).toHaveBeenCalledWith({ video: { facingMode: 'user' }, audio: false });
  await session.start(done, failed);
  for (let i = 1; i <= 400; i++) { video.currentTime = i / 100; await vi.advanceTimersByTimeAsync(10); }
  expect(done).not.toHaveBeenCalled();
  for (let i = 401; i <= 450; i++) { video.currentTime = i / 100; await vi.advanceTimersByTimeAsync(10); }
  expect(done).toHaveBeenCalledTimes(1); expect(done.mock.calls[0][0].length).toBeLessThanOrEqual(60); expect(done.mock.calls[0][0].length).toBeGreaterThan(1);
  expect(stop).toHaveBeenCalledTimes(1); expect(handClose).toHaveBeenCalledTimes(1); expect(faceClose).toHaveBeenCalledTimes(1); expect(video.srcObject).toBeNull(); expect(failed).not.toHaveBeenCalled();
});
it('releases late camera permission after cancellation and never loads detectors', async () => {
  let resolve!: (stream: MediaStream) => void;
  getMedia.mockImplementation(() => new Promise(r => { resolve = r; }));
  const session = new SignCapture(video), preparation = session.prepare(); session.dispose();
  resolve({ getTracks: () => [{ stop }] } as unknown as MediaStream); await preparation;
  expect(stop).toHaveBeenCalledTimes(1); expect(HandLandmarker.createFromOptions).not.toHaveBeenCalled();
});
it('closes partial detector setup and local object URLs on model failure', async () => {
  vi.mocked(FaceLandmarker.createFromOptions).mockRejectedValueOnce(new Error('missing'));
  const session = new SignCapture(video);
  await expect(session.prepare(new File(['video'], 'sign.mp4'))).rejects.toThrow(/model\/WASM unavailable/);
  expect(handClose).toHaveBeenCalledTimes(1); expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob:local'); expect(video.getAttribute('src')).toBeNull(); expect(getMedia).not.toHaveBeenCalled();
});
it('ends frozen/empty media after the wall-clock bound and cancels pending work on dispose', async () => {
  const session = new SignCapture(video), done = vi.fn(); await session.prepare(); await session.start(done, vi.fn());
  await vi.advanceTimersByTimeAsync(4500); expect(done).toHaveBeenCalledTimes(1); expect(done.mock.calls[0][0]).toHaveLength(1);
  session.dispose(); await vi.advanceTimersByTimeAsync(5000); expect(done).toHaveBeenCalledTimes(1);
});
it('closes late detectors after canceled initialization', async () => {
  let resolve!: (model: HandLandmarker) => void;
  vi.mocked(HandLandmarker.createFromOptions).mockImplementationOnce(() => new Promise(r => { resolve = r; }));
  const session = new SignCapture(video), preparation = session.prepare();
  await vi.advanceTimersByTimeAsync(0); session.dispose();
  resolve({ close: handClose } as unknown as HandLandmarker); await preparation;
  expect(handClose).toHaveBeenCalledTimes(1); expect(FaceLandmarker.createFromOptions).not.toHaveBeenCalled(); expect(stop).toHaveBeenCalledTimes(1);
});
it('releases all resources on detector failure without publishing a partial capture', async () => {
  vi.mocked(FaceLandmarker.createFromOptions).mockResolvedValueOnce({ close: faceClose, detectForVideo: () => { throw new Error('detector failure'); } } as unknown as FaceLandmarker);
  const session = new SignCapture(video), done = vi.fn(), failed = vi.fn(); await session.prepare(); await session.start(done, failed);
  expect(done).not.toHaveBeenCalled(); expect(failed).toHaveBeenCalledTimes(1); expect(stop).toHaveBeenCalledTimes(1); expect(handClose).toHaveBeenCalledTimes(1); expect(faceClose).toHaveBeenCalledTimes(1);
});
