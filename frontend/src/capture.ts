import { FaceLandmarker, FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';
import { featureFrame, type Frame } from './landmarks';

// The WASM must come from this exact npm version; all assets are local.
export class SignCapture {
  private hand?: HandLandmarker;
  private face?: FaceLandmarker;
  private stream?: MediaStream;
  private url?: string;
  private disposed = false;
  private raf = 0;
  private timer?: ReturnType<typeof setTimeout>;
  private frames: Frame[] = [];
  private finish?: () => void;
  constructor(private video: HTMLVideoElement) {}
  async prepare(file?: File) {
    let timeout: ReturnType<typeof setTimeout> | undefined;
    try {
      await Promise.race([this.initialize(file), new Promise<never>((_, reject) => {
        timeout = setTimeout(() => reject(new Error('Camera/video or model setup timed out. Check permissions and local assets, then enable again.')), 20000);
      })]);
    } catch (e) { this.dispose(); throw e; }
    finally { clearTimeout(timeout); }
  }
  private async initialize(file?: File) {
    try {
      if (file) { this.url = URL.createObjectURL(file); this.video.src = this.url; }
      else {
        if (!navigator.mediaDevices?.getUserMedia) throw new Error('Camera needs HTTPS or localhost and a supported browser.');
        const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' }, audio: false });
        if (this.disposed) { stream.getTracks().forEach(track => track.stop()); return; }
        this.stream = stream; this.video.srcObject = stream;
      }
      await this.video.play();
      if (this.disposed) return;
      const base = import.meta.env.BASE_URL;
      try {
        const files = await FilesetResolver.forVisionTasks(`${base}mediapipe/wasm`);
        if (this.disposed) return;
        const hand = await HandLandmarker.createFromOptions(files, { baseOptions: { modelAssetPath: `${base}models/hand_landmarker.task` }, runningMode: 'VIDEO', numHands: 2, minHandDetectionConfidence: 0.5, minTrackingConfidence: 0.5 });
        if (this.disposed) { hand.close(); return; } this.hand = hand;
        const face = await FaceLandmarker.createFromOptions(files, { baseOptions: { modelAssetPath: `${base}models/face_landmarker.task` }, runningMode: 'VIDEO', numFaces: 1 });
        if (this.disposed) { face.close(); return; } this.face = face;
      } catch { throw new Error('Vision model/WASM unavailable. Install the pinned local assets in frontend/README.md, check asset CORS, then enable again.'); }
      if (file) { this.video.pause(); this.video.currentTime = 0; }
    } catch (e) { this.dispose(); throw e; }
  }
  async start(done: (frames: Frame[]) => void, failed: (error: unknown) => void) {
    if (this.disposed || !this.hand || !this.face) throw new Error('Enable a camera or local video first.');
    let lastTime = -Infinity, timestamp = -1;
    this.frames = [];
    this.finish = () => { const frames = this.frames; this.dispose(); done(frames); };
    const tick = () => {
      if (this.disposed) return;
      try {
        const time = this.video.currentTime;
        if (this.video.readyState >= 2 && time - lastTime >= 1 / 15) {
          lastTime = time; timestamp = Math.max(timestamp + 1, Math.round(time * 1000));
          this.frames.push(featureFrame(this.hand!.detectForVideo(this.video, timestamp), this.face!.detectForVideo(this.video, timestamp)));
        }
        if (this.frames.length >= 60 || this.video.ended) { this.stop(); return; }
        this.raf = requestAnimationFrame(tick);
      } catch (e) { this.dispose(); failed(e); }
    };
    try { await this.video.play(); if (!this.disposed) { this.timer = setTimeout(() => this.stop(), 4500); tick(); } }
    catch (e) { this.dispose(); failed(e); }
  }
  stop() { this.finish?.(); }
  dispose() {
    this.disposed = true; this.finish = undefined;
    cancelAnimationFrame(this.raf); clearTimeout(this.timer);
    this.stream?.getTracks().forEach(track => track.stop()); this.stream = undefined;
    this.hand?.close(); this.hand = undefined; this.face?.close(); this.face = undefined;
    this.video.pause(); this.video.srcObject = null; this.video.removeAttribute('src'); this.video.load();
    if (this.url) URL.revokeObjectURL(this.url); this.url = undefined;
  }
}
