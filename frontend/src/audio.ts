// Windowed-sinc low-pass interpolation removes frequencies above the target
// Nyquist limit before downsampling (rather than re-labelling recorded WebM).
export function resample(input: Float32Array, fromRate: number, toRate = 16000): Float32Array {
  if (fromRate <= 0 || toRate <= 0) throw new Error('Invalid sample rate.');
  if (fromRate === toRate) return input.slice();
  const output = new Float32Array(Math.floor(input.length * toRate / fromRate));
  const cutoff = Math.min(1, toRate / fromRate) * 0.94;
  const radius = 32;
  for (let i = 0; i < output.length; i++) {
    const position = i * fromRate / toRate;
    let sum = 0, weight = 0;
    for (let j = Math.max(0, Math.ceil(position - radius)); j <= Math.min(input.length - 1, Math.floor(position + radius)); j++) {
      const d = position - j;
      const x = Math.PI * d * cutoff;
      const w = cutoff * (Math.abs(x) < 1e-8 ? 1 : Math.sin(x) / x) * (0.5 + 0.5 * Math.cos(Math.PI * d / radius));
      sum += input[j] * w; weight += w;
    }
    output[i] = weight ? sum / weight : 0;
  }
  return output;
}
export function encodeWav(samples: Float32Array): ArrayBuffer {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const text = (offset: number, value: string) => [...value].forEach((c, i) => view.setUint8(offset + i, c.charCodeAt(0)));
  text(0, 'RIFF'); view.setUint32(4, buffer.byteLength - 8, true); text(8, 'WAVE'); text(12, 'fmt ');
  view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
  view.setUint32(24, 16000, true); view.setUint32(28, 32000, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true);
  text(36, 'data'); view.setUint32(40, samples.length * 2, true);
  samples.forEach((sample, i) => { const n = Math.max(-1, Math.min(1, sample)); view.setInt16(44 + i * 2, Math.round(n * (n < 0 ? 32768 : 32767)), true); });
  return buffer;
}
export function toBase64(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer);
  let binary = '';
  for (let i = 0; i < bytes.length; i += 8192) binary += String.fromCharCode(...bytes.subarray(i, i + 8192));
  return btoa(binary);
}
export class Microphone {
  private stream?: MediaStream;
  private context?: AudioContext;
  private node?: AudioWorkletNode;
  private source?: MediaStreamAudioSourceNode;
  private timer?: ReturnType<typeof setTimeout>;
  private chunks: Float32Array[] = [];
  private cancelled = false;
  private rate = 16000;
  async start(onComplete: () => void) {
    if (!navigator.mediaDevices?.getUserMedia || !window.AudioContext) throw new Error('Recording requires a secure HTTPS context and a browser with Web Audio support.');
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true }, video: false });
      if (this.cancelled) { this.dispose(); return; }
      this.context = new AudioContext();
      this.rate = this.context.sampleRate;
      await this.context.audioWorklet.addModule(`${import.meta.env.BASE_URL}recorder.js`);
      if (this.cancelled) { this.dispose(); return; }
      await this.context.resume();
      if (this.cancelled) { this.dispose(); return; }
      this.source = this.context.createMediaStreamSource(this.stream);
      this.node = new AudioWorkletNode(this.context, 'pcm-recorder');
      this.node.port.onmessage = ({ data }: MessageEvent<Float32Array | string>) => {
        if (this.cancelled) return;
        if (data === 'complete') onComplete();
        else if (data instanceof Float32Array) this.chunks.push(data);
      };
      this.source.connect(this.node);
      // Worklet outputs silence; connecting it keeps processing active without feedback.
      this.node.connect(this.context.destination);
      this.timer = setTimeout(onComplete, 10000);
    } catch (error) { this.dispose(); throw error; }
  }
  finish(): ArrayBuffer {
    const count = Math.min(this.chunks.reduce((n, chunk) => n + chunk.length, 0), this.rate * 10);
    const samples = new Float32Array(count);
    let offset = 0;
    for (const chunk of this.chunks) { const part = chunk.subarray(0, count - offset); samples.set(part, offset); offset += part.length; }
    this.dispose();
    if (!count) throw new Error('No audio was captured. Check microphone permissions and try again.');
    return encodeWav(resample(samples, this.rate));
  }
  dispose() {
    this.cancelled = true;
    clearTimeout(this.timer);
    this.stream?.getTracks().forEach(track => track.stop());
    this.source?.disconnect(); this.node?.disconnect();
    if (this.node) this.node.port.onmessage = null;
    if (this.context && this.context.state !== 'closed') void this.context.close().catch(() => {});
    this.chunks = [];
  }
}
