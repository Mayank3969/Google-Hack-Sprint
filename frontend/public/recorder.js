class PCMRecorder extends AudioWorkletProcessor {
  constructor() { super(); this.remaining = Math.floor(sampleRate * 10); }
  process(inputs) {
    const channels = inputs[0];
    if (!channels?.length || this.remaining <= 0) return true;
    const count = Math.min(channels[0].length, this.remaining);
    const mono = new Float32Array(count);
    for (const channel of channels) for (let i = 0; i < count; i++) mono[i] += channel[i] / channels.length;
    this.port.postMessage(mono, [mono.buffer]);
    this.remaining -= count;
    if (this.remaining === 0) this.port.postMessage('complete');
    return true;
  }
}
registerProcessor('pcm-recorder', PCMRecorder);
