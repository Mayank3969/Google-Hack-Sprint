import { describe, expect, it } from 'vitest';
import { encodeWav, resample, toBase64 } from './audio';
describe('real PCM audio conversion', () => {
  it('resamples 48kHz and 44.1kHz to the correct 16kHz duration', () => {
    for (const rate of [48000, 44100]) {
      const result = resample(new Float32Array(rate).fill(0.5), rate);
      expect(result.length).toBe(16000);
      expect(result[8000]).toBeCloseTo(0.5, 5);
    }
  });
  it('attenuates above-Nyquist content before decimation', () => {
    const tone = (frequency: number) => Float32Array.from({ length: 4800 }, (_, i) => Math.sin(2 * Math.PI * frequency * i / 48000));
    const rms = (samples: Float32Array) => Math.sqrt(samples.slice(32, -32).reduce((n, x) => n + x * x, 0) / (samples.length - 64));
    expect(rms(resample(tone(12000), 48000))).toBeLessThan(0.03);
    expect(rms(resample(tone(1000), 48000))).toBeGreaterThan(0.65);
  });
  it('writes actual mono PCM16 WAV headers, saturates samples, and emits plain base64', () => {
    const wav = encodeWav(new Float32Array([-2, 0, 2])); const view = new DataView(wav);
    expect(new TextDecoder().decode(wav.slice(0, 4))).toBe('RIFF');
    expect(new TextDecoder().decode(wav.slice(8, 12))).toBe('WAVE');
    expect(view.getUint16(20, true)).toBe(1); expect(view.getUint16(22, true)).toBe(1);
    expect(view.getUint32(24, true)).toBe(16000); expect(view.getUint16(34, true)).toBe(16);
    expect(view.getUint32(40, true)).toBe(6); expect(view.getInt16(44, true)).toBe(-32768); expect(view.getInt16(48, true)).toBe(32767);
    expect(toBase64(wav)).not.toContain('data:'); expect(atob(toBase64(wav)).slice(0, 4)).toBe('RIFF');
  });
});
