import { afterEach, expect, it, vi } from 'vitest';
import { Microphone } from './audio';
afterEach(() => { vi.unstubAllGlobals(); vi.useRealTimers(); });
it('releases a stream granted after the user cancels permission', async () => {
  let grant!: (stream: MediaStream) => void;
  const stop = vi.fn();
  vi.stubGlobal('navigator', { mediaDevices: { getUserMedia: () => new Promise<MediaStream>(resolve => { grant = resolve; }) } });
  vi.stubGlobal('window', { AudioContext: vi.fn() });
  const mic = new Microphone(); const pending = mic.start(vi.fn()); mic.dispose();
  grant({ getTracks: () => [{ stop }] } as unknown as MediaStream);
  await pending; expect(stop).toHaveBeenCalledTimes(1);
});
it('stops tracks and closes the context on worklet setup failure', async () => {
  const stop = vi.fn(), close = vi.fn().mockResolvedValue(undefined);
  class Context { sampleRate = 48000; state = 'running'; audioWorklet = { addModule: vi.fn().mockRejectedValue(new Error('Worklet unavailable')) }; close = close; }
  vi.stubGlobal('navigator', { mediaDevices: { getUserMedia: vi.fn().mockResolvedValue({ getTracks: () => [{ stop }] }) } });
  vi.stubGlobal('window', { AudioContext: Context }); vi.stubGlobal('AudioContext', Context);
  await expect(new Microphone().start(vi.fn())).rejects.toThrow('Worklet unavailable');
  expect(stop).toHaveBeenCalledTimes(1); expect(close).toHaveBeenCalledTimes(1);
});
it('caps WAV data at 10 seconds and releases devices before returning it', async () => {
  vi.useFakeTimers();
  const stop = vi.fn(), close = vi.fn().mockResolvedValue(undefined), disconnect = vi.fn();
  let worklet!: Worklet;
  class Worklet { port: { onmessage: null | ((event: { data: Float32Array }) => void) } = { onmessage: null }; connect = vi.fn(); disconnect = disconnect; constructor() { worklet = this; } }
  class Context { sampleRate = 16000; state = 'running'; destination = {}; audioWorklet = { addModule: vi.fn().mockResolvedValue(undefined) }; resume = vi.fn().mockResolvedValue(undefined); close = close; createMediaStreamSource() { return { connect: vi.fn(), disconnect }; } }
  vi.stubGlobal('navigator', { mediaDevices: { getUserMedia: vi.fn().mockResolvedValue({ getTracks: () => [{ stop }] }) } });
  vi.stubGlobal('window', { AudioContext: Context }); vi.stubGlobal('AudioContext', Context); vi.stubGlobal('AudioWorkletNode', Worklet);
  const mic = new Microphone(); const complete = vi.fn(); await mic.start(complete);
  worklet.port.onmessage?.({ data: new Float32Array(200000) });
  await vi.advanceTimersByTimeAsync(10000); expect(complete).toHaveBeenCalledTimes(1);
  const result = mic.finish(); expect(result.byteLength).toBe(44 + 16000 * 10 * 2);
  expect(stop).toHaveBeenCalledTimes(1); expect(close).toHaveBeenCalledTimes(1); expect(worklet.port.onmessage).toBeNull();
});
