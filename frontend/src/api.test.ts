import { describe, expect, it, vi } from 'vitest';
import { demoAdapter, liveAdapter, validateKeywords } from './api';
const config = { demo: false, backend: 'https://api.example.com', ml: 'https://ml.example.com' };
const response = (value: unknown) => new Response(JSON.stringify(value), { status: 200 });
describe('backend contract', () => {
  it('creates without a request body and encodes room identifiers', async () => {
    const fetcher = vi.fn<typeof fetch>().mockResolvedValueOnce(response({ room_id: 'abc' })).mockResolvedValueOnce(response({ messages: [] })).mockResolvedValueOnce(response({ success: true }));
    const api = liveAdapter(config, fetcher); expect(await api.create()).toBe('abc');
    expect(fetcher.mock.calls[0][1]).toMatchObject({ method: 'POST' }); expect(fetcher.mock.calls[0][1]?.body).toBeUndefined();
    await api.messages('a/b'); expect(fetcher.mock.calls[1][0]).toBe('https://api.example.com/session/a%2Fb/messages');
    await api.send('abc', 'signer', 'hello'); expect(JSON.parse(fetcher.mock.calls[2][1]?.body as string)).toEqual({ sender: 'signer', text: 'hello' });
  });
  it('uses backend for process/transcribe and explicit ML base for health', async () => {
    const fetcher = vi.fn<typeof fetch>().mockResolvedValueOnce(response({ sentence: 'Hello.', audio_url: 'https://api.example.com/audio/a.wav' })).mockResolvedValueOnce(response({ text: 'Hi' })).mockResolvedValueOnce(response({ status: 'ok' }));
    const api = liveAdapter(config, fetcher);
    expect(await api.process(['hello'])).toMatchObject({ status: 'success', audio_url: 'https://api.example.com/audio/a.wav' });
    expect(await api.transcribe('UklGRg==')).toBe('Hi');
    expect(JSON.parse(fetcher.mock.calls[1][1]?.body as string)).toEqual({ audio: 'UklGRg==' });
    await api.health('ml'); expect(fetcher.mock.calls[2][0]).toBe('https://ml.example.com/health');
  });
  it('validates inputs and rejects service errors or malformed output', async () => {
    const fetcher = vi.fn<typeof fetch>().mockResolvedValueOnce(new Response('', { status: 503 })).mockResolvedValueOnce(response({ messages: [{ sender: 'unknown' }] })).mockResolvedValueOnce(response({ sentence: 'Hi', audio_url: 'javascript:alert(1)' }));
    const api = liveAdapter(config, fetcher);
    await expect(api.send('a', 'speaker', 'x'.repeat(2001))).rejects.toThrow('2,000');
    expect(fetcher).not.toHaveBeenCalled();
    await expect(api.create()).rejects.toThrow('503'); await expect(api.messages('a')).rejects.toThrow('Invalid'); await expect(api.process(['hello'])).rejects.toThrow('audio URL');
    for (const words of [[], ['a'.repeat(65)], Array(21).fill('a')]) expect(() => validateKeywords(words)).toThrow();
  });
  it('keeps demo isolated, snapshot-based and honest about transcription', async () => {
    const demo = demoAdapter(); const room = await demo.create();
    await demo.send(room, 'speaker', 'Hello'); expect(await demo.messages(room)).toHaveLength(1);
    const snapshot = await demo.messages(room); snapshot.length = 0; expect(await demo.messages(room)).toHaveLength(1);
    expect(await demo.process(['good', 'morning'])).toMatchObject({ status: 'success', sentence: 'good morning.', audio_url: null });
    await expect(demo.transcribe('audio')).rejects.toThrow('unavailable'); await expect(demo.messages('missing')).rejects.toThrow('not found');
  });
});
