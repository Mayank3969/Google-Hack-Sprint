import { describe, expect, it, vi } from 'vitest';
import { liveAdapter } from './api';
import { parseProcess, type ProcessInput } from './process';
import { bilingual } from './testFixtures';
const config = { demo: false, backend: 'https://api.example.com', ml: 'https://ml.example.com' };
describe('process contract parsing', () => {
  it('normalizes legacy success to a one-segment playlist', () => {
    expect(parseProcess({ sentence: 'Hello.', audio_url: bilingual.audio_url })).toEqual({
      status: 'success', sentence: 'Hello.', audio_url: bilingual.audio_url,
      subtitles: [{ language: 'en-IN', text: 'Hello.' }],
      audio_segments: [{ language: 'en-IN', audio_url: bilingual.audio_url }], playback_order: ['en-IN'],
    });
  });
  it('retains labelled bilingual text and server playback order independent of array order', () => {
    expect(parseProcess(bilingual)).toEqual({ status: 'success', ...bilingual });
  });
  it('recognizes HTTP200 recovery without requiring success fields', () => {
    expect(parseProcess({ status: 'record_again', record_again: true })).toEqual({ status: 'record_again', record_again: true });
  });
  it.each([
    null, [], {}, { sentence: '', audio_url: bilingual.audio_url },
    { sentence: 'Hi' }, { sentence: 'Hi', audio_url: null },
    { sentence: 'Hi', audio_url: '/audio/a.mp3' },
    { sentence: 'Hi', audio_url: 'javascript:alert(1)' },
    { sentence: 'Hi', audio_url: 'https://user:secret@example.com/a.mp3' },
    { status: 'record_again', record_again: 'true' }, { record_again: true },
    { ...bilingual, status: 'record_again', record_again: true },
    { ...bilingual, hindi_sentence: undefined }, { ...bilingual, hindi_keywords: [] },
    { ...bilingual, hindi_keywords: [4] }, { ...bilingual, subtitles: [] },
    { ...bilingual, audio_segments: [bilingual.audio_segments[0]] },
    { ...bilingual, playback_order: ['en-IN', 'en-IN'] },
    { ...bilingual, playback_order: ['fr-FR', 'en-IN'] },
    { ...bilingual, audio_segments: [bilingual.audio_segments[0], bilingual.audio_segments[0]] },
    { ...bilingual, sentence: 'Inconsistent text' },
    { ...bilingual, audio_url: 'https://api.example.com/wrong.mp3' },
  ])('rejects incomplete, inconsistent or unsafe payload %#', value => { expect(() => parseProcess(value)).toThrow(); });
  it('passes structured capture results intact without client thresholding', async () => {
    const input: ProcessInput = { capture_id: 'capture-1', results: [
      { segment_index: 2, status: 'uncertain', keyword: null, confidence: 0.2, coverage: 0.4 },
      { segment_index: 0, status: 'recognized', keyword: 'hello', confidence: 0.7, coverage: 0.9, top_k: [{ label: 'hello', confidence: 0.7 }] },
    ] };
    const fetcher = vi.fn<typeof fetch>().mockResolvedValue(new Response(JSON.stringify(bilingual)));
    expect(await liveAdapter(config, fetcher).process(input)).toMatchObject({ status: 'success' });
    expect(JSON.parse(fetcher.mock.calls[0][1]!.body as string)).toEqual(input);
  });
});
describe('safe recoverable API errors', () => {
  it('does not echo malformed JSON response fragments', async () => {
    const fetcher = vi.fn<typeof fetch>().mockResolvedValue(new Response('PRIVATE_AUDIO invalid json'));
    await expect(liveAdapter(config, fetcher).transcribe('audio')).rejects.toThrow('Invalid JSON response from service');
  });
  it.each([400, 413, 422])('maps audio HTTP %i to recording guidance without private details', async status => {
    const fetcher = vi.fn<typeof fetch>().mockResolvedValue(new Response(JSON.stringify({ detail: [{ input: 'PRIVATE_AUDIO', msg: 'private' }] }), { status }));
    await expect(liveAdapter(config, fetcher).transcribe('private')).rejects.toThrow(/Record again/);
    await expect(liveAdapter(config, fetcher).transcribe('private')).rejects.not.toThrow('PRIVATE_AUDIO');
  });
  it('maps missing rooms and temporary provider errors', async () => {
    const fetcher = vi.fn<typeof fetch>().mockResolvedValueOnce(new Response('', { status: 404 })).mockResolvedValueOnce(new Response('', { status: 502 }));
    const api = liveAdapter(config, fetcher);
    await expect(api.messages('missing')).rejects.toThrow('Create a new room');
    await expect(api.process(['hello'])).rejects.toThrow('temporarily unavailable');
  });
  it('accepts an empty transcript as silence, not a malformed response', async () => {
    const fetcher = vi.fn<typeof fetch>().mockResolvedValue(new Response(JSON.stringify({ text: '' })));
    expect(await liveAdapter(config, fetcher).transcribe('audio')).toBe('');
  });
});
