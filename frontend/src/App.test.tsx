// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { App } from './App';
import { bilingual } from './testFixtures';
import { Microphone } from './audio';

vi.mock('./audio', () => ({ Microphone: vi.fn(function () {
  return { start: vi.fn(async () => {}), finish: vi.fn(() => new ArrayBuffer(44)), dispose: vi.fn() };
}), toBase64: vi.fn(() => 'WAV') }));
const config = { demo: false, backend: 'https://api.example.com', ml: 'https://ml.example.com', siteUrl: 'https://signbridge.example' };
const json = (value: unknown, status = 200) => new Response(JSON.stringify(value), { status });
let fetcher: ReturnType<typeof vi.fn<typeof fetch>>;
let audios: HTMLAudioElement[];
let play: ReturnType<typeof vi.fn>, pause: ReturnType<typeof vi.fn>;
beforeEach(() => {
  history.replaceState({}, '', '/');
  fetcher = vi.fn<typeof fetch>(); vi.stubGlobal('fetch', fetcher);
  audios = []; play = vi.fn().mockResolvedValue(undefined); pause = vi.fn();
  vi.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(() => {});
  vi.spyOn(HTMLMediaElement.prototype, 'load').mockImplementation(() => {});
  vi.stubGlobal('Audio', vi.fn(function (src: string) {
    const audio = document.createElement('audio'); audio.src = src;
    audio.play = play; audio.pause = pause; audio.load = vi.fn(); audios.push(audio); return audio;
  }));
  vi.stubGlobal('speechSynthesis', { cancel: vi.fn(), getVoices: vi.fn(() => []), speak: vi.fn() });
  URL.createObjectURL = vi.fn(() => 'blob:recording'); URL.revokeObjectURL = vi.fn();
});
afterEach(() => { cleanup(); vi.unstubAllGlobals(); vi.restoreAllMocks(); vi.clearAllMocks(); });
function generate() {
  fireEvent.change(screen.getByLabelText('What would you like to say?'), { target: { value: 'hello, friend' } });
  fireEvent.click(screen.getByRole('button', { name: 'Generate sentence →' }));
}
describe('bilingual interaction', () => {
  it('plays a legacy English-only success and does not invent Hindi', async () => {
    fetcher.mockResolvedValue(json({ sentence: 'Legacy phrase.', audio_url: bilingual.audio_url }));
    render(<App config={config}/>); generate(); await screen.findByText('Legacy phrase.');
    expect(screen.queryByLabelText('Hindi subtitle')).toBeNull();
    fireEvent.click(screen.getByRole('button', { name: '▶ Play audio' }));
    expect(audios[0].src).toBe(bilingual.audio_url);
    act(() => audios[0].dispatchEvent(new Event('ended')));
    expect(audios).toHaveLength(1); expect(fetcher).toHaveBeenCalledTimes(1);
  });
  it('shows both languages, starts only on click, follows server order and replays without POST', async () => {
    fetcher.mockResolvedValue(json(bilingual));
    render(<App config={config}/>); generate();
    await screen.findByText('Hello, friend.');
    expect(screen.getByText('नमस्ते दोस्त।').lang).toBe('hi-IN');
    expect(audios).toHaveLength(0);
    fireEvent.click(screen.getByRole('button', { name: '▶ Play audio' }));
    expect(audios[0].src).toBe(bilingual.audio_segments[1].audio_url);
    expect(screen.getByText('Playing Hindi')).toBeTruthy();
    act(() => { audios[0].dispatchEvent(new Event('ended')); });
    expect(audios[1].src).toBe(bilingual.audio_url);
    expect(audios[0].getAttribute('src')).toBeNull();
    act(() => { audios[1].dispatchEvent(new Event('ended')); });
    fireEvent.click(screen.getByRole('button', { name: '▶ Play audio' }));
    expect(audios).toHaveLength(3); expect(fetcher).toHaveBeenCalledTimes(1);
    fireEvent.click(screen.getByRole('button', { name: 'Use Hindi as signer draft' }));
    expect((screen.getByLabelText('Your message') as HTMLTextAreaElement).value).toBe(bilingual.hindi_sentence);
  });
  it('discards old audio/subtitles on request and record-again, preserves input, and focuses re-entry', async () => {
    let resolve!: (response: Response) => void;
    fetcher.mockResolvedValueOnce(json(bilingual)).mockImplementationOnce(() => new Promise(r => { resolve = r; }));
    render(<App config={config}/>); generate(); await screen.findByText('Hello, friend.');
    fireEvent.change(screen.getByLabelText('Your message'), { target: { value: 'Keep my draft' } });
    fireEvent.click(screen.getByRole('button', { name: '▶ Play audio' }));
    generate();
    expect(screen.queryByText('Hello, friend.')).toBeNull();
    expect(pause).toHaveBeenCalled(); expect(audios[0].getAttribute('src')).toBeNull();
    act(() => audios[0].dispatchEvent(new Event('ended'))); expect(audios).toHaveLength(1);
    await act(async () => resolve(json({ status: 'record_again', record_again: true })));
    expect(screen.getByText(/Recapture signs or re-enter/)).toBeTruthy();
    expect(screen.queryByRole('button', { name: '▶ Play audio' })).toBeNull();
    fireEvent.click(screen.getByRole('button', { name: 'Re-enter keywords' }));
    expect(document.activeElement).toBe(screen.getByLabelText('What would you like to say?'));
    expect((document.activeElement as HTMLInputElement).value).toBe('hello, friend');
    expect((screen.getByLabelText('Your message') as HTMLTextAreaElement).value).toBe('Keep my draft');
    expect(fetcher).toHaveBeenCalledTimes(2);
  });
  it('ignores a late result after navigation and cleans up playlist listeners on unmount', async () => {
    let resolve!: (response: Response) => void;
    fetcher.mockImplementationOnce(() => new Promise(r => { resolve = r; })).mockResolvedValueOnce(json(bilingual));
    const view = render(<App config={config}/>); generate();
    fireEvent.click(screen.getByRole('link', { name: 'Settings & Help' }));
    await act(async () => resolve(json(bilingual)));
    fireEvent.click(screen.getByRole('link', { name: 'Workspace' }));
    expect(screen.queryByText('Hello, friend.')).toBeNull();
    generate(); await screen.findByText('Hello, friend.');
    fireEvent.click(screen.getByRole('button', { name: '▶ Play audio' }));
    view.unmount(); act(() => audios[0].dispatchEvent(new Event('ended')));
    expect(audios).toHaveLength(1); expect(audios[0].getAttribute('src')).toBeNull();
  });
  it('retains text after playback rejection and exposes replay without regeneration', async () => {
    fetcher.mockResolvedValue(json(bilingual)); play.mockRejectedValueOnce(new Error('blocked'));
    render(<App config={config}/>); generate(); await screen.findByText('Hello, friend.');
    fireEvent.click(screen.getByRole('button', { name: '▶ Play audio' }));
    await screen.findByText(/Generated audio could not play/);
    expect(screen.getByText('Hello, friend.')).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: '▶ Play audio' }));
    expect(fetcher).toHaveBeenCalledTimes(1);
  });
  it('stops the playlist on a media error instead of skipping to another language', async () => {
    fetcher.mockResolvedValue(json(bilingual));
    render(<App config={config}/>); generate(); await screen.findByText('Hello, friend.');
    fireEvent.click(screen.getByRole('button', { name: '▶ Play audio' }));
    act(() => audios[0].dispatchEvent(new Event('error')));
    expect(screen.getByText(/Generated audio could not play/)).toBeTruthy();
    act(() => audios[0].dispatchEvent(new Event('ended')));
    expect(audios).toHaveLength(1); expect(audios[0].getAttribute('src')).toBeNull();
  });
  it('keeps inputs when process fails or returns malformed output', async () => {
    fetcher.mockResolvedValueOnce(json({ detail: { input: 'PRIVATE' } }, 502)).mockResolvedValueOnce(json({ ...bilingual, playback_order: [] }));
    render(<App config={config}/>);
    fireEvent.change(screen.getByLabelText('Your message'), { target: { value: 'Keep draft' } });
    generate(); await screen.findByText(/temporarily unavailable/);
    expect(screen.queryByText(/PRIVATE/)).toBeNull();
    generate(); await screen.findByText(/Invalid sentence response/);
    expect((screen.getByLabelText('Your message') as HTMLTextAreaElement).value).toBe('Keep draft');
    expect((screen.getByLabelText('What would you like to say?') as HTMLInputElement).value).toBe('hello, friend');
  });
});
describe('recording recovery and rooms', () => {
  it.each([200, 400, 413, 422])('preserves a draft after silence or audio error %i', async status => {
    fetcher.mockResolvedValue(json(status === 200 ? { text: '  ' } : { detail: [{ input: 'PRIVATE_AUDIO' }] }, status));
    render(<App config={config}/>);
    fireEvent.change(screen.getByLabelText('Your message'), { target: { value: 'Existing draft' } });
    fireEvent.click(screen.getByRole('button', { name: 'Speech → text' }));
    fireEvent.click(screen.getByRole('button', { name: 'Start microphone' }));
    fireEvent.click(await screen.findByRole('button', { name: 'Stop recording' }));
    await screen.findByText(status === 200 ? /No speech detected/ : /Record again/);
    expect((screen.getByLabelText('Your message') as HTMLTextAreaElement).value).toBe('Existing draft');
    expect(screen.queryByText(/PRIVATE_AUDIO/)).toBeNull();
    expect(Microphone).toHaveBeenCalled();
    expect(fetcher).toHaveBeenCalledTimes(1);
  });
  it('offers existing create/join actions for missing rooms and retains failed-message draft', async () => {
    fetcher.mockImplementation(async (url, init) => {
      if (String(url).endsWith('/create')) return json({ room_id: 'room' });
      if (init?.method === 'POST') return json({ detail: 'missing' }, 404);
      return json({ messages: [] });
    });
    render(<App config={config}/>);
    fireEvent.click(screen.getByRole('button', { name: '＋ Create room' }));
    await screen.findByText('Room created. Share the room ID to connect.');
    fireEvent.change(screen.getByLabelText('Your message'), { target: { value: 'Unsent draft' } });
    fireEvent.click(screen.getByRole('button', { name: 'Send message →' }));
    await screen.findByText(/Room not found/);
    expect((screen.getByLabelText('Your message') as HTMLTextAreaElement).value).toBe('Unsent draft');
    expect(screen.getByRole('button', { name: '＋ Create room' })).toBeTruthy();
  });
  it('keeps demo honest and usable without backend calls', async () => {
    render(<App config={{ ...config, demo: true }}/>); generate();
    await screen.findByText('hello friend.');
    expect(screen.getByText('DEMO PHRASE')).toBeTruthy();
    expect(screen.getByRole('button', { name: '▶ Play with browser voice' })).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: '＋ Create room' }));
    await screen.findByText('Room created. Share the room ID to connect.');
    await waitFor(() => expect(fetcher).not.toHaveBeenCalled());
  });
});
