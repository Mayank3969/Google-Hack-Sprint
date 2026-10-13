import type { Config } from './config';
import { validateLandmarks, type Frame } from './landmarks';
import { parseProcess, type ProcessInput, type ProcessResult } from './process';
export type { Speech, ProcessInput, ProcessResult } from './process';
export type Sender = 'signer' | 'speaker';
export interface Message { sender: Sender; text: string; timestamp: string }
export interface Adapter {
  predict(landmarks: Frame[], signal?: AbortSignal): Promise<string[]>;
  create(): Promise<string>;
  messages(room: string, signal?: AbortSignal): Promise<Message[]>;
  send(room: string, sender: Sender, text: string): Promise<void>;
  process(input: string[] | ProcessInput): Promise<ProcessResult>;
  transcribe(audio: string): Promise<string>;
  health(service: 'backend' | 'ml'): Promise<string>;
}
export function validateKeywords(words: string[]) {
  if (!words.length || words.length > 20 || words.some(w => !w.trim() || w.length > 64)) throw new Error('Enter 1–20 comma-separated keywords, each at most 64 characters.');
}
function validateText(text: string) { if (!text.trim() || text.length > 2000) throw new Error('A message must contain 1–2,000 characters.'); }
const object = (v: unknown): Record<string, unknown> => {
  if (!v || typeof v !== 'object' || Array.isArray(v)) throw new Error('Invalid response from service.');
  return v as Record<string, unknown>;
};
export function liveAdapter(config: Config, fetcher: typeof fetch = fetch): Adapter {
  async function request(path: string, body?: unknown, signal?: AbortSignal, base = config.backend, post = false): Promise<unknown> {
    const controller = new AbortController();
    const abort = () => controller.abort();
    signal?.addEventListener('abort', abort, { once: true });
    if (signal?.aborted) controller.abort();
    const timer = setTimeout(abort, 15000);
    try {
      const response = await fetcher(base + path, { method: post || body !== undefined ? 'POST' : 'GET', ...(body !== undefined ? { headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) } : {}), signal: controller.signal });
      if (!response.ok) {
        // Never echo FastAPI detail/input: it may contain private audio or text.
        if (response.status === 502) throw new Error('The speech or language provider is temporarily unavailable. Your input is kept; please try again shortly.');
        if (response.status === 503 && path === '/predict') throw new Error('Sign model is unavailable (503). Ask the ML host to load its trained 110-point model, then retry.');
        if (response.status === 404 && path.startsWith('/session/')) throw new Error('Room not found. Create a new room or check the room ID and join again. Your draft is kept.');
        if ([400, 413, 422].includes(response.status)) {
          if (path === '/predict') throw new Error('ML rejected the 110-point capture (400/422). Check that the updated ML service is configured, then recapture.');
          if (path === '/transcribe') throw new Error(response.status === 413 ? 'Recording is too large. Record again with a shorter phrase.' : 'Audio could not be processed. Record again with a clear, short phrase.');
          if (path === '/process') throw new Error('The phrase could not be processed. Re-enter your keywords and try again.');
          throw new Error('Check your input and try again. Your draft is kept.');
        }
        throw new Error(`Service request failed (HTTP ${response.status}). Please try again.`);
      }
      if (response.status === 204) return null;
      try { return await response.json(); }
      catch { throw new Error('Invalid JSON response from service. Please try again.'); }
    } catch (error) {
      if (controller.signal.aborted && !signal?.aborted) throw new Error('Service request timed out. Check connection before retrying.');
      if (error instanceof TypeError) throw new Error('Service unreachable. Check endpoint, network and CORS configuration, then retry.');
      throw error;
    } finally { clearTimeout(timer); signal?.removeEventListener('abort', abort); }
  }
  const roomPath = (room: string) => `/session/${encodeURIComponent(room)}`;
  return {
    async predict(landmarks, signal) {
      validateLandmarks(landmarks);
      const data = object(await request('/predict', { landmarks }, signal, config.ml));
      if (!Array.isArray(data.keywords) || data.keywords.some(w => typeof w !== 'string')) throw new Error('Invalid sign prediction response. Recapture or check the ML service.');
      validateKeywords(data.keywords); return data.keywords;
    },
    async create() { const data = object(await request('/session/create', undefined, undefined, config.backend, true)); if (typeof data.room_id !== 'string' || !data.room_id) throw new Error('Service returned no room ID.'); return data.room_id; },
    async messages(room, signal) {
      const data = object(await request(`${roomPath(room)}/messages`, undefined, signal));
      if (!Array.isArray(data.messages)) throw new Error('Invalid message list.');
      return data.messages.map(value => { const m = object(value); if (!['signer', 'speaker'].includes(String(m.sender)) || typeof m.text !== 'string' || typeof m.timestamp !== 'string') throw new Error('Invalid conversation message.'); return m as unknown as Message; });
    },
    async send(room, sender, text) { validateText(text); await request(`${roomPath(room)}/message`, { sender, text }); },
    async process(input) {
      const payload = Array.isArray(input) ? { keywords: input } : input;
      if (payload.keywords) validateKeywords(payload.keywords);
      return parseProcess(await request('/process', payload));
    },
    async transcribe(audio) { const data = object(await request('/transcribe', { audio })); if (typeof data.text !== 'string') throw new Error('Service returned no transcript.'); return data.text; },
    async health(service) { await request('/health', undefined, undefined, config[service]); return 'Reachable · HTTP success'; }
  };
}
export function demoAdapter(): Adapter {
  const rooms = new Map<string, Message[]>();
  const get = (id: string) => { const room = rooms.get(id); if (!room) throw new Error('Demo room not found. Create one in this tab first.'); return room; };
  return {
    async predict() { throw new Error('Sign prediction is disabled in demo mode. Switch to configured live endpoints explicitly.'); },
    async create() { const id = `DEMO-${crypto.randomUUID().slice(0, 8).toUpperCase()}`; rooms.set(id, []); return id; },
    async messages(id) { return [...get(id)]; },
    async send(id, sender, text) { validateText(text); get(id).push({ sender, text, timestamp: new Date().toISOString() }); },
    async process(input) {
      const words = Array.isArray(input) ? input : input.keywords;
      if (!words) throw new Error('Demo mode supports manual keywords only; sign recognition is not implemented.');
      validateKeywords(words);
      const sentence = words.join(' ') + '.';
      return { status: 'success', sentence, audio_url: null, subtitles: [{ language: 'en-IN', text: sentence }], audio_segments: [], playback_order: [] };
    },
    async transcribe() { throw new Error('Cloud transcription is unavailable in demo mode. Type a draft instead.'); },
    async health() { return 'Demo · no network request'; }
  };
}
