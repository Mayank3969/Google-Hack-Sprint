export type Language = 'en-IN' | 'hi-IN';
export interface RecognitionResult {
  segment_index: number;
  status: 'recognized' | 'uncertain';
  keyword?: string | null;
  confidence: number;
  coverage: number;
  top_k?: { label: string; confidence: number }[] | null;
}
export type ProcessInput = { keywords: string[]; capture_id?: never; results?: never }
  | { capture_id: string; results: RecognitionResult[]; keywords?: never };
export interface Subtitle { language: Language; text: string }
export interface AudioSegment { language: Language; audio_url: string }
export interface Speech {
  status: 'success';
  sentence: string;
  audio_url: string | null;
  hindi_sentence?: string;
  hindi_keywords?: string[];
  subtitles: Subtitle[];
  audio_segments: AudioSegment[];
  playback_order: Language[];
}
export type ProcessResult = Speech | { status: 'record_again'; record_again: true };
const invalid = (): never => { throw new Error('Invalid sentence response from service. Please try again.'); };
const record = (value: unknown): Record<string, unknown> => {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return invalid();
  return value as Record<string, unknown>;
};
const text = (value: unknown): string => typeof value === 'string' && !!value.trim() ? value : invalid();
const language = (value: unknown): Language => value === 'en-IN' || value === 'hi-IN' ? value : invalid();
function audioUrl(value: unknown): string {
  try {
    const raw = text(value);
    const url = new URL(raw);
    if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password || /\s/.test(raw)) throw new Error();
    return url.href;
  } catch { throw new Error('Service returned an unsupported audio URL. Expected an absolute HTTP(S) URL.'); }
}
export function parseProcess(value: unknown): ProcessResult {
  const data = record(value);
  if ('status' in data || 'record_again' in data) {
    if (data.status === 'record_again' && data.record_again === true && Object.keys(data).length === 2) {
      return { status: 'record_again', record_again: true };
    }
    return invalid();
  }
  const sentence = text(data.sentence), audio_url = audioUrl(data.audio_url);
  const bilingual = ['hindi_sentence', 'hindi_keywords', 'subtitles', 'audio_segments', 'playback_order'].some(key => key in data);
  if (!bilingual) return { status: 'success', sentence, audio_url,
    subtitles: [{ language: 'en-IN', text: sentence }], audio_segments: [{ language: 'en-IN', audio_url }], playback_order: ['en-IN'] };
  const hindi_sentence = text(data.hindi_sentence);
  if (!Array.isArray(data.hindi_keywords) || !data.hindi_keywords.length || data.hindi_keywords.length > 20) return invalid();
  const hindi_keywords = data.hindi_keywords.map(value => { const word = text(value); return word.length <= 64 ? word : invalid(); });
  if (!Array.isArray(data.subtitles) || data.subtitles.length !== 2 || !Array.isArray(data.audio_segments) || data.audio_segments.length !== 2 || !Array.isArray(data.playback_order) || data.playback_order.length !== 2) return invalid();
  const subtitles = data.subtitles.map(value => { const item = record(value); return { language: language(item.language), text: text(item.text) }; });
  const audio_segments = data.audio_segments.map(value => { const item = record(value); return { language: language(item.language), audio_url: audioUrl(item.audio_url) }; });
  const playback_order = data.playback_order.map(language);
  if ([subtitles.map(s => s.language), audio_segments.map(s => s.language), playback_order].some(languages => new Set(languages).size !== 2)) return invalid();
  if (subtitles.find(s => s.language === 'en-IN')?.text !== sentence || subtitles.find(s => s.language === 'hi-IN')?.text !== hindi_sentence || audio_segments.find(s => s.language === 'en-IN')?.audio_url !== audio_url) return invalid();
  return { status: 'success', sentence, audio_url, hindi_sentence, hindi_keywords, subtitles, audio_segments, playback_order };
}
