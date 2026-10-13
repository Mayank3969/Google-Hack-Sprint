import { useEffect, useRef, useState } from 'react';
import type { Language, Speech } from './process';
const names: Record<Language, string> = { 'en-IN': 'English', 'hi-IN': 'Hindi' };

/** Owns the entire playlist lifetime; replay only reuses returned URLs. */
export function SpeechResult({ speech, demo, onUse }: { speech: Speech; demo: boolean; onUse: (text: string) => void }) {
  const [active, setActive] = useState<Language | null>(null);
  const [error, setError] = useState('');
  const cleanup = useRef<() => void>(() => {});
  const version = useRef(0);
  function stop() { version.current++; cleanup.current(); cleanup.current = () => {}; window.speechSynthesis?.cancel(); setActive(null); }
  useEffect(() => () => { version.current++; cleanup.current(); window.speechSynthesis?.cancel(); }, [speech]);
  function play() {
    stop(); setError('');
    const token = version.current;
    if (demo) {
      if (!window.speechSynthesis) { setError('This browser does not support demo text-to-speech.'); return; }
      const utterance = new SpeechSynthesisUtterance(speech.sentence);
      const voice = speechSynthesis.getVoices().find(v => v.localService);
      if (voice) utterance.voice = voice;
      utterance.onend = () => { if (token === version.current) setActive(null); };
      utterance.onerror = () => { if (token === version.current) { setActive(null); setError('Browser voice playback failed. Try again.'); } };
      cleanup.current = () => { utterance.onend = null; utterance.onerror = null; };
      setActive('en-IN'); speechSynthesis.speak(utterance); return;
    }
    const next = (index: number) => {
      if (token !== version.current) return;
      cleanup.current(); cleanup.current = () => {};
      const language = speech.playback_order[index];
      if (!language) { setActive(null); return; }
      const segment = speech.audio_segments.find(s => s.language === language)!;
      const audio = new Audio(segment.audio_url);
      const failed = () => {
        if (token !== version.current) return;
        stop(); setError('Generated audio could not play or has expired. Your text is kept. Try playback again; if it remains unavailable, generate again.');
      };
      const ended = () => next(index + 1);
      audio.addEventListener('ended', ended);
      audio.addEventListener('error', failed);
      cleanup.current = () => { audio.removeEventListener('ended', ended); audio.removeEventListener('error', failed); audio.pause(); audio.removeAttribute('src'); audio.load(); };
      setActive(language);
      void audio.play().catch(() => { if (token === version.current && audio.getAttribute('src')) failed(); });
    };
    next(0);
  }
  return <div className="generated">
    <span className="eyebrow">{demo ? 'DEMO PHRASE' : 'GENERATED SENTENCE'}</span>
    {speech.subtitles.map(subtitle => <section key={subtitle.language} aria-label={`${names[subtitle.language]} subtitle`}>
      <strong>{names[subtitle.language]}</strong><p lang={subtitle.language}>{subtitle.text}</p>
    </section>)}
    {speech.hindi_keywords && <p lang="hi-IN"><strong>Hindi keywords: </strong>{speech.hindi_keywords.join(', ')}</p>}
    <button className="secondary" onClick={play}>{demo ? '▶ Play with browser voice' : '▶ Play audio'}</button>
    {active && <button className="secondary" onClick={stop}>Stop playback</button>}
    <p role="status">{active ? `Playing ${names[active]}` : demo ? 'Browser voice · demonstration only' : `Playback order: ${speech.playback_order.map(l => names[l]).join(' → ')}`}</p>
    {error && <p role="alert" className="error">{error}</p>}
    <button className="secondary" onClick={() => onUse(speech.sentence)}>Use as signer draft</button>
    {speech.hindi_sentence && <button className="secondary" onClick={() => onUse(speech.hindi_sentence!)}>Use Hindi as signer draft</button>}
  </div>;
}
