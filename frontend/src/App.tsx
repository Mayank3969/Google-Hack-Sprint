import { useEffect, useMemo, useRef, useState } from 'react';
import type { Config } from './config';
import { demoAdapter, liveAdapter, type Message, type Sender, type ProcessResult } from './api';
import { Microphone, toBase64 } from './audio';
import { pollMessages } from './polling';
import { SpeechResult } from './SpeechResult';
import { RecordingPreview } from './RecordingPreview';
import { SignInput } from './SignInput';
import { Analytics } from '@vercel/analytics/react';
import { ConsentBanner, readConsent, type Consent } from './ConsentBanner';
import { LegalPage, NotFoundPage } from './Legal';

export function App({ config }: { config: Config }) {
  const api = useMemo(() => config.demo ? demoAdapter() : liveAdapter(config), [config]);
  const route = location.pathname;
  const [page, setPage] = useState(route.startsWith('/settings') ? 'settings' : route === '/' ? 'workspace' : route.slice(1));
  const [consent, setConsent] = useState<Consent>(() => readConsent());
  const [room, setRoom] = useState(''), [join, setJoin] = useState('');
  const [messages, setMessages] = useState<Message[]>([]), [syncError, setSyncError] = useState('');
  const [paused, setPaused] = useState(false), [busy, setBusy] = useState('');
  const [error, setError] = useState(''), [notice, setNotice] = useState('');
  const [mode, setMode] = useState<'sign' | 'speech'>('sign');
  const [sender, setSender] = useState<Sender>('speaker'), [draft, setDraft] = useState('');
  const [keywords, setKeywords] = useState(''), [result, setResult] = useState<ProcessResult | null>(null);
  const [micState, setMicState] = useState('idle'), [preview, setPreview] = useState('');
  const [health, setHealth] = useState({ backend: 'Not checked', ml: 'Not checked' });
  const microphone = useRef<Microphone | null>(null), generation = useRef(0), processGeneration = useRef(0);
  const keywordInput = useRef<HTMLInputElement>(null);
  const [captureReset, setCaptureReset] = useState(0);
  const poll = useRef<ReturnType<typeof pollMessages> | null>(null);
  const working = useRef(false);
  function clearResult() { processGeneration.current++; setResult(null); window.speechSynthesis?.cancel(); }
  function cancelRecording() { generation.current++; microphone.current?.dispose(); microphone.current = null; setMicState('idle'); }
  useEffect(() => {
     const change = () => { cancelRecording(); clearResult(); setCaptureReset(n => n + 1); const path = location.pathname; setPage(path.startsWith('/settings') ? 'settings' : path === '/' ? 'workspace' : path.slice(1)); };
    window.addEventListener('popstate', change);
    return () => window.removeEventListener('popstate', change);
  }, []);
  useEffect(() => {
    if (!room || paused || page !== 'workspace') return;
    const loop = pollMessages(api, room, setMessages, setSyncError); poll.current = loop;
    return () => { loop.stop(); poll.current = null; };
  }, [api, room, paused, page]);
  useEffect(() => () => { generation.current++; processGeneration.current++; microphone.current?.dispose(); microphone.current = null; window.speechSynthesis?.cancel(); }, []);
  useEffect(() => () => { if (preview) URL.revokeObjectURL(preview); }, [preview]);
  useEffect(() => {
    const legal = page === 'privacy' || page === 'terms';
    const title = page === 'workspace' ? 'Workspace' : page === 'settings' ? 'Settings and Help' : page === 'privacy' ? 'Privacy Policy' : page === 'terms' ? 'Terms and Conditions' : 'Page not found';
    document.title = `${title} | SignBridge`;
    const description = legal ? `SignBridge ${title.toLowerCase()}.` : 'SignBridge is a shared workspace for reviewing sign, speech, and text conversation.';
    const setMeta = (name: string, content: string, property = false) => { const selector = property ? `meta[property="${name}"]` : `meta[name="${name}"]`; let node = document.head.querySelector<HTMLMetaElement>(selector); if (!node) { node = document.createElement('meta'); node.setAttribute(property ? 'property' : 'name', name); document.head.appendChild(node); } node.content = content; };
    const siteUrl = config.siteUrl || location.origin;
    setMeta('description', description); setMeta('og:title', `${title} | SignBridge`, true); setMeta('og:description', description, true); setMeta('og:url', `${siteUrl}${location.pathname}`, true); setMeta('og:image', `${siteUrl}/og-signbridge.svg`, true); setMeta('twitter:card', 'summary_large_image'); setMeta('twitter:title', `${title} | SignBridge`); setMeta('twitter:description', description); setMeta('twitter:image', `${siteUrl}/og-signbridge.svg`);
    let canonical = document.head.querySelector<HTMLLinkElement>('link[rel="canonical"]'); if (!canonical) { canonical = document.createElement('link'); canonical.rel = 'canonical'; document.head.appendChild(canonical); } canonical.href = `${siteUrl}${location.pathname}`;
  }, [page, config.siteUrl]);
  function navigate(target: string) { cancelRecording(); clearResult(); setCaptureReset(n => n + 1); const path = target === 'settings' ? '/settings' : target === 'privacy' ? '/privacy' : target === 'terms' ? '/terms' : '/'; history.pushState({}, '', path); setPage(target); }
  async function run(name: string, task: () => Promise<void>) {
    if (working.current) return;
    working.current = true; setBusy(name); setError(''); setNotice('');
    try { await task(); } catch (e) { setError(e instanceof Error ? e.message : 'Something went wrong. Please try again.'); }
    finally { working.current = false; setBusy(''); }
  }
  async function enter(create: boolean) {
    await run('room', async () => {
      const id = create ? await api.create() : join.trim();
      if (!id || id.length > 128) throw new Error('Enter a room ID up to 128 characters.');
      const snapshot = await api.messages(id);
      poll.current?.stop(); cancelRecording(); clearResult(); setCaptureReset(n => n + 1); setRoom(id); setMessages(snapshot); setSyncError(''); setPaused(false);
      setNotice(create ? 'Room created. Share the room ID to connect.' : 'Room joined.');
    });
  }
  async function send() {
    if (!room) return;
    await run('send', async () => { await api.send(room, sender, draft.trim()); setDraft(''); setNotice('Message sent.'); void poll.current?.refresh(); });
  }
  async function generate() {
    await run('process', async () => {
      clearResult(); setPreview(''); setCaptureReset(n => n + 1);
      const token = processGeneration.current;
      try {
        const value = await api.process({ keywords: keywords.split(',').map(w => w.trim()) });
        if (token === processGeneration.current) setResult(value);
      } catch (e) { if (token === processGeneration.current) throw e; }
    });
  }
  async function stopRecording(token: number) {
    const mic = microphone.current;
    if (!mic || token !== generation.current) return;
    microphone.current = null;
    try {
      const wav = mic.finish(); setPreview(URL.createObjectURL(new Blob([wav], { type: 'audio/wav' })));
      if (config.demo) { setMicState('idle'); setNotice('Local recording ready. Demo mode does not transcribe or upload audio.'); return; }
      setMicState('transcribing');
      const text = await api.transcribe(toBase64(wav));
      if (token !== generation.current) return;
      if (!text.trim()) { setNotice('No speech detected. Record again with a clear phrase. Your draft is kept.'); return; }
      setDraft(text.slice(0, 2000)); setSender('speaker');
      setNotice(text.length > 2000 ? 'Transcript trimmed to 2,000 characters. Review before sending.' : 'Transcript ready. Review your draft before sending.');
    } catch (e) { if (token === generation.current) setError(e instanceof Error ? e.message : 'Recording failed. Record again.'); }
    finally { if (token === generation.current) setMicState('idle'); }
  }
  async function startRecording() {
    if (micState !== 'idle') return;
    clearResult(); setError(''); setNotice(''); setPreview(''); setMicState('requesting');
    const token = ++generation.current, mic = new Microphone(); microphone.current = mic;
    try { await mic.start(() => void stopRecording(token)); if (token === generation.current && microphone.current === mic) setMicState('recording'); }
    catch (e) { if (token === generation.current) { microphone.current = null; setMicState('idle'); setError(e instanceof Error ? e.message : 'Microphone permission was denied.'); } }
  }
  function changeMode(next: 'sign' | 'speech') { cancelRecording(); clearResult(); setMode(next); }
  function useDraft(text: string) { setDraft(text.slice(0, 2000)); setSender('signer'); setNotice('Sentence copied to your editable draft.'); }

  const legal = page === 'privacy' || page === 'terms';
  const unknown = !['workspace', 'settings', 'privacy', 'terms'].includes(page);
  return <>
    <a className="skip" href="#main">Skip to main content</a>
    <header className="topbar">
      <a className="brand" href="/" onClick={e => { e.preventDefault(); navigate('workspace'); }}>SignBridge <small>RELAY</small></a>
      <span className={`badge ${config.demo ? 'terra' : ''}`}>{config.demo ? 'Demo · local session' : 'Live endpoints'}</span>
      <nav aria-label="Main">
        <a href="/" aria-current={page === 'workspace' ? 'page' : undefined} onClick={e => { e.preventDefault(); navigate('workspace'); }}>Workspace</a>
          <a href="/settings" aria-current={page === 'settings' ? 'page' : undefined} onClick={e => { e.preventDefault(); navigate('settings'); }}>Settings & Help</a>
      </nav>
    </header>
    <main id="main">
       {legal || unknown ? <>{legal ? <LegalPage kind={page as 'privacy' | 'terms'} /> : <NotFoundPage />}</> : <><div className="heading">
        <div><span className="eyebrow">A LITTLE TECHNOLOGY. MORE CONNECTION.</span><h1>{page === 'workspace' ? 'A shared space for conversation.' : 'Configuration & trust.'}</h1><p>{page === 'workspace' ? 'Your words, your pace. Meet in the middle.' : 'Understand your connections, permissions, and data.'}</p></div>
        <div className="room-label"><span className="eyebrow">CURRENT ROOM</span><strong>{room || 'Not connected'}</strong>{room && <button className="compact" onClick={() => void run('copy', async () => { await navigator.clipboard.writeText(room); setNotice('Room ID copied.'); })}>Copy room ID</button>}</div>
       </div>
      {config.demo && <div className="demo-note"><strong>Demonstration mode</strong><span>Rooms and messages stay in this tab until reload. Generated phrases simply join your keywords; no sign recognition or cloud transcription.</span></div>}
      {error && <div className="error" role="alert">{error}</div>}
      <div className="notice" role="status">{notice}</div>
      {page === 'workspace' ? <>
        <section className="session-bar" aria-label="Connect to a room">
          <button disabled={!!busy} onClick={() => void enter(true)}>{busy === 'room' ? 'Connecting…' : '＋ Create room'}</button><span className="muted">or join a conversation</span>
           <form onSubmit={e => { e.preventDefault(); void enter(false); }}><label className="sr-only" htmlFor="room-id">Room ID</label><input id="room-id" value={join} onChange={e => setJoin(e.target.value)} placeholder="Enter room ID" maxLength={128} required/><button className="secondary" disabled={!!busy || !join.trim()}>Join room →</button></form>
        </section>
        <div className="workspace-grid">
          <section className="input-panel">
            <div className="section-title"><h2>Make yourself heard</h2><span className="eyebrow">01 / INPUT</span></div>
            <div className="segmented" aria-label="Input mode"><button aria-pressed={mode === 'sign'} onClick={() => changeMode('sign')}>Sign → speech</button><button aria-pressed={mode === 'speech'} onClick={() => changeMode('speech')}>Speech → text</button></div>
            {mode === 'sign' ? <>
              <SignInput key={captureReset} api={api} demo={config.demo} onCapture={clearResult} onAppend={words => {
                clearResult(); setKeywords(previous => [previous.trim(), ...words].filter(Boolean).join(', '));
                setNotice('Reviewed words appended. Capture another isolated sign or generate a sentence.');
              }}/>
              <div className="keyword-panel">
                <span className="eyebrow terra-text">MANUAL KEYWORDS → SPOKEN WORDS</span>
                <label htmlFor="keywords">What would you like to say?</label>
                 <input ref={keywordInput} id="keywords" placeholder="hello, good, morning" value={keywords} onChange={e => setKeywords(e.target.value)} aria-describedby="keyword-help" required/>
                <small id="keyword-help">Separate with commas · up to 20 keywords · 64 characters each</small>
                <button disabled={!!busy || !keywords.trim()} onClick={() => void generate()}>{busy === 'process' ? 'Preparing…' : 'Generate sentence →'}</button>
                {result?.status === 'record_again' && <div className="generated" role="status">
                  <h3>Please try the phrase again.</h3><p>There was not enough recognized information to complete a sentence. Recapture signs or re-enter your keywords.</p>
                  <button className="secondary" onClick={() => { clearResult(); setKeywords(''); setCaptureReset(n => n + 1); setNotice('Capture reset. Enable camera or choose a local video to record the phrase again.'); }}>Recapture signs</button>
                  <button className="secondary" onClick={() => { clearResult(); keywordInput.current?.focus(); keywordInput.current?.select(); }}>Re-enter keywords</button>
                </div>}
                {result?.status === 'success' && <SpeechResult key={processGeneration.current} speech={result} demo={config.demo} onUse={useDraft}/>}
              </div>
            </> : <div className="voice-panel">
              <span className="eyebrow terra-text">YOUR VOICE → EDITABLE TEXT</span><h3>Speak at your own pace.</h3>
              <p>Record up to 10 seconds. Review the transcript before sending. {config.demo && 'Demo mode records locally but does not transcribe.'}</p>
              <p role="status">Microphone: {micState}</p>
              <button disabled={micState !== 'idle' || !!busy} onClick={() => void startRecording()}>Start microphone</button>
              {micState === 'recording' && <button onClick={() => void stopRecording(generation.current)}>Stop recording</button>}
              {micState !== 'idle' && <button className="secondary" onClick={cancelRecording}>Cancel recording</button>}
              {preview && <RecordingPreview key={preview} url={preview}/>}
            </div>}
          </section>
          <section className="conversation">
            <div className="section-title"><div><span className="eyebrow">02 / SHARED WORDS</span><h2>Conversation timeline</h2></div><span className="badge">{messages.length} messages</span></div>
            <div className="sync-bar"><span>{!room ? 'Connect to begin' : paused ? 'Sync paused' : 'Polls every 2s while visible'}</span><button className="compact" disabled={!room} onClick={() => setPaused(!paused)}>{paused ? 'Resume' : 'Pause'}</button></div>
             {syncError && <p role="alert" className="error">{syncError} <button className="compact" onClick={() => void poll.current?.refresh()}>Retry sync</button></p>}
            <div className="timeline" role="log" aria-label="Conversation messages" aria-live="polite" aria-relevant="additions text">
               {messages.length ? messages.map((message, index) => <article className={`message ${message.sender}`} key={`${index}-${message.timestamp}`}>
                <div><strong>{message.sender === 'signer' ? 'Signer' : 'Speaker'}</strong><time dateTime={message.timestamp}>{Number.isNaN(Date.parse(message.timestamp)) ? message.timestamp : new Date(message.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</time></div><p>{message.text}</p>
               </article>) : <div className="empty"><h3>Room for every word.</h3><p>{room ? 'Your conversation will appear here. Send the first message when you’re ready.' : 'Create a room or enter a shared room ID to begin your conversation.'}</p></div>}
            </div>
            <form className="composer" onSubmit={e => { e.preventDefault(); void send(); }}>
              <div className="composer-label"><label htmlFor="draft">Your message</label><label className="sender-select">Send as <select aria-label="Message sender" value={sender} onChange={e => setSender(e.target.value as Sender)}><option value="speaker">Speaker</option><option value="signer">Signer</option></select></label></div>
               <textarea id="draft" rows={4} maxLength={2000} value={draft} onChange={e => setDraft(e.target.value)} placeholder="Type freely, or edit a generated sentence or transcript..." required/>
              <div className="composer-actions"><small>{draft.length}/2,000 · Review before sending</small><button disabled={!room || !!busy || !draft.trim()}>{busy === 'send' ? 'Sending…' : 'Send message →'}</button></div>
            </form>
          </section>
        </div>
      </> : <div className="settings-grid">
        <section className="card"><span className="eyebrow">INPUT & VISION</span><h2>01. Camera</h2><p>Enable camera explicitly or choose a local video in live mode. Install the local MediaPipe assets described in frontend/README.md. Capture extracts 110 unmirrored hand and compact face points at at most 15 fps (60 frames maximum). Only Predict sign uploads landmarks; video is never uploaded. Review isolated words before generating a sentence. No prediction confidence is supplied by this model.</p></section>
        <section className="card"><span className="eyebrow">AUDIO PIPELINE</span><h2>02. Microphone</h2><p>Access is requested only when you press “Start microphone.” Recording stops after at most 10 seconds. Tracks are released on stop, cancel, navigation, or component cleanup.</p><p>Samples are mixed to mono, low-pass resampled to 16 kHz, and encoded as PCM16 WAV. Live mode sends plain base64 WAV to <code>POST /transcribe</code>.</p></section>
        <section className="card">
          <span className="eyebrow">SERVICE PROBES</span><h2>03. Connection</h2>
          {(['backend', 'ml'] as const).map(service => <div className="diagnostic" key={service}><strong>{service === 'backend' ? 'Backend service' : 'ML service'}</strong><span>{health[service]}</span></div>)}
          <button disabled={!!busy} onClick={() => void run('health', async () => {
            setHealth({ backend: 'Checking…', ml: 'Checking…' });
            await Promise.all((['backend', 'ml'] as const).map(async service => {
              try { const value = await api.health(service); setHealth(h => ({ ...h, [service]: value })); }
              catch (e) { setHealth(h => ({ ...h, [service]: e instanceof Error ? e.message : 'Unreachable' })); }
            }));
          })}>{busy === 'health' ? 'Checking…' : 'Check connections'}</button>
          <p className="caption">An HTTP health response indicates reachability only; it does not verify model readiness.</p>
        </section>
        <section className="card"><span className="eyebrow">DATA FLOW & BOUNDARIES</span><h2>04. Privacy, in plain words</h2><p>{config.demo ? 'Demo conversations exist only in this tab’s memory and disappear on reload. Recordings are kept in memory for local playback; no backend calls are made.' : 'Messages, keywords, and recordings are sent to the configured backend. Storage, provider usage, retention, and room access are managed by that deployment.'}</p><p>Generated text is editable before sending. Audio only plays when you choose it. Browser demo voices may use an online speech service. A room ID is not authentication.</p></section>
       </div>}</>}
    </main>
     <footer><span>SignBridge <span className="muted">/ A bridge for conversation.</span></span><span><a href="/privacy" onClick={e => { e.preventDefault(); navigate('privacy'); }}>Privacy</a> <a href="/terms" onClick={e => { e.preventDefault(); navigate('terms'); }}>Terms</a> <a href="/" onClick={e => { e.preventDefault(); navigate('workspace'); }}>Open workspace</a></span></footer>
     <ConsentBanner onChange={setConsent}/>{consent === 'accepted' && <Analytics />}
   </>;
}
