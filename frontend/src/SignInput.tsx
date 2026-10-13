import { useEffect, useRef, useState } from 'react';
import type { Adapter } from './api';
import { SignCapture } from './capture';
import { validateLandmarks, type Frame } from './landmarks';

export function SignInput({ api, demo, onAppend, onCapture }: { api: Adapter; demo: boolean; onAppend: (words: string[]) => void; onCapture: () => void }) {
  const video = useRef<HTMLVideoElement>(null), capture = useRef<SignCapture | null>(null);
  const generation = useRef(0), request = useRef<AbortController | null>(null);
  const [state, setState] = useState('off'), [error, setError] = useState('');
  const [frames, setFrames] = useState<Frame[]>([]), [words, setWords] = useState<string[]>([]);
  function release() { generation.current++; request.current?.abort(); request.current = null; capture.current?.dispose(); capture.current = null; }
  function reset() { release(); setFrames([]); setWords([]); setError(''); setState('off'); }
  useEffect(() => {
    const hidden = () => { if (document.hidden) reset(); };
    document.addEventListener('visibilitychange', hidden);
    return () => { document.removeEventListener('visibilitychange', hidden); release(); };
  }, []);
  async function enable(file?: File) {
    reset(); onCapture();
    const token = generation.current;
    const session = new SignCapture(video.current!); capture.current = session;
    setState('loading');
    try { await session.prepare(file); if (token === generation.current) setState('ready'); }
    catch (e) {
      if (token !== generation.current) return;
      release(); setState('off');
      setError(e instanceof DOMException && e.name === 'NotAllowedError' ? 'Camera permission denied. Allow camera access in browser settings, then enable again; or choose a local video.' : e instanceof Error ? e.message : 'Camera/video could not open. Check permission or choose a supported video and try again.');
    }
  }
  function start() {
    const token = generation.current;
    setError(''); setState('capturing'); onCapture();
    const failed = (e: unknown) => { if (token === generation.current) { release(); setState('off'); setError(e instanceof Error ? e.message : 'Detection failed. Enable again to recapture.'); } };
    void capture.current!.start(value => {
      if (token !== generation.current) return;
      capture.current = null;
      try { validateLandmarks(value); setFrames(value); setState('captured'); }
      catch (e) { setState('off'); setError((e as Error).message); }
    }, failed).catch(failed);
  }
  async function predict() {
    const token = generation.current, controller = new AbortController(); request.current = controller;
    setState('predicting'); setError('');
    try { const value = await api.predict(frames, controller.signal); if (token === generation.current) { setWords(value); setState('review'); } }
    catch (e) { if (token === generation.current) { setState('captured'); setError(e instanceof Error ? e.message : 'Prediction failed. Retry or recapture.'); } }
    finally { if (request.current === controller) request.current = null; }
  }
  return <section className="sign-capture" aria-label="Sign capture">
    <video ref={video} muted playsInline aria-label="Unmirrored sign preview" onError={() => { reset(); setError('Video could not be decoded or the camera stream failed. Enable camera again or choose a supported local video.'); }} style={{ width: '100%', maxHeight: 280, background: '#142c2b', display: state === 'ready' || state === 'capturing' || state === 'loading' ? 'block' : 'none' }}/>
    <h3>Capture one sign at a time.</h3>
    <p>Keep both hands and your face in view. Unmirrored, up to 4.5 seconds / 60 frames at at most 15 fps. Video stays on this device. Only “Predict sign” sends landmarks to the ML service.</p>
    {demo ? <p>Camera sign recognition is disabled in demo mode. Configure live mode to use the trained model; manual demo keywords remain available.</p> : <>
      <p role="status">Sign capture: {state}{frames.length > 0 ? ` · ${frames.length} frames` : ''}</p>
      {state === 'off' && <><button onClick={() => void enable()}>Enable camera</button><label>Or choose local video<input type="file" accept="video/*" onChange={e => { const file = e.target.files?.[0]; e.target.value = ''; if (file) void enable(file); }}/></label></>}
      {state === 'ready' && <button onClick={start}>Capture sign</button>}
      {state === 'capturing' && <button onClick={() => capture.current?.stop()}>Finish capture</button>}
      {state === 'captured' && <button onClick={() => void predict()}>Predict sign</button>}
      {state === 'review' && <div><p>Suggested isolated word(s): <strong>{words.join(', ')}</strong></p><p>Review before appending. The classifier does not return confidence.</p><button onClick={() => { onAppend(words); reset(); }}>Append reviewed words</button></div>}
      {state !== 'off' && <button className="secondary" onClick={reset}>Cancel / reset capture</button>}
    </>}
    {error && <p role="alert" className="error">{error}</p>}
  </section>;
}
