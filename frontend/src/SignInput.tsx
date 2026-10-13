import { useEffect, useRef, useState } from 'react';
import type { Adapter, BatchResult } from './api';
import { SignCapture } from './capture';
import { validateLandmarks, type Frame } from './landmarks';
import { type LandmarkSegment, splitLandmarks } from './segmenting';
import type { RecognitionResult } from './process';

export interface ReviewedCapture {
  capture_id: string;
  results: RecognitionResult[];
}

export function SignInput({ api, demo, onAppend, onCapture }: { api: Adapter; demo: boolean; onAppend: (words: string[], capture?: ReviewedCapture) => void; onCapture: () => void }) {
  const video = useRef<HTMLVideoElement>(null), capture = useRef<SignCapture | null>(null);
  const generation = useRef(0), request = useRef<AbortController | null>(null);
  const [state, setState] = useState('off'), [error, setError] = useState('');
  const [segments, setSegments] = useState<LandmarkSegment[]>([]), [words, setWords] = useState<string[]>([]);
  const [batchResults, setBatchResults] = useState<BatchResult[]>([]), [captureId, setCaptureId] = useState('');
  function release() { generation.current++; request.current?.abort(); request.current = null; capture.current?.dispose(); capture.current = null; }
  function reset() { release(); setSegments([]); setWords([]); setBatchResults([]); setCaptureId(''); setError(''); setState('off'); }
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
      try {
        // SignCapture currently returns one continuous frame stream. Keeping
        // the split at this boundary also accepts a future capture provider
        // that returns completed segments directly without changing the API.
        if (!(value as Frame[]).some(frame => frame.slice(0, 42).some(point => point.some(coordinate => coordinate !== 0)))) throw new Error('No hands detected. Keep your hands in view and recapture.');
        const completed: Frame[][] = Array.isArray((value as unknown as Frame[][])[0]?.[0]?.[0])
          ? value as unknown as Frame[][]
          : splitLandmarks(value as Frame[]).map(segment => segment.landmarks);
        const next = completed.map((frames, index) => { validateLandmarks(frames, 60, false); return { segment_index: index, segment_id: `segment-${index}`, landmarks: frames }; });
        if (!next.length) throw new Error('No sign segment was captured. Keep your hands in view and record again.');
        const id = makeCaptureId();
        setSegments(next); setCaptureId(id); setBatchResults([]); setState('predicting');
        void predict(next, id, token);
      } catch (e) { setState('off'); setError(e instanceof Error ? e.message : 'Capture could not be prepared. Record again.'); }
    }, failed).catch(failed);
  }
  async function predict(next: LandmarkSegment[], id: string, token: number) {
    const controller = new AbortController(); request.current = controller; setError('');
    try {
      const value = await api.predictBatch(id, next, controller.signal);
      if (token !== generation.current) return;
      setBatchResults(value.results);
      setWords(value.results.filter(result => result.status === 'recognized' && !!result.keyword).map(result => result.keyword!));
      if (!value.complete) { setState('retry'); setError('One or more signs were unclear. The whole recording must be repeated; recognized words are shown below for review.'); }
      else setState('review');
    }
    catch (e) { if (token === generation.current) { setState('captured'); setError(e instanceof Error ? e.message : 'Prediction failed. Retry or recapture.'); } }
    finally { if (request.current === controller) request.current = null; }
  }
  function retryPrediction() {
    if (!segments.length || !captureId) return;
    const token = generation.current;
    setState('predicting');
    void predict(segments, captureId, token);
  }
  function retry() { reset(); onCapture(); }
  const orderedResults = batchResults.slice().sort((left, right) => left.segment_index - right.segment_index);
  return <section className="sign-capture" aria-label="Sign capture">
    <video ref={video} muted playsInline aria-label="Unmirrored sign preview" onError={() => { reset(); setError('Video could not be decoded or the camera stream failed. Enable camera again or choose a supported local video.'); }} style={{ width: '100%', maxHeight: 280, background: '#142c2b', display: state === 'ready' || state === 'capturing' || state === 'loading' ? 'block' : 'none' }}/>
      <h3>Capture a phrase in one take.</h3>
      <p>Keep both hands and your face in view. Unmirrored, up to 4.5 seconds / 60 frames at at most 15 fps. Video stays on this device. Stopping sends all completed landmark segments together; video is never uploaded.</p>
    {demo ? <p>Camera sign recognition is disabled in demo mode. Configure live mode to use the trained model; manual demo keywords remain available.</p> : <>
      <p role="status">Sign capture: {state}{segments.length > 0 ? ` · ${segments.length} segment${segments.length === 1 ? '' : 's'}` : ''}</p>
      {state === 'off' && <><button onClick={() => void enable()}>Enable camera</button><label>Or choose local video<input type="file" accept="video/*" onChange={e => { const file = e.target.files?.[0]; e.target.value = ''; if (file) void enable(file); }}/></label></>}
      {state === 'ready' && <button onClick={start}>Capture sign</button>}
      {state === 'capturing' && <button onClick={() => capture.current?.stop()}>Finish capture</button>}
      {['captured', 'predicting', 'review', 'retry'].includes(state) && <button disabled={state !== 'captured'} onClick={retryPrediction}>Predict sign</button>}
      {(state === 'review' || state === 'retry') && <div>
        <p>Ordered word results:</p>
        <ol className="word-results">{orderedResults.map(result => <li key={result.segment_index}><strong>{result.keyword || 'No word recognized'}</strong><small>{result.status === 'recognized' ? 'Recognized' : `Needs another recording (${result.status})`}</small></li>)}</ol>
        {state === 'review' ? <><p>Review the complete recording before appending.</p><button onClick={() => { onAppend(words, batchResults.length && orderedResults[0].confidence === null ? undefined : { capture_id: captureId, results: orderedResults.filter(result => result.status === 'recognized').map(result => ({ segment_index: result.segment_index, status: 'recognized', keyword: result.keyword, confidence: result.confidence as number, coverage: result.coverage, top_k: result.top_k })) }); reset(); }}>Append reviewed words</button></> : <><p>Recognized words remain visible, but partial recordings are not submitted to sentence generation.</p><button className="secondary" onClick={retry}>Repeat whole recording</button></>}
      </div>}
      {state !== 'off' && <button className="secondary" onClick={reset}>Cancel / reset capture</button>}
    </>}
    {error && <p role="alert" className="error">{error}</p>}
  </section>;
}

function makeCaptureId() {
  try { return crypto.randomUUID(); }
  catch { return `capture-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`; }
}
