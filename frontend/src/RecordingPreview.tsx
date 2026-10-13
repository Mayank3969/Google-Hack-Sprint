import { useEffect, useRef } from 'react';

export function RecordingPreview({ url }: { url: string }) {
  const ref = useRef<HTMLAudioElement>(null);
  useEffect(() => {
    const audio = ref.current;
    return () => { if (audio) { audio.pause(); audio.removeAttribute('src'); audio.load(); } };
  }, []);
  return <><label htmlFor="recording-preview">Last recording</label><audio ref={ref} id="recording-preview" controls src={url} preload="none"/></>;
}
