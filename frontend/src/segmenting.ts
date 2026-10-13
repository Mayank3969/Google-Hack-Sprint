import type { Frame } from './landmarks';

export interface LandmarkSegment {
  segment_index: number;
  segment_id?: string;
  landmarks: Frame[];
}
export interface CapturedSegment extends LandmarkSegment {
  status: 'ready' | 'partial' | 'overflow' | 'tracking_lost';
}

const visible = (point: number[]) => point.some(value => value !== 0);
const handsVisible = (frame: Frame) => frame.slice(0, 42).some(visible);
function percentile(values: number[], fraction: number) {
  const ordered = [...values].sort((a, b) => a - b), position = (ordered.length - 1) * fraction;
  return ordered[Math.floor(position)] + (ordered[Math.ceil(position)] - ordered[Math.floor(position)]) * (position % 1);
}
function handMotion(previous: Frame, current: Frame, start: number): number {
  const distances: number[] = [], points: number[][] = [];
  for (let i = start; i < start + 21; i++) {
    if (visible(current[i])) points.push(current[i]);
    if (visible(previous[i]) && visible(current[i])) distances.push(Math.hypot(current[i][0] - previous[i][0], current[i][1] - previous[i][1]));
  }
  if (!distances.length) return 0;
  const scale = points.length < 2 ? 1 : Math.max(0.03, Math.hypot(
    Math.max(...points.map(p => p[0])) - Math.min(...points.map(p => p[0])),
    Math.max(...points.map(p => p[1])) - Math.min(...points.map(p => p[1])),
  ));
  return Math.max(percentile(distances, 0.75), Math.max(...distances) * 0.5) / scale;
}

/** Browser counterpart of ML's pause segmenter: 15fps, 8-frame quiet hold,
 * 3-frame pre/post roll, 2-frame onset and 12-frame quiet calibration. Head
 * motion and detector loss never invent a word boundary. */
export function splitLandmarks(frames: Frame[]): CapturedSegment[] {
  if (frames.length > 450) throw new Error('Recording exceeds 30 seconds. Repeat the whole recording with a shorter phrase.');
  const segments: CapturedSegment[] = [], scores: number[] = [];
  let preRoll: Frame[] = [], current: Frame[] = [];
  let previous: Frame | undefined, calibrating = true, active = false;
  let onset = 0, quiet = 0, lost = 0, pauseThreshold = 0.04, activeThreshold = 0.08;
  let status: CapturedSegment['status'] = 'ready';
  function commit() {
    if (!current.length) return;
    if (segments.length === 10) throw new Error('Recording contains more than 10 signs. Repeat the whole recording with a shorter phrase.');
    const trim = Math.max(0, quiet - 3);
    if (trim) current = current.slice(0, -trim);
    if (status === 'tracking_lost' && quiet >= 8) status = 'ready';
    if (status === 'ready' && current.length < 6) status = 'partial';
    if (status === 'ready' && current.length > 60) status = 'overflow';
    const segment_index = segments.length;
    segments.push({ segment_index, segment_id: `segment-${segment_index}`, landmarks: current, status });
    current = []; preRoll = []; quiet = 0; lost = 0; onset = 0; active = false; status = 'ready';
  }
  frames.forEach((frame, index) => {
    const valid = handsVisible(frame);
    const sameVisibility = previous && previous.slice(0, 42).every((point, i) => visible(point) === visible(frame[i]));
    const motion = previous && valid && sameVisibility && handsVisible(previous)
      ? Math.max(handMotion(previous, frame, 0), handMotion(previous, frame, 21)) : null;
    previous = frame;
    preRoll.push(frame); if (preRoll.length > 3) preRoll.shift();
    if (calibrating && !active) {
      if (motion !== null && motion < 0.04) scores.push(motion);
      if (index >= 11) {
        if (scores.length >= 6) {
          const median = percentile(scores, 0.5), mad = percentile(scores.map(score => Math.abs(score - median)), 0.5);
          pauseThreshold = Math.min(0.08, Math.max(0.04, 2.5 * (median + 3 * mad)));
          activeThreshold = Math.min(0.12, Math.max(1.5 * pauseThreshold, pauseThreshold + 0.02));
        }
        calibrating = false;
      } else if (motion === null || motion < activeThreshold) { onset = 0; return; }
    }
    if (!active) {
      if (!valid || motion === null) { onset = 0; return; }
      onset = motion >= activeThreshold ? onset + 1 : 0;
      if (onset >= 2) { current = [...preRoll]; active = true; calibrating = false; quiet = 0; lost = 0; }
      return;
    }
    if (current.length < 68) current.push(frame);
    else status = 'overflow';
    if (!valid) {
      lost++; quiet = 0;
      if (lost > 4) status = 'tracking_lost';
      return;
    }
    lost = 0;
    if (motion !== null && motion < pauseThreshold) { quiet++; if (quiet >= 8) commit(); }
    else quiet = 0;
  });
  if (active) {
    if ((status as string) === 'tracking_lost' && current.slice(-4).every(frame => !handsVisible(frame)) && current.slice(0, -4).some(handsVisible)) status = 'ready';
    commit();
  }
  // A signer may hold a static pose throughout a short recording. Preserve
  // that completed buffer instead of silently dropping it when onset motion
  // is unavailable.
  if (!segments.length && frames.some(handsVisible)) {
    for (let start = 0; start < frames.length; start += 60) {
      const landmarks = frames.slice(start, start + 60);
      const segment_index = segments.length;
      segments.push({ segment_index, segment_id: `segment-${segment_index}`, landmarks, status: landmarks.length < 6 ? 'partial' : 'ready' });
    }
  }
  return segments;
}
