export type Point = [number, number, number];
export type Frame = Point[];
export const FACE_INDICES = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246,263,249,390,373,374,380,381,382,362,398,384,385,386,387,388,466,70,63,105,66,107,336,296,334,293,300,61,146,91,181,84,17,314,405,321,375,78,95,88,178,87,14,317,402,318,324,1,2,4,5,6,168];
type Landmark = { x: number; y: number; z: number };
export function featureFrame(hands: { landmarks: Landmark[][]; handedness: { categoryName: string }[][] }, faces: { faceLandmarks: Landmark[][] }): Frame {
  const output: Frame = Array.from({ length: 110 }, () => [0, 0, 0]);
  const point = (p: Landmark): Point => { if (![p.x, p.y, p.z].every(Number.isFinite)) throw new Error('Invalid detector coordinates. Recapture.'); return [p.x, p.y, p.z]; };
  hands.landmarks.forEach((hand, i) => {
    if (i >= 2 || hand.length !== 21) throw new Error('Expected two hands with 21 points each.');
    const label = hands.handedness[i]?.[0]?.categoryName.toLowerCase();
    const slot = label === 'left' ? 0 : label === 'right' ? 1 : i;
    hand.forEach((p, j) => { output[slot * 21 + j] = point(p); });
  });
  const face = faces.faceLandmarks[0];
  if (face) {
    if (face.length !== 478) throw new Error('Face model must return 478 source points. Install the documented model.');
    FACE_INDICES.forEach((index, i) => { output[42 + i] = point(face[index]); });
  }
  return output;
}
export function validateLandmarks(frames: Frame[], maxFrames = 60, requireHands = true) {
  if (!Array.isArray(frames) || frames.length < 1 || frames.length > maxFrames || frames.some(frame => !Array.isArray(frame) || frame.length !== 110 || frame.some(p => !Array.isArray(p) || p.length !== 3 || !p.every(v => typeof v === 'number' && Number.isFinite(v))))) throw new Error(`Capture must contain 1 to ${maxFrames} frames of 110 finite [x,y,z] points. Recapture.`);
  if (requireHands && !frames.some(frame => frame.slice(0, 42).some(p => p.some(v => v !== 0)))) throw new Error('No hands detected. Keep your hands in view and recapture.');
}
