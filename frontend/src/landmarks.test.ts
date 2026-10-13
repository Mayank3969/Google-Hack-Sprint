import { describe, expect, it, vi } from 'vitest';
import { FACE_INDICES, featureFrame, validateLandmarks } from './landmarks';
import { liveAdapter } from './api';
const hand = (x: number) => Array.from({ length: 21 }, () => ({ x, y: 0.2, z: -0.3 }));
export const validFrame = () => featureFrame({ landmarks: [hand(0.1)], handedness: [[{ categoryName: 'Left' }]] }, { faceLandmarks: [] });
describe('compact-v1 contract', () => {
  it('uses handedness rather than detection order, preserves unmirrored coordinates/z and exact unsorted face groups', () => {
    const face = Array.from({ length: 478 }, (_, i) => ({ x: i / 478, y: 0.5, z: -i }));
    const result = featureFrame({ landmarks: [hand(0.8), hand(0.1)], handedness: [[{ categoryName: 'Right' }], [{ categoryName: 'Left' }]] }, { faceLandmarks: [face] });
    expect(result).toHaveLength(110); expect(result[0]).toEqual([0.1, 0.2, -0.3]); expect(result[21]).toEqual([0.8, 0.2, -0.3]);
    const expected = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246,263,249,390,373,374,380,381,382,362,398,384,385,386,387,388,466,70,63,105,66,107,336,296,334,293,300,61,146,91,181,84,17,314,405,321,375,78,95,88,178,87,14,317,402,318,324,1,2,4,5,6,168];
    expect(FACE_INDICES).toEqual(expected); expect(result.slice(42)).toEqual(expected.map(i => [i / 478, 0.5, -i]));
  });
  it('zero fills missing blocks, uses training fallback handedness, requires 478 face points and blocks handless prediction', () => {
    const frame = validFrame(); expect(frame.slice(21)).toEqual(Array.from({ length: 89 }, () => [0, 0, 0]));
    expect(featureFrame({ landmarks: [hand(0.1)], handedness: [] }, { faceLandmarks: [] })).toEqual(frame);
    expect(() => featureFrame({ landmarks: [], handedness: [] }, { faceLandmarks: [hand(0.1)] })).toThrow(/478/);
    expect(() => validateLandmarks([featureFrame({ landmarks: [], handedness: [] }, { faceLandmarks: [] })])).toThrow(/No hands/);
  });
  it('bounds input to 1–60 finite 110x3 frames', () => {
    expect(() => validateLandmarks([validFrame()])).not.toThrow();
    expect(() => validateLandmarks(Array.from({ length: 60 }, validFrame))).not.toThrow();
    for (const value of [[], Array.from({ length: 61 }, validFrame), [validFrame().slice(0, 21)], [[[NaN, 0, 0], ...validFrame().slice(1)]]]) expect(() => validateLandmarks(value as ReturnType<typeof validFrame>[])).toThrow();
  });
  it('sends landmarks only and accepts keywords without confidence', async () => {
    const fetcher = vi.fn<typeof fetch>().mockResolvedValue(new Response(JSON.stringify({ keywords: ['pay'] })));
    const api = liveAdapter({ demo: false, backend: 'https://backend.example', ml: 'https://ml.example' }, fetcher);
    expect(await api.predict([validFrame()])).toEqual(['pay']);
    expect(fetcher.mock.calls[0][0]).toBe('https://ml.example/predict');
    expect(JSON.parse(fetcher.mock.calls[0][1]!.body as string)).toEqual({ landmarks: [validFrame()] });
    await expect(api.predict([])).rejects.toThrow(); expect(fetcher).toHaveBeenCalledTimes(1);
  });
  it.each([422, 503])('returns actionable ML %i without private service details', async status => {
    const api = liveAdapter({ demo: false, backend: '', ml: '' }, vi.fn<typeof fetch>().mockResolvedValue(new Response('{"detail":"PRIVATE"}', { status })));
    await expect(api.predict([validFrame()])).rejects.toThrow(status === 503 ? /model is unavailable/ : /110-point capture/);
  });
});
