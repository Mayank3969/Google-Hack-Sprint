import { describe, expect, it, vi } from 'vitest';
import { liveAdapter } from './api';
import type { Frame } from './landmarks';

const config = { demo: false, backend: 'https://api.example.com', ml: 'https://ml.example.com' };
const frame: Frame = Array.from({ length: 110 }, () => [0.1, 0.2, -0.3]);
const segment = (index: number) => ({ segment_index: index, segment_id: `segment-${index}`, landmarks: [frame] });
const response = (value: unknown, status = 200) => new Response(JSON.stringify(value), { status });

describe('ML v2 batch integration', () => {
  it('submits every segment once and returns results in segment order', async () => {
    const fetcher = vi.fn<typeof fetch>().mockResolvedValue(response({
      capture_id: 'capture-1', complete: true, keywords: ['hello', 'friend'],
      results: [
        { segment_index: 1, status: 'recognized', keyword: 'friend', confidence: 0.8, coverage: 0.9 },
        { segment_index: 0, status: 'recognized', keyword: 'hello', confidence: 0.9, coverage: 1 },
      ],
    }));
    const result = await liveAdapter(config, fetcher).predictBatch('capture-1', [segment(0), segment(1)]);
    expect(fetcher).toHaveBeenCalledTimes(1);
    expect(fetcher.mock.calls[0][0]).toBe('https://ml.example.com/v2/predict/batch');
    expect(result.results.map(item => item.keyword)).toEqual(['hello', 'friend']);
    expect(JSON.parse(fetcher.mock.calls[0][1]!.body as string).segments).toHaveLength(2);
  });

  it('keeps incomplete statuses and does not expose provider details', async () => {
    const fetcher = vi.fn<typeof fetch>().mockResolvedValue(response({
      capture_id: 'capture-1', complete: false, keywords: [],
      results: [{ segment_index: 0, status: 'insufficient_data', keyword: null, confidence: null, coverage: 0 }],
    }));
    const result = await liveAdapter(config, fetcher).predictBatch('capture-1', [segment(0)]);
    expect(result.complete).toBe(false);
    expect(result.results[0].status).toBe('insufficient_data');

    fetcher.mockResolvedValueOnce(response({ detail: 'private landmarks' }, 422));
    await expect(liveAdapter(config, fetcher).predictBatch('capture-1', [segment(0)])).rejects.toThrow(/ML rejected/);
    await expect(liveAdapter(config, fetcher).predictBatch('capture-1', [segment(0)])).rejects.not.toThrow('private');
  });

  it('reads coordinate coverage from the ML v2 quality envelope', async () => {
    const fetcher = vi.fn<typeof fetch>().mockResolvedValue(response({
      capture_id: 'capture-1', complete: true, keywords: ['hello'],
      results: [{
        segment_index: 0, status: 'recognized', keyword: 'hello', confidence: 0.9,
        quality: { coordinate_coverage: 0.875, left_hand_coverage: 1, right_hand_coverage: 1, face_coverage: 0.8 },
        top_k: [{ class_id: 1, label: 'hello', confidence: 0.9 }],
      }],
    }));
    const result = await liveAdapter(config, fetcher).predictBatch('capture-1', [segment(0)]);
    expect(result.results[0].coverage).toBe(0.875);
  });
});
