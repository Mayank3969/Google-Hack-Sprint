import { afterEach, expect, it, vi } from 'vitest';
import { demoAdapter, type Message } from './api';
import { pollMessages } from './polling';
afterEach(() => vi.useRealTimers());
it('serializes requests, suspends while hidden and ignores late results after stop', async () => {
  vi.useFakeTimers();
  let visibleChange = () => {};
  const doc = { hidden: false, addEventListener: vi.fn((_type, callback) => { visibleChange = callback; }), removeEventListener: vi.fn() };
  let resolve!: (messages: Message[]) => void;
  const messages = vi.fn(() => new Promise<Message[]>(r => { resolve = r; }));
  const onMessages = vi.fn(); const loop = pollMessages({ ...demoAdapter(), messages }, 'room', onMessages, vi.fn(), doc);
  await loop.refresh(); await vi.advanceTimersByTimeAsync(10000); expect(messages).toHaveBeenCalledTimes(1);
  resolve([]); await Promise.resolve(); doc.hidden = true; visibleChange(); await vi.advanceTimersByTimeAsync(10000); expect(messages).toHaveBeenCalledTimes(1);
  doc.hidden = false; visibleChange(); expect(messages).toHaveBeenCalledTimes(2);
  loop.stop(); resolve([]); await Promise.resolve(); expect(onMessages).toHaveBeenCalledTimes(1); expect(doc.removeEventListener).toHaveBeenCalled();
});
