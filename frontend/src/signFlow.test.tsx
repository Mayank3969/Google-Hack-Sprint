// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { App } from './App';
import { SignCapture } from './capture';
import type { Frame } from './landmarks';
vi.mock('./capture', () => ({ SignCapture: vi.fn() }));
const config = { demo: false, backend: 'https://backend.example', ml: 'https://ml.example' };
const frame: Frame = Array.from({ length: 110 }, () => [0.1, 0.2, -0.3]);
let fetcher: ReturnType<typeof vi.fn<typeof fetch>>, dispose: ReturnType<typeof vi.fn>, prepare: ReturnType<typeof vi.fn>;
let complete: (frames: Frame[]) => void;
beforeEach(() => {
  history.replaceState({}, '', '/'); dispose = vi.fn(); prepare = vi.fn(async () => {});
  vi.mocked(SignCapture).mockImplementation(function () { return { prepare, dispose, start: vi.fn(async (done) => { complete = done; }), stop: vi.fn(() => complete([frame])) } as unknown as SignCapture; });
  fetcher = vi.fn<typeof fetch>(); vi.stubGlobal('fetch', fetcher); vi.stubGlobal('speechSynthesis', { cancel: vi.fn() });
});
afterEach(() => { cleanup(); vi.unstubAllGlobals(); vi.clearAllMocks(); });
async function capture() {
  fireEvent.click(screen.getByRole('button', { name: 'Enable camera' }));
  fireEvent.click(await screen.findByRole('button', { name: 'Capture sign' }));
  fireEvent.click(screen.getByRole('button', { name: 'Finish capture' }));
}
it('requires capture, explicit prediction, reviewed append and explicit generation; record-again recaptures without repeated POST', async () => {
  fetcher.mockResolvedValueOnce(new Response('{"keywords":["pay"]}')).mockResolvedValueOnce(new Response('{"status":"record_again","record_again":true}'));
  render(<App config={config}/>); expect(SignCapture).not.toHaveBeenCalled(); expect(fetcher).not.toHaveBeenCalled();
  await capture(); expect(fetcher).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole('button', { name: 'Predict sign' }));
  await screen.findByText('pay'); expect((screen.getByLabelText('What would you like to say?') as HTMLInputElement).value).toBe('');
  fireEvent.click(screen.getByRole('button', { name: 'Append reviewed words' }));
  expect((screen.getByLabelText('What would you like to say?') as HTMLInputElement).value).toBe('pay'); expect(fetcher).toHaveBeenCalledTimes(1);
  fireEvent.click(screen.getByRole('button', { name: 'Generate sentence →' }));
  fireEvent.click(await screen.findByRole('button', { name: 'Recapture signs' }));
  expect((screen.getByLabelText('What would you like to say?') as HTMLInputElement).value).toBe('');
  expect(screen.queryByRole('button', { name: '▶ Play audio' })).toBeNull(); expect(screen.getByRole('button', { name: 'Enable camera' })).toBeTruthy();
  expect(fetcher).toHaveBeenCalledTimes(2); expect(JSON.parse(fetcher.mock.calls[1][1]!.body as string)).toEqual({ keywords: ['pay'] });
});
it('supports explicit prediction retry on503 and cancels/ignores late predictions on navigation', async () => {
  let resolve!: (response: Response) => void;
  fetcher.mockResolvedValueOnce(new Response('{}', { status: 503 })).mockImplementationOnce(() => new Promise(r => { resolve = r; }));
  render(<App config={config}/>); await capture(); fireEvent.click(screen.getByRole('button', { name: 'Predict sign' }));
  await screen.findByText(/Sign model is unavailable/);
  fireEvent.click(screen.getByRole('button', { name: 'Predict sign' }));
  fireEvent.click(screen.getByRole('link', { name: 'Settings & Help' }));
  expect(fetcher.mock.calls[1][1]!.signal!.aborted).toBe(true);
  await act(async () => resolve(new Response('{"keywords":["pay"]}')));
  fireEvent.click(screen.getByRole('link', { name: 'Workspace' })); expect(screen.queryByText('pay')).toBeNull();
});
it('cleans pending capture on mode changes, handles denied permissions, and blocks all-zero hands locally', async () => {
  prepare.mockRejectedValueOnce(new DOMException('denied', 'NotAllowedError'));
  render(<App config={config}/>); fireEvent.click(screen.getByRole('button', { name: 'Enable camera' }));
  await screen.findByText(/Camera permission denied/); expect(dispose).toHaveBeenCalled();
  fireEvent.click(screen.getByRole('button', { name: 'Enable camera' })); fireEvent.click(await screen.findByRole('button', { name: 'Capture sign' }));
  act(() => complete([Array.from({ length: 110 }, () => [0, 0, 0])])); await screen.findByText(/No hands detected/); expect(fetcher).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole('button', { name: 'Enable camera' })); await screen.findByRole('button', { name: 'Capture sign' });
  fireEvent.click(screen.getByRole('button', { name: 'Speech → text' })); expect(dispose).toHaveBeenCalledTimes(2);
});
it('never initializes models/camera or network in demo', async () => {
  render(<App config={{ ...config, demo: true }}/>);
  expect(screen.queryByRole('button', { name: 'Enable camera' })).toBeNull();
  await waitFor(() => { expect(SignCapture).not.toHaveBeenCalled(); expect(fetcher).not.toHaveBeenCalled(); });
});
