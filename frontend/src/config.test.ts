import { describe, expect, it } from 'vitest';
import { parseConfig } from './config';
describe('explicit public configuration', () => {
  it('never silently chooses a mode or localhost', () => {
    expect(() => parseConfig({})).toThrow('explicitly');
    expect(() => parseConfig({ VITE_DEMO_MODE: 'TRUE' })).toThrow();
    expect(() => parseConfig({ VITE_DEMO_MODE: 'false' })).toThrow('required');
    expect(parseConfig({ VITE_DEMO_MODE: 'true' })).toEqual({ demo: true, backend: '', ml: '', siteUrl: 'http://localhost:5173' });
  });
  it('normalizes configured endpoints and rejects unsafe URLs', () => {
    expect(parseConfig({ VITE_DEMO_MODE: 'false', VITE_SITE_URL: 'https://signbridge.example', VITE_BACKEND_BASE_URL: 'https://api.example.com/', VITE_ML_BASE_URL: 'https://ml.example.com' }).backend).toBe('https://api.example.com');
    expect(() => parseConfig({ VITE_DEMO_MODE: 'false', VITE_SITE_URL: 'http://signbridge.example', VITE_BACKEND_BASE_URL: 'https://api.example.com', VITE_ML_BASE_URL: 'https://ml.example.com' })).toThrow('HTTPS');
    expect(parseConfig({ VITE_DEMO_MODE: 'false', VITE_SITE_URL: 'https://signbridge.example', VITE_BACKEND_BASE_URL: 'http://localhost:8000', VITE_ML_BASE_URL: 'http://127.0.0.1:8001' }).backend).toBe('http://localhost:8000');
    for (const url of ['javascript:alert(1)', 'https://user:secret@example.com', '/api', 'https://example.com/?key=a']) expect(() => parseConfig({ VITE_DEMO_MODE: 'true', VITE_BACKEND_BASE_URL: url })).toThrow();
  });
});
