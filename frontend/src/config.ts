export interface PublicEnv { VITE_DEMO_MODE?: string; VITE_BACKEND_BASE_URL?: string; VITE_ML_BASE_URL?: string; VITE_SITE_URL?: string }
export interface Config { demo: boolean; backend: string; ml: string; siteUrl?: string }
export function parseConfig(env: PublicEnv): Config {
  if (env.VITE_DEMO_MODE !== 'true' && env.VITE_DEMO_MODE !== 'false') throw new Error('Set VITE_DEMO_MODE explicitly to true or false. See frontend/.env.example.');
  const demo = env.VITE_DEMO_MODE === 'true';
  function endpoint(value: string | undefined, name: string) {
    if (!value?.trim()) { if (demo) return ''; throw new Error(`${name} is required in live mode.`); }
    let url: URL;
    try { url = new URL(value.trim()); } catch { throw new Error(`${name} must be an absolute HTTP(S) URL.`); }
    const local = url.hostname === 'localhost' || url.hostname === '127.0.0.1' || url.hostname === '[::1]';
    if (!['https:', 'http:'].includes(url.protocol) || (url.protocol === 'http:' && !local) || url.username || url.password || url.search || url.hash) throw new Error(`${name} must use HTTPS in production and contain no credentials, query or fragment.`);
    return url.href.replace(/\/$/, '');
  }
  const siteValue = env.VITE_SITE_URL?.trim() || (demo ? 'http://localhost:5173' : '');
  if (!siteValue) throw new Error('VITE_SITE_URL is required in live mode.');
  const siteUrl = endpoint(siteValue, 'VITE_SITE_URL');
  return { demo, backend: endpoint(env.VITE_BACKEND_BASE_URL, 'VITE_BACKEND_BASE_URL'), ml: endpoint(env.VITE_ML_BASE_URL, 'VITE_ML_BASE_URL'), siteUrl };
}
