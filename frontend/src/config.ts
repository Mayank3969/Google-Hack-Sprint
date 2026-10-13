export interface PublicEnv { VITE_DEMO_MODE?: string; VITE_BACKEND_BASE_URL?: string; VITE_ML_BASE_URL?: string }
export interface Config { demo: boolean; backend: string; ml: string }
export function parseConfig(env: PublicEnv): Config {
  if (env.VITE_DEMO_MODE !== 'true' && env.VITE_DEMO_MODE !== 'false') throw new Error('Set VITE_DEMO_MODE explicitly to true or false. See frontend/.env.example.');
  const demo = env.VITE_DEMO_MODE === 'true';
  function endpoint(value: string | undefined, name: string) {
    if (!value?.trim()) { if (demo) return ''; throw new Error(`${name} is required in live mode.`); }
    let url: URL;
    try { url = new URL(value.trim()); } catch { throw new Error(`${name} must be an absolute HTTP(S) URL.`); }
    if (!['https:', 'http:'].includes(url.protocol) || url.username || url.password || url.search || url.hash) throw new Error(`${name} must be a public HTTP(S) base URL without credentials, query or fragment.`);
    return url.href.replace(/\/$/, '');
  }
  return { demo, backend: endpoint(env.VITE_BACKEND_BASE_URL, 'VITE_BACKEND_BASE_URL'), ml: endpoint(env.VITE_ML_BASE_URL, 'VITE_ML_BASE_URL') };
}
