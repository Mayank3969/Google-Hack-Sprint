import { useEffect, useState } from 'react';

export const ANALYTICS_CONSENT_KEY = 'signbridge-analytics-consent';
export type Consent = 'accepted' | 'declined' | null;
export function readConsent(): Consent {
  const value = localStorage.getItem(ANALYTICS_CONSENT_KEY);
  return value === 'accepted' || value === 'declined' ? value : null;
}
export function ConsentBanner({ onChange }: { onChange: (value: Consent) => void }) {
  const [visible, setVisible] = useState(false);
  useEffect(() => setVisible(readConsent() === null), []);
  if (!visible) return null;
  const choose = (value: Consent) => { localStorage.setItem(ANALYTICS_CONSENT_KEY, value!); setVisible(false); onChange(value); };
  return <aside className="cookie-banner" role="dialog" aria-label="Analytics privacy choices">
    <div><strong>Help us improve SignBridge</strong><p>Optional, privacy-friendly analytics help us understand page visits. They stay off unless you choose to allow them. Read our <a href="/privacy">Privacy Policy</a>.</p></div>
    <div className="cookie-actions"><button className="secondary" onClick={() => choose('declined')}>Decline analytics</button><button onClick={() => choose('accepted')}>Allow analytics</button></div>
  </aside>;
}
