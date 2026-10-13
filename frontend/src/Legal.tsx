export function LegalPage({ kind }: { kind: 'privacy' | 'terms' }) {
  const privacy = kind === 'privacy';
  return <article className="legal-page">
    <span className="eyebrow">SIGNBRIDGE / {privacy ? 'PRIVACY' : 'TERMS'}</span>
    <h1>{privacy ? 'Privacy Policy' : 'Terms and Conditions'}</h1>
    <p className="lead">{privacy ? 'A plain-language overview of the information this frontend handles.' : 'The rules for using this demonstration and communication workspace.'}</p>
    {privacy ? <>
      <h2>What we process</h2><p>When you use live mode, the text, keywords, room messages, sign landmarks, and audio you submit are sent to the configured deployment so it can provide the requested feature. Demo mode keeps its room data and recordings in this browser tab.</p>
      <h2>Analytics choices</h2><p>Optional Vercel Analytics is loaded only after you select “Allow analytics” in the consent banner. Your choice is stored locally as <code>signbridge-analytics-consent</code>. Declining keeps analytics disabled.</p>
      <h2>Storage and retention</h2><p>The backend operator controls live deployment storage, retention, access, and provider usage. A room ID is not authentication. Do not submit information you are not permitted to share.</p>
      <h2>Your controls</h2><p>You can decline analytics, clear this site’s local storage, leave a room, and stop camera or microphone capture. Contact the deployment operator for requests about backend data.</p>
    </> : <>
      <h2>Use of the service</h2><p>Use SignBridge lawfully and only with content you have permission to process. Review generated text and transcripts before sending them to a shared room.</p>
      <h2>Availability and output</h2><p>The workspace depends on browser permissions and configured services. It may be unavailable or return incomplete results. Generated text and predictions require human review and are not professional advice.</p>
      <h2>Your responsibility</h2><p>You are responsible for the content you submit, the people you invite, and the security of any room ID you share. Do not rely on a room ID as an access credential.</p>
      <h2>Changes</h2><p>These terms may be updated as the project changes. Continued use after an update means you accept the revised text.</p>
    </>}
    <p className="caption">Last updated: October 13, 2026. Formal legal review is recommended before production use.</p>
  </article>;
}

export function NotFoundPage() { return <article className="legal-page"><span className="eyebrow">SIGNBRIDGE / 404</span><h1>That page is not here.</h1><p className="lead">The address may be outdated or typed incorrectly.</p><a className="button-link" href="/">Return to the workspace</a></article>; }
