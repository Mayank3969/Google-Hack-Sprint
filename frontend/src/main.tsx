import React from 'react';
import { createRoot } from 'react-dom/client';
import { App } from './App';
import { parseConfig } from './config';
import '@fontsource/newsreader/500.css';
import '@fontsource/plus-jakarta-sans/400.css';
import '@fontsource/plus-jakarta-sans/600.css';
import './styles.css';

const root = createRoot(document.getElementById('root')!);
try { root.render(<React.StrictMode><App config={parseConfig(import.meta.env)} /></React.StrictMode>); }
catch (error) { root.render(<main className="setup"><span className="eyebrow">SIGNBRIDGE / CONFIGURATION</span><h1>Let’s connect thoughtfully.</h1><p role="alert">{error instanceof Error ? error.message : 'Configuration could not be loaded.'}</p><p>Copy <code>.env.example</code> to <code>.env.local</code>, choose demo or live mode explicitly, then restart the development server or rebuild your deployment.</p></main>); }
