// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { ANALYTICS_CONSENT_KEY, ConsentBanner } from './ConsentBanner';
import { App } from './App';

afterEach(() => { cleanup(); localStorage.clear(); });
beforeEach(() => localStorage.clear());
describe('privacy consent and routes', () => {
  it('stores an explicit analytics choice', () => {
    render(<ConsentBanner onChange={() => undefined}/>);
    fireEvent.click(screen.getByRole('button', { name: 'Decline analytics' }));
    expect(localStorage.getItem(ANALYTICS_CONSENT_KEY)).toBe('declined');
    expect(screen.queryByRole('dialog')).toBeNull();
  });
  it.each(['/privacy', '/terms', '/missing'])('renders the route %s', path => {
    history.replaceState({}, '', path);
    render(<App config={{ demo: true, backend: '', ml: '', siteUrl: 'http://localhost:5173' }}/>);
    expect(screen.getByRole('main')).toBeTruthy();
    expect(path === '/missing' ? screen.getByText('That page is not here.') : screen.getByRole('heading', { name: path === '/privacy' ? 'Privacy Policy' : 'Terms and Conditions' })).toBeTruthy();
  });
});
