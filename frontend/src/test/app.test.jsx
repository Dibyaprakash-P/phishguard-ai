import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { act, render, screen, waitFor } from '@testing-library/react'
import App from '../App'
import AppBackground from '../components/AppBackground'

const HEALTH = {
  status: 'ok',
  app: 'PhishGuard AI',
  version: '1.0.0',
  environment: 'test',
  model_loaded: true,
  model_name: 'hybrid_ensemble',
  llm: { configured: false, provider: 'none', model: null },
  agent_enabled: false,
}

beforeEach(() => {
  window.localStorage.clear()
  window.location.hash = ''
  vi.stubGlobal(
    'fetch',
    vi.fn(() =>
      Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve(HEALTH) }),
    ),
  )
  // jsdom does not implement matchMedia, which the risk gauge queries.
  vi.stubGlobal(
    'matchMedia',
    vi.fn(() => ({ matches: false, addEventListener: vi.fn(), removeEventListener: vi.fn() })),
  )
  window.scrollTo = vi.fn()
})

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('AppBackground', () => {
  it('paints the supplied artwork as a fixed, cover-sized layer', () => {
    const { container } = render(
      <AppBackground>
        <p>content</p>
      </AppBackground>,
    )
    const layer = container.querySelector('[style*="background-image"]')

    expect(layer).toBeTruthy()
    // The exact asset shipped in public/assets must be the one used.
    expect(layer.getAttribute('style')).toContain('/assets/image.png')
    expect(layer.className).toContain('fixed')
    expect(layer.className).toContain('bg-cover')
    expect(layer.className).toContain('bg-center')
  })

  it('keeps the artwork behind the content and out of the accessibility tree', () => {
    const { container } = render(
      <AppBackground>
        <p>content</p>
      </AppBackground>,
    )
    const layer = container.querySelector('[style*="background-image"]')
    expect(layer.getAttribute('aria-hidden')).toBe('true')
    expect(layer.className).toContain('-z-30')
    expect(screen.getByText('content')).toBeInTheDocument()
  })
})

describe('App shell', () => {
  /**
   * App probes /api/health on mount. Rendering inside act() and awaiting lets
   * that state update settle, so the tests assert on a settled tree instead of
   * racing it.
   */
  const renderApp = async () => {
    let result
    await act(async () => {
      result = render(<App />)
    })
    return result
  }

  it('renders the hero and the scanner by default', async () => {
    await renderApp()

    expect(screen.getByText(/AI-Powered Cybersecurity/i)).toBeInTheDocument()
    expect(screen.getByText(/Detect Phishing\./i)).toBeInTheDocument()
    expect(screen.getByLabelText(/Enter a URL to analyze/i)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Analyze URL/i })).toBeInTheDocument()
  })

  it('states the static-analysis guarantee prominently', async () => {
    await renderApp()
    expect(screen.getAllByText(/never visits the URL|static analysis/i).length).toBeGreaterThan(0)
  })

  it('reflects backend health once the probe resolves', async () => {
    await renderApp()
    await waitFor(() => expect(screen.getByText(/Model online/i)).toBeInTheDocument())
  })

  it('reports a connection problem instead of pretending to be healthy', async () => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.reject(new Error('offline'))))
    await renderApp()
    await waitFor(() => expect(screen.getByText(/Connecting…/i)).toBeInTheDocument())
  })

  it('applies the glass system to the scanner card', async () => {
    const { container } = await renderApp()
    expect(container.querySelector('.glass-strong')).toBeTruthy()
  })
})
