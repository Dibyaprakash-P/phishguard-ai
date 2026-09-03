/**
 * Single point of contact with the PhishGuard AI backend.
 *
 * No component builds a URL or calls fetch directly — everything goes through
 * here, so the base URL lives in exactly one place (the VITE_API_URL
 * environment variable) and error shapes are normalised once.
 */

const BASE_URL = (import.meta.env.VITE_API_URL || 'http://localhost:8000').replace(/\/+$/, '')

/** Requests are abandoned after this long so the UI can never hang forever. */
const DEFAULT_TIMEOUT_MS = 45_000

/**
 * Error carrying the backend's structured `{ error, message, detail }` payload
 * so the UI can show a specific message instead of a generic failure.
 */
export class ApiError extends Error {
  constructor(message, { code = 'request_failed', status = 0, detail = null } = {}) {
    super(message)
    this.name = 'ApiError'
    this.code = code
    this.status = status
    this.detail = detail
  }
}

async function request(path, { method = 'GET', body, timeout = DEFAULT_TIMEOUT_MS } = {}) {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeout)

  let response
  try {
    response = await fetch(`${BASE_URL}${path}`, {
      method,
      headers: body ? { 'Content-Type': 'application/json' } : undefined,
      body: body ? JSON.stringify(body) : undefined,
      signal: controller.signal,
    })
  } catch (error) {
    clearTimeout(timer)
    if (error.name === 'AbortError') {
      throw new ApiError(
        'The request timed out. The backend may be busy or unreachable.',
        { code: 'timeout' },
      )
    }
    throw new ApiError(
      `Cannot reach the PhishGuard API at ${BASE_URL}. Make sure the backend is running.`,
      { code: 'network_error', detail: error.message },
    )
  }
  clearTimeout(timer)

  let payload = null
  try {
    payload = await response.json()
  } catch {
    payload = null
  }

  if (!response.ok) {
    throw new ApiError(
      payload?.message || `Request failed with status ${response.status}.`,
      { code: payload?.error || 'request_failed', status: response.status, detail: payload?.detail },
    )
  }

  return payload
}

export const api = {
  baseUrl: BASE_URL,

  /** Service health, including whether the model and an LLM are available. */
  health: () => request('/api/health', { timeout: 8000 }),

  /** Full analysis of one URL. */
  analyze: (url, { includeAiExplanation = true } = {}) =>
    request('/api/analyze', {
      method: 'POST',
      body: { url, include_ai_explanation: includeAiExplanation },
    }),

  /** Classify several URLs at once (no LLM explanations). */
  batchAnalyze: (urls) =>
    request('/api/batch-analyze', { method: 'POST', body: { urls } }),

  /** Trained-model metadata and measured evaluation metrics. */
  modelInfo: () => request('/api/model-info'),

  /** Optional LangGraph agent path. */
  agentAnalyze: (url) =>
    request('/api/agent-analyze', { method: 'POST', body: { url } }),
}
