// frontend/src/lib/api.ts (VERSION 12.4 - SYNCED & MINIMAL)

// API_BASE_URL:
// - In local development, leave as empty string so Next.js proxies to http://localhost:8000
// - In production, set NEXT_PUBLIC_API_URL to "https://your-domain.io/api"
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || ''

/**
 * Custom error type for API failures
 */
class APIError extends Error {
  constructor(
    message: string,
    public status?: number,
    public response?: Response
  ) {
    super(message)
    this.name = 'APIError'
  }
}

/**
 * fetch wrapper with timeout support
 */
async function fetchWithTimeout(
  url: string,
  options: RequestInit,
  timeout = 15000 // default 15 seconds
): Promise<Response> {
  const controller = new AbortController()
  const timeoutId = setTimeout(() => controller.abort(), timeout)

  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
      headers: {
        'Content-Type': 'application/json',
        Accept: 'application/json',
        ...options.headers,
      },
    })
    clearTimeout(timeoutId)

    if (!response.ok) {
      const errorText = await response.text().catch(() => '')
      throw new APIError(
        `HTTP ${response.status}: ${response.statusText}${errorText ? ' – ' + errorText : ''}`,
        response.status,
        response
      )
    }
    return response
  } catch (err) {
    clearTimeout(timeoutId)
    if (err instanceof APIError) throw err

    const isAbort = err instanceof Error && err.name === 'AbortError'
    const message = isAbort
      ? 'Request timed out—please check your connection.'
      : err instanceof Error
        ? `Network error: ${err.message}`
        : 'Unknown network error'
    throw new APIError(message)
  }
}

/**
 * Minimal API client, aligned with Backend V12
 */
export const api = {
  /**
   * Check the health of the backend service.
   * - Local: GET http://localhost:3000/health  (Next.js proxy → backend)
   * - Prod:  GET https://your-domain.io/api/health
   */
  async healthCheck() {
    try {
      const res = await fetchWithTimeout(`${API_BASE_URL}/health`, { method: 'GET' }, 5000)
      return await res.json()
    } catch (err) {
      if (process.env.NODE_ENV === 'development') {
        console.error('Health check failed:', err)
      }
      return { status: 'unhealthy', error: 'Unable to connect to backend' }
    }
  },

  /**
   * Load the full message history for a given session ID.
   * Backend endpoint: GET /get-session-history/{sessionId}
   */
  async getSessionHistory(sessionId: string) {
    if (!sessionId) {
      throw new APIError('Session ID is required to fetch history.')
    }
    try {
      const res = await fetchWithTimeout(
        `${API_BASE_URL}/get-session-history/${encodeURIComponent(sessionId)}`,
        { method: 'GET' }
      )
      return await res.json()
    } catch (err) {
      console.error(`Failed to load history for session ${sessionId}:`, err)
      throw new APIError('Failed to load conversation history.')
    }
  },
}
