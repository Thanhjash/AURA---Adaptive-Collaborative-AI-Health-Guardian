// src/lib/api.ts
import type { ChatRequest, ChatResponse } from '@/lib/types'

// Production-ready API configuration
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

// Request timeout configuration
const DEFAULT_TIMEOUT = 30000 // 30 seconds
const STREAM_TIMEOUT = 180000 // 3 minutes for streaming

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

// Enhanced request helper with timeout and retries
async function fetchWithTimeout(
  url: string, 
  options: RequestInit, 
  timeout = DEFAULT_TIMEOUT,
  retries = 1
): Promise<Response> {
  const controller = new AbortController()
  const timeoutId = setTimeout(() => controller.abort(), timeout)
  
  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        ...options.headers,
      },
    })
    
    clearTimeout(timeoutId)
    
    if (!response.ok) {
      let errorMessage = `HTTP ${response.status}: ${response.statusText}`
      try {
        const errorText = await response.text()
        if (errorText) {
          errorMessage += ` - ${errorText}`
        }
      } catch {
        // Ignore error text parsing failures
      }
      throw new APIError(errorMessage, response.status, response)
    }
    
    return response
  } catch (error) {
    clearTimeout(timeoutId)
    
    if (error instanceof APIError) {
      throw error
    }
    
    // Retry logic for network errors
    const isAbortError = error instanceof Error && error.name === 'AbortError'
    if (retries > 0 && isAbortError) {
      console.log(`Request timeout, retrying... (${retries} retries left)`)
      await new Promise(resolve => setTimeout(resolve, 1000))
      return fetchWithTimeout(url, options, timeout, retries - 1)
    }
    
    const errorMessage = isAbortError
      ? 'Request timeout - please check your connection'
      : error instanceof Error 
        ? `Network error: ${error.message}`
        : 'Unknown network error'
    
    throw new APIError(errorMessage, undefined, undefined)
  }
}

export const api = {
  // === STREAMING CHAT (Primary) ===
  // Note: Streaming is handled directly in ChatInterface.tsx using fetchEventSource
  
  // === NON-STREAMING CHAT (Fallback/Testing) ===
  async chat(data: ChatRequest): Promise<ChatResponse> {
    try {
      const response = await fetchWithTimeout(
        `${API_BASE_URL}/api/chat`,
        {
          method: 'POST',
          body: JSON.stringify(data),
        },
        STREAM_TIMEOUT, // Use longer timeout for complex processing
        2 // Retry twice for chat requests
      )
      
      return await response.json()
    } catch (error) {
      console.error('Chat API error:', error)
      throw error
    }
  },

  // === SESSION MANAGEMENT ===
  async getSessionHistory(sessionId: string) {
    try {
      const response = await fetchWithTimeout(
        `${API_BASE_URL}/api/chat/session/${encodeURIComponent(sessionId)}/history`,
        { method: 'GET' }
      )
      
      return await response.json()
    } catch (error) {
      console.error('Session history failed:', error)
      throw new APIError('Failed to load conversation history')
    }
  },

  async continueChat(data: { session_id: string; message: string }) {
    try {
      const response = await fetchWithTimeout(
        `${API_BASE_URL}/api/chat/continue`,
        {
          method: 'POST',
          body: JSON.stringify(data),
        }
      )
      
      return await response.json()
    } catch (error) {
      console.error('Continue chat failed:', error)
      throw error
    }
  },

  // === SYSTEM HEALTH ===
  async healthCheck() {
    try {
      const response = await fetchWithTimeout(
        `${API_BASE_URL}/health`,
        { 
          method: 'GET',
          headers: { 'Accept': 'application/json' }
        },
        5000 // Short timeout for health checks
      )
      
      return await response.json()
    } catch (error) {
      // Silent fail for health checks in production
      if (process.env.NODE_ENV === 'development') {
        console.error('Health check failed:', error)
      }
      return { 
        status: 'unhealthy', 
        error: 'Connection failed',
        service: 'aura_main'
      }
    }
  },

  async getServicesStatus() {
    try {
      const response = await fetchWithTimeout(
        `${API_BASE_URL}/api/services/status`,
        { method: 'GET' }
      )
      
      return await response.json()
    } catch (error) {
      if (process.env.NODE_ENV === 'development') {
        console.error('Services status failed:', error)
      }
      return { error: 'Connection failed' }
    }
  },

  // === EXPERT COUNCIL OBSERVABILITY ===
  async getCouncilSessions(limit = 10, successfulOnly = false) {
    try {
      const params = new URLSearchParams({
        limit: limit.toString(),
        include_successful_only: successfulOnly.toString()
      })
      
      const response = await fetchWithTimeout(
        `${API_BASE_URL}/api/council-sessions/recent?${params}`,
        { method: 'GET' }
      )
      
      return await response.json()
    } catch (error) {
      console.error('Council sessions failed:', error)
      throw error
    }
  },

  async getCouncilSession(sessionId: string) {
    try {
      const response = await fetchWithTimeout(
        `${API_BASE_URL}/api/council-sessions/${encodeURIComponent(sessionId)}`,
        { method: 'GET' }
      )
      
      return await response.json()
    } catch (error) {
      console.error('Council session fetch failed:', error)
      throw error
    }
  },

  async getCouncilAnalytics(days = 30) {
    try {
      const response = await fetchWithTimeout(
        `${API_BASE_URL}/api/analytics/council-performance?days=${days}`,
        { method: 'GET' }
      )
      
      return await response.json()
    } catch (error) {
      console.error('Council analytics failed:', error)
      throw error
    }
  },

  // === USER FEEDBACK & ANALYTICS ===
  async submitFeedback(data: {
    user_id: string;
    interaction_id: string;
    feedback: Record<string, any>;
  }) {
    try {
      const response = await fetchWithTimeout(
        `${API_BASE_URL}/api/feedback`,
        {
          method: 'POST',
          body: JSON.stringify(data),
        }
      )
      
      return await response.json()
    } catch (error) {
      console.error('Feedback submission failed:', error)
      throw error
    }
  },

  async getFeedbackAnalytics() {
    try {
      const response = await fetchWithTimeout(
        `${API_BASE_URL}/api/analytics/feedback`,
        { method: 'GET' }
      )
      
      return await response.json()
    } catch (error) {
      console.error('Feedback analytics failed:', error)
      throw error
    }
  },

  // === USER DATA MANAGEMENT ===
  async exportUserData(userId: string) {
    try {
      const response = await fetchWithTimeout(
        `${API_BASE_URL}/api/user/${encodeURIComponent(userId)}/export`,
        { method: 'GET' }
      )
      
      return await response.json()
    } catch (error) {
      console.error('User data export failed:', error)
      throw error
    }
  },

  async deleteUserAccount(userId: string) {
    try {
      const response = await fetchWithTimeout(
        `${API_BASE_URL}/api/user/${encodeURIComponent(userId)}`,
        { method: 'DELETE' }
      )
      
      return await response.json()
    } catch (error) {
      console.error('User account deletion failed:', error)
      throw error
    }
  },

  // === TESTING & DEBUGGING ===
  async testTriage(query: string, context = '') {
    try {
      const params = new URLSearchParams({
        query,
        context
      })
      
      const response = await fetchWithTimeout(
        `${API_BASE_URL}/api/triage/test?${params}`,
        { method: 'GET' }
      )
      
      return await response.json()
    } catch (error) {
      console.error('Triage test failed:', error)
      throw error
    }
  },

  async debugTriage() {
    try {
      const response = await fetchWithTimeout(
        `${API_BASE_URL}/api/triage/debug`,
        { method: 'GET' }
      )
      
      return await response.json()
    } catch (error) {
      console.error('Triage debug failed:', error)
      throw error
    }
  },

  // === KNOWLEDGE SEARCH ===
  async searchKnowledge(query: string, limit = 3) {
    try {
      const params = new URLSearchParams({
        query,
        limit: limit.toString()
      })
      
      const response = await fetchWithTimeout(
        `${API_BASE_URL}/api/knowledge/search?${params}`,
        { method: 'GET' }
      )
      
      return await response.json()
    } catch (error) {
      console.error('Knowledge search failed:', error)
      throw error
    }
  }
}

// Export configuration for external use
export { API_BASE_URL, APIError }

// Type exports for consistency
export type { ChatRequest, ChatResponse } from '@/lib/types'