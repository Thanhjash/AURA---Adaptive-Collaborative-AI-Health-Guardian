// src/lib/api.ts
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export const api = {
  async chat(data: { 
    query: string; 
    user_id: string; 
    session_id?: string; 
    force_expert_council?: boolean 
  }) {
    const response = await fetch(`${API_BASE_URL}/api/chat`, {
      method: 'POST',
      headers: { 
        'Content-Type': 'application/json',
        'Accept': 'application/json'
      },
      body: JSON.stringify(data),
    });
    
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`API Error ${response.status}: ${errorText}`);
    }
    
    return response.json();
  },

  async getSessionHistory(sessionId: string) {
    try {
      const response = await fetch(`${API_BASE_URL}/api/chat/session/${sessionId}/history`, {
        method: 'GET',
        headers: { 'Accept': 'application/json' }
      });
      
      if (!response.ok) {
        throw new Error(`Failed to load session: ${response.status}`);
      }
      
      return response.json();
    } catch (error) {
      console.error('Session history failed:', error);
      throw error;
    }
  },

  async continueChat(data: {
    session_id: string;
    message: string;
  }) {
    const response = await fetch(`${API_BASE_URL}/api/chat/continue`, {
      method: 'POST',
      headers: { 
        'Content-Type': 'application/json',
        'Accept': 'application/json'
      },
      body: JSON.stringify(data),
    });
    
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`API Error ${response.status}: ${errorText}`);
    }
    
    return response.json();
  },

  async healthCheck() {
    try {
      const response = await fetch(`${API_BASE_URL}/health`, {
        method: 'GET',
        headers: { 'Accept': 'application/json' }
      });
      return response.json();
    } catch (error) {
      console.error('Health check failed:', error);
      return { status: 'unhealthy', error: 'Connection failed' };
    }
  },

  async getCouncilSessions(limit = 10) {
    const response = await fetch(
      `${API_BASE_URL}/api/council-sessions/recent?limit=${limit}`
    );
    return response.json();
  },

  async submitFeedback(data: {
    user_id: string;
    interaction_id: string;
    feedback: Record<string, any>;
  }) {
    const response = await fetch(`${API_BASE_URL}/api/feedback`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });
    return response.json();
  },

  async getServicesStatus() {
    try {
      const response = await fetch(`${API_BASE_URL}/api/services/status`);
      return response.json();
    } catch (error) {
      console.error('Services status failed:', error);
      return { error: 'Connection failed' };
    }
  }
};

export { API_BASE_URL };