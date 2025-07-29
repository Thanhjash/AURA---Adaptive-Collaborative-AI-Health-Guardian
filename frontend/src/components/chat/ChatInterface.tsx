'use client'

import React, { useState, useEffect, useRef, useCallback } from 'react'
import { fetchEventSource } from '@microsoft/fetch-event-source'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { 
  Send, 
  Bot, 
  User, 
  AlertTriangle, 
  CheckCircle, 
  Clock,
  Activity,
  Brain,
  Menu,
  Plus,
  History,
  Loader2,
  Trash2
} from 'lucide-react'
import { api } from '@/lib/api'
import type { 
  SystemHealth,
  StructuredAnalysis,
  InteractiveComponents
} from '@/lib/types'

interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: string
  data?: {
    structured_analysis?: StructuredAnalysis
    interactive_components?: InteractiveComponents
    reasoning_trace?: any
    confidence?: number
  }
}

interface ChatHistory {
  session_id: string
  title: string
  timestamp: string
}

interface CouncilStep {
  step: number
  status: string
  description: string
}

// Production API endpoint configuration
const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export function ChatInterface() {
  // === CORE STATE ===
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  
  // === SESSION MANAGEMENT (FIXED) ===
  const [currentSession, setCurrentSession] = useState<string | null>(() => {
    // Only access sessionStorage on client side
    if (typeof window !== 'undefined') {
      const saved = window.sessionStorage.getItem('aura_active_session')
      return saved || null
    }
    return null
  })
  
  // === STREAMING STATE ===
  const [isStreaming, setIsStreaming] = useState(false)
  const [streamStatus, setStreamStatus] = useState('')
  const [councilStep, setCouncilStep] = useState<CouncilStep | null>(null)
  const [lastRequestTime, setLastRequestTime] = useState(0)
  const REQUEST_DEBOUNCE_MS = 2000
  
  // === UI STATE ===
  const [systemHealth, setSystemHealth] = useState<SystemHealth | null>(null)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [chatHistory, setChatHistory] = useState<ChatHistory[]>([])
  
  // === REFS ===
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)
  const abortControllerRef = useRef<AbortController | null>(null)
  const isMountedRef = useRef(true)

  // === SESSION MANAGEMENT (IMPROVED) ===
  const setActiveSession = useCallback((sessionId: string | null) => {
    if (sessionId !== currentSession) {
      setCurrentSession(sessionId)
      if (typeof window !== 'undefined') {
        if (sessionId) {
          window.sessionStorage.setItem('aura_active_session', sessionId)
        } else {
          window.sessionStorage.removeItem('aura_active_session')
        }
      }
    }
  }, [currentSession])

  // === EFFECTS ===
  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [])

  useEffect(() => {
    scrollToBottom()
  }, [messages, isStreaming])

  useEffect(() => {
    isMountedRef.current = true
    
    // Load system health (silent fail in production)
    api.healthCheck()
      .then(setSystemHealth)
      .catch(() => {
        if (process.env.NODE_ENV === 'development') {
          console.error('Health check failed')
        }
      })
    
    loadChatHistory()
    
    // Focus input after mount
    setTimeout(() => {
      inputRef.current?.focus()
    }, 100)
    
    return () => {
      isMountedRef.current = false
      if (abortControllerRef.current) {
        abortControllerRef.current.abort()
      }
    }
  }, [])

  // === CHAT HISTORY (IMPROVED) ===
  const loadChatHistory = async () => {
    if (typeof window === 'undefined') return
    
    try {
      const savedHistory = localStorage.getItem('aura_session_list')
      if (savedHistory) {
        const parsed = JSON.parse(savedHistory)
        setChatHistory(Array.isArray(parsed) ? parsed : [])
      }
    } catch (error) {
      // Silent fail, clear corrupted data
      localStorage.removeItem('aura_session_list')
      setChatHistory([])
    }
  }

  const saveSessionToList = useCallback((sessionId: string, title: string) => {
    if (typeof window === 'undefined') return
    
    const sessionItem: ChatHistory = {
      session_id: sessionId,
      title: title.slice(0, 50) + (title.length > 50 ? '...' : ''),
      timestamp: new Date().toISOString()
    }
    
    setChatHistory(prev => {
      const filtered = prev.filter(h => h.session_id !== sessionId)
      const updated = [sessionItem, ...filtered].slice(0, 20) // Keep max 20 sessions
      
      try {
        localStorage.setItem('aura_session_list', JSON.stringify(updated))
      } catch (error) {
        // Storage full, keep only recent 10
        const reduced = updated.slice(0, 10)
        localStorage.setItem('aura_session_list', JSON.stringify(reduced))
        return reduced
      }
      
      return updated
    })
  }, [])

  const clearAllHistory = () => {
    if (typeof window === 'undefined') return
    
    localStorage.removeItem('aura_session_list')
    setChatHistory([])
    handleNewChat()
  }

  // === MAIN STREAMING LOGIC (IMPROVED) ===

  const handleSend = async () => {
    const now = Date.now()
    
    // Enhanced debouncing
    if (now - lastRequestTime < REQUEST_DEBOUNCE_MS) {
      console.log('Request debounced')
      return
    }
    
    if (!input.trim() || isStreaming) {
      return
    }
    
    if (!isMountedRef.current) {
      return
    }

    setLastRequestTime(now)

    // Abort any existing request
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
      await new Promise(resolve => setTimeout(resolve, 100))
    }

    const userMessage: ChatMessage = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: input,
      timestamp: new Date().toISOString()
    }

    const currentInput = input
    setInput('')
    setMessages(prev => [...prev, userMessage])

    // Create assistant message placeholder
    const assistantId = `assistant-${Date.now()}`
    const assistantMessage: ChatMessage = {
      id: assistantId,
      role: 'assistant',
      content: '',
      timestamp: new Date().toISOString()
    }

    setMessages(prev => [...prev, assistantMessage])
    setIsStreaming(true)
    setStreamStatus('Connecting...')
    setCouncilStep(null)

    const controller = new AbortController()
    abortControllerRef.current = controller

    let finalContent = ''
    let structuredData: any = null
    let newSessionId: string | null = null

    // 🔍 FIXED: Properly handle session_id
    const requestPayload = {
      query: currentInput,
      user_id: 'demo_user',
      session_id: currentSession, // Remove || undefined - send null if no session
      force_expert_council: false
    }

    // 🔍 DEBUG: Log what we're sending
    console.log('🔍 FRONTEND DEBUG - Sending request:')
    console.log('   - Current session:', currentSession)
    console.log('   - Request payload:', requestPayload)

    try {
      await fetchEventSource(`${API_BASE}/api/chat-stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'text/event-stream',
          'Cache-Control': 'no-cache',
        },
        body: JSON.stringify(requestPayload),
        signal: controller.signal,

        async onopen(response) {
          if (response.ok) {
            setStreamStatus('Connected')
            console.log('✅ Stream connection established')
            return
          } else {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`)
          }
        },

        onmessage(event) {
          if (!isMountedRef.current) return

          try {
            switch (event.event) {
              case 'task_started':
                const taskData = JSON.parse(event.data)
                setStreamStatus(taskData.message || 'Task started')
                break

              case 'session_ready':
                const sessionData = JSON.parse(event.data)
                newSessionId = sessionData.session_id
                
                // 🔍 FIXED: Only update if different
                if (sessionData.session_id !== currentSession) {
                  console.log('🔄 Updating session:', currentSession, '→', sessionData.session_id)
                  setActiveSession(sessionData.session_id)
                } else {
                  console.log('✅ Session confirmed:', sessionData.session_id)
                }
                
                setStreamStatus(sessionData.message || 'Session ready')
                break

              case 'analysis_start':
                const analysisData = JSON.parse(event.data)
                setStreamStatus(analysisData.message || 'Analyzing...')
                break

              case 'analysis_complete':
                const analysisCompleteData = JSON.parse(event.data)
                setStreamStatus(`Analysis: ${analysisCompleteData.category || 'Complete'}`)
                break

              case 'knowledge_search':
                const knowledgeData = JSON.parse(event.data)
                setStreamStatus(knowledgeData.message || 'Searching knowledge...')
                break

              case 'council_step':
                const stepData = JSON.parse(event.data)
                setCouncilStep({
                  step: stepData.step || 0,
                  status: stepData.status || 'Processing',
                  description: stepData.description || ''
                })
                setStreamStatus(`Expert Council: ${stepData.status || 'Processing'}`)
                break

              case 'text_token':
                const tokenData = JSON.parse(event.data)
                if (tokenData.token) {
                  finalContent += tokenData.token
                  setMessages(prev => prev.map(msg => 
                    msg.id === assistantId 
                      ? { ...msg, content: finalContent }
                      : msg
                  ))
                }
                break

              case 'council_complete':
                structuredData = JSON.parse(event.data)
                setCouncilStep(null)
                setStreamStatus('Analysis complete')
                break

              case 'error':
                const errorData = JSON.parse(event.data)
                setStreamStatus(`Error: ${errorData.error || 'Unknown error'}`)
                throw new Error(errorData.error || 'Stream error')

              case 'stream_end':
                const endData = JSON.parse(event.data)
                console.log('🏁 Stream ended:', endData)
                setStreamStatus('Complete')
                break

              default:
                if (process.env.NODE_ENV === 'development') {
                  console.log('Unknown event:', event.event, event.data)
                }
                break
            }
          } catch (parseError) {
            if (process.env.NODE_ENV === 'development') {
              console.error('Error parsing event:', parseError, event)
            }
          }
        },

        onclose() {
          if (!isMountedRef.current) return
          
          console.log('🔚 Stream closed')
          
          // Apply structured data if available
          if (structuredData) {
            setMessages(prev => prev.map(msg => 
              msg.id === assistantId 
                ? { ...msg, data: structuredData }
                : msg
            ))
          }

          // 🔍 FIXED: Save to history logic
          const finalSessionId = newSessionId || currentSession
          console.log('💾 Saving to history - Session:', finalSessionId, 'Content length:', finalContent.length)
          
          if (finalSessionId && finalContent && currentInput) {
            saveSessionToList(finalSessionId, currentInput)
          }

          // Reset streaming state
          setIsStreaming(false)
          setStreamStatus('')
          setCouncilStep(null)
          abortControllerRef.current = null
          
          // Focus input for next message
          setTimeout(() => {
            inputRef.current?.focus()
          }, 100)
        },

        onerror(error) {
          if (!isMountedRef.current) return
          
          console.error('❌ Stream error:', error)
          
          setIsStreaming(false)
          setStreamStatus('Connection error')
          setCouncilStep(null)
          abortControllerRef.current = null
          
          // If no content was received, show error message
          if (!finalContent) {
            const errorMessage: ChatMessage = {
              id: `error-${Date.now()}`,
              role: 'assistant',
              content: `I'm experiencing connection difficulties. Please ensure the backend is running and try again.`,
              timestamp: new Date().toISOString()
            }
            setMessages(prev => [...prev.slice(0, -1), errorMessage])
          }
          
          throw error
        }
      })

    } catch (error) {
      if (isMountedRef.current) {
        setIsStreaming(false)
        setStreamStatus('')
        setCouncilStep(null)
        
        console.error('❌ Send error:', error)
      }
    }
  }
  
  // === UI HANDLERS ===
  const handleNewChat = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
    }
    setMessages([])
    setActiveSession(null)
    setIsStreaming(false)
    setStreamStatus('')
    setCouncilStep(null)
    setLastRequestTime(0)
    inputRef.current?.focus()
  }

  const loadChatSession = async (sessionId: string) => {
    if (isStreaming) return

    try {
      setMessages([])
      setActiveSession(sessionId)
      
      const sessionData = await api.getSessionHistory(sessionId)
      
      if (sessionData.messages && Array.isArray(sessionData.messages)) {
        const chatMessages: ChatMessage[] = sessionData.messages.map((msg: any, index: number) => ({
          id: `${sessionId}_${index}`,
          role: msg.role,
          content: msg.content || '',
          timestamp: msg.timestamp || new Date().toISOString(),
          data: msg.role === 'assistant' && msg.metadata ? {
            structured_analysis: msg.metadata.structured_analysis,
            interactive_components: msg.metadata.interactive_components,
            reasoning_trace: msg.metadata.reasoning_trace,
            confidence: msg.metadata.confidence
          } : undefined
        }))
        
        setMessages(chatMessages)
      }
    } catch (error) {
      if (process.env.NODE_ENV === 'development') {
        console.error('Failed to load session:', error)
      }
      
      // Show error to user
      const errorMessage: ChatMessage = {
        id: `error-${Date.now()}`,
        role: 'assistant',
        content: 'Failed to load conversation history. Starting fresh.',
        timestamp: new Date().toISOString()
      }
      setMessages([errorMessage])
    }
  }

  const handleQuickResponse = (text: string) => {
    if (isStreaming) return
    setInput(text)
    setTimeout(() => {
      handleSend()
    }, 100)
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  // === RENDERING HELPERS ===
  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
    if (confidence >= 0.6) return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200'
    return 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
  }

  const renderMarkdownText = (text: string) => {
    const parts = text.split(/(\*\*.*?\*\*)/g)
    return parts.map((part, index) => {
      if (part.startsWith('**') && part.endsWith('**')) {
        return <strong key={index}>{part.slice(2, -2)}</strong>
      }
      return <span key={index}>{part}</span>
    })
  }

  const renderExpertCouncilReport = (data: any) => (
    <div className="mt-4 space-y-4 border-l-4 border-blue-500 pl-4 bg-blue-50/50 rounded-r-lg p-4">
      <div className="text-sm font-medium flex items-center gap-2 text-blue-700">
        <Brain className="h-4 w-4" />
        Expert Council Analysis
      </div>
      
      <Tabs defaultValue="summary" className="w-full">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="summary">Summary</TabsTrigger>
          <TabsTrigger value="diagnosis">Diagnosis</TabsTrigger>
          <TabsTrigger value="actions">Actions</TabsTrigger>
        </TabsList>
        
        <TabsContent value="summary" className="space-y-3">
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-lg flex items-center gap-2">
                Expert Council Assessment
                <Badge variant="default">
                  {data?.structured_analysis?.clinical_summary?.urgency_level || "assessment"}
                </Badge>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-gray-600 dark:text-gray-300 mb-3">
                {data?.structured_analysis?.clinical_summary?.primary_assessment || 
                 "Expert analysis completed"}
              </p>
              
              <div className="space-y-2">
                {(data?.structured_analysis?.clinical_summary?.key_findings || []).slice(0, 3).map((finding: string, idx: number) => (
                  <div key={idx} className="flex items-start gap-2 text-sm">
                    <CheckCircle className="h-4 w-4 text-green-500 mt-0.5 flex-shrink-0" />
                    <span>{finding}</span>
                  </div>
                ))}
              </div>
              
              <div className="mt-3 flex items-center gap-2">
                <span className="text-xs text-gray-500">Confidence:</span>
                <Badge className={getConfidenceColor(data?.confidence || 0.7)}>
                  {((data?.confidence || 0.7) * 100).toFixed(0)}%
                </Badge>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
        
        <TabsContent value="diagnosis" className="space-y-3">
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-base">Primary Assessment</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex items-center justify-between mb-2">
                <span className="font-medium">
                  {data?.structured_analysis?.differential_diagnoses?.[0]?.condition ||
                   data?.structured_analysis?.clinical_summary?.primary_assessment ||
                   "Assessment completed"}
                </span>
                <Badge>Expert Analysis</Badge>
              </div>
              <p className="text-sm text-gray-600 dark:text-gray-300">
                Based on comprehensive multi-agent analysis
              </p>
            </CardContent>
          </Card>
        </TabsContent>
        
        <TabsContent value="actions" className="space-y-3">
          <div className="space-y-4">
            {(data?.structured_analysis?.recommendations?.immediate_actions?.length > 0) && (
              <Card>
                <CardHeader className="pb-3">
                  <CardTitle className="text-base flex items-center gap-2">
                    <AlertTriangle className="h-4 w-4 text-orange-500" />
                    Immediate Actions
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-2">
                  {data.structured_analysis.recommendations.immediate_actions.map((action: string, idx: number) => (
                    <div key={idx} className="flex items-start gap-3 p-2 bg-orange-50 dark:bg-orange-900/20 rounded">
                      <Clock className="h-4 w-4 text-orange-600 mt-0.5 flex-shrink-0" />
                      <div className="flex-1">
                        <p className="text-sm font-medium">{action}</p>
                      </div>
                    </div>
                  ))}
                </CardContent>
              </Card>
            )}
          </div>
        </TabsContent>
      </Tabs>
    </div>
  )

  // === UI LOGIC ===
  const isDebounced = Date.now() - lastRequestTime < REQUEST_DEBOUNCE_MS
  const canSendMessage = !isStreaming && input.trim() && !isDebounced && isMountedRef.current
  const inputDisabled = isStreaming || !isMountedRef.current

  return (
    <div className="flex h-screen bg-white dark:bg-gray-900">
      {/* Sidebar */}
      <div className={`${sidebarOpen ? 'w-64' : 'w-0'} transition-all duration-300 overflow-hidden bg-gray-50 dark:bg-gray-800 border-r`}>
        <div className="p-4 space-y-3">
          <Button 
            onClick={handleNewChat}
            className="w-full justify-start gap-2"
            variant="outline"
            disabled={inputDisabled}
          >
            <Plus className="h-4 w-4" />
            New Chat
          </Button>
          
          <Button 
            onClick={clearAllHistory}
            className="w-full justify-start gap-2"
            variant="outline"
            size="sm"
          >
            <Trash2 className="h-4 w-4" />
            Clear All
          </Button>
          
          <div className="space-y-2">
            <div className="flex items-center gap-2 text-sm font-medium text-gray-600 dark:text-gray-300">
              <History className="h-4 w-4" />
              Recent Chats
            </div>
            <ScrollArea className="h-[250px]">
              {chatHistory.map((chat) => (
                <Button
                  key={chat.session_id}
                  variant="ghost"
                  size="sm"
                  className="w-full justify-start p-2 h-auto text-left mb-1"
                  onClick={() => loadChatSession(chat.session_id)}
                  disabled={inputDisabled}
                >
                  <div className="truncate">
                    <div className="text-xs font-medium truncate">{chat.title}</div>
                    <div className="text-xs text-gray-500 truncate">
                      {new Date(chat.timestamp).toLocaleDateString()}
                    </div>
                  </div>
                </Button>
              ))}
              {chatHistory.length === 0 && (
                <div className="text-xs text-gray-500 text-center py-4">
                  No previous conversations
                </div>
              )}
            </ScrollArea>
          </div>
          
          {systemHealth && (
            <div className="p-3 bg-white dark:bg-gray-700 rounded-lg text-xs">
              <div className="flex items-center gap-2 mb-1">
                <div className={`w-2 h-2 rounded-full ${
                  systemHealth.status === 'healthy' ? 'bg-green-500' : 'bg-red-500'
                }`} />
                <span className="font-medium">System Status</span>
              </div>
              <div className="text-gray-600 dark:text-gray-300">
                Expert Council: {systemHealth.systems?.expert_council?.status || 'ready'}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b bg-white dark:bg-gray-900">
          <div className="flex items-center gap-3">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setSidebarOpen(!sidebarOpen)}
            >
              <Menu className="h-4 w-4" />
            </Button>
            <div className="flex items-center gap-2">
              <Bot className="h-6 w-6 text-blue-600" />
              <h1 className="text-xl font-semibold">AURA Health Guardian</h1>
            </div>
          </div>
          
          <div className="flex items-center gap-2">
            {currentSession && (
              <Badge variant="outline" className="text-xs">
                Session: {currentSession.split('_')[1] || 'Active'}
              </Badge>
            )}
            {isStreaming && (
              <Badge variant="default" className="text-xs">
                <Loader2 className="h-3 w-3 animate-spin mr-1" />
                Processing
              </Badge>
            )}
          </div>
        </div>

        {/* Messages Area */}
        <div className="flex-1 overflow-hidden">
          <ScrollArea className="h-full">
            <div className="max-w-4xl mx-auto">
              {messages.length === 0 ? (
                <div className="flex flex-col items-center justify-center h-full text-center p-8">
                  <Brain className="h-16 w-16 text-gray-300 mb-4" />
                  <h2 className="text-2xl font-semibold mb-2">Welcome to AURA</h2>
                  <p className="text-gray-600 mb-4">Your AI Health Guardian with Expert Council analysis</p>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3 max-w-2xl">
                    <Button 
                      variant="outline" 
                      className="p-4 h-auto text-left justify-start"
                      onClick={() => handleQuickResponse("I have chest pain and shortness of breath")}
                      disabled={inputDisabled}
                    >
                      <AlertTriangle className="h-4 w-4 mr-2 text-red-500" />
                      Emergency symptoms
                    </Button>
                    <Button 
                      variant="outline"
                      className="p-4 h-auto text-left justify-start"
                      onClick={() => handleQuickResponse("I have a persistent headache for 3 days")}
                      disabled={inputDisabled}
                    >
                      <Activity className="h-4 w-4 mr-2 text-blue-500" />
                      Ongoing symptoms
                    </Button>
                  </div>
                </div>
              ) : (
                <div className="space-y-6 p-6">
                  {messages.map((message) => (
                    <div
                      key={message.id}
                      className={`flex gap-4 ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                    >
                      <div className={`flex gap-4 max-w-[80%] ${message.role === 'user' ? 'flex-row-reverse' : ''}`}>
                        <div className="flex-shrink-0 mt-1">
                          {message.role === 'user' ? (
                            <div className="w-8 h-8 bg-blue-600 rounded-full flex items-center justify-center">
                              <User className="h-4 w-4 text-white" />
                            </div>
                          ) : (
                            <div className="w-8 h-8 bg-green-600 rounded-full flex items-center justify-center">
                              <Bot className="h-4 w-4 text-white" />
                            </div>
                          )}
                        </div>
                        
                        <div className="flex-1">
                          <div
                            className={`rounded-2xl px-4 py-3 ${
                              message.role === 'user'
                                ? 'bg-blue-600 text-white'
                                : 'bg-gray-100 dark:bg-gray-800 text-gray-900 dark:text-gray-100'
                            }`}
                          >
                            <div className="whitespace-pre-wrap leading-relaxed">
                              {renderMarkdownText(message.content)}
                            </div>
                          </div>
                          
                          {message.role === 'assistant' && message.data && 
                            renderExpertCouncilReport(message.data)
                          }
                        </div>
                      </div>
                    </div>
                  ))}
                  
                  {/* Streaming Status */}
                  {isStreaming && (
                    <div className="flex gap-4 justify-start">
                      <div className="flex gap-4 max-w-[80%]">
                        <div className="flex-shrink-0 mt-1">
                          <div className="w-8 h-8 bg-green-600 rounded-full flex items-center justify-center">
                            <Bot className="h-4 w-4 text-white" />
                          </div>
                        </div>
                        <div className="flex-1">
                          <div className="rounded-2xl px-4 py-3 bg-gray-100 dark:bg-gray-800">
                            <div className="flex items-center gap-2 text-sm">
                              <Loader2 className="h-4 w-4 animate-spin" />
                              <span>{streamStatus || 'Processing...'}</span>
                            </div>
                            
                            {councilStep && (
                              <div className="mt-2 p-2 bg-blue-50 dark:bg-blue-900/20 rounded">
                                <div className="flex items-center gap-2 text-xs text-blue-700 dark:text-blue-300">
                                  <Brain className="h-3 w-3" />
                                  <span>Step {councilStep.step}/7: {councilStep.status}</span>
                                </div>
                                <div className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                                  {councilStep.description}
                                </div>
                              </div>
                            )}
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                  
                  <div ref={messagesEndRef} />
                </div>
              )}
            </div>
          </ScrollArea>
        </div>

        {/* Input Area */}
        <div className="border-t bg-white dark:bg-gray-900 p-4">
          <div className="max-w-4xl mx-auto">
            <div className="flex gap-3">
              <div className="flex-1 relative">
                <Input
                  ref={inputRef}
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder="Ask AURA about your health concerns..."
                  disabled={inputDisabled}
                  className="pr-12 py-3 text-base"
                />
                <Button 
                  onClick={handleSend}
                  disabled={!canSendMessage}
                  size="sm"
                  className="absolute right-2 top-1/2 transform -translate-y-1/2"
                >
                  {isStreaming ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <Send className="h-4 w-4" />
                  )}
                </Button>
              </div>
            </div>
            <p className="text-xs text-gray-500 mt-2 text-center">
              ⚠️ For medical emergencies, call emergency services immediately
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}