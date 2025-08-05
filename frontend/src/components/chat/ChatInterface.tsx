'use client'

import React, { useState, useEffect, useRef, useCallback } from 'react'
import { fetchEventSource } from '@microsoft/fetch-event-source'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import { Switch } from '@/components/ui/switch'
import { Label } from '@/components/ui/label'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Alert, AlertDescription } from '@/components/ui/alert'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { 
  Send, 
  Bot, 
  User, 
  Menu,
  Plus,
  Loader2,
  Trash2,
  Brain,
  Activity,
  AlertTriangle,
  CheckCircle,
  Clock
} from 'lucide-react'

interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: string
  data?: {
    confidence?: number
    reasoning_trace?: any
    duration_seconds?: number
    session_id?: string
  }
}

interface ChatHistory {
  session_id: string
  title: string
  timestamp: string
}

interface CouncilStep {
  step: string
  status: string
  description?: string
}

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

// Lock management helpers
const getRequestLock = (): boolean => {
  if (typeof window === 'undefined') return false
  return window.sessionStorage.getItem('aura_request_lock') === 'true'
}

const setRequestLock = (locked: boolean) => {
  if (typeof window === 'undefined') return
  if (locked) {
    window.sessionStorage.setItem('aura_request_lock', 'true')
  } else {
    window.sessionStorage.removeItem('aura_request_lock')
  }
}

const getActiveSession = (): string | null => {
  if (typeof window === 'undefined') return null
  return window.sessionStorage.getItem('aura_active_session')
}

const setActiveSession = (sessionId: string | null) => {
  if (typeof window === 'undefined') return
  if (sessionId) {
    window.sessionStorage.setItem('aura_active_session', sessionId)
  } else {
    window.sessionStorage.removeItem('aura_active_session')
  }
}

export function ChatInterface() {
  // Core state
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [currentSession, setCurrentSession] = useState<string | null>(null)
  
  // Streaming state
  const [isStreaming, setIsStreaming] = useState(false)
  const [streamStatus, setStreamStatus] = useState('')
  const [councilStep, setCouncilStep] = useState<CouncilStep | null>(null)
  const [councilActive, setCouncilActive] = useState(false)
  
  // UI state
  const [mounted, setMounted] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [chatHistory, setChatHistory] = useState<ChatHistory[]>([])
  const [forceCouncil, setForceCouncil] = useState(false)
  
  // Refs
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)
  const abortControllerRef = useRef<AbortController | null>(null)
  const isMountedRef = useRef(false)
  const sessionIdRef = useRef<string | null>(null)

  // Initialization
  useEffect(() => {
    isMountedRef.current = true
    setMounted(true)
    const storedSession = getActiveSession()
    setCurrentSession(storedSession)
    sessionIdRef.current = storedSession
    loadChatHistory()
    inputRef.current?.focus()
    
    return () => {
      isMountedRef.current = false
      if (abortControllerRef.current) {
        abortControllerRef.current.abort()
      }
      setRequestLock(false)
    }
  }, [])

  // Session sync
  useEffect(() => {
    if (!mounted) return
    
    const syncSession = () => {
      if (getRequestLock() || isStreaming) return
      
      const storedSession = getActiveSession()
      if (storedSession !== currentSession) {
        setCurrentSession(storedSession)
        sessionIdRef.current = storedSession
      }
    }

    window.addEventListener('focus', syncSession)
    return () => window.removeEventListener('focus', syncSession)
  }, [currentSession, mounted, isStreaming])

  useEffect(() => {
    sessionIdRef.current = currentSession
    setActiveSession(currentSession)
  }, [currentSession])

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  // Emergency lock release
  useEffect(() => {
    if (!isStreaming && getRequestLock()) {
      console.log('🔧 Emergency lock release detected')
      setRequestLock(false)
    }
  }, [isStreaming])

  const loadChatHistory = () => {
    if (typeof window === 'undefined') return
    try {
      const saved = localStorage.getItem('aura_session_list')
      setChatHistory(saved ? JSON.parse(saved) : [])
    } catch (error) {
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
      const updated = [sessionItem, ...filtered].slice(0, 20)
      
      try {
        localStorage.setItem('aura_session_list', JSON.stringify(updated))
      } catch (error) {
        console.error('Storage error:', error)
      }
      
      return updated
    })
  }, [])

  const cleanupStream = useCallback(() => {
    if (!isMountedRef.current) return
    
    console.log('🧹 Centralizing cleanup...')
    setIsStreaming(false)
    setRequestLock(false)
    setStreamStatus('')
    setCouncilStep(null)
    setCouncilActive(false)
    setForceCouncil(false)
    abortControllerRef.current = null
    
    setTimeout(() => {
      if (isMountedRef.current && inputRef.current) {
        inputRef.current.focus()
      }
    }, 100)
  }, [])

  const loadChatSession = async (sessionId: string) => {
    if (isStreaming || getRequestLock()) return

    console.log(`[FE] Loading session: ${sessionId}`)
    setMessages([])
    setCurrentSession(sessionId)

    try {
      const response = await fetch(`${API_BASE}/api/session/${sessionId}`)
      if (!response.ok) {
        if (response.status === 404) {
          console.log('Session not found, starting fresh')
          return
        }
        throw new Error(`HTTP ${response.status}: ${response.statusText}`)
      }
      
      const sessionData = await response.json()
      console.log('📥 Session data received:', sessionData)
      
      if (sessionData && Array.isArray(sessionData.message_history)) {
        const loadedMessages: ChatMessage[] = sessionData.message_history.map((msg: any, index: number) => ({
          id: `${sessionId}-${index}`,
          role: msg.role,
          content: msg.content,
          timestamp: msg.timestamp,
          data: msg.data
        }))
        setMessages(loadedMessages)
        console.log(`✅ Loaded ${loadedMessages.length} messages`)
      }
    } catch (error) {
      console.error("Failed to load session history:", error)
      setMessages([{
        id: 'error-load',
        role: 'assistant',
        content: 'Sorry, I was unable to load this conversation.',
        timestamp: new Date().toISOString()
      }])
    }
  }

  const handleSend = useCallback(async () => {
    const canSend = input.trim() && !isStreaming && !getRequestLock() && isMountedRef.current
    
    if (!canSend) return

    const currentInput = input
    setInput('')
    
    setRequestLock(true)
    setIsStreaming(true)

    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
      await new Promise(resolve => setTimeout(resolve, 100))
    }

    const userMessage: ChatMessage = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: currentInput,
      timestamp: new Date().toISOString()
    }

    const assistantId = `assistant-${Date.now()}`
    const assistantMessage: ChatMessage = {
      id: assistantId,
      role: 'assistant',
      content: '',
      timestamp: new Date().toISOString()
    }

    setMessages(prev => [...prev, userMessage, assistantMessage])
    setStreamStatus('Connecting...')
    setCouncilStep(null)
    setCouncilActive(false)

    const controller = new AbortController()
    abortControllerRef.current = controller

    let finalContent = ''
    let finalData: any = null
    let newSessionId: string | null = null

    const requestPayload = {
      query: currentInput,
      user_id: 'demo_user_v12_5',
      session_id: sessionIdRef.current,
      force_expert_council: forceCouncil
    }

    try {
      await fetchEventSource(`${API_BASE}/chat-stream`, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Accept': 'text/event-stream',
          'Cache-Control': 'no-cache'
        },
        body: JSON.stringify(requestPayload),
        signal: controller.signal,

        async onopen(response) {
          if (response.ok) {
            setStreamStatus('Connected')
            return
          }
          throw new Error(`HTTP ${response.status}: ${response.statusText}`)
        },

        onmessage(event) {
          if (!isMountedRef.current) return

          try {
            const data = JSON.parse(event.data)

            switch (event.event) {
              case 'session_ready':
                if (data.session_id && data.session_id !== sessionIdRef.current) {
                  setCurrentSession(data.session_id)
                  sessionIdRef.current = data.session_id
                  newSessionId = data.session_id
                }
                setStreamStatus(data.message || 'Session ready')
                break

              case 'progress':
                setStreamStatus(`${data.step}: ${data.status}`)
                break
              
              case 'council_started':
                setCouncilActive(true)
                setStreamStatus(`Expert Council: ${data.message}`)
                break

              case 'council_step':
                setCouncilStep({
                  step: data.step,
                  status: data.status,
                  description: data.description,
                })
                setStreamStatus(`Expert Council: ${data.status}`)
                break

              case 'text_token':
                if (data.token) {
                  finalContent += data.token
                  setMessages(prev => prev.map(msg =>
                    msg.id === assistantId ? { ...msg, content: finalContent } : msg
                  ))
                }
                break
                
              case 'stream_end':
                if (data.session_id) {
                  newSessionId = data.session_id
                  setCurrentSession(data.session_id)
                  sessionIdRef.current = data.session_id
                }
                setStreamStatus('Complete')
                break
                
              case 'result':
                finalData = data.data
                finalContent = finalData.user_response || finalContent
                setMessages(prev => prev.map(msg =>
                  msg.id === assistantId ? { ...msg, content: finalContent, data: finalData } : msg
                ))
                break

              case 'error':
                throw new Error(data.message || 'Unknown stream error')
            }
          } catch (parseError) {
            console.error('Event parsing error:', parseError, event)
          }
        },

        onclose() {
          if (finalData && isMountedRef.current) {
            setMessages(prev => prev.map(msg => 
              msg.id === assistantId 
                ? { ...msg, data: finalData } 
                : msg
            ))
          }

          const finalSessionId = newSessionId || sessionIdRef.current
          if (finalSessionId && finalContent && currentInput) {
            saveSessionToList(finalSessionId, currentInput)
          }
        },

        onerror(error) {
          console.error('❌ Stream error:', error)
          
          if (isMountedRef.current && error.name !== 'AbortError') {
            setMessages(prev => prev.map(msg =>
              msg.id === assistantId ? {
                ...msg,
                content: 'Sorry, there was a connection error. Please try again.'
              } : msg
            ))
          }
          
          controller.abort()
          throw error
        }
      })
    } catch (error) {
      if (isMountedRef.current && abortControllerRef.current?.signal.aborted !== true) {
        console.error('Send error:', error)
        
        const errorMessage: ChatMessage = {
          id: `error-${Date.now()}`,
          role: 'assistant',
          content: 'Connection failed. Please check your network and try again.',
          timestamp: new Date().toISOString()
        }
        setMessages(prev => [...prev.slice(0, -1), errorMessage])
      }
    } finally {
      cleanupStream()
    }
  }, [input, isStreaming, forceCouncil, cleanupStream, saveSessionToList])

  const handleNewChat = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
    }
    setMessages([])
    setCurrentSession(null)
    sessionIdRef.current = null
    cleanupStream()
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const clearAllHistory = () => {
    if (typeof window === 'undefined') return
    localStorage.removeItem('aura_session_list')
    setChatHistory([])
    handleNewChat()
  }

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'bg-green-100 text-green-800'
    if (confidence >= 0.6) return 'bg-yellow-100 text-yellow-800'
    return 'bg-red-100 text-red-800'
  }

  const getCouncilStepIcon = (step: string) => {
    switch (step.toLowerCase()) {
      case 'health_check':
        return <Activity className="h-4 w-4" />
      case 'parallel_analysis':
        return <Brain className="h-4 w-4" />
      case 'synthesis':
        return <CheckCircle className="h-4 w-4" />
      case 'formatting':
        return <Clock className="h-4 w-4" />
      default:
        return <Loader2 className="h-4 w-4 animate-spin" />
    }
  }

  if (!mounted) {
    return <div className="flex h-screen bg-gray-50">Loading...</div>
  }

  const isButtonDisabled = isStreaming || getRequestLock() || !input.trim()

  return (
    <div className="flex h-screen bg-gray-50">
      {/* Sidebar */}
      <div className={`${sidebarOpen ? 'translate-x-0' : '-translate-x-full'} fixed inset-y-0 left-0 z-50 w-64 bg-white shadow-lg transform transition-transform duration-300 ease-in-out lg:translate-x-0 lg:static lg:inset-0`}>
        <div className="flex items-center justify-between h-16 px-4 border-b">
          <h1 className="text-xl font-bold text-gray-900">AURA</h1>
          <Button 
            variant="ghost" 
            size="sm"
            onClick={() => setSidebarOpen(false)}
            className="lg:hidden"
          >
            ✕
          </Button>
        </div>
        
        <div className="p-4">
          <Button onClick={handleNewChat} className="w-full mb-4">
            <Plus className="h-4 w-4 mr-2" />
            New Chat
          </Button>
          
          <div className="flex justify-between items-center mb-2">
            <h2 className="text-sm font-medium text-gray-700">Chat History</h2>
            <Button 
              variant="ghost" 
              size="sm" 
              onClick={clearAllHistory}
              className="text-xs"
            >
              <Trash2 className="h-3 w-3" />
            </Button>
          </div>
          
          <ScrollArea className="h-[calc(100vh-200px)]">
            <div className="space-y-2">
              {chatHistory.map((item) => (
                <div
                  key={item.session_id}
                  onClick={() => loadChatSession(item.session_id)}
                  className={`p-3 rounded cursor-pointer transition-colors ${
                    currentSession === item.session_id
                      ? 'bg-blue-100 text-blue-800'
                      : 'hover:bg-gray-100'
                  }`}
                >
                  <div className="text-sm font-medium truncate">
                    {item.title}
                  </div>
                  <div className="text-xs text-gray-500 mt-1">
                    {new Date(item.timestamp).toLocaleDateString()}
                  </div>
                </div>
              ))}
            </div>
          </ScrollArea>
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <div className="h-16 bg-white border-b flex items-center justify-between px-4">
          <div className="flex items-center">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setSidebarOpen(true)}
              className="lg:hidden mr-2"
            >
              <Menu className="h-4 w-4" />
            </Button>
            <h2 className="text-lg font-semibold">
              AURA Health Assistant
            </h2>
          </div>
          
          {/* Force Expert Council Toggle */}
          <div className="flex items-center space-x-2">
            <Switch
              id="force-council"
              checked={forceCouncil}
              onCheckedChange={setForceCouncil}
              disabled={isStreaming || getRequestLock()}
            />
            <Label htmlFor="force-council" className="text-sm">
              Force Expert Council
            </Label>
          </div>
        </div>

        {/* Messages Area */}
        <ScrollArea className="flex-1 p-4">
          <div className="max-w-4xl mx-auto space-y-4">
            {messages.length === 0 && (
              <div className="text-center text-gray-500 mt-8">
                <Bot className="h-12 w-12 mx-auto mb-4 text-gray-400" />
                <h3 className="text-lg font-medium mb-2">Welcome to AURA</h3>
                <p>Your AI Health Guardian. Ask me about your health concerns.</p>
              </div>
            )}
            
            {messages.map((message) => (
              <div key={message.id} className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                <div className={`max-w-[80%] rounded-lg p-4 ${
                  message.role === 'user' 
                    ? 'bg-blue-600 text-white' 
                    : 'bg-white border shadow-sm'
                }`}>
                  <div className="flex items-start space-x-2">
                    <div className="flex-shrink-0">
                      {message.role === 'user' ? (
                        <User className="h-5 w-5" />
                      ) : (
                        <Bot className="h-5 w-5" />
                      )}
                    </div>
                    <div className="flex-1">
                      {/* FIXED: Use ReactMarkdown instead of split/map */}
                      <div className="prose prose-sm max-w-none break-words">
                        <ReactMarkdown 
                          remarkPlugins={[remarkGfm]}
                          components={{
                            // Customize markdown components for better styling
                            h1: ({children}) => <h1 className="text-lg font-bold mb-2">{children}</h1>,
                            h2: ({children}) => <h2 className="text-md font-semibold mb-2">{children}</h2>,
                            h3: ({children}) => <h3 className="text-sm font-semibold mb-1">{children}</h3>,
                            p: ({children}) => <p className="mb-2 last:mb-0">{children}</p>,
                            ul: ({children}) => <ul className="list-disc pl-4 mb-2">{children}</ul>,
                            ol: ({children}) => <ol className="list-decimal pl-4 mb-2">{children}</ol>,
                            li: ({children}) => <li className="mb-1">{children}</li>,
                            strong: ({children}) => <strong className="font-semibold">{children}</strong>,
                            code: ({children}) => <code className="bg-gray-100 px-1 py-0.5 rounded text-xs">{children}</code>,
                            blockquote: ({children}) => <blockquote className="border-l-4 border-gray-300 pl-4 italic">{children}</blockquote>
                          }}
                        >
                          {message.content}
                        </ReactMarkdown>
                      </div>
                      
                      {/* Enhanced Message Metadata */}
                      {message.role === 'assistant' && message.data && (
                        <div className="mt-3 space-y-2">
                          <div className="flex flex-wrap gap-2">
                            {typeof message.data.confidence === 'number' && (
                              <Badge 
                                variant="secondary" 
                                className={`text-xs ${getConfidenceColor(message.data.confidence)}`}
                              >
                                {Math.round(message.data.confidence * 100)}% confident
                              </Badge>
                            )}
                            {message.data.duration_seconds && (
                              <Badge variant="outline" className="text-xs">
                                <Clock className="h-3 w-3 mr-1" />
                                {message.data.duration_seconds}s
                              </Badge>
                            )}
                          </div>
                          
                          {/* Expert Council Reasoning Trace */}
                          {message.data.reasoning_trace && (
                            <Tabs defaultValue="summary" className="w-full">
                              <TabsList className="grid w-full grid-cols-3">
                                <TabsTrigger value="summary">Summary</TabsTrigger>
                                <TabsTrigger value="evidence">Evidence</TabsTrigger>
                                <TabsTrigger value="reasoning">Reasoning</TabsTrigger>
                              </TabsList>
                              <TabsContent value="summary" className="mt-2">
                                <Card>
                                  <CardContent className="pt-4">
                                    <p className="text-sm text-gray-600">
                                      Expert Council analysis completed with confidence level: {Math.round((message.data.confidence || 0.7) * 100)}%
                                    </p>
                                  </CardContent>
                                </Card>
                              </TabsContent>
                              <TabsContent value="evidence" className="mt-2">
                                <Card>
                                  <CardContent className="pt-4">
                                    <p className="text-sm text-gray-600">
                                      Analysis based on medical guidelines and evidence-based practices.
                                    </p>
                                  </CardContent>
                                </Card>
                              </TabsContent>
                              <TabsContent value="reasoning" className="mt-2">
                                <Card>
                                  <CardContent className="pt-4">
                                    <div className="space-y-2 text-sm">
                                      {Object.entries(message.data.reasoning_trace).map(([key, value]) => (
                                        <div key={key}>
                                          <span className="font-medium capitalize">{key.replace('_', ' ')}:</span>
                                          <p className="text-gray-600 mt-1">{typeof value === 'string' ? value.slice(0, 200) + '...' : 'Complex analysis completed'}</p>
                                        </div>
                                      ))}
                                    </div>
                                  </CardContent>
                                </Card>
                              </TabsContent>
                            </Tabs>
                          )}
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            ))}
            
            {/* Enhanced Streaming Status */}
            {isStreaming && (
              <div className="flex justify-start">
                <Card className="max-w-md">
                  <CardContent className="pt-4">
                    <div className="flex items-center space-x-2 mb-2">
                      <Loader2 className="h-4 w-4 animate-spin" />
                      <span className="text-sm font-medium">{streamStatus}</span>
                    </div>
                    
                    {councilActive && (
                      <Alert className="mt-2">
                        <Brain className="h-4 w-4" />
                        <AlertDescription>
                          Expert Council is analyzing your query...
                        </AlertDescription>
                      </Alert>
                    )}
                    
                    {councilStep && (
                      <div className="mt-2 p-2 bg-blue-50 rounded-lg">
                        <div className="flex items-center space-x-2 text-xs">
                          {getCouncilStepIcon(councilStep.step)}
                          <span className="font-medium">Step {councilStep.step}:</span>
                          <span>{councilStep.status}</span>
                        </div>
                        {councilStep.description && (
                          <p className="text-xs text-gray-600 mt-1 ml-6">
                            {councilStep.description}
                          </p>
                        )}
                      </div>
                    )}
                  </CardContent>
                </Card>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        </ScrollArea>

        {/* Input Area */}
        <div className="bg-white border-t p-4">
          <div className="max-w-4xl mx-auto flex space-x-2">
            <Input
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Type your health question..."
              disabled={isStreaming || getRequestLock()}
              onKeyDown={handleKeyDown}
              className="flex-1"
            />
            <Button 
              onClick={handleSend} 
              disabled={isButtonDisabled}
              className="px-6"
            >
              {isStreaming ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Send className="h-4 w-4" />
              )}
            </Button>
          </div>
        </div>
      </div>
    </div>
  )
}