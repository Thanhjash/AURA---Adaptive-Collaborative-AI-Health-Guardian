// frontend/src/components/chat/ChatInterface.tsx (VERSION 12.2 - SYNCED)
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

// Updated API configuration
const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export function ChatInterface() {
  // Core state
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [currentSession, setCurrentSession] = useState<string | null>(null)
  
  // Streaming state
  const [isStreaming, setIsStreaming] = useState(false)
  const [streamStatus, setStreamStatus] = useState('')
  const [councilStep, setCouncilStep] = useState<CouncilStep | null>(null)
  
  // UI state
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [chatHistory, setChatHistory] = useState<ChatHistory[]>([])
  
  // Refs
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)
  const abortControllerRef = useRef<AbortController | null>(null)
  const isMountedRef = useRef(true)

  // Utility functions
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  useEffect(() => {
    inputRef.current?.focus()
    loadChatHistory()
    return () => { isMountedRef.current = false }
  }, [])

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

  // Main streaming logic - Updated for V12.2
  const handleSend = async () => {
    if (!input.trim() || isStreaming) return
    if (!isMountedRef.current) return

    // Abort existing request
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
    let finalData: any = null
    let newSessionId: string | null = null

    const requestPayload = {
      query: currentInput,
      user_id: 'demo_user_v12',
      session_id: currentSession,
      force_expert_council: false
    }

    try {
      await fetchEventSource(`${API_BASE}/chat-stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestPayload),
        signal: controller.signal,

        async onopen(response) {
          if (response.ok) {
            setStreamStatus('Connected. Processing...')
            return
          }
          throw new Error(`HTTP ${response.status}: ${response.statusText}`)
        },

        onmessage(event) {
          if (!isMountedRef.current) return

          try {
            const data = JSON.parse(event.data)

            switch (event.event) {
              // V12.2 Backend Events
              case 'progress':
                setStreamStatus(`${data.step}: ${data.status}`)
                break
              
              case 'council_started':
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
                newSessionId = data.session_id
                if (data.session_id && data.session_id !== currentSession) {
                  setCurrentSession(data.session_id)
                }
                setStreamStatus('Complete.')
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

              default:
                console.log('Unknown event:', event.event, data)
                break
            }
          } catch (parseError) {
            console.error('Event parsing error:', parseError, event)
          }
        },

        onclose() {
          if (!isMountedRef.current) return
          
          if (finalData) {
            setMessages(prev => prev.map(msg => 
              msg.id === assistantId ? { ...msg, data: finalData } : msg
            ))
          }

          const finalSessionId = newSessionId || currentSession
          if (finalSessionId && finalContent && currentInput) {
            saveSessionToList(finalSessionId, currentInput)
          }

          setIsStreaming(false)
          setStreamStatus('')
          setCouncilStep(null)
          abortControllerRef.current = null
          
          setTimeout(() => inputRef.current?.focus(), 100)
        },

        onerror(error) {
          console.error('Stream error:', error)
          if (isMountedRef.current) {
            setStreamStatus('Connection error')
            setTimeout(() => {
              setIsStreaming(false)
              setStreamStatus('')
              setCouncilStep(null)
            }, 2000)
          }
          return 5000 // Retry in 5 seconds
        }
      })
    } catch (error) {
      if (isMountedRef.current) {
        setIsStreaming(false)
        setStreamStatus('')
        setCouncilStep(null)
        console.error('Send error:', error)
        
        // Add error message
        const errorMessage: ChatMessage = {
          id: `error-${Date.now()}`,
          role: 'assistant',
          content: 'Connection failed. Please check your network and try again.',
          timestamp: new Date().toISOString()
        }
        setMessages(prev => [...prev.slice(0, -1), errorMessage])
      }
    }
  }

  const handleNewChat = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
    }
    setMessages([])
    setCurrentSession(null)
    setIsStreaming(false)
    setStreamStatus('')
    setCouncilStep(null)
    inputRef.current?.focus()
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
    if (confidence >= 0.8) return 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
    if (confidence >= 0.6) return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200'
    return 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
  }

  const renderMarkdownText = (text: string) => {
    return text.split(/(\*\*.*?\*\*)/).map((part, index) => {
      if (part.startsWith('**') && part.endsWith('**')) {
        return <strong key={index}>{part.slice(2, -2)}</strong>
      }
      return part
    })
  }

  const renderExpertCouncilReport = (data: any) => {
    if (!data.reasoning_trace) return null

    return (
      <div className="mt-3 border rounded-lg overflow-hidden">
        <Tabs defaultValue="summary" className="w-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="summary">Summary</TabsTrigger>
            <TabsTrigger value="reasoning">Reasoning</TabsTrigger>
            <TabsTrigger value="confidence">Details</TabsTrigger>
          </TabsList>
          
          <TabsContent value="summary" className="p-4">
            <div className="space-y-2">
              {data.confidence && (
                <Badge className={getConfidenceColor(data.confidence)}>
                  Confidence: {Math.round(data.confidence * 100)}%
                </Badge>
              )}
              {data.duration_seconds && (
                <Badge variant="outline">
                  <Clock className="w-3 h-3 mr-1" />
                  {data.duration_seconds}s
                </Badge>
              )}
            </div>
          </TabsContent>
          
          <TabsContent value="reasoning" className="p-4">
            <div className="space-y-3 text-sm">
              {data.reasoning_trace.coordinator && (
                <div>
                  <strong>Coordinator:</strong>
                  <p className="mt-1 text-gray-600 dark:text-gray-300">
                    {data.reasoning_trace.coordinator.slice(0, 200)}...
                  </p>
                </div>
              )}
              {data.reasoning_trace.reasoner && (
                <div>
                  <strong>Reasoner:</strong>
                  <p className="mt-1 text-gray-600 dark:text-gray-300">
                    {data.reasoning_trace.reasoner.slice(0, 200)}...
                  </p>
                </div>
              )}
            </div>
          </TabsContent>
          
          <TabsContent value="confidence" className="p-4">
            <div className="text-sm space-y-2">
              <div>Session ID: {data.session_id}</div>
              <div>Processing Time: {data.duration_seconds}s</div>
              <div>Analysis Depth: Expert Council</div>
            </div>
          </TabsContent>
        </Tabs>
      </div>
    )
  }

  return (
    <div className="flex h-screen bg-gray-50 dark:bg-gray-900">
      {/* Sidebar */}
      <div className={`${sidebarOpen ? 'translate-x-0' : '-translate-x-full'} fixed inset-y-0 left-0 z-50 w-64 bg-white dark:bg-gray-800 border-r border-gray-200 dark:border-gray-700 transform transition-transform duration-200 ease-in-out lg:translate-x-0 lg:static lg:inset-0`}>
        <div className="flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-700">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white">AURA Chat</h2>
          <Button variant="ghost" size="sm" onClick={() => setSidebarOpen(false)} className="lg:hidden">
            ×
          </Button>
        </div>
        
        <div className="p-4">
          <Button onClick={handleNewChat} className="w-full mb-4">
            <Plus className="w-4 h-4 mr-2" />
            New Chat
          </Button>
          
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400">Recent</h3>
              {chatHistory.length > 0 && (
                <Button variant="ghost" size="sm" onClick={clearAllHistory}>
                  <Trash2 className="w-3 h-3" />
                </Button>
              )}
            </div>
            
            <ScrollArea className="h-64">
              {chatHistory.map((session) => (
                <div
                  key={session.session_id}
                  className="p-2 text-sm cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-700 rounded"
                  onClick={() => setCurrentSession(session.session_id)}
                >
                  <div className="font-medium truncate">{session.title}</div>
                  <div className="text-xs text-gray-500 dark:text-gray-400">
                    {new Date(session.timestamp).toLocaleDateString()}
                  </div>
                </div>
              ))}
            </ScrollArea>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <div className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 p-4">
          <div className="flex items-center justify-between">
            <Button variant="ghost" size="sm" onClick={() => setSidebarOpen(true)} className="lg:hidden">
              <Menu className="w-4 h-4" />
            </Button>
            <div className="flex items-center gap-2">
              <Activity className="w-5 h-5 text-green-500" />
              <span className="font-medium">AURA Health Assistant</span>
            </div>
            <div></div>
          </div>
        </div>

        {/* Messages */}
        <ScrollArea className="flex-1 p-4">
          <div className="max-w-4xl mx-auto space-y-4">
            {messages.map((message) => (
              <div key={message.id} className="flex gap-4">
                <div className="flex-shrink-0 mt-1">
                  <div className={`w-8 h-8 rounded-full flex items-center justify-center ${
                    message.role === 'user' 
                      ? 'bg-blue-600' 
                      : 'bg-green-600'
                  }`}>
                    {message.role === 'user' ? (
                      <User className="h-4 w-4 text-white" />
                    ) : (
                      <Bot className="h-4 w-4 text-white" />
                    )}
                  </div>
                </div>
                <div className="flex-1 min-w-0">
                  <div className={`rounded-2xl px-4 py-3 ${
                    message.role === 'user'
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-100 dark:bg-gray-800 text-gray-900 dark:text-gray-100'
                  }`}>
                    <div className="whitespace-pre-wrap leading-relaxed">
                      {renderMarkdownText(message.content)}
                    </div>
                  </div>
                  
                  {message.role === 'assistant' && message.data && 
                    renderExpertCouncilReport(message.data)
                  }
                </div>
              </div>
            ))}
            
            {/* Streaming Status */}
            {isStreaming && (
              <div className="flex gap-4">
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
                          <span>{councilStep.step}: {councilStep.status}</span>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}
            
            <div ref={messagesEndRef} />
          </div>
        </ScrollArea>

        {/* Input */}
        <div className="border-t border-gray-200 dark:border-gray-700 p-4 bg-white dark:bg-gray-800">
          <div className="max-w-4xl mx-auto">
            <div className="flex gap-2">
              <Input
                ref={inputRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask about your health..."
                disabled={isStreaming}
                className="flex-1"
              />
              <Button 
                onClick={handleSend} 
                disabled={isStreaming || !input.trim()}
                size="icon"
              >
                {isStreaming ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Send className="h-4 w-4" />
                )}
              </Button>
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