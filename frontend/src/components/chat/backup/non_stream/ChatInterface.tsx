'use client'

import React, { useState, useEffect, useRef } from 'react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Separator } from '@/components/ui/separator'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { 
  Send, 
  Bot, 
  User, 
  Heart, 
  AlertTriangle, 
  CheckCircle, 
  Clock,
  Activity,
  Brain,
  Shield,
  Menu,
  Plus,
  Settings,
  MessageSquare,
  History
} from 'lucide-react'
import { api } from '@/lib/api'
import type { 
  ChatMessage, 
  ChatResponse, 
  SystemHealth,
  StructuredAnalysis,
  InteractiveComponents
} from '@/lib/types'

interface ChatHistory {
  session_id: string
  title: string
  last_message: string
  timestamp: string
  message_count: number
}

export function ChatInterface() {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [systemHealth, setSystemHealth] = useState<SystemHealth | null>(null)
  const [currentSession, setCurrentSession] = useState<string | null>(null)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [chatHistory, setChatHistory] = useState<ChatHistory[]>([])
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  useEffect(() => {
    // Check system health and load chat history
    api.healthCheck().then(setSystemHealth).catch(console.error)
    loadChatHistory()
    inputRef.current?.focus()
  }, [])

  const loadChatHistory = async () => {
    try {
      // Load from localStorage
      const savedHistory = localStorage.getItem('aura_chat_history')
      if (savedHistory) {
        setChatHistory(JSON.parse(savedHistory))
      }
    } catch (error) {
      console.error('Failed to load chat history:', error)
    }
  }

  const saveToHistory = (sessionId: string, title: string, lastMessage: string) => {
    const newHistoryItem: ChatHistory = {
      session_id: sessionId,
      title: title.slice(0, 50) + (title.length > 50 ? '...' : ''),
      last_message: lastMessage.slice(0, 100),
      timestamp: new Date().toISOString(),
      message_count: 2
    }
    
    const updatedHistory = [newHistoryItem, ...chatHistory.filter(h => h.session_id !== sessionId)]
    setChatHistory(updatedHistory)
    localStorage.setItem('aura_chat_history', JSON.stringify(updatedHistory))
  }

  const updateHistoryItem = (sessionId: string, lastMessage: string, incrementCount: boolean = true) => {
    const updatedHistory = chatHistory.map(item => 
      item.session_id === sessionId 
        ? { 
            ...item, 
            last_message: lastMessage.slice(0, 100), 
            message_count: incrementCount ? item.message_count + 1 : item.message_count
          }
        : item
    )
    setChatHistory(updatedHistory)
    localStorage.setItem('aura_chat_history', JSON.stringify(updatedHistory))
  }

  const formatExpertResponse = (content: string): string => {
    // Enhanced markdown formatting for Expert Council responses
    return content
      .replace(/\*\*(.*?)\*\*/g, '**$1**') // Bold
      .replace(/\*Key Recommendations:\*/g, '\n**Key Recommendations:**')
      .replace(/\*Primary Finding:\*/g, '\n**Primary Finding:**')
      .replace(/\*Confidence Level:\*/g, '\n**Confidence Level:**')
      .replace(/\*Important:\*/g, '\n**Important:**')
      .replace(/(\d+\.\s)/g, '\n$1') // Number lists
      .replace(/^([A-Z][^:]*:)/gm, '**$1**') // Headers
      .trim()
  }

  const handleSend = async () => {
    if (!input.trim() || isLoading) return

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: input,
      timestamp: new Date().toISOString()
    }

    setMessages(prev => [...prev, userMessage])
    const currentInput = input // Store for history
    setInput('')
    setIsLoading(true)

    try {
      const response: ChatResponse = await api.chat({
        query: currentInput,
        user_id: 'demo_user',
        session_id: currentSession || undefined,
        force_expert_council: false
      })

      // Format response for better display
      let formattedResponse = response.response
      if (response.service_used.includes('expert_council')) {
        formattedResponse = formatExpertResponse(response.response)
      }

      const assistantMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: formattedResponse,
        timestamp: response.timestamp,
        metadata: {
          service_used: response.service_used,
          confidence: response.confidence,
          structured_analysis: response.structured_analysis,
          interactive_components: response.interactive_components,
          reasoning_trace: response.reasoning_trace,
          triage_category: response.triage_analysis?.category
        }
      }

      setMessages(prev => [...prev, assistantMessage])

      // Handle session and history updates
      if (response.session_id) {
        if (!currentSession) {
          // New session - save to history
          setCurrentSession(response.session_id)
          saveToHistory(response.session_id, currentInput, response.response)
        } else if (response.session_id === currentSession) {
          // Existing session - update history
          updateHistoryItem(currentSession, formattedResponse, true)
        }
      }

    } catch (error) {
      console.error('Chat error:', error)
      const errorMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: `Connection error: ${error instanceof Error ? error.message : 'Unknown error'}. Please ensure Docker services are running.`,
        timestamp: new Date().toISOString(),
        metadata: {
          service_used: 'error_handler',
          confidence: 0
        }
      }
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  const handleNewChat = () => {
    setMessages([])
    setCurrentSession(null)
    loadChatHistory() // Reload history to show updated list
    inputRef.current?.focus()
  }

  const loadChatSession = async (sessionId: string) => {
    try {
      setIsLoading(true)
      const sessionData = await api.getSessionHistory(sessionId)
      
      if (sessionData.messages) {
        const chatMessages: ChatMessage[] = sessionData.messages.map((msg: any, index: number) => ({
          id: `${sessionId}_${index}`,
          role: msg.role,
          content: msg.role === 'assistant' && msg.service?.includes('expert_council') 
            ? formatExpertResponse(msg.content)
            : msg.content,
          timestamp: msg.timestamp,
          metadata: msg.role === 'assistant' ? {
            service_used: msg.service || 'unknown',
            confidence: msg.metadata?.confidence || 0
          } : undefined
        }))
        
        setMessages(chatMessages)
        setCurrentSession(sessionId)
      }
    } catch (error) {
      console.error('Failed to load session:', error)
    } finally {
      setIsLoading(false)
    }
  }

  const getServiceIcon = (service: string) => {
    if (service.includes('expert_council')) return <Brain className="h-3 w-3" />
    if (service.includes('progressive')) return <Activity className="h-3 w-3" />
    if (service.includes('emergency')) return <AlertTriangle className="h-3 w-3" />
    return <Bot className="h-3 w-3" />
  }

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
    if (confidence >= 0.6) return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200'
    return 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
  }

  const renderMarkdownText = (text: string) => {
    // Simple markdown rendering for better display
    const parts = text.split(/(\*\*.*?\*\*)/g)
    return parts.map((part, index) => {
      if (part.startsWith('**') && part.endsWith('**')) {
        return <strong key={index}>{part.slice(2, -2)}</strong>
      }
      return <span key={index}>{part}</span>
    })
  }

  const renderExpertCouncilReport = (
    structured: StructuredAnalysis,
    interactive: InteractiveComponents
  ) => (
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
                {interactive?.summary_card?.title || "Expert Council Assessment"}
                <Badge variant="default">
                  {(structured?.clinical_summary as any)?.urgency_level || "assessment"}
                </Badge>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-gray-600 dark:text-gray-300 mb-3">
                {(structured?.clinical_summary as any)?.primary_assessment || 
                 "Expert analysis completed"}
              </p>
              
              <div className="space-y-2">
                {((structured?.clinical_summary as any)?.key_findings || 
                  (structured?.clinical_summary as any)?.safety_considerations || 
                  []).slice(0, 3).map((finding: string, idx: number) => (
                  <div key={idx} className="flex items-start gap-2 text-sm">
                    <CheckCircle className="h-4 w-4 text-green-500 mt-0.5 flex-shrink-0" />
                    <span>{finding}</span>
                  </div>
                ))}
              </div>
              
              <div className="mt-3 flex items-center gap-2">
                <span className="text-xs text-gray-500">Confidence:</span>
                <Badge className={getConfidenceColor(
                  (structured as any)?.confidence_assessment?.overall_confidence || 0.7
                )}>
                  {(((structured as any)?.confidence_assessment?.overall_confidence || 0.7) * 100).toFixed(0)}%
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
                  {(structured?.differential_diagnoses as any)?.[0]?.condition ||
                   (structured?.clinical_summary as any)?.primary_assessment ||
                   "Assessment completed"}
                </span>
                <Badge>High confidence</Badge>
              </div>
              <p className="text-sm text-gray-600 dark:text-gray-300">
                Based on comprehensive expert analysis
              </p>
            </CardContent>
          </Card>
        </TabsContent>
        
        <TabsContent value="actions" className="space-y-3">
          <div className="space-y-4">
            {((structured?.recommendations as any)?.immediate_actions?.length > 0) && (
              <Card>
                <CardHeader className="pb-3">
                  <CardTitle className="text-base flex items-center gap-2">
                    <AlertTriangle className="h-4 w-4 text-orange-500" />
                    Immediate Actions
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-2">
                  {(structured?.recommendations as any)?.immediate_actions?.map((action: string, idx: number) => (
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

  return (
    <div className="flex h-screen bg-white dark:bg-gray-900">
      {/* Sidebar */}
      <div className={`${sidebarOpen ? 'w-64' : 'w-0'} transition-all duration-300 overflow-hidden bg-gray-50 dark:bg-gray-800 border-r`}>
        <div className="p-4 space-y-3">
          <Button 
            onClick={handleNewChat}
            className="w-full justify-start gap-2"
            variant="outline"
          >
            <Plus className="h-4 w-4" />
            New Chat
          </Button>
          
          {/* Chat History */}
          <div className="space-y-2">
            <div className="flex items-center gap-2 text-sm font-medium text-gray-600 dark:text-gray-300">
              <History className="h-4 w-4" />
              Recent Chats
            </div>
            <ScrollArea className="h-[300px]">
              {chatHistory.map((chat) => (
                <Button
                  key={chat.session_id}
                  variant="ghost"
                  size="sm"
                  className="w-full justify-start p-2 h-auto text-left mb-1"
                  onClick={() => loadChatSession(chat.session_id)}
                >
                  <div className="truncate">
                    <div className="text-xs font-medium truncate">{chat.title}</div>
                    <div className="text-xs text-gray-500 truncate">{chat.last_message}</div>
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
          
          {/* System Status */}
          {systemHealth && (
            <div className="p-3 bg-white dark:bg-gray-700 rounded-lg text-xs">
              <div className="flex items-center gap-2 mb-1">
                <div className={`w-2 h-2 rounded-full ${
                  systemHealth.status === 'healthy' ? 'bg-green-500' : 'bg-red-500'
                }`} />
                <span className="font-medium">System Status</span>
              </div>
              <div className="text-gray-600 dark:text-gray-300">
                Expert Council: {systemHealth.systems?.expert_council?.status || 'checking...'}
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
                Session: {currentSession.split('_')[1]}
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
                      onClick={() => setInput("I have chest pain and shortness of breath")}
                    >
                      <AlertTriangle className="h-4 w-4 mr-2 text-red-500" />
                      Emergency symptoms
                    </Button>
                    <Button 
                      variant="outline"
                      className="p-4 h-auto text-left justify-start"
                      onClick={() => setInput("I have a persistent headache for 3 days")}
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
                          
                          {message.metadata && (
                            <div className="mt-2 space-y-2">
                              <div className="flex items-center gap-2 text-xs text-gray-500">
                                {getServiceIcon(message.metadata.service_used)}
                                <span>{message.metadata.service_used.replace(/_/g, ' ')}</span>
                                {message.metadata.confidence > 0 && (
                                  <Badge className={getConfidenceColor(message.metadata.confidence)} variant="outline">
                                    {(message.metadata.confidence * 100).toFixed(0)}%
                                  </Badge>
                                )}
                              </div>
                              
                              {message.metadata.structured_analysis && message.metadata.interactive_components && 
                                renderExpertCouncilReport(
                                  message.metadata.structured_analysis,
                                  message.metadata.interactive_components
                                )
                              }
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  ))}
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
                  placeholder="Ask AURA about your health concerns..."
                  onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && handleSend()}
                  disabled={isLoading}
                  className="pr-12 py-3 text-base"
                />
                <Button 
                  onClick={handleSend} 
                  disabled={isLoading || !input.trim()}
                  size="sm"
                  className="absolute right-2 top-1/2 transform -translate-y-1/2"
                >
                  {isLoading ? (
                    <div className="h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" />
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