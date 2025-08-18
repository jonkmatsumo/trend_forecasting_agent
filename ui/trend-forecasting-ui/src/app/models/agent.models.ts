export interface AgentRequest {
  message: string;
  context?: string;
  options?: {
    maxTokens?: number;
    temperature?: number;
  };
}

export interface AgentResponse {
  response: string;
  messageId: string;
  timestamp: string;
  metadata?: {
    tokensUsed?: number;
    processingTime?: number;
  };
}

export interface ChatMessage {
  id: number;
  type: 'user' | 'agent' | 'system' | 'error';
  text: string;
  data?: any;
  metadata?: any;
  timestamp: Date;
}

export interface AgentConfig {
  model: string;
  temperature: number;
  maxTokens: number;
  timeout?: number;
  retryAttempts?: number;
  systemPrompt?: string;
} 