export interface AgentRequest {
  query: string;
  context?: any;
  user_id?: string;
  session_id?: string;
}

export interface AgentResponse {
  text: string;
  data?: any;
  metadata?: {
    confidence?: number;
    sources?: string[];
    [key: string]: any;
  };
  timestamp: string;
  request_id: string;
}

export interface ChatMessage {
  id: number;
  type: 'user' | 'agent' | 'system' | 'error';
  text: string;
  data?: any;
  metadata?: any;
  timestamp: Date;
} 