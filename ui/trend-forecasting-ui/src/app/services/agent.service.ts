import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { AgentRequest, AgentResponse, AgentConfig, ChatMessage } from '../models/agent.models';
import { environment } from '../../environments/environment';

@Injectable({
  providedIn: 'root'
})
export class AgentService {
  private baseUrl = environment.agentUrl;

  constructor(private http: HttpClient) {}

  sendMessage(request: AgentRequest): Observable<AgentResponse> {
    return this.http.post<AgentResponse>(`${this.baseUrl}/chat`, request);
  }

  askAgent(request: AgentRequest): Observable<AgentResponse> {
    return this.http.post<AgentResponse>(`${this.baseUrl}/ask`, request);
  }

  getHealth(): Observable<any> {
    return this.http.get(`${this.baseUrl}/health`);
  }

  getCapabilities(): Observable<any> {
    return this.http.get(`${this.baseUrl}/capabilities`);
  }

  getAgentConfig(): Observable<AgentConfig> {
    return this.http.get<AgentConfig>(`${this.baseUrl}/config`);
  }

  updateAgentConfig(config: AgentConfig): Observable<AgentConfig> {
    return this.http.put<AgentConfig>(`${this.baseUrl}/config`, config);
  }

  getChatHistory(): Observable<ChatMessage[]> {
    return this.http.get<ChatMessage[]>(`${this.baseUrl}/chat/history`);
  }

  clearChatHistory(): Observable<any> {
    return this.http.delete(`${this.baseUrl}/chat/history`);
  }

  getAgentStatus(): Observable<any> {
    return this.http.get(`${this.baseUrl}/status`);
  }
} 