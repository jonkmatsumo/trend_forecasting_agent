import { Injectable } from '@angular/core';
import { HttpClient, HttpErrorResponse } from '@angular/common/http';
import { Observable, catchError } from 'rxjs';
import { AgentRequest, AgentResponse, AgentConfig, ChatMessage } from '../models/agent.models';
import { environment } from '../../environments/environment';
import { ErrorHandlerService } from './error-handler.service';
import { LoadingService } from './loading.service';

@Injectable({
  providedIn: 'root'
})
export class AgentService {
  private baseUrl = environment.agentUrl;

  constructor(
    private http: HttpClient,
    private errorHandler: ErrorHandlerService,
    private loadingService: LoadingService
  ) {}

  sendMessage(request: AgentRequest): Observable<AgentResponse> {
    return this.errorHandler.createRetryableRequest<AgentResponse>(
      this.http.post<AgentResponse>(`${this.baseUrl}/chat`, request),
      { retryAttempts: 2, customMessage: 'Failed to send message to agent' }
    );
  }

  askAgent(request: AgentRequest): Observable<AgentResponse> {
    return this.errorHandler.createRetryableRequest<AgentResponse>(
      this.http.post<AgentResponse>(`${this.baseUrl}/ask`, request),
      { retryAttempts: 3, customMessage: 'Failed to get response from agent' }
    );
  }

  getHealth(): Observable<any> {
    return this.errorHandler.createRetryableRequest(
      this.http.get(`${this.baseUrl}/health`).pipe(
        catchError((error: HttpErrorResponse) => 
          this.errorHandler.handleHttpError(error, {
            customMessage: 'Failed to check agent health',
            retryAttempts: 1
          })
        )
      ),
      { retryAttempts: 1, customMessage: 'Failed to check agent health' }
    );
  }

  getCapabilities(): Observable<any> {
    return this.errorHandler.createRetryableRequest(
      this.http.get(`${this.baseUrl}/capabilities`).pipe(
        catchError((error: HttpErrorResponse) => 
          this.errorHandler.handleHttpError(error, {
            customMessage: 'Failed to get agent capabilities',
            retryAttempts: 2
          })
        )
      ),
      { retryAttempts: 2, customMessage: 'Failed to get agent capabilities' }
    );
  }

  getAgentConfig(): Observable<AgentConfig> {
    return this.errorHandler.createRetryableRequest<AgentConfig>(
      this.http.get<AgentConfig>(`${this.baseUrl}/config`),
      { retryAttempts: 2, customMessage: 'Failed to get agent configuration' }
    );
  }

  updateAgentConfig(config: AgentConfig): Observable<AgentConfig> {
    return this.errorHandler.createRetryableRequest<AgentConfig>(
      this.http.put<AgentConfig>(`${this.baseUrl}/config`, config),
      { retryAttempts: 2, customMessage: 'Failed to update agent configuration' }
    );
  }

  getChatHistory(): Observable<ChatMessage[]> {
    return this.errorHandler.createRetryableRequest<ChatMessage[]>(
      this.http.get<ChatMessage[]>(`${this.baseUrl}/chat/history`),
      { retryAttempts: 2, customMessage: 'Failed to get chat history' }
    );
  }

  clearChatHistory(): Observable<any> {
    return this.errorHandler.createRetryableRequest(
      this.http.delete(`${this.baseUrl}/chat/history`).pipe(
        catchError((error: HttpErrorResponse) => 
          this.errorHandler.handleHttpError(error, {
            customMessage: 'Failed to clear chat history',
            retryAttempts: 1
          })
        )
      ),
      { retryAttempts: 1, customMessage: 'Failed to clear chat history' }
    );
  }

  getAgentStatus(): Observable<any> {
    return this.errorHandler.createRetryableRequest(
      this.http.get(`${this.baseUrl}/status`).pipe(
        catchError((error: HttpErrorResponse) => 
          this.errorHandler.handleHttpError(error, {
            customMessage: 'Failed to get agent status',
            retryAttempts: 1
          })
        )
      ),
      { retryAttempts: 1, customMessage: 'Failed to get agent status' }
    );
  }
} 