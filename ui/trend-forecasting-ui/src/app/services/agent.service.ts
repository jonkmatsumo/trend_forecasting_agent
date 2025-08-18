import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { AgentRequest, AgentResponse } from '../models/agent.models';
import { environment } from '../../environments/environment';

@Injectable({
  providedIn: 'root'
})
export class AgentService {
  private baseUrl = environment.agentUrl;

  constructor(private http: HttpClient) {}

  askAgent(request: AgentRequest): Observable<AgentResponse> {
    return this.http.post<AgentResponse>(`${this.baseUrl}/ask`, request);
  }

  getHealth(): Observable<any> {
    return this.http.get(`${this.baseUrl}/health`);
  }

  getCapabilities(): Observable<any> {
    return this.http.get(`${this.baseUrl}/capabilities`);
  }
} 