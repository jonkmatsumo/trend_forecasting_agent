import { Injectable } from '@angular/core';
import { environment } from '../../environments/environment';

export interface AppConfig {
  apiUrl: string;
  agentUrl: string;
  enableLogging: boolean;
  defaultTimeout: number;
  maxRetries: number;
}

@Injectable({
  providedIn: 'root'
})
export class ConfigService {
  private config: AppConfig = {
    apiUrl: environment.apiUrl,
    agentUrl: environment.agentUrl,
    enableLogging: environment.enableLogging,
    defaultTimeout: 30000, // 30 seconds
    maxRetries: 3
  };

  constructor() {}

  getConfig(): AppConfig {
    return { ...this.config };
  }

  getApiUrl(): string {
    return this.config.apiUrl;
  }

  getAgentUrl(): string {
    return this.config.agentUrl;
  }

  isLoggingEnabled(): boolean {
    return this.config.enableLogging;
  }

  getDefaultTimeout(): number {
    return this.config.defaultTimeout;
  }

  getMaxRetries(): number {
    return this.config.maxRetries;
  }

  updateConfig(updates: Partial<AppConfig>): void {
    this.config = { ...this.config, ...updates };
  }
} 