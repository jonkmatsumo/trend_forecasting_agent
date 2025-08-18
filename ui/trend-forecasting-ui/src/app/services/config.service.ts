import { Injectable } from '@angular/core';
import { BehaviorSubject, Observable } from 'rxjs';
import { environment } from '../../environments/environment';

export interface AppConfig {
  apiUrl: string;
  agentUrl: string;
  enableLogging: boolean;
  defaultTimeout: number;
  maxRetries: number;
  production: boolean;
  apiTimeout?: number;
  apiRetryAttempts?: number;
  environment?: string;
  debug?: boolean;
  features?: {
    notifications?: { enabled: boolean };
    analytics?: { enabled: boolean };
    logging?: { enabled: boolean };
  };
  ui?: {
    theme?: string;
    language?: string;
    timezone?: string;
    dateFormat?: string;
    timeFormat?: string;
  };
  security?: {
    cors?: { enabled: boolean };
    auth?: { enabled: boolean };
    rateLimiting?: { enabled: boolean };
  };
  performance?: {
    cache?: { enabled: boolean };
    compression?: { enabled: boolean };
    optimization?: { enabled: boolean };
  };
  monitoring?: {
    healthCheck?: { enabled: boolean };
    metrics?: { enabled: boolean };
    tracing?: { enabled: boolean };
  };
  validation?: {
    input?: { enabled: boolean };
    output?: { enabled: boolean };
    schema?: { enabled: boolean };
  };
  errorHandling?: {
    reporting?: { enabled: boolean };
    logging?: { enabled: boolean };
    recovery?: { enabled: boolean };
  };
  [key: string]: any; // Allow dynamic property access
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
    maxRetries: 3,
    production: environment.production,
    apiTimeout: 30000,
    apiRetryAttempts: 3,
    environment: environment.production ? 'production' : 'development',
    debug: false,
    features: {
      notifications: { enabled: true },
      analytics: { enabled: false },
      logging: { enabled: true }
    },
    ui: {
      theme: 'light',
      language: 'en',
      timezone: 'UTC',
      dateFormat: 'YYYY-MM-DD',
      timeFormat: 'HH:mm:ss'
    },
    security: {
      cors: { enabled: true },
      auth: { enabled: false },
      rateLimiting: { enabled: true }
    },
    performance: {
      cache: { enabled: true },
      compression: { enabled: true },
      optimization: { enabled: true }
    },
    monitoring: {
      healthCheck: { enabled: true },
      metrics: { enabled: false },
      tracing: { enabled: false }
    },
    validation: {
      input: { enabled: true },
      output: { enabled: true },
      schema: { enabled: true }
    },
    errorHandling: {
      reporting: { enabled: false },
      logging: { enabled: true },
      recovery: { enabled: true }
    }
  };

  private _configChanged$ = new BehaviorSubject<{ key: string; value: any } | null>(null);
  private _configReset$ = new BehaviorSubject<void>(undefined);

  constructor() {}

  // Property getters for tests
  get apiUrl(): string { return this.config.apiUrl; }
  get agentUrl(): string { return this.config.agentUrl; }
  get apiTimeout(): number { return this.config.apiTimeout || 30000; }
  get apiRetryAttempts(): number { return this.config.apiRetryAttempts || 3; }
  get environment(): string { return this.config.environment || 'development'; }
  get isProduction(): boolean { return this.config.production; }
  get isDevelopment(): boolean { return !this.config.production; }
  get debug(): boolean { return this.config.debug || false; }
  get features(): any { return this.config.features; }
  get notificationsEnabled(): boolean { return this.config.features?.notifications?.enabled || false; }
  get analyticsEnabled(): boolean { return this.config.features?.analytics?.enabled || false; }
  get loggingEnabled(): boolean { return this.config.features?.logging?.enabled || true; }
  get theme(): string { return this.config.ui?.theme || 'light'; }
  get language(): string { return this.config.ui?.language || 'en'; }
  get timezone(): string { return this.config.ui?.timezone || 'UTC'; }
  get dateFormat(): string { return this.config.ui?.dateFormat || 'YYYY-MM-DD'; }
  get timeFormat(): string { return this.config.ui?.timeFormat || 'HH:mm:ss'; }
  get security(): any { return this.config.security; }
  get corsEnabled(): boolean { return this.config.security?.cors?.enabled || true; }
  get authEnabled(): boolean { return this.config.security?.auth?.enabled || false; }
  get rateLimitingEnabled(): boolean { return this.config.security?.rateLimiting?.enabled || true; }
  get performance(): any { return this.config.performance; }
  get cacheEnabled(): boolean { return this.config.performance?.cache?.enabled || true; }
  get compressionEnabled(): boolean { return this.config.performance?.compression?.enabled || true; }
  get optimizationEnabled(): boolean { return this.config.performance?.optimization?.enabled || true; }
  get monitoring(): any { return this.config.monitoring; }
  get healthCheckEnabled(): boolean { return this.config.monitoring?.healthCheck?.enabled || true; }
  get metricsEnabled(): boolean { return this.config.monitoring?.metrics?.enabled || false; }
  get tracingEnabled(): boolean { return this.config.monitoring?.tracing?.enabled || false; }
  get validation(): any { return this.config.validation; }
  get inputValidationEnabled(): boolean { return this.config.validation?.input?.enabled || true; }
  get outputValidationEnabled(): boolean { return this.config.validation?.output?.enabled || true; }
  get schemaValidationEnabled(): boolean { return this.config.validation?.schema?.enabled || true; }
  get errorHandling(): any { return this.config.errorHandling; }
  get errorReportingEnabled(): boolean { return this.config.errorHandling?.reporting?.enabled || false; }
  get errorLoggingEnabled(): boolean { return this.config.errorHandling?.logging?.enabled || true; }
  get errorRecoveryEnabled(): boolean { return this.config.errorHandling?.recovery?.enabled || true; }

  // Method getters
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

  // Configuration methods
  getConfig(key?: string, defaultValue?: any): any {
    if (!key) return { ...this.config };
    
    const keys = key.split('.');
    let value = this.config;
    
    for (const k of keys) {
      if (value && typeof value === 'object' && k in value) {
        value = value[k];
      } else {
        return defaultValue !== undefined ? defaultValue : undefined;
      }
    }
    
    return value;
  }

  getAllConfig(): AppConfig {
    return { ...this.config };
  }

  hasConfig(key: string): boolean {
    return this.getConfig(key) !== undefined;
  }

  isEnabled(key: string): boolean {
    const value = this.getConfig(key);
    return typeof value === 'boolean' ? value : false;
  }

  updateConfig(key: string, value: any): void {
    const keys = key.split('.');
    let current = this.config;
    
    for (let i = 0; i < keys.length - 1; i++) {
      if (!current[keys[i]]) {
        current[keys[i]] = {};
      }
      current = current[keys[i]];
    }
    
    current[keys[keys.length - 1]] = value;
    this._configChanged$.next({ key, value });
  }

  resetConfig(): void {
    // Reset to default values
    this.config = {
      apiUrl: environment.apiUrl,
      agentUrl: environment.agentUrl,
      enableLogging: environment.enableLogging,
      defaultTimeout: 30000,
      maxRetries: 3,
      production: environment.production,
      apiTimeout: 30000,
      apiRetryAttempts: 3,
      environment: environment.production ? 'production' : 'development',
      debug: false,
      features: {
        notifications: { enabled: true },
        analytics: { enabled: false },
        logging: { enabled: true }
      },
      ui: {
        theme: 'light',
        language: 'en',
        timezone: 'UTC',
        dateFormat: 'YYYY-MM-DD',
        timeFormat: 'HH:mm:ss'
      },
      security: {
        cors: { enabled: true },
        auth: { enabled: false },
        rateLimiting: { enabled: true }
      },
      performance: {
        cache: { enabled: true },
        compression: { enabled: true },
        optimization: { enabled: true }
      },
      monitoring: {
        healthCheck: { enabled: true },
        metrics: { enabled: false },
        tracing: { enabled: false }
      },
      validation: {
        input: { enabled: true },
        output: { enabled: true },
        schema: { enabled: true }
      },
      errorHandling: {
        reporting: { enabled: false },
        logging: { enabled: true },
        recovery: { enabled: true }
      }
    };
    this._configReset$.next();
  }

  // Observables
  get configChanged$(): Observable<{ key: string; value: any } | null> {
    return this._configChanged$.asObservable();
  }

  get configReset$(): Observable<void> {
    return this._configReset$.asObservable();
  }

  // Utility methods
  saveConfig(): void {
    localStorage.setItem('appConfig', JSON.stringify(this.config));
  }

  loadConfig(): void {
    const saved = localStorage.getItem('appConfig');
    if (saved) {
      this.config = { ...this.config, ...JSON.parse(saved) };
    }
  }

  clearConfig(): void {
    localStorage.removeItem('appConfig');
  }

  mergeConfig(config1: Partial<AppConfig>, config2: Partial<AppConfig>): AppConfig {
    return { ...config1, ...config2 } as AppConfig;
  }

  deepMergeConfig(config1: Partial<AppConfig>, config2: Partial<AppConfig>): AppConfig {
    const merged = { ...config1 };
    for (const key in config2) {
      if (config2[key] && typeof config2[key] === 'object' && !Array.isArray(config2[key])) {
        merged[key] = this.deepMergeConfig(merged[key] || {}, config2[key]);
      } else {
        merged[key] = config2[key];
      }
    }
    return merged as AppConfig;
  }

  cloneConfig(config: AppConfig): AppConfig {
    return JSON.parse(JSON.stringify(config));
  }

  flattenConfig(obj: any, prefix = ''): Record<string, any> {
    const flattened: Record<string, any> = {};
    for (const key in obj) {
      if (obj[key] && typeof obj[key] === 'object' && !Array.isArray(obj[key])) {
        Object.assign(flattened, this.flattenConfig(obj[key], `${prefix}${key}.`));
      } else {
        flattened[`${prefix}${key}`] = obj[key];
      }
    }
    return flattened;
  }

  unflattenConfig(flattened: Record<string, any>): any {
    const unflattened: any = {};
    for (const key in flattened) {
      const keys = key.split('.');
      let current = unflattened;
      for (let i = 0; i < keys.length - 1; i++) {
        if (!current[keys[i]]) {
          current[keys[i]] = {};
        }
        current = current[keys[i]];
      }
      current[keys[keys.length - 1]] = flattened[key];
    }
    return unflattened;
  }

  validateConfig(requiredConfig: string[]): boolean {
    return requiredConfig.every(key => this.hasConfig(key));
  }

  validateConfigStructure(structure: any): boolean {
    // Simple structure validation
    return typeof structure === 'object' && structure !== null;
  }
} 