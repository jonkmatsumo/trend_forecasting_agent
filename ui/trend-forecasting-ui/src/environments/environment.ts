export const environment = {
  production: false,
  apiUrl: 'http://localhost:5000',
  agentUrl: 'http://localhost:5000/agent',
  enableLogging: true,
  apiTimeout: 30000,
  apiRetryAttempts: 3,
  environment: 'development',
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