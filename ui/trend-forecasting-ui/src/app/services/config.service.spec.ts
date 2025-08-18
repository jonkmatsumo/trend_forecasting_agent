import { TestBed } from '@angular/core/testing';
import { ConfigService } from './config.service';
import { environment } from '../../environments/environment';

describe('ConfigService', () => {
  let service: ConfigService;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [ConfigService]
    });
    service = TestBed.inject(ConfigService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });

  describe('API Configuration', () => {
    it('should return correct API URL', () => {
      expect(service.apiUrl).toBe(environment.apiUrl);
    });

    it('should return correct API base URL', () => {
      expect(service.apiUrl).toBe(environment.apiUrl);
    });

    it('should return correct API timeout', () => {
      expect(service.apiTimeout).toBe(environment.apiTimeout || 30000);
    });

    it('should return correct API retry attempts', () => {
      expect(service.apiRetryAttempts).toBe(environment.apiRetryAttempts || 3);
    });
  });

  describe('Environment Configuration', () => {
    it('should return correct environment name', () => {
      expect(service.environment).toBe(environment.environment);
    });

    it('should return correct production flag', () => {
      expect(service.isProduction).toBe(environment.production);
    });

    it('should return correct development flag', () => {
      expect(service.isDevelopment).toBe(!environment.production);
    });

    it('should return correct debug flag', () => {
      expect(service.debug).toBe(environment.debug || false);
    });
  });

  describe('Feature Flags', () => {
    it('should return correct feature flags', () => {
      expect(service.features).toBeDefined();
      expect(typeof service.features).toBe('object');
    });

    it('should return correct notifications enabled flag', () => {
      expect(service.notificationsEnabled).toBe(environment.features?.notifications?.enabled || true);
    });

    it('should return correct analytics enabled flag', () => {
      expect(service.analyticsEnabled).toBe(environment.features?.analytics?.enabled || false);
    });

    it('should return correct logging enabled flag', () => {
      expect(service.loggingEnabled).toBe(environment.features?.logging?.enabled || true);
    });
  });

  describe('UI Configuration', () => {
    it('should return correct theme', () => {
      expect(service.theme).toBe(environment.ui?.theme || 'light');
    });

    it('should return correct language', () => {
      expect(service.language).toBe(environment.ui?.language || 'en');
    });

    it('should return correct timezone', () => {
      expect(service.timezone).toBe(environment.ui?.timezone || 'UTC');
    });

    it('should return correct date format', () => {
      expect(service.dateFormat).toBe(environment.ui?.dateFormat || 'YYYY-MM-DD');
    });

    it('should return correct time format', () => {
      expect(service.timeFormat).toBe(environment.ui?.timeFormat || 'HH:mm:ss');
    });
  });

  describe('Security Configuration', () => {
    it('should return correct security settings', () => {
      expect(service.security).toBeDefined();
      expect(typeof service.security).toBe('object');
    });

    it('should return correct CORS settings', () => {
      expect(service.corsEnabled).toBe(environment.security?.cors?.enabled || true);
    });

    it('should return correct authentication settings', () => {
      expect(service.authEnabled).toBe(environment.security?.auth?.enabled || false);
    });

    it('should return correct rate limiting settings', () => {
      expect(service.rateLimitingEnabled).toBe(environment.security?.rateLimiting?.enabled || false);
    });
  });

  describe('Performance Configuration', () => {
    it('should return correct performance settings', () => {
      expect(service.performance).toBeDefined();
      expect(typeof service.performance).toBe('object');
    });

    it('should return correct cache settings', () => {
      expect(service.cacheEnabled).toBe(environment.performance?.cache?.enabled || true);
    });

    it('should return correct compression settings', () => {
      expect(service.compressionEnabled).toBe(environment.performance?.compression?.enabled || true);
    });

    it('should return correct optimization settings', () => {
      expect(service.optimizationEnabled).toBe(environment.performance?.optimization?.enabled || true);
    });
  });

  describe('Monitoring Configuration', () => {
    it('should return correct monitoring settings', () => {
      expect(service.monitoring).toBeDefined();
      expect(typeof service.monitoring).toBe('object');
    });

    it('should return correct health check settings', () => {
      expect(service.healthCheckEnabled).toBe(environment.monitoring?.healthCheck?.enabled || true);
    });

    it('should return correct metrics settings', () => {
      expect(service.metricsEnabled).toBe(environment.monitoring?.metrics?.enabled || false);
    });

    it('should return correct tracing settings', () => {
      expect(service.tracingEnabled).toBe(environment.monitoring?.tracing?.enabled || false);
    });
  });

  describe('Validation Configuration', () => {
    it('should return correct validation settings', () => {
      expect(service.validation).toBeDefined();
      expect(typeof service.validation).toBe('object');
    });

    it('should return correct input validation settings', () => {
      expect(service.inputValidationEnabled).toBe(environment.validation?.input?.enabled || true);
    });

    it('should return correct output validation settings', () => {
      expect(service.outputValidationEnabled).toBe(environment.validation?.output?.enabled || true);
    });

    it('should return correct schema validation settings', () => {
      expect(service.schemaValidationEnabled).toBe(environment.validation?.schema?.enabled || true);
    });
  });

  describe('Error Handling Configuration', () => {
    it('should return correct error handling settings', () => {
      expect(service.errorHandling).toBeDefined();
      expect(typeof service.errorHandling).toBe('object');
    });

    it('should return correct error reporting settings', () => {
      expect(service.errorReportingEnabled).toBe(environment.errorHandling?.reporting?.enabled || false);
    });

    it('should return correct error logging settings', () => {
      expect(service.errorLoggingEnabled).toBe(environment.errorHandling?.logging?.enabled || true);
    });

    it('should return correct error recovery settings', () => {
      expect(service.errorRecoveryEnabled).toBe(environment.errorHandling?.recovery?.enabled || true);
    });
  });

  describe('Configuration Methods', () => {
    it('should get configuration value by key', () => {
      const apiUrl = service.getConfig('apiUrl');
      expect(apiUrl).toBe(environment.apiUrl);
    });

    it('should get nested configuration value by path', () => {
      const notificationsEnabled = service.getConfig('features.notifications.enabled');
      expect(notificationsEnabled).toBe(environment.features?.notifications?.enabled || true);
    });

    it('should return undefined for non-existent configuration', () => {
      const nonExistent = service.getConfig('non.existent.config');
      expect(nonExistent).toBeUndefined();
    });

    it('should return default value for non-existent configuration', () => {
      const defaultValue = service.getConfig('non.existent.config', 'default');
      expect(defaultValue).toBe('default');
    });

    it('should get all configuration', () => {
      const allConfig = service.getAllConfig();
      expect(allConfig).toBeDefined();
      expect(typeof allConfig).toBe('object');
      expect(allConfig.apiUrl).toBe(environment.apiUrl);
    });

    it('should check if configuration exists', () => {
      expect(service.hasConfig('apiUrl')).toBe(true);
      expect(service.hasConfig('non.existent.config')).toBe(false);
    });

    it('should check if configuration is enabled', () => {
      expect(service.isEnabled('features.notifications.enabled')).toBe(environment.features?.notifications?.enabled || true);
      expect(service.isEnabled('non.existent.config')).toBe(false);
    });
  });

  describe('Environment-Specific Configuration', () => {
    it('should return correct configuration for production', () => {
      if (environment.production) {
        expect(service.isProduction).toBe(true);
        expect(service.isDevelopment).toBe(false);
      }
    });

    it('should return correct configuration for development', () => {
      if (!environment.production) {
        expect(service.isProduction).toBe(false);
        expect(service.isDevelopment).toBe(true);
      }
    });

    it('should return correct debug configuration', () => {
      if (environment.debug) {
        expect(service.debug).toBe(true);
      }
    });
  });

  describe('Configuration Validation', () => {
    it('should validate required configuration', () => {
      const requiredConfig = ['apiUrl', 'environment'];
      const isValid = service.validateConfig(requiredConfig);
      expect(isValid).toBe(true);
    });

    it('should fail validation for missing required configuration', () => {
      const requiredConfig = ['apiUrl', 'non.existent.config'];
      const isValid = service.validateConfig(requiredConfig);
      expect(isValid).toBe(false);
    });

    it('should validate configuration structure', () => {
      const configStructure = {
        apiUrl: 'string',
        environment: 'string',
        production: 'boolean'
      };
      const isValid = service.validateConfigStructure(configStructure);
      expect(isValid).toBe(true);
    });

    it('should fail validation for incorrect configuration structure', () => {
      const configStructure = {
        apiUrl: 'number', // Should be string
        environment: 'string',
        production: 'boolean'
      };
      const isValid = service.validateConfigStructure(configStructure);
      expect(isValid).toBe(false);
    });
  });

  describe('Configuration Updates', () => {
    it('should update configuration value', () => {
      const newApiUrl = 'https://new-api.example.com';
      service.updateConfig('apiUrl', newApiUrl);
      expect(service.apiUrl).toBe(newApiUrl);
    });

    it('should update nested configuration value', () => {
      const newTheme = 'dark';
      service.updateConfig('ui.theme', newTheme);
      expect(service.theme).toBe(newTheme);
    });

    it('should not update non-existent configuration', () => {
      const originalConfig = service.getAllConfig();
      service.updateConfig('non.existent.config', 'value');
      expect(service.getAllConfig()).toEqual(originalConfig);
    });

    it('should reset configuration to default', () => {
      const originalApiUrl = service.apiUrl;
      service.updateConfig('apiUrl', 'https://modified-api.example.com');
      expect(service.apiUrl).toBe('https://modified-api.example.com');
      
      service.resetConfig();
      expect(service.apiUrl).toBe(originalApiUrl);
    });
  });

  describe('Configuration Events', () => {
    it('should emit configuration change events', (done) => {
      service.configChanged$.subscribe((change) => {
        expect(change).toBeDefined();
        expect(change!.key).toBe('apiUrl');
        expect(change!.value).toBe('https://new-api.example.com');
        done();
      });

      service.updateConfig('apiUrl', 'https://new-api.example.com');
    });

    it('should emit configuration reset events', (done) => {
      service.configReset$.subscribe(() => {
        done();
      });

      service.resetConfig();
    });
  });

  describe('Configuration Persistence', () => {
    it('should save configuration to storage', () => {
      spyOn(localStorage, 'setItem');
      service.saveConfig();
      expect(localStorage.setItem).toHaveBeenCalled();
    });

    it('should load configuration from storage', () => {
      const mockConfig = { apiUrl: 'https://saved-api.example.com' };
      spyOn(localStorage, 'getItem').and.returnValue(JSON.stringify(mockConfig));
      service.loadConfig();
      expect(localStorage.getItem).toHaveBeenCalled();
    });

    it('should clear configuration from storage', () => {
      spyOn(localStorage, 'removeItem');
      service.clearConfig();
      expect(localStorage.removeItem).toHaveBeenCalled();
    });
  });

  describe('Configuration Utilities', () => {
    it('should merge configurations', () => {
      const config1: any = { a: 1, b: 2 };
      const config2: any = { b: 3, c: 4 };
      const merged = service.mergeConfig(config1, config2);
      expect(merged).toEqual({ a: 1, b: 3, c: 4 });
    });

    it('should deep merge configurations', () => {
      const config1: any = { a: { b: 1, c: 2 } };
      const config2: any = { a: { c: 3, d: 4 } };
      const merged = service.deepMergeConfig(config1, config2);
      expect(merged).toEqual({ a: { b: 1, c: 3, d: 4 } });
    });

    it('should clone configuration', () => {
      const original: any = { apiUrl: 'https://example.com', features: { notifications: { enabled: true } } };
      const cloned = service.cloneConfig(original);
      expect(cloned).toEqual(original);
      expect(cloned).not.toBe(original);
    });

    it('should flatten configuration', () => {
      const nested = { a: { b: { c: 1 } }, d: 2 };
      const flattened = service.flattenConfig(nested);
      expect(flattened).toEqual({ 'a.b.c': 1, d: 2 });
    });

    it('should unflatten configuration', () => {
      const flattened = { 'a.b.c': 1, d: 2 };
      const nested = service.unflattenConfig(flattened);
      expect(nested).toEqual({ a: { b: { c: 1 } }, d: 2 });
    });
  });
});