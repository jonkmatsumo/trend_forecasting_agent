import { TestBed } from '@angular/core/testing';
import { provideZoneChangeDetection } from '@angular/core';
import { ConfigService } from './config.service';
import { environment } from '../../environments/environment';

describe('ConfigService', () => {
  let service: ConfigService;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [
        ConfigService,
        provideZoneChangeDetection()
      ]
    });
    service = TestBed.inject(ConfigService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });

  describe('getConfig', () => {
    it('should return a copy of the configuration', () => {
      const config = service.getConfig();
      expect(config).toBeDefined();
      expect(config.apiUrl).toBe(environment.apiUrl);
      expect(config.agentUrl).toBe(environment.agentUrl);
      expect(config.enableLogging).toBe(environment.enableLogging);
      expect(config.defaultTimeout).toBe(30000);
      expect(config.maxRetries).toBe(3);
    });

    it('should return a new object each time (immutable)', () => {
      const config1 = service.getConfig();
      const config2 = service.getConfig();
      expect(config1).not.toBe(config2);
    });
  });

  describe('getApiUrl', () => {
    it('should return the API URL from environment', () => {
      expect(service.getApiUrl()).toBe(environment.apiUrl);
    });
  });

  describe('getAgentUrl', () => {
    it('should return the agent URL from environment', () => {
      expect(service.getAgentUrl()).toBe(environment.agentUrl);
    });
  });

  describe('isLoggingEnabled', () => {
    it('should return logging status from environment', () => {
      expect(service.isLoggingEnabled()).toBe(environment.enableLogging);
    });
  });

  describe('getDefaultTimeout', () => {
    it('should return the default timeout value', () => {
      expect(service.getDefaultTimeout()).toBe(30000);
    });
  });

  describe('getMaxRetries', () => {
    it('should return the max retries value', () => {
      expect(service.getMaxRetries()).toBe(3);
    });
  });

  describe('updateConfig', () => {
    it('should update configuration with partial updates', () => {
      const originalTimeout = service.getDefaultTimeout();
      const newTimeout = 60000;

      service.updateConfig({ defaultTimeout: newTimeout });

      expect(service.getDefaultTimeout()).toBe(newTimeout);
      expect(service.getMaxRetries()).toBe(3); // Should remain unchanged
    });

    it('should update multiple configuration values', () => {
      service.updateConfig({
        defaultTimeout: 45000,
        maxRetries: 5
      });

      expect(service.getDefaultTimeout()).toBe(45000);
      expect(service.getMaxRetries()).toBe(5);
    });

    it('should preserve existing values when updating partial config', () => {
      const originalConfig = service.getConfig();
      
      service.updateConfig({ defaultTimeout: 90000 });

      const updatedConfig = service.getConfig();
      expect(updatedConfig.defaultTimeout).toBe(90000);
      expect(updatedConfig.maxRetries).toBe(originalConfig.maxRetries);
      expect(updatedConfig.apiUrl).toBe(originalConfig.apiUrl);
    });
  });
}); 