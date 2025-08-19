// Test setup file for Zone.js
import 'zone.js/testing';

// Override the app config for testing to use Zone.js instead of zoneless change detection
import { appConfig as originalConfig } from './app/app.config';
import { appConfig as testConfig } from './app/app.config.test';

// Replace the original config with test config for testing
Object.defineProperty(originalConfig, 'providers', {
  value: testConfig.providers,
  writable: false
}); 