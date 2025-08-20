// Test setup file for Zone.js
import 'zone.js/testing';
import { getTestBed } from '@angular/core/testing';
import { BrowserDynamicTestingModule, platformBrowserDynamicTesting } from '@angular/platform-browser-dynamic/testing';

// Initialize the Angular testing environment
getTestBed().initTestEnvironment(
  BrowserDynamicTestingModule,
  platformBrowserDynamicTesting(),
);

// Ensure Zone.js is properly configured for testing
import 'zone.js/fesm2015/fake-async-test.js';
import 'zone.js/fesm2015/task-tracking.js';

// Set a global flag to indicate we're in test mode
(globalThis as any).__TESTING__ = true; 