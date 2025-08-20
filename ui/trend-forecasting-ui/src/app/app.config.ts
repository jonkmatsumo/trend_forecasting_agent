import { ApplicationConfig, provideBrowserGlobalErrorListeners, provideZonelessChangeDetection, provideZoneChangeDetection } from '@angular/core';
import { provideRouter } from '@angular/router';
import { provideHttpClient, withInterceptorsFromDi } from '@angular/common/http';
import { provideAnimations } from '@angular/platform-browser/animations';
import { provideNativeDateAdapter } from '@angular/material/core';
import { provideServiceWorker } from '@angular/service-worker';
import { provideClientHydration, withEventReplay } from '@angular/platform-browser';

import { routes } from './app.routes';

// Detect if we're in test environment
const isTestEnvironment = typeof (globalThis as any).__TESTING__ !== 'undefined';

export const appConfig: ApplicationConfig = {
  providers: [
    // Use Zone.js for testing, zoneless for production
    ...(isTestEnvironment ? [provideZoneChangeDetection()] : [provideZonelessChangeDetection()]),
    
    // Router configuration
    provideRouter(routes),
    
    // HTTP client configuration
    provideHttpClient(withInterceptorsFromDi()),
    
    // Browser animations
    provideAnimations(),
    
    // Material date adapter
    provideNativeDateAdapter(),
    
    // Service worker (disabled for testing)
    provideServiceWorker('ngsw-worker.js', {
      enabled: !isTestEnvironment,
      registrationStrategy: 'registerWhenStable:30000'
    }),
    
    // Client hydration
    provideClientHydration(withEventReplay()),
    
    // Global error listeners
    provideBrowserGlobalErrorListeners()
  ]
};
