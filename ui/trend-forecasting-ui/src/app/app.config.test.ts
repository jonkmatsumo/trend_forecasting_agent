import { ApplicationConfig, provideZoneChangeDetection } from '@angular/core';
import { provideRouter } from '@angular/router';
import { provideHttpClient, withInterceptorsFromDi } from '@angular/common/http';
import { provideAnimations } from '@angular/platform-browser/animations';
import { provideNativeDateAdapter } from '@angular/material/core';
import { provideServiceWorker } from '@angular/service-worker';
import { provideClientHydration, withEventReplay } from '@angular/platform-browser';

import { routes } from './app.routes';

// Test-specific configuration that explicitly uses Zone.js
export const appConfig: ApplicationConfig = {
  providers: [
    // Explicitly use Zone.js for testing
    provideZoneChangeDetection(),
    
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
      enabled: false,
      registrationStrategy: 'registerWhenStable:30000'
    }),
    
    // Client hydration
    provideClientHydration(withEventReplay())
  ]
}; 