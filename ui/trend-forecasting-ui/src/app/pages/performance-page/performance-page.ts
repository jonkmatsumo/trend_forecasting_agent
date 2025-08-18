import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { MatCardModule } from '@angular/material/card';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatTabsModule } from '@angular/material/tabs';

import { PerformanceMonitorComponent } from '../../components/performance-monitor/performance-monitor';

@Component({
  selector: 'app-performance-page',
  standalone: true,
  imports: [
    CommonModule,
    MatCardModule,
    MatButtonModule,
    MatIconModule,
    MatTabsModule,
    PerformanceMonitorComponent
  ],
  template: `
    <div class="performance-page">
      <mat-card class="page-header">
        <mat-card-header>
          <mat-card-title>
            <mat-icon>speed</mat-icon>
            Performance & Optimization
          </mat-card-title>
          <mat-card-subtitle>
            Monitor application performance and optimize for better user experience
          </mat-card-subtitle>
        </mat-card-header>
      </mat-card>

      <mat-tab-group class="performance-tabs">
        <mat-tab label="Performance Monitor">
          <app-performance-monitor></app-performance-monitor>
        </mat-tab>
        
        <mat-tab label="Optimization Guide">
          <div class="optimization-guide">
            <mat-card>
              <mat-card-header>
                <mat-card-title>Performance Optimization Tips</mat-card-title>
              </mat-card-header>
              <mat-card-content>
                <div class="tip-section">
                  <h3>🚀 Lazy Loading</h3>
                  <p>Routes are now lazy-loaded to reduce initial bundle size and improve load times.</p>
                  <ul>
                    <li>Components load only when needed</li>
                    <li>Reduces initial JavaScript bundle</li>
                    <li>Improves first contentful paint</li>
                  </ul>
                </div>

                <div class="tip-section">
                  <h3>⚡ Service Worker</h3>
                  <p>Service worker provides offline capabilities and intelligent caching.</p>
                  <ul>
                    <li>Offline functionality</li>
                    <li>Background sync</li>
                    <li>Push notifications</li>
                    <li>Intelligent caching strategies</li>
                  </ul>
                </div>

                <div class="tip-section">
                  <h3>📊 Web Vitals Monitoring</h3>
                  <p>Real-time monitoring of Core Web Vitals for optimal user experience.</p>
                  <ul>
                    <li>First Contentful Paint (FCP)</li>
                    <li>Largest Contentful Paint (LCP)</li>
                    <li>Cumulative Layout Shift (CLS)</li>
                    <li>First Input Delay (FID)</li>
                  </ul>
                </div>

                <div class="tip-section">
                  <h3>🎯 Bundle Optimization</h3>
                  <p>Production builds are optimized for maximum performance.</p>
                  <ul>
                    <li>Tree shaking removes unused code</li>
                    <li>Code splitting for better caching</li>
                    <li>Minification and compression</li>
                    <li>Source maps disabled in production</li>
                  </ul>
                </div>

                <div class="tip-section">
                  <h3>📱 PWA Features</h3>
                  <p>Progressive Web App features for better mobile experience.</p>
                  <ul>
                    <li>Web app manifest</li>
                    <li>Installable on mobile devices</li>
                    <li>App-like experience</li>
                    <li>Offline functionality</li>
                  </ul>
                </div>
              </mat-card-content>
            </mat-card>
          </div>
        </mat-tab>

        <mat-tab label="Best Practices">
          <div class="best-practices">
            <mat-card>
              <mat-card-header>
                <mat-card-title>Performance Best Practices</mat-card-title>
              </mat-card-header>
              <mat-card-content>
                <div class="practice-section">
                  <h3>✅ Do's</h3>
                  <ul>
                    <li>Use lazy loading for routes and components</li>
                    <li>Optimize images and use appropriate formats</li>
                    <li>Minimize bundle size with tree shaking</li>
                    <li>Use service workers for caching</li>
                    <li>Monitor Core Web Vitals</li>
                    <li>Implement proper error boundaries</li>
                    <li>Use OnPush change detection strategy</li>
                    <li>Optimize third-party scripts</li>
                  </ul>
                </div>

                <div class="practice-section">
                  <h3>❌ Don'ts</h3>
                  <ul>
                    <li>Don't load unnecessary dependencies</li>
                    <li>Avoid large bundle sizes</li>
                    <li>Don't block rendering with heavy scripts</li>
                    <li>Avoid layout shifts during loading</li>
                    <li>Don't ignore mobile performance</li>
                    <li>Avoid synchronous operations in main thread</li>
                    <li>Don't forget to test on slow networks</li>
                    <li>Avoid memory leaks in components</li>
                  </ul>
                </div>
              </mat-card-content>
            </mat-card>
          </div>
        </mat-tab>
      </mat-tab-group>
    </div>
  `,
  styles: [`
    .performance-page {
      padding: 1rem;
      max-width: 1200px;
      margin: 0 auto;
    }

    .page-header {
      margin-bottom: 2rem;
    }

    .performance-tabs {
      margin-top: 1rem;
    }

    .optimization-guide,
    .best-practices {
      padding: 1rem;
    }

    .tip-section,
    .practice-section {
      margin-bottom: 2rem;
      padding: 1rem;
      border-left: 4px solid #1976d2;
      background: #f5f5f5;
      border-radius: 0 8px 8px 0;
    }

    .tip-section h3,
    .practice-section h3 {
      margin-top: 0;
      color: #1976d2;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }

    .tip-section ul,
    .practice-section ul {
      margin: 1rem 0;
      padding-left: 1.5rem;
    }

    .tip-section li,
    .practice-section li {
      margin-bottom: 0.5rem;
      line-height: 1.5;
    }

    .practice-section h3:first-child {
      color: #4caf50;
    }

    .practice-section h3:last-child {
      color: #f44336;
    }

    @media (max-width: 768px) {
      .performance-page {
        padding: 0.5rem;
      }

      .tip-section,
      .practice-section {
        padding: 0.75rem;
      }
    }
  `]
})
export class PerformancePageComponent {} 