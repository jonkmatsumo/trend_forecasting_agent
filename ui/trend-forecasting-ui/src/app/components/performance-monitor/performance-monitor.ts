import { Component, OnInit, OnDestroy, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { MatCardModule } from '@angular/material/card';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { MatChipsModule } from '@angular/material/chips';
import { MatDividerModule } from '@angular/material/divider';
import { MatSnackBar } from '@angular/material/snack-bar';

import { PerformanceService, PerformanceMetrics, CacheStats } from '../../services/performance.service';

@Component({
  selector: 'app-performance-monitor',
  standalone: true,
  imports: [
    CommonModule,
    MatCardModule,
    MatButtonModule,
    MatIconModule,
    MatProgressBarModule,
    MatChipsModule,
    MatDividerModule
  ],
  template: `
    <div class="performance-monitor">
      <!-- Real-time Metrics -->
      <mat-card class="metrics-card">
        <mat-card-header>
          <mat-card-title>
            <mat-icon>monitor</mat-icon>
            Real-time Performance Metrics
          </mat-card-title>
          <mat-card-subtitle>
            Core Web Vitals and performance indicators
          </mat-card-subtitle>
        </mat-card-header>
        <mat-card-content>
          <div class="metrics-grid">
            <div class="metric-item">
              <div class="metric-header">
                <span class="metric-label">Load Time</span>
                <mat-chip [color]="getMetricColor(metrics.loadTime, 3000)" selected>
                  {{ metrics.loadTime | number:'1.0-0' }}ms
                </mat-chip>
              </div>
              <mat-progress-bar 
                [value]="getMetricPercentage(metrics.loadTime, 3000)"
                [color]="getMetricColor(metrics.loadTime, 3000)">
              </mat-progress-bar>
            </div>

            <div class="metric-item">
              <div class="metric-header">
                <span class="metric-label">First Contentful Paint</span>
                <mat-chip [color]="getMetricColor(metrics.firstContentfulPaint, 1800)" selected>
                  {{ metrics.firstContentfulPaint | number:'1.0-0' }}ms
                </mat-chip>
              </div>
              <mat-progress-bar 
                [value]="getMetricPercentage(metrics.firstContentfulPaint, 1800)"
                [color]="getMetricColor(metrics.firstContentfulPaint, 1800)">
              </mat-progress-bar>
            </div>

            <div class="metric-item">
              <div class="metric-header">
                <span class="metric-label">Largest Contentful Paint</span>
                <mat-chip [color]="getMetricColor(metrics.largestContentfulPaint, 2500)" selected>
                  {{ metrics.largestContentfulPaint | number:'1.0-0' }}ms
                </mat-chip>
              </div>
              <mat-progress-bar 
                [value]="getMetricPercentage(metrics.largestContentfulPaint, 2500)"
                [color]="getMetricColor(metrics.largestContentfulPaint, 2500)">
              </mat-progress-bar>
            </div>

            <div class="metric-item">
              <div class="metric-header">
                <span class="metric-label">Cumulative Layout Shift</span>
                <mat-chip [color]="getMetricColor(metrics.cumulativeLayoutShift, 0.1, true)" selected>
                  {{ metrics.cumulativeLayoutShift | number:'1.3-3' }}
                </mat-chip>
              </div>
              <mat-progress-bar 
                [value]="getMetricPercentage(metrics.cumulativeLayoutShift, 0.1, true)"
                [color]="getMetricColor(metrics.cumulativeLayoutShift, 0.1, true)">
              </mat-progress-bar>
            </div>

            <div class="metric-item">
              <div class="metric-header">
                <span class="metric-label">First Input Delay</span>
                <mat-chip [color]="getMetricColor(metrics.firstInputDelay, 100)" selected>
                  {{ metrics.firstInputDelay | number:'1.0-0' }}ms
                </mat-chip>
              </div>
              <mat-progress-bar 
                [value]="getMetricPercentage(metrics.firstInputDelay, 100)"
                [color]="getMetricColor(metrics.firstInputDelay, 100)">
              </mat-progress-bar>
            </div>
          </div>
        </mat-card-content>
      </mat-card>

      <!-- Cache Management -->
      <mat-card class="cache-card">
        <mat-card-header>
          <mat-card-title>
            <mat-icon>storage</mat-icon>
            Cache Management
          </mat-card-title>
          <mat-card-subtitle>
            Service worker cache statistics and controls
          </mat-card-subtitle>
        </mat-card-header>
        <mat-card-content>
          <div class="cache-stats">
            <div class="stat-item">
              <span class="stat-label">Total Cache Size:</span>
              <span class="stat-value">{{ formatBytes(cacheStats.totalSize) }}</span>
            </div>
            <div class="stat-item">
              <span class="stat-label">Cached Items:</span>
              <span class="stat-value">{{ cacheStats.itemCount }}</span>
            </div>
            <div class="stat-item">
              <span class="stat-label">Last Updated:</span>
              <span class="stat-value">{{ cacheStats.lastUpdated | date:'short' }}</span>
            </div>
          </div>
          
          <mat-divider></mat-divider>
          
          <div class="cache-actions">
            <button mat-raised-button color="primary" (click)="checkForUpdates()">
              <mat-icon>refresh</mat-icon>
              Check for Updates
            </button>
            <button mat-raised-button color="warn" (click)="clearCache()">
              <mat-icon>clear_all</mat-icon>
              Clear Cache
            </button>
            <button mat-raised-button color="accent" (click)="refreshMetrics()">
              <mat-icon>update</mat-icon>
              Refresh Metrics
            </button>
          </div>
        </mat-card-content>
      </mat-card>

      <!-- Performance Recommendations -->
      <mat-card class="recommendations-card">
        <mat-card-header>
          <mat-card-title>
            <mat-icon>lightbulb</mat-icon>
            Performance Recommendations
          </mat-card-title>
          <mat-card-subtitle>
            Suggestions for improving application performance
          </mat-card-subtitle>
        </mat-card-header>
        <mat-card-content>
          <div class="recommendations-list">
            <div *ngFor="let recommendation of recommendations" class="recommendation-item">
              <mat-icon [color]="recommendation.priority">{{ recommendation.icon }}</mat-icon>
              <div class="recommendation-content">
                <h4>{{ recommendation.title }}</h4>
                <p>{{ recommendation.description }}</p>
              </div>
            </div>
          </div>
        </mat-card-content>
      </mat-card>
    </div>
  `,
  styles: [`
    .performance-monitor {
      display: flex;
      flex-direction: column;
      gap: 1.5rem;
      padding: 1rem;
    }

    .metrics-card,
    .cache-card,
    .recommendations-card {
      margin-bottom: 1rem;
    }

    .metrics-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
      gap: 1.5rem;
      margin-top: 1rem;
    }

    .metric-item {
      padding: 1rem;
      border: 1px solid #e0e0e0;
      border-radius: 8px;
      background: #fafafa;
    }

    .metric-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 0.5rem;
    }

    .metric-label {
      font-weight: 500;
      color: #333;
    }

    .cache-stats {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 1rem;
      margin-bottom: 1rem;
    }

    .stat-item {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 0.5rem;
      background: #f5f5f5;
      border-radius: 4px;
    }

    .stat-label {
      font-weight: 500;
      color: #666;
    }

    .stat-value {
      font-weight: 600;
      color: #1976d2;
    }

    .cache-actions {
      display: flex;
      gap: 1rem;
      margin-top: 1rem;
      flex-wrap: wrap;
    }

    .recommendations-list {
      display: flex;
      flex-direction: column;
      gap: 1rem;
    }

    .recommendation-item {
      display: flex;
      align-items: flex-start;
      gap: 1rem;
      padding: 1rem;
      border: 1px solid #e0e0e0;
      border-radius: 8px;
      background: #fafafa;
    }

    .recommendation-content h4 {
      margin: 0 0 0.5rem 0;
      color: #333;
    }

    .recommendation-content p {
      margin: 0;
      color: #666;
      line-height: 1.5;
    }

    @media (max-width: 768px) {
      .performance-monitor {
        padding: 0.5rem;
      }

      .metrics-grid {
        grid-template-columns: 1fr;
      }

      .cache-stats {
        grid-template-columns: 1fr;
      }

      .cache-actions {
        flex-direction: column;
      }

      .recommendation-item {
        flex-direction: column;
        text-align: center;
      }
    }
  `]
})
export class PerformanceMonitorComponent implements OnInit, OnDestroy {
  private performanceService = inject(PerformanceService);
  private snackBar = inject(MatSnackBar);

  metrics: PerformanceMetrics = {
    loadTime: 0,
    firstContentfulPaint: 0,
    largestContentfulPaint: 0,
    cumulativeLayoutShift: 0,
    firstInputDelay: 0
  };

  cacheStats: CacheStats = {
    totalSize: 0,
    itemCount: 0,
    lastUpdated: new Date()
  };

  recommendations = [
    {
      icon: 'warning',
      priority: 'warn',
      title: 'Optimize Images',
      description: 'Consider using WebP format and implementing lazy loading for better performance.'
    },
    {
      icon: 'info',
      priority: 'primary',
      title: 'Enable Compression',
      description: 'Ensure gzip compression is enabled on your server for faster loading.'
    },
    {
      icon: 'check_circle',
      priority: 'accent',
      title: 'Lazy Loading Active',
      description: 'Routes are lazy-loaded, which helps reduce initial bundle size.'
    }
  ];

  private updateInterval: any;

  ngOnInit(): void {
    this.refreshMetrics();
    this.updateCacheStats();
    
    // Update metrics every 5 seconds
    this.updateInterval = setInterval(() => {
      this.refreshMetrics();
      this.updateCacheStats();
    }, 5000);
  }

  ngOnDestroy(): void {
    if (this.updateInterval) {
      clearInterval(this.updateInterval);
    }
  }

  async refreshMetrics(): Promise<void> {
    try {
      this.metrics = await this.performanceService.getPerformanceMetrics();
    } catch (error) {
      console.error('Error refreshing metrics:', error);
    }
  }

  async updateCacheStats(): Promise<void> {
    try {
      this.cacheStats = await this.performanceService.getCacheStats();
    } catch (error) {
      console.error('Error updating cache stats:', error);
    }
  }

  async checkForUpdates(): Promise<void> {
    try {
      await this.performanceService.checkForUpdate();
      this.snackBar.open('Update check completed', 'Close', { duration: 3000 });
    } catch (error) {
      this.snackBar.open('Error checking for updates', 'Close', { duration: 3000 });
    }
  }

  async clearCache(): Promise<void> {
    try {
      await this.performanceService.clearCache();
      await this.updateCacheStats();
      this.snackBar.open('Cache cleared successfully', 'Close', { duration: 3000 });
    } catch (error) {
      this.snackBar.open('Error clearing cache', 'Close', { duration: 3000 });
    }
  }

  getMetricColor(value: number, threshold: number, reverse: boolean = false): string {
    const percentage = this.getMetricPercentage(value, threshold, reverse);
    if (percentage <= 33) return 'primary';
    if (percentage <= 66) return 'accent';
    return 'warn';
  }

  getMetricPercentage(value: number, threshold: number, reverse: boolean = false): number {
    if (reverse) {
      return Math.min(100, (value / threshold) * 100);
    }
    return Math.min(100, (value / threshold) * 100);
  }

  formatBytes(bytes: number): string {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }
} 