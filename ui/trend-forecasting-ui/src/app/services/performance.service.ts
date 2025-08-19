import { Injectable, inject } from '@angular/core';
import { SwUpdate, VersionReadyEvent } from '@angular/service-worker';
import { filter, map } from 'rxjs/operators';

export interface PerformanceMetrics {
  loadTime: number;
  firstContentfulPaint: number;
  largestContentfulPaint: number;
  cumulativeLayoutShift: number;
  firstInputDelay: number;
}

export interface CacheStats {
  totalSize: number;
  itemCount: number;
  lastUpdated: Date;
}

@Injectable({
  providedIn: 'root'
})
export class PerformanceService {
  private swUpdate = inject(SwUpdate);

  constructor() {
    this.initializeServiceWorker();
  }

  /**
   * Initialize service worker update handling
   */
  private initializeServiceWorker(): void {
    if (this.swUpdate.isEnabled) {
      this.swUpdate.versionUpdates
        .pipe(
          filter((evt): evt is VersionReadyEvent => evt.type === 'VERSION_READY'),
          map(() => this.promptUser())
        )
        .subscribe();
    }
  }

  /**
   * Prompt user to update the application
   */
  private promptUser(): void {
    if (confirm('New version available. Load new version?')) {
      window.location.reload();
    }
  }

  /**
   * Check for service worker updates
   */
  async checkForUpdate(): Promise<void> {
    if (this.swUpdate.isEnabled) {
      try {
        await this.swUpdate.checkForUpdate();
      } catch (error) {
        console.error('Error checking for updates:', error);
      }
    }
  }

  /**
   * Get performance metrics using Web Vitals API
   */
  getPerformanceMetrics(): Promise<PerformanceMetrics> {
    return new Promise((resolve) => {
      const metrics: PerformanceMetrics = {
        loadTime: 0,
        firstContentfulPaint: 0,
        largestContentfulPaint: 0,
        cumulativeLayoutShift: 0,
        firstInputDelay: 0
      };

      // Load time
      if (performance.timing) {
        metrics.loadTime = performance.timing.loadEventEnd - performance.timing.navigationStart;
      }

      // First Contentful Paint
      const fcpObserver = new PerformanceObserver((list) => {
        const entries = list.getEntries();
        const fcp = entries.find(entry => entry.name === 'first-contentful-paint');
        if (fcp) {
          metrics.firstContentfulPaint = fcp.startTime;
        }
      });
      fcpObserver.observe({ entryTypes: ['paint'] });

      // Largest Contentful Paint
      const lcpObserver = new PerformanceObserver((list) => {
        const entries = list.getEntries();
        const lcp = entries[entries.length - 1];
        if (lcp) {
          metrics.largestContentfulPaint = lcp.startTime;
        }
      });
      lcpObserver.observe({ entryTypes: ['largest-contentful-paint'] });

      // Cumulative Layout Shift
      const clsObserver = new PerformanceObserver((list) => {
        let cls = 0;
        for (const entry of list.getEntries()) {
          const layoutShiftEntry = entry as any;
          if (!layoutShiftEntry.hadRecentInput) {
            cls += layoutShiftEntry.value;
          }
        }
        metrics.cumulativeLayoutShift = cls;
      });
      clsObserver.observe({ entryTypes: ['layout-shift'] });

      // First Input Delay
      const fidObserver = new PerformanceObserver((list) => {
        const entries = list.getEntries();
        const fid = entries[0] as any;
        if (fid) {
          metrics.firstInputDelay = fid.processingStart - fid.startTime;
        }
      });
      fidObserver.observe({ entryTypes: ['first-input'] });

      // Resolve after a short delay to allow metrics to be collected
      setTimeout(() => {
        resolve(metrics);
      }, 1000);
    });
  }

  /**
   * Preload critical resources
   */
  preloadResources(resources: string[]): void {
    resources.forEach(resource => {
      const link = document.createElement('link');
      link.rel = 'preload';
      link.href = resource;
      link.as = this.getResourceType(resource);
      document.head.appendChild(link);
    });
  }

  /**
   * Determine resource type for preloading
   */
  private getResourceType(resource: string): string {
    if (resource.endsWith('.css')) return 'style';
    if (resource.endsWith('.js')) return 'script';
    if (resource.endsWith('.woff2')) return 'font';
    if (resource.match(/\.(jpg|jpeg|png|gif|webp|svg)$/)) return 'image';
    return 'fetch';
  }

  /**
   * Optimize images using lazy loading
   */
  setupLazyLoading(): void {
    if ('IntersectionObserver' in window) {
      const imageObserver = new IntersectionObserver((entries, observer) => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            const img = entry.target as HTMLImageElement;
            img.src = img.dataset['src'] || '';
            img.classList.remove('lazy');
            observer.unobserve(img);
          }
        });
      });

      document.querySelectorAll('img[data-src]').forEach(img => {
        imageObserver.observe(img);
      });
    }
  }

  /**
   * Get cache statistics
   */
  async getCacheStats(): Promise<CacheStats> {
    if ('caches' in window) {
      const cacheNames = await caches.keys();
      let totalSize = 0;
      let itemCount = 0;

      for (const cacheName of cacheNames) {
        const cache = await caches.open(cacheName);
        const keys = await cache.keys();
        itemCount += keys.length;

        for (const request of keys) {
          const response = await cache.match(request);
          if (response) {
            const blob = await response.blob();
            totalSize += blob.size;
          }
        }
      }

      return {
        totalSize,
        itemCount,
        lastUpdated: new Date()
      };
    }

    return {
      totalSize: 0,
      itemCount: 0,
      lastUpdated: new Date()
    };
  }

  /**
   * Clear application cache
   */
  async clearCache(): Promise<void> {
    if ('caches' in window) {
      const cacheNames = await caches.keys();
      await Promise.all(
        cacheNames.map(cacheName => caches.delete(cacheName))
      );
    }
  }

  /**
   * Debounce function for performance optimization
   */
  debounce<T extends (...args: any[]) => any>(
    func: T,
    wait: number
  ): (...args: Parameters<T>) => void {
    let timeout: NodeJS.Timeout;
    return (...args: Parameters<T>) => {
      clearTimeout(timeout);
      timeout = setTimeout(() => func(...args), wait);
    };
  }

  /**
   * Throttle function for performance optimization
   */
  throttle<T extends (...args: any[]) => any>(
    func: T,
    limit: number
  ): (...args: Parameters<T>) => void {
    let inThrottle: boolean;
    return (...args: Parameters<T>) => {
      if (!inThrottle) {
        func(...args);
        inThrottle = true;
        setTimeout(() => inThrottle = false, limit);
      }
    };
  }
} 