import { Injectable } from '@angular/core';
import { BehaviorSubject, Observable } from 'rxjs';

export interface LoadingState {
  id: string;
  isLoading: boolean;
  message?: string;
  progress?: number;
  startTime: number;
}

export interface LoadingConfig {
  message?: string;
  showProgress?: boolean;
  timeout?: number;
  allowMultiple?: boolean;
}

@Injectable({
  providedIn: 'root'
})
export class LoadingService {
  private loadingStates$ = new BehaviorSubject<Map<string, LoadingState>>(new Map());
  private globalLoading$ = new BehaviorSubject<boolean>(false);
  private defaultTimeout = 30000; // 30 seconds

  constructor() {}

  /**
   * Get loading states as observable
   */
  get loadingStates(): Observable<Map<string, LoadingState>> {
    return this.loadingStates$.asObservable();
  }

  /**
   * Get global loading state
   */
  get globalLoading(): Observable<boolean> {
    return this.globalLoading$.asObservable();
  }

  /**
   * Get current loading states
   */
  get currentLoadingStates(): Map<string, LoadingState> {
    return this.loadingStates$.value;
  }

  /**
   * Check if any loading is active
   */
  get isAnyLoading(): boolean {
    return this.globalLoading$.value;
  }

  /**
   * Start loading for a specific operation
   */
  startLoading(id: string, config: LoadingConfig = {}): void {
    const {
      message = 'Loading...',
      showProgress = false,
      timeout = this.defaultTimeout,
      allowMultiple = false
    } = config;

    const currentStates = this.loadingStates$.value;
    
    // Check if operation is already loading
    if (currentStates.has(id) && !allowMultiple) {
      console.warn(`Loading operation '${id}' is already in progress`);
      return;
    }

    const loadingState: LoadingState = {
      id,
      isLoading: true,
      message,
      progress: showProgress ? 0 : undefined,
      startTime: Date.now()
    };

    currentStates.set(id, loadingState);
    this.loadingStates$.next(new Map(currentStates));
    this.updateGlobalLoading();

    // Set timeout
    if (timeout > 0) {
      setTimeout(() => {
        if (currentStates.get(id)?.isLoading) {
          this.stopLoading(id, 'Operation timed out');
        }
      }, timeout);
    }
  }

  /**
   * Stop loading for a specific operation
   */
  stopLoading(id: string, message?: string): void {
    const currentStates = this.loadingStates$.value;
    const loadingState = currentStates.get(id);

    if (!loadingState) {
      console.warn(`No loading operation found for id: ${id}`);
      return;
    }

    if (message) {
      loadingState.message = message;
    }

    loadingState.isLoading = false;
    loadingState.progress = 100;

    // Remove from map after a short delay to show completion
    setTimeout(() => {
      const updatedStates = this.loadingStates$.value;
      updatedStates.delete(id);
      this.loadingStates$.next(new Map(updatedStates));
      this.updateGlobalLoading();
    }, 500);
  }

  /**
   * Update progress for a loading operation
   */
  updateProgress(id: string, progress: number, message?: string): void {
    const currentStates = this.loadingStates$.value;
    const loadingState = currentStates.get(id);

    if (!loadingState) {
      console.warn(`No loading operation found for id: ${id}`);
      return;
    }

    loadingState.progress = Math.min(Math.max(progress, 0), 100);
    if (message) {
      loadingState.message = message;
    }

    this.loadingStates$.next(new Map(currentStates));
  }

  /**
   * Stop all loading operations
   */
  stopAllLoading(message?: string): void {
    const currentStates = this.loadingStates$.value;
    currentStates.forEach((state, id) => {
      this.stopLoading(id, message);
    });
  }

  /**
   * Check if a specific operation is loading
   */
  isLoading(id: string): boolean {
    return this.loadingStates$.value.get(id)?.isLoading || false;
  }

  /**
   * Get loading state for a specific operation
   */
  getLoadingState(id: string): LoadingState | undefined {
    return this.loadingStates$.value.get(id);
  }

  /**
   * Create a loading wrapper for async operations
   */
  async withLoading<T>(
    id: string,
    operation: () => Promise<T>,
    config: LoadingConfig = {}
  ): Promise<T> {
    this.startLoading(id, config);

    try {
      const result = await operation();
      this.stopLoading(id, 'Completed successfully');
      return result;
    } catch (error) {
      this.stopLoading(id, 'Operation failed');
      throw error;
    }
  }

  /**
   * Create a loading wrapper for observables
   */
  withLoadingObservable<T>(
    id: string,
    operation: Observable<T>,
    config: LoadingConfig = {}
  ): Observable<T> {
    this.startLoading(id, config);

    return new Observable(observer => {
      operation.subscribe({
        next: (value) => {
          observer.next(value);
        },
        error: (error) => {
          this.stopLoading(id, 'Operation failed');
          observer.error(error);
        },
        complete: () => {
          this.stopLoading(id, 'Completed successfully');
          observer.complete();
        }
      });
    });
  }

  /**
   * Show skeleton loading for a component
   */
  showSkeleton(id: string, message = 'Loading content...'): void {
    this.startLoading(id, {
      message,
      showProgress: false,
      timeout: 0 // No timeout for skeleton loading
    });
  }

  /**
   * Hide skeleton loading
   */
  hideSkeleton(id: string): void {
    this.stopLoading(id);
  }

  /**
   * Update global loading state
   */
  private updateGlobalLoading(): void {
    const currentStates = this.loadingStates$.value;
    const hasAnyLoading = Array.from(currentStates.values()).some(state => state.isLoading);
    this.globalLoading$.next(hasAnyLoading);
  }

  /**
   * Get loading statistics
   */
  getLoadingStats(): {
    activeCount: number;
    totalOperations: number;
    averageDuration: number;
  } {
    const currentStates = this.loadingStates$.value;
    const activeStates = Array.from(currentStates.values()).filter(state => state.isLoading);
    const completedStates = Array.from(currentStates.values()).filter(state => !state.isLoading);
    
    const activeCount = activeStates.length;
    const totalOperations = activeCount + completedStates.length;
    
    let averageDuration = 0;
    if (completedStates.length > 0) {
      const totalDuration = completedStates.reduce((sum, state) => {
        return sum + (Date.now() - state.startTime);
      }, 0);
      averageDuration = totalDuration / completedStates.length;
    }

    return {
      activeCount,
      totalOperations,
      averageDuration
    };
  }
} 