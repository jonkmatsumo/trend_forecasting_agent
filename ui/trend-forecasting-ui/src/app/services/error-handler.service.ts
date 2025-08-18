import { Injectable } from '@angular/core';
import { HttpErrorResponse } from '@angular/common/http';
import { Observable, throwError, timer } from 'rxjs';
import { retryWhen, delayWhen, tap, catchError } from 'rxjs/operators';
import { NotificationService } from './notification.service';

export interface ErrorConfig {
  showNotification?: boolean;
  retryAttempts?: number;
  retryDelay?: number;
  customMessage?: string;
  logError?: boolean;
}

export interface ValidationError {
  field: string;
  message: string;
  code?: string;
}

export interface ApiError {
  status: number;
  statusText: string;
  message: string;
  details?: any;
  timestamp: Date;
}

@Injectable({
  providedIn: 'root'
})
export class ErrorHandlerService {
  private readonly defaultRetryAttempts = 3;
  private readonly defaultRetryDelay = 1000;

  constructor(private notificationService: NotificationService) {}

  /**
   * Handle HTTP errors with retry mechanism
   */
  handleHttpError<T>(error: HttpErrorResponse, config: ErrorConfig = {}): Observable<T> {
    const {
      showNotification = true,
      retryAttempts = this.defaultRetryAttempts,
      retryDelay = this.defaultRetryDelay,
      customMessage,
      logError = true
    } = config;

    if (logError) {
      console.error('HTTP Error:', error);
    }

    const errorMessage = this.getErrorMessage(error, customMessage);

    if (showNotification) {
      this.notificationService.error(errorMessage, 'Request Failed');
    }

    return throwError(() => this.createApiError(error, errorMessage));
  }

  /**
   * Create a retryable HTTP request with error handling
   */
  createRetryableRequest<T>(
    request: Observable<T>,
    config: ErrorConfig = {}
  ): Observable<T> {
    const {
      retryAttempts = this.defaultRetryAttempts,
      retryDelay = this.defaultRetryDelay,
      showNotification = true
    } = config;

    return request.pipe(
      retryWhen(errors =>
        errors.pipe(
          tap((error, index) => {
            if (index >= retryAttempts - 1) {
              if (showNotification) {
                this.notificationService.error(
                  `Request failed after ${retryAttempts} attempts. Please try again later.`,
                  'Connection Error'
                );
              }
            } else {
              console.warn(`Retry attempt ${index + 1}/${retryAttempts}`);
            }
          }),
          delayWhen((_, index) => timer(retryDelay * (index + 1)))
        )
      ),
      catchError(error => this.handleHttpError(error, config))
    );
  }

  /**
   * Handle validation errors
   */
  handleValidationError(errors: ValidationError[], title = 'Validation Error'): void {
    const errorMessages = errors.map(error => `${error.field}: ${error.message}`).join('\n');
    this.notificationService.error(errorMessages, title);
  }

  /**
   * Handle generic errors
   */
  handleGenericError(error: any, context = 'Application Error'): void {
    console.error(`${context}:`, error);
    
    const message = error?.message || error?.toString() || 'An unexpected error occurred';
    this.notificationService.error(message, context);
  }

  /**
   * Get user-friendly error message from HTTP error
   */
  private getErrorMessage(error: HttpErrorResponse, customMessage?: string): string {
    if (customMessage) {
      return customMessage;
    }

    switch (error.status) {
      case 0:
        return 'Network error. Please check your internet connection.';
      case 400:
        return 'Invalid request. Please check your input and try again.';
      case 401:
        return 'Authentication required. Please log in again.';
      case 403:
        return 'Access denied. You don\'t have permission to perform this action.';
      case 404:
        return 'Resource not found. The requested data is not available.';
      case 408:
        return 'Request timeout. The server took too long to respond.';
      case 429:
        return 'Too many requests. Please wait a moment before trying again.';
      case 500:
        return 'Server error. Please try again later.';
      case 502:
        return 'Bad gateway. The server is temporarily unavailable.';
      case 503:
        return 'Service unavailable. The server is temporarily down for maintenance.';
      case 504:
        return 'Gateway timeout. The server took too long to respond.';
      default:
        return error.error?.message || error.message || 'An unexpected error occurred';
    }
  }

  /**
   * Create API error object
   */
  private createApiError(error: HttpErrorResponse, message: string): ApiError {
    return {
      status: error.status,
      statusText: error.statusText,
      message,
      details: error.error,
      timestamp: new Date()
    };
  }

  /**
   * Check if error is retryable
   */
  isRetryableError(error: HttpErrorResponse): boolean {
    // Retry on network errors, 5xx server errors, and 408 timeout
    return error.status === 0 || 
           (error.status >= 500 && error.status < 600) || 
           error.status === 408;
  }

  /**
   * Get retry delay based on error type
   */
  getRetryDelay(error: HttpErrorResponse, attempt: number): number {
    if (error.status === 429) {
      // Exponential backoff for rate limiting
      return Math.min(1000 * Math.pow(2, attempt), 30000);
    }
    return this.defaultRetryDelay * (attempt + 1);
  }
} 