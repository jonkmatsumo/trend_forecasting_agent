import { Injectable } from '@angular/core';
import { BehaviorSubject, Observable } from 'rxjs';
import { NotificationComponent, NotificationAction, NotificationType } from '../components/shared/notification/notification';

export interface NotificationConfig {
  type: NotificationType;
  title?: string;
  message: string;
  duration?: number;
  autoClose?: boolean;
  dismissible?: boolean;
  actions?: NotificationAction[];
  id?: string;
}

export interface NotificationInstance extends NotificationConfig {
  id: string;
  timestamp: number;
}

@Injectable({
  providedIn: 'root'
})
export class NotificationService {
  private notifications$ = new BehaviorSubject<NotificationInstance[]>([]);
  private maxNotifications = 5;

  constructor() {}

  /**
   * Get notifications as observable
   */
  get notifications(): Observable<NotificationInstance[]> {
    return this.notifications$.asObservable();
  }

  /**
   * Get current notifications
   */
  get currentNotifications(): NotificationInstance[] {
    return this.notifications$.value;
  }

  /**
   * Show a success notification
   */
  success(message: string, title?: string, config?: Partial<NotificationConfig>): string {
    return this.show({
      type: 'success',
      title,
      message,
      duration: 5000,
      autoClose: true,
      dismissible: true,
      ...config
    });
  }

  /**
   * Show an error notification
   */
  error(message: string, title?: string, config?: Partial<NotificationConfig>): string {
    return this.show({
      type: 'error',
      title,
      message,
      duration: 8000,
      autoClose: true,
      dismissible: true,
      ...config
    });
  }

  /**
   * Show a warning notification
   */
  warning(message: string, title?: string, config?: Partial<NotificationConfig>): string {
    return this.show({
      type: 'warning',
      title,
      message,
      duration: 6000,
      autoClose: true,
      dismissible: true,
      ...config
    });
  }

  /**
   * Show an info notification
   */
  info(message: string, title?: string, config?: Partial<NotificationConfig>): string {
    return this.show({
      type: 'info',
      title,
      message,
      duration: 4000,
      autoClose: true,
      dismissible: true,
      ...config
    });
  }

  /**
   * Show a custom notification
   */
  show(config: NotificationConfig): string {
    const id = config.id || this.generateId();
    const notification: NotificationInstance = {
      ...config,
      id,
      timestamp: Date.now()
    };

    const currentNotifications = this.notifications$.value;
    const updatedNotifications = [notification, ...currentNotifications].slice(0, this.maxNotifications);

    this.notifications$.next(updatedNotifications);

    // Auto-remove if autoClose is enabled
    if (notification.autoClose && notification.duration) {
      setTimeout(() => {
        this.remove(id);
      }, notification.duration);
    }

    return id;
  }

  /**
   * Remove a specific notification
   */
  remove(id: string): void {
    const currentNotifications = this.notifications$.value;
    const updatedNotifications = currentNotifications.filter(n => n.id !== id);
    this.notifications$.next(updatedNotifications);
  }

  /**
   * Remove all notifications
   */
  clear(): void {
    this.notifications$.next([]);
  }

  /**
   * Remove notifications by type
   */
  clearByType(type: NotificationType): void {
    const currentNotifications = this.notifications$.value;
    const updatedNotifications = currentNotifications.filter(n => n.type !== type);
    this.notifications$.next(updatedNotifications);
  }

  /**
   * Update a notification
   */
  update(id: string, updates: Partial<NotificationConfig>): void {
    const currentNotifications = this.notifications$.value;
    const updatedNotifications = currentNotifications.map(n => 
      n.id === id ? { ...n, ...updates } : n
    );
    this.notifications$.next(updatedNotifications);
  }

  /**
   * Get notification by ID
   */
  getById(id: string): NotificationInstance | undefined {
    return this.notifications$.value.find(n => n.id === id);
  }

  /**
   * Check if notification exists
   */
  exists(id: string): boolean {
    return this.getById(id) !== undefined;
  }

  /**
   * Get notification count
   */
  get count(): number {
    return this.notifications$.value.length;
  }

  /**
   * Get notification count by type
   */
  getCountByType(type: NotificationType): number {
    return this.notifications$.value.filter(n => n.type === type).length;
  }

  /**
   * Generate unique ID for notifications
   */
  private generateId(): string {
    return `notification_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  /**
   * Show notification for API errors
   */
  showApiError(error: any, context?: string): string {
    let message = 'An unexpected error occurred';
    let title = 'API Error';

    if (error?.error?.message) {
      message = error.error.message;
    } else if (error?.message) {
      message = error.message;
    } else if (typeof error === 'string') {
      message = error;
    }

    if (context) {
      title = `${context} Error`;
    }

    return this.error(message, title, {
      duration: 10000,
      actions: [
        {
          label: 'Retry',
          color: 'primary',
          action: () => {
            // Retry logic can be implemented here
            console.log('Retry action clicked');
          }
        }
      ]
    });
  }

  /**
   * Show notification for successful operations
   */
  showSuccess(message: string, context?: string): string {
    const title = context ? `${context} Success` : 'Success';
    return this.success(message, title);
  }

  /**
   * Show notification for warnings
   */
  showWarning(message: string, context?: string): string {
    const title = context ? `${context} Warning` : 'Warning';
    return this.warning(message, title);
  }

  /**
   * Show notification for information
   */
  showInfo(message: string, context?: string): string {
    const title = context ? `${context} Info` : 'Information';
    return this.info(message, title);
  }

  /**
   * Show loading notification (non-dismissible)
   */
  showLoading(message: string, title?: string): string {
    return this.show({
      type: 'info',
      title: title || 'Loading',
      message,
      autoClose: false,
      dismissible: false,
      duration: undefined
    });
  }

  /**
   * Update loading notification to success
   */
  updateToSuccess(id: string, message: string): void {
    this.update(id, {
      type: 'success',
      message,
      autoClose: true,
      dismissible: true,
      duration: 3000
    });
  }

  /**
   * Update loading notification to error
   */
  updateToError(id: string, message: string): void {
    this.update(id, {
      type: 'error',
      message,
      autoClose: true,
      dismissible: true,
      duration: 8000
    });
  }
} 