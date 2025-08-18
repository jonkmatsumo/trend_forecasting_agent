import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { NotificationService } from '../../services/notification.service';

@Component({
  selector: 'app-styling-demo-page',
  standalone: true,
  imports: [CommonModule, MatButtonModule, MatIconModule],
  templateUrl: './styling-demo-page.html',
  styleUrls: ['./styling-demo-page.scss']
})
export class StylingDemoPageComponent {
  constructor(private notificationService: NotificationService) {}

  showSuccessNotification() {
    this.notificationService.success(
      'This is a success notification with enhanced styling and glass morphism effects!',
      'Success',
      {
        duration: 5000,
        actions: [
          {
            label: 'View Details',
            color: 'primary',
            action: () => {
              console.log('View details clicked');
            }
          }
        ]
      }
    );
  }

  showErrorNotification() {
    this.notificationService.error(
      'This is an error notification demonstrating the error styling and longer duration.',
      'Error',
      {
        duration: 8000,
        actions: [
          {
            label: 'Retry',
            color: 'primary',
            action: () => {
              console.log('Retry clicked');
            }
          },
          {
            label: 'Dismiss',
            color: 'accent',
            action: () => {
              console.log('Dismiss clicked');
            }
          }
        ]
      }
    );
  }

  showWarningNotification() {
    this.notificationService.warning(
      'This is a warning notification with custom styling and action buttons.',
      'Warning',
      {
        duration: 6000,
        actions: [
          {
            label: 'Acknowledge',
            color: 'primary',
            action: () => {
              console.log('Acknowledged');
            }
          }
        ]
      }
    );
  }

  showInfoNotification() {
    this.notificationService.info(
      'This is an informational notification showcasing the info styling and auto-dismiss functionality.',
      'Information',
      {
        duration: 4000
      }
    );
  }

  showNotificationWithActions() {
    this.notificationService.show({
      type: 'info',
      title: 'Custom Notification',
      message: 'This notification demonstrates custom actions and advanced features.',
      duration: 10000,
      autoClose: true,
      dismissible: true,
      actions: [
        {
          label: 'Primary Action',
          color: 'primary',
          action: () => {
            this.notificationService.success('Primary action executed!');
          }
        },
        {
          label: 'Secondary Action',
          color: 'accent',
          action: () => {
            this.notificationService.info('Secondary action executed!');
          }
        },
        {
          label: 'Cancel',
          color: 'warn',
          action: () => {
            this.notificationService.warning('Action cancelled!');
          }
        }
      ]
    });
  }
} 