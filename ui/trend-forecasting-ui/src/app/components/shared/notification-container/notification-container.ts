import { Component, OnInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Subscription } from 'rxjs';
import { NotificationComponent } from '../notification/notification';
import { NotificationService, NotificationInstance } from '../../../services/notification.service';
import { NotificationAction } from '../notification/notification';

@Component({
  selector: 'app-notification-container',
  standalone: true,
  imports: [CommonModule, NotificationComponent],
  templateUrl: './notification-container.html',
  styleUrls: ['./notification-container.scss']
})
export class NotificationContainerComponent implements OnInit, OnDestroy {
  notifications: NotificationInstance[] = [];
  private subscription?: Subscription;

  constructor(private notificationService: NotificationService) {}

  ngOnInit() {
    this.subscription = this.notificationService.notifications.subscribe(
      notifications => {
        this.notifications = notifications;
      }
    );
  }

  ngOnDestroy() {
    if (this.subscription) {
      this.subscription.unsubscribe();
    }
  }

  onNotificationDismissed(id: string) {
    this.notificationService.remove(id);
  }

  onActionClicked(action: NotificationAction, notificationId: string) {
    // Handle action click
    if (action.action) {
      action.action();
    }
    
    // Remove notification after action if specified
    if (action.action) {
      this.notificationService.remove(notificationId);
    }
  }
} 