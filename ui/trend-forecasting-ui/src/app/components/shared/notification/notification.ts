import { Component, Input, Output, EventEmitter, OnInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { MatIconModule } from '@angular/material/icon';
import { MatButtonModule } from '@angular/material/button';
import { trigger, state, style, transition, animate, keyframes } from '@angular/animations';

export interface NotificationAction {
  label: string;
  color?: 'primary' | 'accent' | 'warn';
  action?: () => void;
}

export type NotificationType = 'success' | 'error' | 'warning' | 'info';

@Component({
  selector: 'app-notification',
  standalone: true,
  imports: [CommonModule, MatIconModule, MatButtonModule],
  templateUrl: './notification.html',
  styleUrls: ['./notification.scss'],
  animations: [
    trigger('notificationAnimation', [
      state('void', style({
        transform: 'translateX(100%)',
        opacity: 0
      })),
      state('visible', style({
        transform: 'translateX(0)',
        opacity: 1
      })),
      state('hidden', style({
        transform: 'translateX(100%)',
        opacity: 0
      })),
      transition('void => visible', [
        animate('300ms cubic-bezier(0.68, -0.55, 0.265, 1.55)', keyframes([
          style({ transform: 'translateX(100%)', opacity: 0, offset: 0 }),
          style({ transform: 'translateX(-10px)', opacity: 0.8, offset: 0.8 }),
          style({ transform: 'translateX(0)', opacity: 1, offset: 1 })
        ]))
      ]),
      transition('visible => hidden', [
        animate('200ms ease-out', keyframes([
          style({ transform: 'translateX(0)', opacity: 1, offset: 0 }),
          style({ transform: 'translateX(100%)', opacity: 0, offset: 1 })
        ]))
      ])
    ]),
    trigger('progressAnimation', [
      state('visible', style({
        width: '0%'
      })),
      transition('visible => *', [
        animate('{{duration}}ms linear', style({
          width: '100%'
        }))
      ], { params: { duration: 5000 } })
    ])
  ]
})
export class NotificationComponent implements OnInit, OnDestroy {
  @Input() type: NotificationType = 'info';
  @Input() title?: string;
  @Input() message: string = '';
  @Input() duration?: number;
  @Input() autoClose: boolean = true;
  @Input() dismissible: boolean = true;
  @Input() actions?: NotificationAction[];
  
  @Output() dismissed = new EventEmitter<void>();
  @Output() actionClicked = new EventEmitter<NotificationAction>();

  animationState: 'void' | 'visible' | 'hidden' = 'void';
  private autoCloseTimeout?: any;

  ngOnInit() {
    // Start animation
    setTimeout(() => {
      this.animationState = 'visible';
    }, 100);

    // Auto-close functionality
    if (this.autoClose && this.duration) {
      this.autoCloseTimeout = setTimeout(() => {
        this.dismiss();
      }, this.duration);
    }
  }

  ngOnDestroy() {
    if (this.autoCloseTimeout) {
      clearTimeout(this.autoCloseTimeout);
    }
  }

  dismiss() {
    this.animationState = 'hidden';
    if (this.autoCloseTimeout) {
      clearTimeout(this.autoCloseTimeout);
    }
  }

  handleAction(action: NotificationAction) {
    this.actionClicked.emit(action);
    if (action.action) {
      action.action();
    }
  }

  onAnimationComplete() {
    if (this.animationState === 'hidden') {
      this.dismissed.emit();
    }
  }

  getIcon(): string {
    const icons = {
      success: 'check_circle',
      error: 'error',
      warning: 'warning',
      info: 'info'
    };
    return icons[this.type] || 'info';
  }

  getIconColor(): string {
    const colors = {
      success: 'primary',
      error: 'warn',
      warning: 'accent',
      info: 'primary'
    };
    return colors[this.type] || 'primary';
  }
} 