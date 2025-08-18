import { ComponentFixture, TestBed } from '@angular/core/testing';
import { NoopAnimationsModule } from '@angular/platform-browser/animations';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { StylingDemoPageComponent } from './styling-demo-page';
import { NotificationService, NotificationConfig } from '../../services/notification.service';
import { NotificationAction } from '../../components/shared/notification/notification';

describe('StylingDemoPageComponent', () => {
  let component: StylingDemoPageComponent;
  let fixture: ComponentFixture<StylingDemoPageComponent>;
  let notificationService: jasmine.SpyObj<NotificationService>;

  beforeEach(async () => {
    const spy = jasmine.createSpyObj('NotificationService', [
      'success',
      'error',
      'warning',
      'info',
      'show'
    ]);

    await TestBed.configureTestingModule({
      imports: [
        NoopAnimationsModule,
        MatButtonModule,
        MatIconModule,
        StylingDemoPageComponent
      ],
      providers: [
        { provide: NotificationService, useValue: spy }
      ]
    }).compileComponents();

    fixture = TestBed.createComponent(StylingDemoPageComponent);
    component = fixture.componentInstance;
    notificationService = TestBed.inject(NotificationService) as jasmine.SpyObj<NotificationService>;
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  describe('Success Notification Demo', () => {
    it('should call notification service success method', () => {
      component.showSuccessNotification();
      
      expect(notificationService.success).toHaveBeenCalledWith(
        'This is a success notification with enhanced styling and glass morphism effects!',
        'Success',
        {
          duration: 5000,
          actions: [
            {
              label: 'View Details',
              color: 'primary',
              action: jasmine.any(Function)
            }
          ]
        }
      );
    });

    it('should include action with correct configuration', () => {
      component.showSuccessNotification();
      
      const callArgs = notificationService.success.calls.mostRecent().args;
      const config = callArgs[2] as NotificationConfig;
      
      expect(config).toBeDefined();
      expect(config.actions).toBeDefined();
      expect(config.actions!.length).toBe(1);
      expect(config.actions![0].label).toBe('View Details');
      expect(config.actions![0].color).toBe('primary');
      expect(config.actions![0].action).toBeDefined();
    });

    it('should execute action function when called', () => {
      component.showSuccessNotification();
      
      const callArgs = notificationService.success.calls.mostRecent().args;
      const config = callArgs[2] as NotificationConfig;
      const action = config.actions![0].action;
      
      expect(action).toBeDefined();
      spyOn(console, 'log');
      action!();
      
      expect(console.log).toHaveBeenCalledWith('View details clicked');
    });
  });

  describe('Error Notification Demo', () => {
    it('should call notification service error method', () => {
      component.showErrorNotification();
      
      expect(notificationService.error).toHaveBeenCalledWith(
        'This is an error notification demonstrating the error styling and longer duration.',
        'Error',
        {
          duration: 8000,
          actions: [
            {
              label: 'Retry',
              color: 'primary',
              action: jasmine.any(Function)
            },
            {
              label: 'Dismiss',
              color: 'accent',
              action: jasmine.any(Function)
            }
          ]
        }
      );
    });

    it('should include multiple actions', () => {
      component.showErrorNotification();
      
      const callArgs = notificationService.error.calls.mostRecent().args;
      const config = callArgs[2] as NotificationConfig;
      
      expect(config).toBeDefined();
      expect(config.actions).toBeDefined();
      expect(config.actions!.length).toBe(2);
      expect(config.actions![0].label).toBe('Retry');
      expect(config.actions![1].label).toBe('Dismiss');
    });

    it('should execute retry action function', () => {
      component.showErrorNotification();
      
      const callArgs = notificationService.error.calls.mostRecent().args;
      const config = callArgs[2] as NotificationConfig;
      const retryAction = config.actions![0].action;
      
      expect(retryAction).toBeDefined();
      spyOn(console, 'log');
      retryAction!();
      
      expect(console.log).toHaveBeenCalledWith('Retry clicked');
    });

    it('should execute dismiss action function', () => {
      component.showErrorNotification();
      
      const callArgs = notificationService.error.calls.mostRecent().args;
      const config = callArgs[2] as NotificationConfig;
      const dismissAction = config.actions![1].action;
      
      expect(dismissAction).toBeDefined();
      spyOn(console, 'log');
      dismissAction!();
      
      expect(console.log).toHaveBeenCalledWith('Dismiss clicked');
    });
  });

  describe('Warning Notification Demo', () => {
    it('should call notification service warning method', () => {
      component.showWarningNotification();
      
      expect(notificationService.warning).toHaveBeenCalledWith(
        'This is a warning notification with custom styling and action buttons.',
        'Warning',
        {
          duration: 6000,
          actions: [
            {
              label: 'Acknowledge',
              color: 'primary',
              action: jasmine.any(Function)
            }
          ]
        }
      );
    });

    it('should include action with correct configuration', () => {
      component.showWarningNotification();
      
      const callArgs = notificationService.warning.calls.mostRecent().args;
      const config = callArgs[2] as NotificationConfig;
      
      expect(config).toBeDefined();
      expect(config.actions).toBeDefined();
      expect(config.actions!.length).toBe(1);
      expect(config.actions![0].label).toBe('Acknowledge');
      expect(config.actions![0].color).toBe('primary');
      expect(config.actions![0].action).toBeDefined();
    });

    it('should execute acknowledge action function', () => {
      component.showWarningNotification();
      
      const callArgs = notificationService.warning.calls.mostRecent().args;
      const config = callArgs[2] as NotificationConfig;
      const acknowledgeAction = config.actions![0].action;
      
      expect(acknowledgeAction).toBeDefined();
      spyOn(console, 'log');
      acknowledgeAction!();
      
      expect(console.log).toHaveBeenCalledWith('Acknowledged');
    });
  });

  describe('Info Notification Demo', () => {
    it('should call notification service info method', () => {
      component.showInfoNotification();
      
      expect(notificationService.info).toHaveBeenCalledWith(
        'This is an informational notification with default styling.',
        'Information'
      );
    });

    it('should not include actions for info notification', () => {
      component.showInfoNotification();
      
      const callArgs = notificationService.info.calls.mostRecent().args;
      const config = callArgs[2] as NotificationConfig | undefined;
      
      expect(config?.actions).toBeUndefined();
    });
  });

  describe('Custom Notification Demo', () => {
    it('should call notification service show method with custom config', () => {
      component.showCustomNotification();
      
      expect(notificationService.show).toHaveBeenCalledWith({
        type: 'success',
        title: 'Custom Notification',
        message: 'This is a custom notification with multiple actions and extended duration.',
        duration: 10000,
        autoClose: false,
        dismissible: true,
        actions: [
          {
            label: 'Primary Action',
            color: 'primary',
            action: jasmine.any(Function)
          },
          {
            label: 'Secondary Action',
            color: 'accent',
            action: jasmine.any(Function)
          },
          {
            label: 'Cancel',
            color: 'warn',
            action: jasmine.any(Function)
          }
        ]
      });
    });

    it('should include multiple actions with correct configuration', () => {
      component.showCustomNotification();
      
      const callArgs = notificationService.show.calls.mostRecent().args;
      const config = callArgs[0] as NotificationConfig;
      
      expect(config).toBeDefined();
      expect(config.actions).toBeDefined();
      expect(config.actions!.length).toBe(3);
      expect(config.actions![0].label).toBe('Primary Action');
      expect(config.actions![1].label).toBe('Secondary Action');
      expect(config.actions![2].label).toBe('Cancel');
    });

    it('should execute primary action function', () => {
      component.showCustomNotification();
      
      const callArgs = notificationService.show.calls.mostRecent().args;
      const config = callArgs[0] as NotificationConfig;
      const primaryAction = config.actions![0].action;
      
      expect(primaryAction).toBeDefined();
      spyOn(console, 'log');
      primaryAction!();
      
      expect(console.log).toHaveBeenCalledWith('Primary action executed');
    });

    it('should execute secondary action function', () => {
      component.showCustomNotification();
      
      const callArgs = notificationService.show.calls.mostRecent().args;
      const config = callArgs[0] as NotificationConfig;
      const secondaryAction = config.actions![1].action;
      
      expect(secondaryAction).toBeDefined();
      spyOn(console, 'log');
      secondaryAction!();
      
      expect(console.log).toHaveBeenCalledWith('Secondary action executed');
    });

    it('should execute cancel action function', () => {
      component.showCustomNotification();
      
      const callArgs = notificationService.show.calls.mostRecent().args;
      const config = callArgs[0] as NotificationConfig;
      const cancelAction = config.actions![2].action;
      
      expect(cancelAction).toBeDefined();
      spyOn(console, 'log');
      cancelAction!();
      
      expect(console.log).toHaveBeenCalledWith('Cancel action executed');
    });
  });

  describe('Error Handling Demo', () => {
    it('should call notification service error method for error handling demo', () => {
      component.showErrorHandlingDemo();
      
      expect(notificationService.error).toHaveBeenCalledWith(
        'This demonstrates error handling with custom actions and extended duration.',
        'Error Handling Demo',
        {
          duration: 12000,
          autoClose: false,
          dismissible: true,
          actions: [
            {
              label: 'Retry Operation',
              color: 'primary',
              action: jasmine.any(Function)
            }
          ]
        }
      );
    });

    it('should execute retry action function for error handling demo', () => {
      component.showErrorHandlingDemo();
      
      const callArgs = notificationService.error.calls.mostRecent().args;
      const config = callArgs[2] as NotificationConfig;
      const action = config.actions![0].action;
      
      expect(action).toBeDefined();
      spyOn(console, 'log');
      expect(() => action!()).toThrow();
      
      expect(console.log).toHaveBeenCalledWith('Retry operation clicked');
    });
  });

  describe('Accessibility Demo', () => {
    it('should call notification service info method for accessibility demo', () => {
      component.showAccessibilityDemo();
      
      expect(notificationService.info).toHaveBeenCalledWith(
        'This notification demonstrates accessibility features with proper ARIA labels and keyboard navigation support.',
        'Accessibility Demo',
        {
          duration: 8000,
          autoClose: true,
          dismissible: true
        }
      );
    });
  });

  describe('Performance Demo', () => {
    it('should call notification service success method for performance demo', () => {
      component.showPerformanceDemo();
      
      expect(notificationService.success).toHaveBeenCalledWith(
        'This notification demonstrates performance optimizations with efficient animations and minimal DOM updates.',
        'Performance Demo',
        {
          duration: 6000,
          autoClose: true,
          dismissible: true
        }
      );
    });
  });

  describe('Responsive Demo', () => {
    it('should call notification service warning method for responsive demo', () => {
      component.showResponsiveDemo();
      
      expect(notificationService.warning).toHaveBeenCalledWith(
        'This notification demonstrates responsive design with adaptive layouts for different screen sizes.',
        'Responsive Demo',
        {
          duration: 7000,
          autoClose: true,
          dismissible: true
        }
      );
    });
  });

  describe('Theme Demo', () => {
    it('should call notification service info method for theme demo', () => {
      component.showThemeDemo();
      
      expect(notificationService.info).toHaveBeenCalledWith(
        'This notification demonstrates theme support with dark mode, high contrast, and reduced motion preferences.',
        'Theme Demo',
        {
          duration: 9000,
          autoClose: true,
          dismissible: true
        }
      );
    });
  });
}); 