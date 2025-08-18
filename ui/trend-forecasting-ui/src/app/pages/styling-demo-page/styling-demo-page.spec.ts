import { ComponentFixture, TestBed } from '@angular/core/testing';
import { NoopAnimationsModule } from '@angular/platform-browser/animations';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { StylingDemoPageComponent } from './styling-demo-page';
import { NotificationService } from '../../services/notification.service';

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
      const config = callArgs[2];
      
      expect(config.actions).toBeDefined();
      expect(config.actions.length).toBe(1);
      expect(config.actions[0].label).toBe('View Details');
      expect(config.actions[0].color).toBe('primary');
      expect(config.actions[0].action).toBeDefined();
    });

    it('should execute action function when called', () => {
      component.showSuccessNotification();
      
      const callArgs = notificationService.success.calls.mostRecent().args;
      const action = callArgs[2].actions[0].action;
      
      spyOn(console, 'log');
      action();
      
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
      const config = callArgs[2];
      
      expect(config.actions.length).toBe(2);
      expect(config.actions[0].label).toBe('Retry');
      expect(config.actions[1].label).toBe('Dismiss');
    });

    it('should execute retry action function', () => {
      component.showErrorNotification();
      
      const callArgs = notificationService.error.calls.mostRecent().args;
      const retryAction = callArgs[2].actions[0].action;
      
      spyOn(console, 'log');
      retryAction();
      
      expect(console.log).toHaveBeenCalledWith('Retry clicked');
    });

    it('should execute dismiss action function', () => {
      component.showErrorNotification();
      
      const callArgs = notificationService.error.calls.mostRecent().args;
      const dismissAction = callArgs[2].actions[1].action;
      
      spyOn(console, 'log');
      dismissAction();
      
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

    it('should include acknowledge action', () => {
      component.showWarningNotification();
      
      const callArgs = notificationService.warning.calls.mostRecent().args;
      const config = callArgs[2];
      
      expect(config.actions.length).toBe(1);
      expect(config.actions[0].label).toBe('Acknowledge');
      expect(config.actions[0].color).toBe('primary');
    });

    it('should execute acknowledge action function', () => {
      component.showWarningNotification();
      
      const callArgs = notificationService.warning.calls.mostRecent().args;
      const acknowledgeAction = callArgs[2].actions[0].action;
      
      spyOn(console, 'log');
      acknowledgeAction();
      
      expect(console.log).toHaveBeenCalledWith('Acknowledged');
    });
  });

  describe('Info Notification Demo', () => {
    it('should call notification service info method', () => {
      component.showInfoNotification();
      
      expect(notificationService.info).toHaveBeenCalledWith(
        'This is an informational notification showcasing the info styling and auto-dismiss functionality.',
        'Information',
        {
          duration: 4000
        }
      );
    });

    it('should not include actions for info notification', () => {
      component.showInfoNotification();
      
      const callArgs = notificationService.info.calls.mostRecent().args;
      const config = callArgs[2];
      
      expect(config.actions).toBeUndefined();
    });
  });

  describe('Custom Notification Demo', () => {
    it('should call notification service show method with custom config', () => {
      component.showNotificationWithActions();
      
      expect(notificationService.show).toHaveBeenCalledWith({
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

    it('should include three custom actions', () => {
      component.showNotificationWithActions();
      
      const callArgs = notificationService.show.calls.mostRecent().args;
      const config = callArgs[0];
      
      expect(config.actions.length).toBe(3);
      expect(config.actions[0].label).toBe('Primary Action');
      expect(config.actions[1].label).toBe('Secondary Action');
      expect(config.actions[2].label).toBe('Cancel');
    });

    it('should execute primary action and show success notification', () => {
      component.showNotificationWithActions();
      
      const callArgs = notificationService.show.calls.mostRecent().args;
      const primaryAction = callArgs[0].actions[0].action;
      
      primaryAction();
      
      expect(notificationService.success).toHaveBeenCalledWith('Primary action executed!');
    });

    it('should execute secondary action and show info notification', () => {
      component.showNotificationWithActions();
      
      const callArgs = notificationService.show.calls.mostRecent().args;
      const secondaryAction = callArgs[0].actions[1].action;
      
      secondaryAction();
      
      expect(notificationService.info).toHaveBeenCalledWith('Secondary action executed!');
    });

    it('should execute cancel action and show warning notification', () => {
      component.showNotificationWithActions();
      
      const callArgs = notificationService.show.calls.mostRecent().args;
      const cancelAction = callArgs[0].actions[2].action;
      
      cancelAction();
      
      expect(notificationService.warning).toHaveBeenCalledWith('Action cancelled!');
    });
  });

  describe('Template Rendering', () => {
    it('should render demo buttons', () => {
      fixture.detectChanges();
      
      const buttons = fixture.nativeElement.querySelectorAll('button');
      expect(buttons.length).toBeGreaterThan(0);
    });

    it('should render success button with correct text and icon', () => {
      fixture.detectChanges();
      
      const successButton = fixture.nativeElement.querySelector('button[color="primary"]');
      expect(successButton).toBeTruthy();
      expect(successButton.textContent).toContain('Success');
      expect(successButton.querySelector('mat-icon')).toBeTruthy();
    });

    it('should render error button with correct text and icon', () => {
      fixture.detectChanges();
      
      const errorButton = fixture.nativeElement.querySelector('button[color="warn"]');
      expect(errorButton).toBeTruthy();
      expect(errorButton.textContent).toContain('Error');
      expect(errorButton.querySelector('mat-icon')).toBeTruthy();
    });

    it('should render warning button with correct text and icon', () => {
      fixture.detectChanges();
      
      const warningButton = fixture.nativeElement.querySelector('button[color="accent"]');
      expect(warningButton).toBeTruthy();
      expect(warningButton.textContent).toContain('Warning');
      expect(warningButton.querySelector('mat-icon')).toBeTruthy();
    });
  });

  describe('Button Interactions', () => {
    it('should call showSuccessNotification when success button is clicked', () => {
      fixture.detectChanges();
      
      spyOn(component, 'showSuccessNotification');
      
      const successButton = fixture.nativeElement.querySelector('button[color="primary"]');
      successButton.click();
      
      expect(component.showSuccessNotification).toHaveBeenCalled();
    });

    it('should call showErrorNotification when error button is clicked', () => {
      fixture.detectChanges();
      
      spyOn(component, 'showErrorNotification');
      
      const errorButton = fixture.nativeElement.querySelector('button[color="warn"]');
      errorButton.click();
      
      expect(component.showErrorNotification).toHaveBeenCalled();
    });

    it('should call showWarningNotification when warning button is clicked', () => {
      fixture.detectChanges();
      
      spyOn(component, 'showWarningNotification');
      
      const warningButton = fixture.nativeElement.querySelector('button[color="accent"]');
      warningButton.click();
      
      expect(component.showWarningNotification).toHaveBeenCalled();
    });

    it('should call showInfoNotification when info button is clicked', () => {
      fixture.detectChanges();
      
      spyOn(component, 'showInfoNotification');
      
      const infoButtons = fixture.nativeElement.querySelectorAll('button[color="primary"]');
      const infoButton = infoButtons[1]; // Second primary button
      infoButton.click();
      
      expect(component.showInfoNotification).toHaveBeenCalled();
    });

    it('should call showNotificationWithActions when actions button is clicked', () => {
      fixture.detectChanges();
      
      spyOn(component, 'showNotificationWithActions');
      
      const actionButtons = fixture.nativeElement.querySelectorAll('button[color="primary"]');
      const actionsButton = actionButtons[2]; // Third primary button
      actionsButton.click();
      
      expect(component.showNotificationWithActions).toHaveBeenCalled();
    });
  });

  describe('Demo Content', () => {
    it('should render demo sections', () => {
      fixture.detectChanges();
      
      const sections = fixture.nativeElement.querySelectorAll('.demo-section');
      expect(sections.length).toBeGreaterThan(0);
    });

    it('should render notification system demo section', () => {
      fixture.detectChanges();
      
      const section = fixture.nativeElement.querySelector('.demo-section');
      expect(section.textContent).toContain('Notification System');
      expect(section.textContent).toContain('Advanced notification system with glass morphism effects');
    });

    it('should render enhanced cards demo section', () => {
      fixture.detectChanges();
      
      const sections = fixture.nativeElement.querySelectorAll('.demo-section');
      const cardsSection = sections[1]; // Second section
      expect(cardsSection.textContent).toContain('Enhanced Card Components');
    });

    it('should render enhanced buttons demo section', () => {
      fixture.detectChanges();
      
      const sections = fixture.nativeElement.querySelectorAll('.demo-section');
      const buttonsSection = sections[2]; // Third section
      expect(buttonsSection.textContent).toContain('Enhanced Button Styles');
    });

    it('should render typography demo section', () => {
      fixture.detectChanges();
      
      const sections = fixture.nativeElement.querySelectorAll('.demo-section');
      const typographySection = sections[3]; // Fourth section
      expect(typographySection.textContent).toContain('Enhanced Typography');
    });

    it('should render status indicators demo section', () => {
      fixture.detectChanges();
      
      const sections = fixture.nativeElement.querySelectorAll('.demo-section');
      const statusSection = sections[4]; // Fifth section
      expect(statusSection.textContent).toContain('Status Indicators');
    });

    it('should render responsive design demo section', () => {
      fixture.detectChanges();
      
      const sections = fixture.nativeElement.querySelectorAll('.demo-section');
      const responsiveSection = sections[5]; // Sixth section
      expect(responsiveSection.textContent).toContain('Responsive Design');
    });

    it('should render animation effects demo section', () => {
      fixture.detectChanges();
      
      const sections = fixture.nativeElement.querySelectorAll('.demo-section');
      const animationSection = sections[6]; // Seventh section
      expect(animationSection.textContent).toContain('Animation Effects');
    });
  });

  describe('Accessibility', () => {
    it('should have proper button structure', () => {
      fixture.detectChanges();
      
      const buttons = fixture.nativeElement.querySelectorAll('button');
      buttons.forEach((button: any) => {
        expect(button).toBeTruthy();
        expect(button.textContent.trim()).not.toBe('');
      });
    });

    it('should have proper heading structure', () => {
      fixture.detectChanges();
      
      const headings = fixture.nativeElement.querySelectorAll('h1, h2');
      expect(headings.length).toBeGreaterThan(0);
    });
  });

  describe('Error Handling', () => {
    it('should handle notification service errors gracefully', () => {
      notificationService.success.and.throwError('Service error');
      
      expect(() => component.showSuccessNotification()).toThrow();
    });

    it('should handle action function errors gracefully', () => {
      component.showSuccessNotification();
      
      const callArgs = notificationService.success.calls.mostRecent().args;
      const action = callArgs[2].actions[0].action;
      
      spyOn(console, 'log').and.throwError('Action error');
      
      expect(() => action()).toThrow();
    });
  });

  describe('Performance', () => {
    it('should render quickly', () => {
      const startTime = performance.now();
      fixture.detectChanges();
      const endTime = performance.now();
      
      expect(endTime - startTime).toBeLessThan(100); // Should render quickly
    });

    it('should handle multiple rapid button clicks', () => {
      fixture.detectChanges();
      
      const successButton = fixture.nativeElement.querySelector('button[color="primary"]');
      
      // Rapid clicks
      for (let i = 0; i < 10; i++) {
        successButton.click();
      }
      
      expect(notificationService.success).toHaveBeenCalledTimes(10);
    });
  });
}); 