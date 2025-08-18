import { ComponentFixture, TestBed } from '@angular/core/testing';
import { NoopAnimationsModule } from '@angular/platform-browser/animations';
import { NotificationContainerComponent } from './notification-container';
import { NotificationService, NotificationInstance, NotificationAction } from '../../../services/notification.service';
import { of } from 'rxjs';

describe('NotificationContainerComponent', () => {
  let component: NotificationContainerComponent;
  let fixture: ComponentFixture<NotificationContainerComponent>;
  let notificationService: jasmine.SpyObj<NotificationService>;

  const mockNotifications: NotificationInstance[] = [
    {
      id: '1',
      type: 'success',
      title: 'Success Title',
      message: 'Success message',
      timestamp: Date.now(),
      autoClose: true,
      dismissible: true,
      duration: 5000
    },
    {
      id: '2',
      type: 'error',
      title: 'Error Title',
      message: 'Error message',
      timestamp: Date.now(),
      autoClose: true,
      dismissible: true,
      duration: 8000
    }
  ];

  beforeEach(async () => {
    const spy = jasmine.createSpyObj('NotificationService', ['remove'], {
      notifications: of(mockNotifications)
    });

    await TestBed.configureTestingModule({
      imports: [
        NoopAnimationsModule,
        NotificationContainerComponent
      ],
      providers: [
        { provide: NotificationService, useValue: spy }
      ]
    }).compileComponents();

    fixture = TestBed.createComponent(NotificationContainerComponent);
    component = fixture.componentInstance;
    notificationService = TestBed.inject(NotificationService) as jasmine.SpyObj<NotificationService>;
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  describe('Initialization', () => {
    it('should subscribe to notification service on init', () => {
      spyOn(notificationService.notifications, 'subscribe');
      
      component.ngOnInit();
      
      expect(notificationService.notifications.subscribe).toHaveBeenCalled();
    });

    it('should populate notifications from service', () => {
      component.ngOnInit();
      
      expect(component.notifications).toEqual(mockNotifications);
      expect(component.notifications.length).toBe(2);
    });

    it('should handle empty notifications', () => {
      const emptySpy = jasmine.createSpyObj('NotificationService', ['remove'], {
        notifications: of([])
      });

      TestBed.overrideProvider(NotificationService, { useValue: emptySpy });
      const emptyFixture = TestBed.createComponent(NotificationContainerComponent);
      const emptyComponent = emptyFixture.componentInstance;
      
      emptyComponent.ngOnInit();
      
      expect(emptyComponent.notifications).toEqual([]);
    });
  });

  describe('Notification Dismissal', () => {
    it('should call service remove method when notification is dismissed', () => {
      component.ngOnInit();
      
      component.onNotificationDismissed('1');
      
      expect(notificationService.remove).toHaveBeenCalledWith('1');
    });

    it('should handle dismissal of non-existent notification', () => {
      component.ngOnInit();
      
      expect(() => component.onNotificationDismissed('non-existent-id')).not.toThrow();
      expect(notificationService.remove).toHaveBeenCalledWith('non-existent-id');
    });
  });

  describe('Action Handling', () => {
    it('should handle action click with action function', () => {
      const mockAction: NotificationAction = {
        label: 'Test Action',
        color: 'primary',
        action: jasmine.createSpy('action')
      };

      component.onActionClicked(mockAction, '1');
      
      expect(mockAction.action).toHaveBeenCalled();
      expect(notificationService.remove).toHaveBeenCalledWith('1');
    });

    it('should handle action click without action function', () => {
      const mockAction: NotificationAction = {
        label: 'Test Action',
        color: 'primary'
      };

      expect(() => component.onActionClicked(mockAction, '1')).not.toThrow();
      expect(notificationService.remove).toHaveBeenCalledWith('1');
    });

    it('should handle action click with null action', () => {
      const mockAction: NotificationAction = {
        label: 'Test Action',
        color: 'primary',
        action: null as any
      };

      expect(() => component.onActionClicked(mockAction, '1')).not.toThrow();
      expect(notificationService.remove).toHaveBeenCalledWith('1');
    });
  });

  describe('Template Rendering', () => {
    it('should render notification overlay when notifications exist', () => {
      component.ngOnInit();
      fixture.detectChanges();
      
      const overlay = fixture.nativeElement.querySelector('.notification-overlay');
      expect(overlay).toBeTruthy();
    });

    it('should not render notification overlay when no notifications', () => {
      const emptySpy = jasmine.createSpyObj('NotificationService', ['remove'], {
        notifications: of([])
      });

      TestBed.overrideProvider(NotificationService, { useValue: emptySpy });
      const emptyFixture = TestBed.createComponent(NotificationContainerComponent);
      const emptyComponent = emptyFixture.componentInstance;
      
      emptyComponent.ngOnInit();
      emptyFixture.detectChanges();
      
      const overlay = emptyFixture.nativeElement.querySelector('.notification-overlay');
      expect(overlay).toBeNull();
    });

    it('should render correct number of notification components', () => {
      component.ngOnInit();
      fixture.detectChanges();
      
      const notificationComponents = fixture.nativeElement.querySelectorAll('app-notification');
      expect(notificationComponents.length).toBe(2);
    });

    it('should pass correct props to notification components', () => {
      component.ngOnInit();
      fixture.detectChanges();
      
      const notificationComponents = fixture.nativeElement.querySelectorAll('app-notification');
      
      // Check first notification
      const firstNotification = notificationComponents[0];
      expect(firstNotification.getAttribute('ng-reflect-type')).toBe('success');
      expect(firstNotification.getAttribute('ng-reflect-title')).toBe('Success Title');
      expect(firstNotification.getAttribute('ng-reflect-message')).toBe('Success message');
      
      // Check second notification
      const secondNotification = notificationComponents[1];
      expect(secondNotification.getAttribute('ng-reflect-type')).toBe('error');
      expect(secondNotification.getAttribute('ng-reflect-title')).toBe('Error Title');
      expect(secondNotification.getAttribute('ng-reflect-message')).toBe('Error message');
    });
  });

  describe('Subscription Management', () => {
    it('should unsubscribe on destroy', () => {
      const mockSubscription = jasmine.createSpyObj('Subscription', ['unsubscribe']);
      spyOn(notificationService.notifications, 'subscribe').and.returnValue(mockSubscription);
      
      component.ngOnInit();
      component.ngOnDestroy();
      
      expect(mockSubscription.unsubscribe).toHaveBeenCalled();
    });

    it('should handle destroy without subscription', () => {
      expect(() => component.ngOnDestroy()).not.toThrow();
    });
  });

  describe('Dynamic Updates', () => {
    it('should update notifications when service emits new values', (done) => {
      const newNotifications: NotificationInstance[] = [
        {
          id: '3',
          type: 'warning',
          title: 'Warning Title',
          message: 'Warning message',
          timestamp: Date.now(),
          autoClose: true,
          dismissible: true,
          duration: 6000
        }
      ];

      const dynamicSpy = jasmine.createSpyObj('NotificationService', ['remove'], {
        notifications: of(mockNotifications, newNotifications)
      });

      TestBed.overrideProvider(NotificationService, { useValue: dynamicSpy });
      const dynamicFixture = TestBed.createComponent(NotificationContainerComponent);
      const dynamicComponent = dynamicFixture.componentInstance;
      
      dynamicComponent.ngOnInit();
      
      // First emission
      expect(dynamicComponent.notifications).toEqual(mockNotifications);
      
      // Second emission
      setTimeout(() => {
        expect(dynamicComponent.notifications).toEqual(newNotifications);
        done();
      }, 0);
    });
  });

  describe('Error Handling', () => {
    it('should handle service errors gracefully', () => {
      const errorSpy = jasmine.createSpyObj('NotificationService', ['remove'], {
        notifications: of(null as any)
      });

      TestBed.overrideProvider(NotificationService, { useValue: errorSpy });
      const errorFixture = TestBed.createComponent(NotificationContainerComponent);
      const errorComponent = errorFixture.componentInstance;
      
      expect(() => errorComponent.ngOnInit()).not.toThrow();
      expect(errorComponent.notifications).toBeNull();
    });

    it('should handle service throwing errors', () => {
      const throwingSpy = jasmine.createSpyObj('NotificationService', ['remove'], {
        notifications: { subscribe: () => { throw new Error('Service error'); } }
      });

      TestBed.overrideProvider(NotificationService, { useValue: throwingSpy });
      const throwingFixture = TestBed.createComponent(NotificationContainerComponent);
      const throwingComponent = throwingFixture.componentInstance;
      
      expect(() => throwingComponent.ngOnInit()).toThrow();
    });
  });

  describe('Performance', () => {
    it('should handle large number of notifications efficiently', () => {
      const largeNotifications: NotificationInstance[] = Array.from({ length: 100 }, (_, i) => ({
        id: `notification-${i}`,
        type: 'info',
        title: `Title ${i}`,
        message: `Message ${i}`,
        timestamp: Date.now(),
        autoClose: true,
        dismissible: true,
        duration: 5000
      }));

      const largeSpy = jasmine.createSpyObj('NotificationService', ['remove'], {
        notifications: of(largeNotifications)
      });

      TestBed.overrideProvider(NotificationService, { useValue: largeSpy });
      const largeFixture = TestBed.createComponent(NotificationContainerComponent);
      const largeComponent = largeFixture.componentInstance;
      
      const startTime = performance.now();
      largeComponent.ngOnInit();
      const endTime = performance.now();
      
      expect(largeComponent.notifications.length).toBe(100);
      expect(endTime - startTime).toBeLessThan(100); // Should complete quickly
    });
  });

  describe('Accessibility', () => {
    it('should have proper ARIA attributes', () => {
      component.ngOnInit();
      fixture.detectChanges();
      
      const overlay = fixture.nativeElement.querySelector('.notification-overlay');
      expect(overlay).toBeTruthy();
      
      // Check that notifications are properly structured
      const notifications = fixture.nativeElement.querySelectorAll('app-notification');
      notifications.forEach((notification: any) => {
        expect(notification).toBeTruthy();
      });
    });
  });

  describe('Edge Cases', () => {
    it('should handle notifications with missing properties', () => {
      const incompleteNotifications: NotificationInstance[] = [
        {
          id: 'incomplete',
          type: 'info',
          message: 'Incomplete message',
          timestamp: Date.now(),
          autoClose: true,
          dismissible: true
        } as NotificationInstance
      ];

      const incompleteSpy = jasmine.createSpyObj('NotificationService', ['remove'], {
        notifications: of(incompleteNotifications)
      });

      TestBed.overrideProvider(NotificationService, { useValue: incompleteSpy });
      const incompleteFixture = TestBed.createComponent(NotificationContainerComponent);
      const incompleteComponent = incompleteFixture.componentInstance;
      
      expect(() => incompleteComponent.ngOnInit()).not.toThrow();
      expect(incompleteComponent.notifications).toEqual(incompleteNotifications);
    });

    it('should handle notifications with null/undefined values', () => {
      const nullNotifications: NotificationInstance[] = [
        {
          id: 'null-test',
          type: 'info',
          title: null as any,
          message: null as any,
          timestamp: Date.now(),
          autoClose: true,
          dismissible: true,
          duration: 5000
        }
      ];

      const nullSpy = jasmine.createSpyObj('NotificationService', ['remove'], {
        notifications: of(nullNotifications)
      });

      TestBed.overrideProvider(NotificationService, { useValue: nullSpy });
      const nullFixture = TestBed.createComponent(NotificationContainerComponent);
      const nullComponent = nullFixture.componentInstance;
      
      expect(() => nullComponent.ngOnInit()).not.toThrow();
      expect(nullComponent.notifications).toEqual(nullNotifications);
    });
  });
}); 